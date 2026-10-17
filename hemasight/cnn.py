"""Pretrained ResNet-18 feature extractor in pure numpy.

PyTorch has no wheels for IBM Z (s390x), so this module reads the official
torchvision checkpoint (resnet18-f37072fd.pth) directly - it is a zip of
pickled tensor metadata plus little-endian float32 blobs - and runs the
forward pass with numpy. It works identically on x86 and s390x (big-endian).

Features: tissue-masked global average pooling of layer2, layer3 and layer4
(128 + 256 + 512 = 896 values), so the background of a cut-out never counts.
"""
from __future__ import annotations

import pickle
import zipfile

import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------- checkpoint loading without torch ----------
class _Storage:
    def __init__(self, data):
        self.data = data


def _rebuild_tensor(storage, offset, size, stride, *args):
    a = storage.data
    if len(size) == 0:
        return a[offset:offset + 1].reshape(())
    item = a.itemsize
    return np.lib.stride_tricks.as_strided(a[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def load_state_dict(path: str) -> dict:
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]
    dtypes = {"FloatStorage": "<f4", "LongStorage": "<i8", "HalfStorage": "<f2",
              "IntStorage": "<i4", "DoubleStorage": "<f8"}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if mod.startswith("torch") and name in dtypes:
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            raise pickle.UnpicklingError(f"refusing to load {mod}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _, _ = pid
            raw = zf.read(f"{root}/data/{key}")
            return _Storage(np.frombuffer(raw, dtype=dtypes[stype]).astype(np.float32))

    with zf.open(f"{root}/data.pkl") as f:
        sd = U(f).load()
    return {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}


# ---------- numpy layers ----------
def _conv(x, w, stride=1, pad=0):
    """x: C,H,W  w: O,C,k,k -> O,H',W' via im2col + matmul."""
    C, H, W = x.shape
    O, _, k, _ = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (x.shape[1] - k) // stride + 1
    Wo = (x.shape[2] - k) // stride + 1
    s0, s1, s2 = x.strides
    cols = np.lib.stride_tricks.as_strided(
        x, shape=(C, k, k, Ho, Wo), strides=(s0, s1, s2, s1 * stride, s2 * stride))
    cols = cols.reshape(C * k * k, Ho * Wo)
    return (w.reshape(O, -1) @ cols).reshape(O, Ho, Wo)


def _bn(x, p, name, eps=1e-5):
    g, b = p[name + ".weight"], p[name + ".bias"]
    m, v = p[name + ".running_mean"], p[name + ".running_var"]
    scale = g / np.sqrt(v + eps)
    return x * scale[:, None, None] + (b - m * scale)[:, None, None]


def _maxpool(x, k=3, s=2, pad=1):
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), constant_values=-np.inf)
    C, H, W = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    s0, s1, s2 = x.strides
    v = np.lib.stride_tricks.as_strided(x, shape=(C, Ho, Wo, k, k),
                                        strides=(s0, s1 * s, s2 * s, s1, s2))
    return v.max(axis=(3, 4))


def _block(x, p, name, stride):
    out = _conv(x, p[name + ".conv1.weight"], stride, 1)
    out = np.maximum(_bn(out, p, name + ".bn1"), 0)
    out = _conv(out, p[name + ".conv2.weight"], 1, 1)
    out = _bn(out, p, name + ".bn2")
    if name + ".downsample.0.weight" in p:
        x = _bn(_conv(x, p[name + ".downsample.0.weight"], stride, 0), p, name + ".downsample.1")
    return np.maximum(out + x, 0)


class ResNet18Features:
    SIZE = 224

    def __init__(self, weights_path: str):
        self.p = load_state_dict(weights_path)

    def _prep(self, path):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
        im.thumbnail((self.SIZE, self.SIZE))
        canvas = Image.new("RGBA", (self.SIZE, self.SIZE), (0, 0, 0, 0))
        canvas.paste(im, ((self.SIZE - im.width) // 2, (self.SIZE - im.height) // 2))
        a = np.asarray(canvas, dtype=np.float32) / 255.0
        mask = a[..., 3] > 0.5
        rgb = (a[..., :3] - MEAN) / STD
        rgb[~mask] = 0.0  # background = ImageNet mean colour
        return rgb.transpose(2, 0, 1).copy(), mask

    @staticmethod
    def _masked_gap(fmap, mask):
        C, H, W = fmap.shape
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 64
        if m.sum() == 0:
            m[:] = True
        return fmap[:, m].mean(axis=1)

    def __call__(self, path: str) -> np.ndarray:
        p = self.p
        x, mask = self._prep(path)
        x = np.maximum(_bn(_conv(x, p["conv1.weight"], 2, 3), p, "bn1"), 0)
        x = _maxpool(x)
        feats = []
        for li, stride in ((1, 1), (2, 2), (3, 2), (4, 2)):
            x = _block(x, p, f"layer{li}.0", stride)
            x = _block(x, p, f"layer{li}.1", 1)
            if li >= 2:
                feats.append(self._masked_gap(x, mask))
        return np.concatenate(feats).astype(np.float32)

    def logits(self, path: str) -> np.ndarray:
        """ImageNet class scores (used only to verify the port)."""
        p = self.p
        x, _ = self._prep(path)
        x = np.maximum(_bn(_conv(x, p["conv1.weight"], 2, 3), p, "bn1"), 0)
        x = _maxpool(x)
        for li, stride in ((1, 1), (2, 2), (3, 2), (4, 2)):
            x = _block(x, p, f"layer{li}.0", stride)
            x = _block(x, p, f"layer{li}.1", 1)
        return p["fc.weight"] @ x.mean(axis=(1, 2)) + p["fc.bias"]
