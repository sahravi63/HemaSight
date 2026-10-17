# HemaSight

**Needle-free anaemia screening from phone photos, trained and served on IBM LinuxONE.**

Needle-free anaemia screening from photos of the inner eyelid, fingernails and palm.
Team zeroBytes, IBM Z Datathon 2026 (Wildcard: Beyond the Obvious).

## How it works

1. **Colour features** (`hemasight/features.py`): tissue pixels are described in RGB,
   chromaticity, CIELAB, HSV and the erythema index. Pale (anaemic) tissue has lower
   redness (a*) and red/green contrast.
2. **Leakage-free grouping** (`hemasight/data.py`): flipped or rotated copies of the
   same photo get the same group, so they never sit in both train and test.
3. **Per-site models** (`train.py`): grouped, stratified 5-fold cross-validation of
   logistic regression, random forest, extra trees, gradient boosting and, when
   available, IBM Snap ML. The best model by AUC is kept, with a threshold chosen to
   catch at least 90% of anaemic cases.
4. **Fusion and decision** (`hemasight/predict.py`): each site votes by its margin
   above or below its own threshold. Disagreement or a small margin gives an
   "uncertain" result, which means retake the photo or refer for a blood test.
5. **Haemoglobin estimate** (`train_hb.py`): 710 eyelid photos (CP-AnemiC, `Image_001`–`Image_710`)
   are matched to lab Hb values in `Anemia_Data_Collection_Sheet.xlsx`, and a regressor learns
   Hb in g/dL. Reported against a predict-the-average baseline.
6. **Demo app** (`app.py`): upload photos and get the screening decision in a browser.

7. **Camera robustness** (`hemasight/robust.py`): every model is trained on colour-jittered copies
   of each photo and tested on simulated new cameras; models are chosen by robustness, not best case.
   See `RESULTS.md`.

Everything uses numpy, Pillow and scikit-learn, so it runs natively on IBM LinuxONE (s390x).
`hemasight/cnn.py` is a pure-numpy ResNet-18 (PyTorch has no s390x build), used in the experiments
that showed colour features beat deep features for this task.

## Results (measured on IBM LinuxONE)

| Model | Validation | Result |
|---|---|---|
| Eyelid Hb (5-seed ensemble) | 10 hospitals, leave-one-hospital-out | **AUC 0.93**, MAE 1.39 g/dL; 90% sensitivity at 79% specificity |
| Eyelid Hb on a simulated new camera | ±4% / ±8% colour shift | AUC 0.91 / 0.86 |
| Palm (5-seed ensemble) | grouped 5-fold | AUC 0.88 |
| Fingernails (supporting only) | grouped 5-fold | AUC 0.78 |

Full details, experiments and limitations: [`RESULTS.md`](RESULTS.md).

## Run on IBM LinuxONE (s390x)

```bash
bash setup_server.sh                          # distro packages + venv (+ IBM Snap ML if available)
source ~/hemasight-venv/bin/activate

# deployed models: 5-seed ensembles (about 1 hour on 3 CPUs)
python3 train_seeds.py --data ~/data --out models_seeds --seeds 5
python3 app.py --models models_seeds --port 5000      # open http://<server-ip>:5000

# single-seed training and the design experiments
python3 train_hb.py --data ~/data --out models
python3 train.py    --data ~/data --out models
wget -q https://download.pytorch.org/models/resnet18-f37072fd.pth
python3 experiments.py --data ~/data --out models --resnet resnet18-f37072fd.pth
```

## Data

The image datasets are **not included** in this repository; they are licensed for academic use.
Download them from Mendeley Data and place them in one folder (`~/data`):

- CP-AnemiC conjunctiva with lab Hb: 10.17632/m53vz6b7fx.1 (`Image_###.png` + `Anemia_Data_Collection_Sheet.xlsx`)
- Conjunctiva images, Ghana: 10.17632/nt7r8hv2pz.1
- Palm images: 10.17632/ccr8cm22vz.1
- Fingernail images: 10.17632/2xx4j3kjg2.1

Folders are labelled by name: a folder or file name containing `Anemic` / `Non-anemic`, and a site
folder containing `Conjunctiva`/`eye`, `Fingernail`/`nail` or `Palm`.

## Disclaimer

HemaSight is a research prototype and screening aid, not a diagnostic device. Any flagged, uncertain
or severe result needs a confirmatory laboratory blood test.

Team zeroBytes · IBM Z Datathon 2026
