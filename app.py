"""HemaSight demo web app.

    python3 app.py --models models --port 5000

Open http://<server-ip>:5000, upload eyelid / nail / palm photos and get a
screening decision. Photos are processed in memory-backed temp files and
deleted straight after scoring.
"""
from __future__ import annotations

import argparse
import os
import tempfile

from flask import Flask, render_template_string, request

from hemasight.predict import SITE_NAMES, Screener

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
SCREENER: Screener | None = None

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HemaSight</title><style>
:root{--ink:#1d1b22;--ink2:#4a4752;--line:#e4e0e6;--red:#a8202b;--rose:#fbeeee;
--teal:#1f7a6d;--tealbg:#e7f3f1;--amber:#b86b0a;--amberbg:#fdf3e3;--soft:#f7f5f6}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
color:var(--ink);background:#fff}
header{background:var(--ink);color:#fff;padding:22px 16px}
header h1{margin:0;font-size:24px}header p{margin:4px 0 0;color:#d9d3db;font-size:14px}
main{max-width:760px;margin:0 auto;padding:20px 16px 40px}
.sites{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.site{border:1px dashed #cfc8d1;border-radius:10px;padding:14px;display:block;background:var(--soft)}
.site b{display:block;margin-bottom:6px}.site small{color:var(--ink2)}
.pick{display:flex;gap:8px;margin-top:10px}
.pbtn{flex:1;text-align:center;background:var(--ink);color:#fff;border-radius:8px;padding:10px 6px;font-size:14px;font-weight:600;cursor:pointer}
.pbtn.alt{background:#fff;color:var(--ink);border:1px solid var(--line)}
.status.ok{color:var(--teal);font-weight:600}
button{margin-top:16px;background:var(--red);color:#fff;border:0;border-radius:8px;
padding:12px 18px;font-size:15px;font-weight:600;cursor:pointer;width:100%}
.result{margin-top:24px;border-radius:12px;padding:18px}
.likely_anaemic{background:var(--rose)}.unlikely{background:var(--tealbg)}.uncertain{background:var(--amberbg)}
.result h2{margin:0 0 6px;font-size:20px}.result p{margin:0;color:var(--ink2)}
table{width:100%;border-collapse:collapse;margin-top:14px;font-size:14px}
td,th{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}
.bar{height:8px;background:#eee;border-radius:5px;position:relative;min-width:120px}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--red)}
.bar s{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink)}
.demo{display:flex;gap:12px;margin-bottom:12px}.demo label{font-size:14px;color:var(--ink2);display:flex;flex-direction:column;gap:4px}
.demo input,.demo select{padding:8px;border:1px solid var(--line);border-radius:6px;font-size:15px;width:140px}
.hb{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;margin:12px 0 4px}.hb b{font-size:34px;letter-spacing:-.02em;white-space:nowrap}
.hb span{color:var(--ink2);font-size:14px}
.needs_outline{background:var(--amberbg)}
.cv{margin-top:10px;display:none}.cv canvas{width:100%;border-radius:6px;cursor:crosshair;touch-action:none;background:#222}
.cv .tools{display:flex;gap:6px;margin-top:6px;align-items:center}.cv .tools small{flex:1}
.cv .tools a{font-size:12px;color:var(--red);cursor:pointer;text-decoration:underline}
.ok{color:var(--teal);font-weight:600}
.note{font-size:12px;color:#7a7682;margin-top:18px}.err{color:var(--red);margin-top:14px}
</style></head><body>
<header><h1>HemaSight</h1><p>Needle-free anaemia screening from photos &middot; running on IBM LinuxONE</p></header>
<main>
<form id="f" method="post" enctype="multipart/form-data">
<div class="demo"><label>Age (months)<input type="number" name="age" min="6" max="59" value="{{ age }}"></label>
<label>Sex<select name="sex"><option value="0" {% if sex==0 %}selected{% endif %}>Female</option>
<option value="1" {% if sex==1 %}selected{% endif %}>Male</option></select></label></div>
<div class="sites">
{% for key, name in sites %}
<div class="site-box" data-site="{{ key }}"><div class="site"><b>{{ name }}</b><small>{{ hints[key] }}</small>
<div class="pick"><label class="pbtn">&#128247; Take photo<input type="file" accept="image/*" capture="environment" hidden></label>
<label class="pbtn alt">&#8682; Upload<input type="file" accept="image/*" hidden></label></div></div>
<div class="cv"><canvas></canvas>
<div class="tools"><small class="status">Tap around the region to outline it</small><a data-act="undo">undo</a><a data-act="clear">clear</a></div></div></div>
{% endfor %}
</div>
<button type="submit">Screen</button>
</form>
{% if error %}<p class="err">{{ error }}</p>{% endif %}
{% if r %}
<div class="result {{ r.decision }}">
<h2>{{ {"likely_anaemic":"Likely anaemic","unlikely":"Anaemia unlikely","uncertain":"Uncertain","needs_outline":"Outline needed"}[r.decision] }}</h2>
<p>{{ r.action }}</p>
{% if r.hb %}<div class="hb"><b>{{ r.hb.hb }} g/dL</b><span>estimated haemoglobin (&plusmn;{{ "%.1f"|format(r.hb.hb_mae) }} typical error)
&middot; indicative WHO grade: {{ r.hb.grade }}</span></div>{% endif %}
<table><tr><th>Site</th><th>Result</th><th></th></tr>
{% for s in r.per_site %}
{% if s.skipped is defined %}
<tr><td>{{ s.name }}</td><td colspan="2">Skipped: {{ s.skipped }}</td></tr>
{% elif s.hb is defined %}
<tr><td>{{ s.name }}</td><td>Hb {{ s.hb }} g/dL{% if s.unusual %} &middot; <b>unusual photo, ignored</b>{% endif %}</td><td>flag below {{ "%.1f"|format(s.screen_cut) }}</td></tr>
{% else %}
<tr><td>{{ s.name }}{% if s.supporting %}<br><small>supporting only</small>{% endif %}</td><td>{{ "%.0f"|format(s.probability*100) }}% anaemia probability{% if s.unusual %} &middot; <b>unusual photo, ignored</b>{% endif %}</td>
<td><div class="bar"><i style="width:{{ (s.probability*100)|round(0) }}%"></i>
<s style="left:{{ (s.threshold*100)|round(0) }}%" title="screening threshold"></s></div></td></tr>
{% endif %}
{% endfor %}</table>
</div>
{% endif %}
<p class="note">Screening aid only, not a diagnosis. Thresholds are set so the models catch at least 90% of anaemic
cases in validation. The Hb estimate was trained on children aged 6&ndash;59 months and tends to under-state severe
anaemia, so every flagged child needs a confirmatory blood test. Photos are deleted right after analysis.</p>
</main>
<script>
// Outline tool: the models were trained on hand-cut regions (transparent background),
// so the browser cuts the outlined polygon out of the photo before upload.
const res = document.querySelector('.result, .err');
if (res) res.scrollIntoView({behavior: 'smooth', block: 'start'});
const state = {};
const MAX_SIDE = 1600;  // phone photos are large; the model only needs a few hundred pixels
document.querySelectorAll('.site-box[data-site]').forEach(siteBox => {
  const site = siteBox.dataset.site, box = siteBox.querySelector('.cv');
  const cv = box.querySelector('canvas'), ctx = cv.getContext('2d'), status = box.querySelector('.status');
  state[site] = {pts: [], img: null, file: null};
  const redraw = () => {
    const st = state[site]; if (!st.img) return;
    ctx.drawImage(st.img, 0, 0, cv.width, cv.height);
    const n = st.pts.length;
    status.textContent = n >= 3 ? `Outlined (${n} points) - ready` : `Tap around the region to outline it (${n} points)`;
    status.classList.toggle('ok', n >= 3);
    if (!n) return;
    ctx.lineWidth = Math.max(2, cv.width / 250); ctx.strokeStyle = '#ffde59'; ctx.fillStyle = 'rgba(255,222,89,.25)';
    ctx.beginPath(); st.pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    if (n > 2) { ctx.closePath(); ctx.fill(); } ctx.stroke();
    st.pts.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], ctx.lineWidth * 1.6, 0, 7); ctx.fillStyle = '#ffde59'; ctx.fill(); });
  };
  siteBox.querySelectorAll('input[type=file]').forEach(inp => inp.addEventListener('change', async () => {
    const f = inp.files[0]; if (!f) return;
    let img = await createImageBitmap(f);
    const sc = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
    if (sc < 1) img = await createImageBitmap(img, {resizeWidth: Math.round(img.width * sc),
                                                    resizeHeight: Math.round(img.height * sc), resizeQuality: 'high'});
    Object.assign(state[site], {img, file: f, pts: []});
    cv.width = img.width; cv.height = img.height; box.style.display = 'block'; redraw();
    inp.value = '';
  }));
  cv.addEventListener('pointerdown', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    state[site].pts.push([(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]);
    redraw();
  });
  box.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
    if (a.dataset.act === 'undo') state[site].pts.pop(); else state[site].pts = [];
    redraw();
  }));
});
const cutout = st => new Promise(res => {
  const xs = st.pts.map(p => p[0]), ys = st.pts.map(p => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs))), y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const w = Math.ceil(Math.max(...xs)) - x0, h = Math.ceil(Math.max(...ys)) - y0;
  const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d');
  g.beginPath(); st.pts.forEach((p, i) => i ? g.lineTo(p[0] - x0, p[1] - y0) : g.moveTo(p[0] - x0, p[1] - y0));
  g.closePath(); g.clip(); g.drawImage(st.img, -x0, -y0); c.toBlob(res, 'image/png');
});
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData();
  fd.append('age', document.querySelector('[name=age]').value);
  fd.append('sex', document.querySelector('[name=sex]').value);
  for (const [site, st] of Object.entries(state)) {
    if (!st.file) continue;
    if (st.pts.length >= 3) fd.append(site, await cutout(st), site + '.png');
    else fd.append(site, await new Promise(res => { const c = document.createElement('canvas');
      c.width = st.img.width; c.height = st.img.height; c.getContext('2d').drawImage(st.img, 0, 0);
      c.toBlob(res, 'image/jpeg', 0.92); }), site + '.jpg');
  }
  const btn = e.target.querySelector('button'); btn.disabled = true; btn.textContent = 'Screening...';
  const html = await (await fetch('/', {method: 'POST', body: fd})).text();
  document.open(); document.write(html); document.close();
});
</script>
</body></html>"""

HINTS = {"eye": "Gently pull the lower eyelid down; photograph the inner red lining.",
         "nail": "Bare nails, good daylight; outline one nail. Supporting evidence only.",
         "palm": "Open palm, good daylight; outline a square patch of palm skin (no fingers or background)."}


@app.route("/", methods=["GET", "POST"])
def index():
    sites = [(k, SITE_NAMES[k]) for k in SCREENER.sites]
    r, error, tmp = None, None, []
    age = request.form.get("age", 24, type=float)
    sex = request.form.get("sex", 0, type=int)
    if request.method == "POST":
        try:
            images = {}
            for key, _ in sites:
                f = request.files.get(key)
                if f and f.filename:
                    fd, path = tempfile.mkstemp(suffix=os.path.splitext(f.filename)[1] or ".jpg")
                    os.close(fd); f.save(path); tmp.append(path); images[key] = path
            if not images:
                error = "Add at least one photo."
            else:
                r = SCREENER.screen(images, age_months=age, male=sex)
        except Exception as e:
            error = f"Could not analyse the photos: {e}"
        finally:
            for p in tmp:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return render_template_string(PAGE, sites=sites, hints=HINTS, r=r, error=error,
                                  age=int(age), sex=sex)


@app.route("/health")
def health():
    return {"status": "ok", "sites": SCREENER.sites, "hb_model": bool(SCREENER.hb)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    SCREENER = Screener(a.models)
    app.run(host=a.host, port=a.port)
