"""Builds scripts/zenodo_release.ipynb. Usage: python build_zenodo_nb.py"""
import json, os

cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md(r'''
# Zenodo release: build the zip files from Drive, then upload them  (CPU runtime is enough)

Run in the Google account whose Drive holds `MyDrive/SSL_AgriStudy/`. Nothing under `SSL_AgriStudy/` is changed.

**Part A (cells 1–4, ~15–40 min)** writes to a NEW Drive folder `MyDrive/ssl-agri-scale-zenodo/`:

| File | Contents |
|---|---|
| `ssl-agri-scale-checkpoints.zip` | the teacher backbone of every finished laboratory-scale DINO ViT-S/16 run (`<run>_teacher_backbone.pth`, ~87 MB each), `checkpoint_manifest.csv`, `README.txt` |
| `checkpoint_manifest.csv` | SHA-256, size, final epoch and training arguments of every backbone (also inside the zip) |
| `ssl-agri-scale-results.zip` | splits, result JSONs, per-image correctness (`correct/`), paper tables and figures |

**Part B (cells 5–8, ~10–30 min)** creates a Zenodo **draft**, reserves its DOI, uploads the three files and checks
every upload against its MD5. It never publishes: you review the draft on zenodo.org and press *Publish* yourself.
Part B reads the files from the Drive folder, so it also works in a fresh runtime after a disconnect
(run cell 1, then cells 5–8).
''')

code(r'''
# ── 1. Setup ──
import os, sys, json, shutil, hashlib, glob, zipfile, time
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
import torch, pandas as pd
BASE = os.environ.get('SSL_BASE', '/content/drive/MyDrive/SSL_AgriStudy')
REL  = os.environ.get('SSL_RELEASE', '/content/drive/MyDrive/ssl-agri-scale-zenodo')   # output folder on Drive
WORK = os.environ.get('SSL_WORK', '/content/zenodo_work')                              # local scratch
RES  = f'{BASE}/results_v2'
assert os.path.isdir(RES), f'{RES} not found — wrong Google account or folder?'
os.makedirs(REL, exist_ok=True)
def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()
def md5(path, buf=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()
print('source:', BASE, '\noutput:', REL)
''')

code(r'''
# ── 2. Results zip (Part A) ──
shutil.rmtree(WORK, ignore_errors=True)
R = f'{WORK}/results'; os.makedirs(R)
for f in sorted(glob.glob(f'{RES}/*.json')):
    shutil.copy(f, R)
for d in sorted(glob.glob(f'{RES}/*_runs')) + [f'{RES}/splits', f'{RES}/correct']:
    if os.path.isdir(d):
        shutil.copytree(d, f'{R}/{os.path.basename(d)}')
for src, dst in [(f'{BASE}/paper_v2/tables', 'paper_tables'), (f'{BASE}/paper_v2/figures', 'paper_figures')]:
    if os.path.isdir(src):
        shutil.copytree(src, f'{R}/{dst}')
pd_info = f'{BASE}/datasets/PlantDoc/SPLIT_INFO.json'
if os.path.exists(pd_info):
    os.makedirs(f'{R}/splits', exist_ok=True); shutil.copy(pd_info, f'{R}/splits/PlantDoc_SPLIT_INFO.json')
assert os.path.isdir(f'{R}/correct'), 'results_v2/correct/ missing — per-image predictions are needed for the paired tests'
n = sum(len(fs) for _, _, fs in os.walk(R))
zp = f'{WORK}/ssl-agri-scale-results.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for p, _, fs in os.walk(R):
        for f in fs:
            z.write(os.path.join(p, f), os.path.join('results', os.path.relpath(os.path.join(p, f), R)))
print(f'✓ results zip: {n} files, {os.path.getsize(zp) / 1e6:.1f} MB')
''')

code(r'''
# ── 3. Teacher backbones + manifest (Part A) ──
runs = json.load(open(f'{RES}/08_10_pretraining.json')) if os.path.exists(f'{RES}/08_10_pretraining.json') else {}
for f in glob.glob(f'{RES}/08_10_pretraining_runs/*.json'):
    try:
        rec = json.load(open(f)); runs[rec['__key']] = rec['record']
    except Exception:
        pass
CK = f'{WORK}/checkpoints'; os.makedirs(CK, exist_ok=True)
rows, skipped = [], []
for sub in ('domain_dino', 'domain_dino_v2'):
    for d in sorted(glob.glob(f'{BASE}/checkpoints/{sub}/*')):
        name, ck = os.path.basename(d), f'{d}/checkpoint.pth'
        complete = os.path.exists(f'{d}/COMPLETE') or sub == 'domain_dino'
        if not os.path.exists(ck) or not complete:
            skipped.append(name); print(f'skip {name}: ' + ('no checkpoint' if not os.path.exists(ck) else 'not COMPLETE')); continue
        t0 = time.time()
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        ep = sd.get('epoch') if isinstance(sd, dict) else None
        teacher = sd['teacher'] if isinstance(sd, dict) and 'teacher' in sd else sd
        bb = {k.replace('module.', '').replace('backbone.', ''): v for k, v in teacher.items()
              if not k.replace('module.', '').replace('backbone.', '').startswith('head')}
        assert len(bb) == 150, f'{name}: expected 150 ViT-S/16 backbone tensors, got {len(bb)}'
        out = f'{CK}/{name}_teacher_backbone.pth'
        torch.save(bb, out)
        r = runs.get(name, {})
        rows.append({'run': name, 'folder': sub, 'file': os.path.basename(out), 'sha256': sha256(out),
                     'size_MB': round(os.path.getsize(out) / 1e6, 1), 'final_epoch': ep,
                     'n_images': r.get('n_images'), 'steps': r.get('steps'), 'init': r.get('init'),
                     'lr': r.get('lr'), 'epochs': r.get('epochs'), 'seed': r.get('seed'),
                     'extra_args': ' '.join(r.get('extra_args', []))})
        print(f'✓ {name:28s} epoch {ep}  {rows[-1]["size_MB"]} MB  ({time.time() - t0:.0f}s)')
        del sd, teacher, bb
if not rows:
    raise RuntimeError('No finished checkpoints under checkpoints/domain_dino*/ — wrong Drive account?')
M = pd.DataFrame(rows); M.to_csv(f'{CK}/checkpoint_manifest.csv', index=False)
print(f'\n{len(M)} backbones ({M.size_MB.sum() / 1e3:.2f} GB); skipped {len(skipped)}: {skipped}')
if len(M) != 27:
    print(f'⚠ The paper reports 27 laboratory-scale checkpoints; {len(M)} were found. Check the skipped runs before uploading.')
open(f'{CK}/README.txt', 'w').write("""Teacher backbones (DINO ViT-S/16, patch 16, 384-d CLS) of the laboratory-scale runs in
"Domain Match Helps, Scale Decides" (IEEE Access). Code: https://github.com/munna237/munna237-SSL-AgriStudy

Each <run>_teacher_backbone.pth is a plain state_dict (150 tensors, no DINO head) loadable with
    import torch, vision_transformer as vits      # from the official DINO repo, commit 7c446df
    m = vits.vit_small(patch_size=16, num_classes=0)
    m.load_state_dict(torch.load("<run>_teacher_backbone.pth", map_location="cpu"), strict=True)
checkpoint_manifest.csv lists each file's SHA-256, final epoch, image count, steps, initialization and arguments.
Licence: CC BY 4.0.
""")
''')

code(r'''
# ── 4. Checkpoint zip + copy everything to Drive (Part A) ──
# .pth weights do not compress, so the zip is stored (fast) rather than deflated.
zc = f'{WORK}/ssl-agri-scale-checkpoints.zip'
with zipfile.ZipFile(zc, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
    for f in sorted(os.listdir(CK)):
        z.write(f'{CK}/{f}', f'checkpoints/{f}')
with zipfile.ZipFile(zc) as z:                       # integrity check before anything leaves the runtime
    bad = z.testzip(); assert bad is None, f'corrupt member {bad}'
    assert len([n for n in z.namelist() if n.endswith('.pth')]) == len(M)
for f in [zc, f'{WORK}/ssl-agri-scale-results.zip', f'{CK}/checkpoint_manifest.csv']:
    shutil.copy(f, REL)
    assert md5(f) == md5(f'{REL}/{os.path.basename(f)}'), f'copy to Drive failed for {f}'
    print(f'✓ {os.path.basename(f):34s} {os.path.getsize(f) / 1e6:9.1f} MB  → {REL}')
print('\nPart A done. Part B (Zenodo upload) can run now or later in a fresh runtime.')
''')

md(r'''
## Part B — upload to Zenodo

**Before running cell 5 (2 minutes, once):**
1. Log in at <https://zenodo.org> (GitHub or ORCID login is fine).
2. Avatar (top right) → **Applications** → *Personal access tokens* → **New token**. Name it `ssl-agri-scale`,
   tick **`deposit:write`** and **`deposit:actions`**, press *Create*, copy the token (it is shown only once).
3. In Colab: the **key icon** (Secrets) in the left bar → *Add new secret* → name `ZENODO_TOKEN`, paste the token,
   switch *Notebook access* on.

Want a rehearsal first? Set `SANDBOX = True` (needs a separate account and token at <https://sandbox.zenodo.org>,
saved as the secret `ZENODO_SANDBOX_TOKEN`); sandbox DOIs are not real.
''')

code(r'''
# ── 5. Zenodo connection + metadata ──
import requests
SANDBOX = False
HOST = 'https://sandbox.zenodo.org' if SANDBOX else 'https://zenodo.org'
if IN_COLAB:
    from google.colab import userdata
    TOKEN = userdata.get('ZENODO_SANDBOX_TOKEN' if SANDBOX else 'ZENODO_TOKEN')
else:
    TOKEN = os.environ['ZENODO_TOKEN']
P = {'access_token': TOKEN}
r = requests.get(f'{HOST}/api/deposit/depositions', params=P, timeout=60)
assert r.status_code == 200, f'token rejected ({r.status_code}): {r.text[:300]} — needs deposit:write and deposit:actions'
print('✓ token works;', len(r.json()), 'existing uploads in this account')

FILES = [f'{REL}/ssl-agri-scale-checkpoints.zip', f'{REL}/checkpoint_manifest.csv', f'{REL}/ssl-agri-scale-results.zip']
for f in FILES:
    assert os.path.exists(f), f'{f} missing — run Part A first'

NSU = 'North South University, Dhaka, Bangladesh'
CREATORS = [
    {'name': 'Munna, Mohammad Mehedi Hasan', 'affiliation': NSU, 'orcid': '0009-0000-0860-9337'},
    {'name': 'Kabir, Acramul Haque',         'affiliation': NSU, 'orcid': '0000-0002-4899-5786'},
    {'name': 'Tulona, Reefka Fabliha',       'affiliation': 'Independent University, Bangladesh, Dhaka, Bangladesh', 'orcid': '0009-0002-3557-0938'},
    {'name': 'Zahid, Tanvir Bin',            'affiliation': NSU, 'orcid': '0009-0007-2595-3720'},
    {'name': 'Antu, Nahid Hossain',          'affiliation': NSU, 'orcid': '0009-0007-5924-6901'},
    {'name': 'ul Mulk, Md. Mansur',          'affiliation': NSU, 'orcid': '0009-0000-0743-3309'},
    {'name': 'Chowdhury, Arvid',             'affiliation': NSU},
]
n_ck = len(pd.read_csv(f'{REL}/checkpoint_manifest.csv'))
META = {'metadata': {
    'title': 'Laboratory-scale DINO ViT-S/16 checkpoints and per-image results for "Domain Match Helps, Scale Decides"',
    'upload_type': 'dataset',
    'description': (
        f'<p>Teacher backbones of the {n_ck} laboratory-scale DINO ViT-S/16 pretraining runs, and the per-image '
        'evaluation results, of the study "Domain Match Helps, Scale Decides: A Budget-Matched Study of Laboratory-Scale '
        'Self-Supervised Pretraining and Near-Duplicate Inflation in Agricultural and Botanical Recognition" (IEEE Access, '
        'under review).</p>'
        '<ul><li><b>ssl-agri-scale-checkpoints.zip</b>: one <code>&lt;run&gt;_teacher_backbone.pth</code> per run '
        '(plain ViT-S/16 state_dict, no DINO head), README.txt with loading code.</li>'
        '<li><b>checkpoint_manifest.csv</b>: SHA-256, final epoch, image count, optimizer steps, initialization and '
        'training arguments of every file.</li>'
        '<li><b>ssl-agri-scale-results.zip</b>: split files, result JSONs, per-image correctness for every backbone, '
        'dataset and protocol (needed for the paired bootstrap and McNemar tests), and the paper tables and figures.</li></ul>'
        '<p>Code and instructions: <a href="https://github.com/munna237/munna237-SSL-AgriStudy">'
        'github.com/munna237/munna237-SSL-AgriStudy</a>. Images are not redistributed; FloraBD and RiceLeafDiseaseBD are '
        'on Mendeley Data, the other datasets at their original sources.</p>'),
    'creators': CREATORS,
    'access_right': 'open',
    'license': 'cc-by-4.0',
    'keywords': ['self-supervised learning', 'DINO', 'vision transformer', 'agricultural computer vision',
                 'plant disease recognition', 'near-duplicate detection', 'pretrained models', 'Bangladesh'],
    'related_identifiers': [
        {'identifier': 'https://github.com/munna237/munna237-SSL-AgriStudy', 'relation': 'isSupplementTo', 'scheme': 'url'},
        {'identifier': '10.17632/r9whm6nz8n.2', 'relation': 'isDerivedFrom', 'scheme': 'doi'},
        {'identifier': '10.17632/86s4jzj2m4.2', 'relation': 'isDerivedFrom', 'scheme': 'doi'},
    ],
    'version': '1.0',
    'prereserve_doi': True,
}}
print(json.dumps({k: META['metadata'][k] for k in ('title', 'upload_type', 'license', 'version')}, indent=1))
print(f'{len(CREATORS)} creators; files:', [os.path.basename(f) for f in FILES])
''')

code(r'''
# ── 6. Create the draft (or reuse the one made earlier) and reserve the DOI ──
state = f'{REL}/zenodo_draft{"_sandbox" if SANDBOX else ""}.json'
dep = None
if os.path.exists(state):                            # re-run after a disconnect: keep the same draft and DOI
    old = json.load(open(state))
    r = requests.get(f'{HOST}/api/deposit/depositions/{old["id"]}', params=P, timeout=60)
    if r.status_code == 200 and not r.json().get('submitted'):
        dep = r.json(); print('reusing draft', dep['id'])
if dep is None:
    r = requests.post(f'{HOST}/api/deposit/depositions', params=P, json={}, timeout=60)
    assert r.status_code == 201, r.text[:500]
    dep = r.json()
r = requests.put(f'{HOST}/api/deposit/depositions/{dep["id"]}', params=P, json=META, timeout=60)
assert r.status_code == 200, f'metadata rejected: {r.text[:800]}'
dep = r.json()
DOI = dep['metadata']['prereserve_doi']['doi']
json.dump({'id': dep['id'], 'doi': DOI, 'bucket': dep['links']['bucket'], 'host': HOST}, open(state, 'w'), indent=1)
print('✓ draft', dep['id'], '\n  reserved DOI:', DOI, '\n  edit/review:', f'{HOST}/uploads/{dep["id"]}')
''')

code(r'''
# ── 7. Upload the files (skips files already uploaded with the right checksum) ──
BUCKET = dep['links']['bucket']
def remote_files():
    r = requests.get(f'{HOST}/api/deposit/depositions/{dep["id"]}/files', params=P, timeout=60)
    return {f['filename']: f['checksum'].replace('md5:', '') for f in r.json()} if r.status_code == 200 else {}
have = remote_files()
for f in FILES:
    name, local = os.path.basename(f), md5(f)
    if have.get(name) == local:
        print(f'= {name} already uploaded'); continue
    for attempt in range(1, 4):
        t0 = time.time()
        with open(f, 'rb') as fh:
            r = requests.put(f'{BUCKET}/{name}', data=fh, params=P, timeout=None)
        if r.status_code in (200, 201) and r.json().get('checksum', '').replace('md5:', '') == local:
            print(f'✓ {name:34s} {os.path.getsize(f) / 1e6:8.1f} MB  md5 ok  ({time.time() - t0:.0f}s)'); break
        print(f'  attempt {attempt} failed ({r.status_code}): {r.text[:200]}'); time.sleep(20)
    else:
        raise RuntimeError(f'{name} did not upload after 3 attempts — re-run this cell; finished files are skipped')
''')

code(r'''
# ── 8. Final check ──
have = remote_files()
ok = all(have.get(os.path.basename(f), '').replace('md5:', '') == md5(f) for f in FILES)
print('files on Zenodo:', list(have)); print('all checksums match:', ok)
print(f"""
DOI (reserved, becomes live when you publish): {DOI}
Review the draft:                              {HOST}/uploads/{dep["id"]}

Next: open the link, check authors/licence/description, then press "Publish".
Send the DOI back so it can go into the paper, README and CITATION.cff.
""")
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
      'language_info': {'name': 'python'}, 'colab': {'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zenodo_release.ipynb')
json.dump(nb, open(out, 'w'), indent=1)
print('wrote', out, len(cells), 'cells')
