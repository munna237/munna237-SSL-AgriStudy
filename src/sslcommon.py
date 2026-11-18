"""
sslcommon.py — shared, validated code for the SSL_AgriStudy re-runs.

Every notebook writes this exact file and imports it, so all experiments use
ONE loader, ONE split definition, ONE linear-probe protocol and ONE set of
statistics. Fixes relative to the audited July runs:

  * Backbone loading is STRICT: no key filtering beyond the DINO projection
    head; any missing/unexpected backbone key raises an error. (The old
    Phase 2 / Phase 4 loaders dropped every 'mlp' key with strict=False.)
  * Splits: model selection on VAL, reporting on TEST. For Flowers102 and CUB
    the folder called 'val' is the official test set, so a stratified
    validation split is carved from 'train' (fixed seed, saved to JSON).
  * Linear probe: C chosen on val, refit on train+val, scored once on test.
  * Variance comes from things that actually vary (label subsets, fine-tune
    seeds); full-data linear probes get bootstrap CIs and paired tests.
"""
import os, json, time, random, shutil, hashlib, platform, subprocess, sys, copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ───────────────────────── configuration ─────────────────────────
BASE      = os.environ.get('SSL_BASE', '/content/drive/MyDrive/SSL_AgriStudy')
LOCAL     = os.environ.get('SSL_LOCAL', '/content/ssl_data')
RESULTS   = f'{BASE}/results_v2'           # new, clean results folder
FEATS     = f'{BASE}/features_v2'          # new feature cache (never reuse old)
SPLITS    = f'{RESULTS}/splits'
CKPT_PRE  = f'{BASE}/checkpoints/pretrained'
CKPT_DOM  = f'{BASE}/checkpoints/domain_dino'      # 9 clean from-scratch runs
CKPT_DOM2 = f'{BASE}/checkpoints/domain_dino_v2'   # new runs from notebook 02
DINO_REPO = os.environ.get('SSL_DINO_REPO', '/content/dino_vanilla')

# dataset key -> folder under BASE/datasets
DATASETS = {
    'FloraBD':      'FloraBD',
    'RiceLeaf':     'RiceLeafDiseaseBD',
    'PlantVillage': 'PlantVillage',
    'Flowers102':   'Flowers102',
    'CUB':          'CUB',
}
# datasets whose 'val' folder is really the official TEST set
VAL_IS_TEST = {'Flowers102', 'CUB'}
CARVE_VAL_FRAC = 0.20     # fraction of train per class held out as val there
SPLIT_SEED = 0

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = min(8, os.cpu_count() or 2)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    for d in (RESULTS, FEATS, SPLITS, CKPT_DOM2):
        os.makedirs(d, exist_ok=True)


# ───────────────────────── local data copy ─────────────────────────
# Image counts seen when these datasets were copied successfully in July (SSL_SR notebook).
# A different count means Drive returned an incomplete listing or the folder is not the full dataset.
EXPECTED_IMAGES = {'FloraBD': 19426, 'RiceLeaf': 9769, 'PlantVillage': 44014,
                   'Flowers102': 8189, 'CUB': 11788}
# If a dataset lives elsewhere on Drive, point to it here before calling localize(), e.g.
#   S.SOURCE_OVERRIDE['FloraBD'] = '/content/drive/MyDrive/FloraBD/FloraBD_224_split'
_SRC_FILE = f'{BASE}/results_v2/data_sources.json'      # shared by all notebooks
SOURCE_OVERRIDE = json.load(open(_SRC_FILE)) if os.path.exists(_SRC_FILE) else {}
ALLOW_COUNT_MISMATCH = False


def set_source(key, path):
    """Use a different Drive folder for a dataset; saved so every notebook uses it."""
    assert key in DATASETS, key
    assert os.path.isdir(path), f'not a folder: {path}'
    SOURCE_OVERRIDE[key] = path
    os.makedirs(os.path.dirname(_SRC_FILE), exist_ok=True)
    json.dump(SOURCE_OVERRIDE, open(_SRC_FILE, 'w'), indent=1)
    print(f'{key} -> {path}  (saved to {_SRC_FILE})')


def source_dir(key):
    return SOURCE_OVERRIDE.get(key, f'{BASE}/datasets/{DATASETS[key]}')


def _count_imgs(p):
    if not os.path.isdir(p):
        return 0
    return sum(1 for _, _, fs in os.walk(p, followlinks=True)
               for f in fs if f.lower().endswith(IMG_EXT))


def _scan(src, retries=3):
    """One walk of the source (follows Drive shortcuts). Retries because the
    Drive FUSE mount can return empty or partial listings right after mounting."""
    best = []
    for attempt in range(retries):
        files = [os.path.relpath(os.path.join(r, f), src)
                 for r, _, fs in os.walk(src, followlinks=True)
                 for f in fs if f.lower().endswith(IMG_EXT)]
        if len(files) > len(best):
            best = files
        if best and len(files) == len(best) and attempt > 0:
            break                       # two identical listings in a row -> stable
        time.sleep(3)
    return sorted(best)


def describe_source(key):
    """Per-split image counts of the Drive source folder (diagnostics)."""
    src = source_dir(key)
    print(f'{key}: {src}')
    if not os.path.isdir(src):
        print('   folder does not exist'); return
    for e in sorted(os.listdir(src)):
        p = os.path.join(src, e)
        kind = 'shortcut/symlink' if os.path.islink(p) else 'dir' if os.path.isdir(p) else 'file'
        n = _count_imgs(p) if os.path.isdir(p) else ''
        print(f'   {e:20s} {kind:16s} {n}')


TAR_CACHE = f'{BASE}/datasets_tar'     # one uncompressed .tar per dataset (built once, reused by every session)


def _local_count(dst):
    return _count_imgs(dst)


def _split_counts(paths):
    c = {}
    for r in paths:
        k = r.split(os.sep)[0]
        c[k] = c.get(k, 0) + 1
    return c


def _check_count(key, n, src):
    exp = EXPECTED_IMAGES.get(key)
    if (exp and n != exp and not ALLOW_COUNT_MISMATCH and os.environ.get('SSL_SMOKE') != '1'):
        describe_source(key)
        raise RuntimeError(
            f'{key}: found {n} images in {src}, expected {exp}.\n'
            f'Either Drive has not finished listing the folder (re-run the cell), or this is not the '
            f'full dataset. Point to the right folder with S.set_source("{key}", "..."), or set '
            f'S.ALLOW_COUNT_MISMATCH = True only if you are sure the new count is correct.')


def _from_tar(key, src, dst):
    """Fast path: copy ONE large file from Drive and extract locally (seconds-minutes
    instead of tens of thousands of small Drive reads). Returns True on success."""
    tar_f, man_f = f'{TAR_CACHE}/{key}.tar', f'{TAR_CACHE}/{key}.json'
    if not (os.path.exists(tar_f) and os.path.exists(man_f)):
        return False
    man = json.load(open(man_f))
    if man.get('source') != src:
        print(f'  [{key}] tar cache was built from a different folder — rebuilding'); return False
    t0 = time.time()
    tmp = f'/tmp/_{key}.tar'
    shutil.copyfile(tar_f, tmp)
    t1 = time.time()
    os.makedirs(dst, exist_ok=True)
    subprocess.run(['tar', '-xf', tmp, '-C', dst], check=True)
    os.remove(tmp)
    n = _local_count(dst)
    if n != man['n']:
        print(f'  [{key}] tar extract gave {n} images, manifest says {man["n"]} — falling back to file copy')
        shutil.rmtree(dst, ignore_errors=True); return False
    gb = os.path.getsize(tar_f) / 1e9
    print(f'  [{key}] ✓ {n} images local from tar cache ({gb:.2f} GB: copy {t1-t0:.0f}s, '
          f'extract {time.time()-t1:.0f}s)  {man["splits"]}')
    return True


def _build_tar(key, src, dst, files):
    """After a file-by-file copy, pack the LOCAL copy into one tar on Drive for next time."""
    try:
        os.makedirs(TAR_CACHE, exist_ok=True)
        tmp = f'/tmp/_{key}_build.tar'
        lst = f'/tmp/_{key}_files.txt'
        open(lst, 'w').write('\n'.join(files) + '\n')
        subprocess.run(['tar', '-cf', tmp, '-C', dst, '-T', lst], check=True)
        tar_f = f'{TAR_CACHE}/{key}.tar'
        shutil.copyfile(tmp, tar_f + '.part')
        os.replace(tar_f + '.part', tar_f)
        json.dump({'source': src, 'n': len(files), 'splits': _split_counts(files),
                   'built': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(f'{TAR_CACHE}/{key}.json', 'w'), indent=1)
        os.remove(tmp); os.remove(lst)
        print(f'  [{key}] tar cache written to Drive ({os.path.getsize(tar_f)/1e9:.2f} GB) — '
              f'future sessions will load it in one copy')
    except Exception as e:                        # cache is an optimisation only
        print(f'  [{key}] ⚠ could not write tar cache ({e}); next session will copy file-by-file again')


def localize(names=None, workers=64):
    """
    Put datasets on local disk. Order of preference:
      1. already local in this VM  -> nothing to do
      2. tar cache on Drive        -> copy one big file + extract (fast)
      3. file-by-file parallel copy from Drive (slow, first time only), then
         build the tar cache so every later session uses path 2.
    Image counts are always checked against the expected totals.
    """
    from tqdm.auto import tqdm
    names = names or list(DATASETS)
    os.makedirs(LOCAL, exist_ok=True)
    for key in names:
        src, dst = source_dir(key), f'{LOCAL}/{key}'
        exp = EXPECTED_IMAGES.get(key)
        n_local = _local_count(dst)
        if n_local and (exp is None or n_local == exp or os.environ.get('SSL_SMOKE') == '1'):
            print(f'  [{key}] ✓ already local ({n_local} images)')
            if not os.path.exists(f'{TAR_CACHE}/{key}.tar') and os.path.isdir(src):
                files = sorted(os.path.relpath(os.path.join(r, f), dst)
                               for r, _, fs in os.walk(dst) for f in fs if f.lower().endswith(IMG_EXT))
                _build_tar(key, src, dst, files)
            continue
        if _from_tar(key, src, dst):
            _check_count(key, _local_count(dst), src)
            continue
        if not os.path.isdir(src):
            raise FileNotFoundError(f'{key}: source folder missing: {src}')
        files = _scan(src)
        _check_count(key, len(files), src)
        todo = [r for r in files if not os.path.exists(os.path.join(dst, r))]
        for d in {os.path.dirname(os.path.join(dst, r)) for r in todo}:
            os.makedirs(d, exist_ok=True)
        if todo:
            def _cp(r):
                shutil.copyfile(os.path.join(src, r), os.path.join(dst, r))
                return 1
            with ThreadPoolExecutor(workers) as ex:
                list(tqdm(ex.map(_cp, todo), total=len(todo), desc=f'{key} (first-time copy)',
                          unit='img', mininterval=2))
        missing = [r for r in files if not os.path.exists(os.path.join(dst, r))]
        assert not missing, f'{key}: {len(missing)} files failed to copy, e.g. {missing[:3]}'
        print(f'  [{key}] ✓ {len(files)} images local  {_split_counts(files)}')
        _build_tar(key, src, dst, files)


# ───────────────────────── splits ─────────────────────────
def _list_folder(folder):
    """Return {class_name: [sorted image paths]} for non-empty class dirs."""
    out = {}
    for c in sorted(os.listdir(folder)):
        p = os.path.join(folder, c)
        if os.path.isdir(p):
            imgs = sorted(os.path.join(p, f) for f in os.listdir(p)
                          if f.lower().endswith(IMG_EXT))
            if imgs:
                out[c] = imgs
    return out


def get_splits(key, verbose=True):
    """
    Returns dict with 'classes' and 'train'/'val'/'test' lists of (path, label).
    Class indices come from TRAIN; val/test images of classes absent from train
    are excluded and reported (never silently re-indexed).
    """
    root = f'{LOCAL}/{key}'
    tr = _list_folder(f'{root}/train')
    classes = sorted(tr)
    cidx = {c: i for i, c in enumerate(classes)}

    def to_samples(d, name):
        s, dropped = [], 0
        for c, imgs in d.items():
            if c in cidx:
                s += [(p, cidx[c]) for p in imgs]
            else:
                dropped += len(imgs)
        if dropped:
            print(f'  ⚠ {key}/{name}: {dropped} images in classes not present in train — excluded')
        return s

    if key in VAL_IS_TEST:
        test = to_samples(_list_folder(f'{root}/val'), 'val(=official test)')
        split_f = f'{SPLITS}/{key}_carved_val.json'
        def _rel(p):            # 'train/<class>/<file>' — independent of where LOCAL is
            p = p.replace('\\', '/')
            i = p.find(f'/{key}/train/')
            return p[i + len(key) + 2:] if i >= 0 else p
        if os.path.exists(split_f):
            val_rel = {_rel(p) for p in json.load(open(split_f))['val_paths']}
        else:
            rng = random.Random(SPLIT_SEED)
            val_rel = set()
            for c, imgs in tr.items():
                imgs = list(imgs); rng.shuffle(imgs)
                n = max(1, int(round(len(imgs) * CARVE_VAL_FRAC)))
                val_rel.update(_rel(p) for p in imgs[:n])
            os.makedirs(SPLITS, exist_ok=True)
            json.dump({'seed': SPLIT_SEED, 'frac': CARVE_VAL_FRAC,
                       'val_paths': sorted(val_rel)}, open(split_f, 'w'))
        val_paths = {p for p, _ in to_samples(tr, 'train') if _rel(p) in val_rel}
        if len(val_paths) != len(val_rel):
            raise RuntimeError(f'{key}: carved val split lists {len(val_rel)} images but only '
                               f'{len(val_paths)} were found locally — dataset copy incomplete?')
        all_tr = to_samples(tr, 'train')
        train = [s for s in all_tr if s[0] not in val_paths]
        val = [s for s in all_tr if s[0] in val_paths]
    else:
        for need in ('val', 'test'):
            if not os.path.isdir(f'{root}/{need}'):
                raise FileNotFoundError(f'{key}: expected {root}/{need}')
        train = to_samples(tr, 'train')
        val = to_samples(_list_folder(f'{root}/val'), 'val')
        test = to_samples(_list_folder(f'{root}/test'), 'test')

    # guard against overlap between splits (same file path)
    a, b, c = {p for p, _ in train}, {p for p, _ in val}, {p for p, _ in test}
    assert not (a & b) and not (a & c) and not (b & c), f'{key}: split overlap'
    if verbose:
        print(f'  {key:12s} classes={len(classes):3d}  train={len(train):6d}  '
              f'val={len(val):5d}  test={len(test):5d}'
              + ('   (val carved from train; test = official test)' if key in VAL_IS_TEST else ''))
    return {'classes': classes, 'train': train, 'val': val, 'test': test}


def split_fingerprint(samples):
    h = hashlib.md5()
    for p, y in samples:
        h.update(f'{os.path.relpath(p, LOCAL)}|{y}\n'.encode())
    return h.hexdigest()


class ListDataset(Dataset):
    """(path, label) list -> (img, label, ok). Unreadable images get ok=False."""
    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        try:
            img = Image.open(p).convert('RGB')
            return self.tf(img), y, True
        except Exception:
            return torch.zeros(3, 224, 224), y, False


# ───────────────────────── backbones ─────────────────────────
def _vits():
    if DINO_REPO not in sys.path:
        sys.path.insert(0, DINO_REPO)
    import vision_transformer as vits
    return vits


def _strip_dino(sd):
    for k in ('teacher', 'student', 'model', 'state_dict'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]; break
    out = {}
    for k, v in sd.items():
        k = k.replace('module.', '').replace('backbone.', '')
        if k.startswith('head.') or k.startswith('head'):  # DINO projection head only
            continue
        out[k] = v
    return out


def _load_strict(model, sd, name):
    msg = model.load_state_dict(sd, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(f'{name}: load mismatch\n missing={msg.missing_keys[:10]}'
                           f'\n unexpected={msg.unexpected_keys[:10]}')
    n = len(model.state_dict())
    print(f'  ✓ {name}: {n}/{n} backbone tensors loaded (strict)')
    return model


DINO_IN_URL = ('https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/'
               'dino_deitsmall16_pretrain.pth')
MAE_URL = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'

# Domain checkpoints: name -> path. Missing files are skipped by callers.
def domain_checkpoints():
    d = {}
    for n in ['florabd_1k', 'florabd_5k', 'florabd_10k', 'rice_1k', 'rice_5k',
              'rice_full', 'pv_5k', 'pv_10k', 'pv_full']:
        d[n] = f'{CKPT_DOM}/{n}/checkpoint.pth'
    for n in ['florabd_full_scratch', 'florabd_cont_in', 'rice_cont_in'] + REVISION_RUNS:
        d[n] = f'{CKPT_DOM2}/{n}/checkpoint.pth'
    return d


# Revision runs (notebooks 08-10). Used only once their COMPLETE marker exists.
REVISION_RUNS = [
    # 08: generic-data control, from scratch, image counts and steps matched to the RiceLeaf arms
    'inet_n996', 'inet_n4671', 'inet_n6835', 'inet_n30829',
    # 09: extra pretraining seeds for the key arms
    'rice_full_s1', 'rice_full_s2', 'rice_cont_in_s1', 'rice_cont_in_s2',
    # 10: training budget / recipe
    'rice_full_ep400', 'rice_full_inrecipe', 'rice_1k_stepmatch', 'rice_5k_stepmatch',
    # 10: continued-pretraining variants
    'rice_cont_fullhead', 'rice_cont_fullhead_lowlr', 'rice_cont_replay',
]

# pretraining images per checkpoint (for scaling plots)
PRETRAIN_IMAGES = {
    'imagenet_dino': 1281167, 'florabd_1k': 990, 'florabd_5k': 4995,
    'florabd_10k': 9671, 'florabd_full_scratch': 13598, 'rice_1k': 996,
    'rice_5k': 4671, 'rice_full': 6835, 'pv_5k': 4920, 'pv_10k': 9444,
    'pv_full': 30829, 'florabd_cont_in': 1281167 + 13598,
    'rice_cont_in': 1281167 + 6835,
    'inet_n996': 996, 'inet_n4671': 4671, 'inet_n6835': 6835, 'inet_n30829': 30829,
    'rice_full_s1': 6835, 'rice_full_s2': 6835, 'rice_cont_in_s1': 1281167 + 6835,
    'rice_cont_in_s2': 1281167 + 6835, 'rice_full_ep400': 6835, 'rice_full_inrecipe': 6835,
    'rice_1k_stepmatch': 996, 'rice_5k_stepmatch': 4671, 'rice_cont_fullhead': 1281167 + 6835,
    'rice_cont_fullhead_lowlr': 1281167 + 6835, 'rice_cont_replay': 1281167 + 6835,
}


def is_complete(name):
    """Domain checkpoint usable? The 9 July runs are complete; notebook-02 runs only
    once notebook 02 has written its COMPLETE marker (DINO writes checkpoint.pth
    every epoch, so file existence alone would expose half-trained models)."""
    p = domain_checkpoints()[name]
    if not os.path.exists(p):
        return False
    if p.startswith(CKPT_DOM2):
        return os.path.exists(os.path.join(os.path.dirname(p), 'COMPLETE'))
    return True


def backbone_id(name):
    """Identity of the exact weights behind a backbone name. For domain checkpoints
    it changes whenever checkpoint.pth is rewritten (e.g. DINO resumed training),
    so results and feature caches from an older/incomplete checkpoint are rejected."""
    if name in domain_checkpoints():
        p = domain_checkpoints()[name]
        if not os.path.exists(p):
            return f'{name}:missing'
        st = os.stat(p)
        return f'{name}:{st.st_size}:{int(st.st_mtime)}'
    return name


def _is_dom2(name):
    return name in domain_checkpoints() and domain_checkpoints()[name].startswith(CKPT_DOM2)


def result_valid(rec, name):
    """A saved result counts as done only if it came from the current weights.
    Records written before ckpt_id existed are trusted only for checkpoints that
    can no longer change (ImageNet/DINOv2/MAE and the 9 finished July domain runs)."""
    if 'ckpt_id' in rec:
        return rec['ckpt_id'] == backbone_id(name)
    return not _is_dom2(name)


def available_backbones(include_external=True):
    names = ['imagenet_dino'] + [n for n in domain_checkpoints() if is_complete(n)]
    if include_external:
        names += ['dinov2_vits14', 'dinov2_vitb14', 'mae_vitb16']
    return names


def load_backbone(name):
    """Return (model in eval mode, feature_dim). Raises on any load mismatch."""
    if name == 'imagenet_dino':
        vits = _vits()
        m = vits.vit_small(patch_size=16, num_classes=0)
        f = f'{CKPT_PRE}/dino_deitsmall16_pretrain.pth'
        if not os.path.exists(f):
            os.makedirs(CKPT_PRE, exist_ok=True)
            torch.hub.download_url_to_file(DINO_IN_URL, f)
        sd = torch.load(f, map_location='cpu', weights_only=False)
        return _load_strict(m, _strip_dino(sd), name).eval(), 384
    if name in domain_checkpoints():
        if not is_complete(name):
            raise RuntimeError(f'{name}: checkpoint missing or pretraining not finished (notebook 02).')
        vits = _vits()
        m = vits.vit_small(patch_size=16, num_classes=0)
        ck = torch.load(domain_checkpoints()[name], map_location='cpu', weights_only=False)
        ep = ck.get('epoch', None) if isinstance(ck, dict) else None
        if ep is not None:
            print(f'  {name}: checkpoint epoch = {ep}')
        return _load_strict(m, _strip_dino(ck), name).eval(), 384
    if name == 'dinov2_vits14':
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval(), 384
    if name == 'dinov2_vitb14':
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval(), 768
    if name == 'mae_vitb16':
        import timm
        m = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        f = f'{CKPT_PRE}/mae_vitbase16.pth'
        if not os.path.exists(f):
            os.makedirs(CKPT_PRE, exist_ok=True)
            torch.hub.download_url_to_file(MAE_URL, f)
        sd = torch.load(f, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)
        sd = {k: v for k, v in sd.items()
              if not k.startswith('decoder') and k != 'mask_token'}
        return _load_strict(m, sd, name).eval(), 768
    raise KeyError(name)


def _forward_feats(model, x):
    out = model(x)
    if isinstance(out, dict):
        out = out['x_norm_clstoken']
    elif isinstance(out, (list, tuple)):
        out = out[0]
    return out


# ───────────────────────── feature cache ─────────────────────────
@torch.no_grad()
def extract(model, samples, batch=256, workers=None):
    workers = NUM_WORKERS if workers is None else workers
    dl = DataLoader(ListDataset(samples, EVAL_TF), batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=torch.cuda.is_available())
    model = model.to(DEVICE).eval()
    X, Y, OK = [], [], []
    for x, y, ok in dl:
        with torch.autocast('cuda', enabled=torch.cuda.is_available()):
            f = _forward_feats(model, x.to(DEVICE, non_blocking=True))
        X.append(f.float().cpu()); Y.append(y); OK.append(ok)
    X, Y, OK = torch.cat(X), torch.cat(Y), torch.cat(OK)
    return X[OK].numpy(), Y[OK].numpy(), int((~OK).sum())


def get_features(backbone_name, key, splits=None, model=None):
    """
    Cached features for train/val/test of dataset `key`.
    Cache is keyed on the exact sample list, so a changed split forces re-extraction.
    """
    splits = splits or get_splits(key, verbose=False)
    out = {}
    for sp in ('train', 'val', 'test'):
        f = f'{FEATS}/{backbone_name}/{key}_{sp}.npz'
        fp = split_fingerprint(splits[sp])
        bid = backbone_id(backbone_name)
        if os.path.exists(f):
            z = np.load(f, allow_pickle=False)
            same_w = (str(z['bid']) == bid) if 'bid' in z.files else not _is_dom2(backbone_name)
            if str(z['fp']) == fp and same_w:
                out[sp] = (z['X'], z['y']); continue
            print(f'  cache stale for {backbone_name}/{key}/{sp} (split or weights changed) — re-extracting')
        if model is None:
            model, _ = load_backbone(backbone_name)
        t0 = time.time()
        X, y, bad = extract(model, splits[sp])
        if bad:
            print(f'  ⚠ {key}/{sp}: {bad} unreadable images skipped')
        os.makedirs(os.path.dirname(f), exist_ok=True)
        tmp = f[:-4] + f'.tmp{os.getpid()}.npz'          # atomic: parallel sessions never read half a file
        np.savez(tmp, X=X.astype(np.float32), y=y, fp=fp, bad=bad, bid=bid)
        os.replace(tmp, f)
        print(f'  cached {backbone_name}/{key}/{sp}: {X.shape} in {time.time()-t0:.0f}s')
        out[sp] = (X, y)
    return out, model


# ───────────────────────── linear probe ─────────────────────────
C_GRID = [0.01, 0.0316, 0.1, 0.316, 1.0, 3.16]


def linear_probe(Xtr, ytr, Xva, yva, Xte, yte, Cs=C_GRID, refit_trainval=True):
    """
    Standardize on train, choose C on val, refit on train+val, score test once.
    Returns dict(test_acc, val_acc, C, correct[bool array over test]).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    best = (-1, None)
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(sc.transform(Xtr), ytr)
        acc = clf.score(sc.transform(Xva), yva)
        if acc > best[0]:
            best = (acc, C)
    val_acc, C = best
    if refit_trainval:
        Xf, yf = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
        sc = StandardScaler().fit(Xf)
    else:
        Xf, yf = Xtr, ytr
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xf), yf)
    pred = clf.predict(sc.transform(Xte))
    correct = pred == yte
    return {'test_acc': 100 * correct.mean(), 'val_acc': 100 * val_acc,
            'C': C, 'correct': correct}


def stratified_subset(labels, fraction, seed, min_per_class=1):
    """Indices of a class-stratified subset; returns (idx, n_per_class_list)."""
    rng = np.random.RandomState(seed)
    idx, per = [], []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        n = max(min_per_class, int(round(len(ci) * fraction)))
        n = min(n, len(ci))
        idx.extend(ci[:n].tolist()); per.append(n)
    return np.array(sorted(idx)), per


# ───────────────────────── statistics ─────────────────────────
def bootstrap_ci(correct, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    c = np.asarray(correct, dtype=float)
    accs = c[rng.randint(0, len(c), (n_boot, len(c)))].mean(1) * 100
    return float(np.percentile(accs, 100 * alpha / 2)), float(np.percentile(accs, 100 * (1 - alpha / 2)))


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on paired per-image correctness. Returns (b, c, p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue)


def paired_bootstrap_diff(correct_a, correct_b, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    i = rng.randint(0, len(a), (n_boot, len(a)))
    d = (a[i].mean(1) - b[i].mean(1)) * 100
    return float((a.mean() - b.mean()) * 100), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()


# ───────────────────────── results I/O ─────────────────────────
def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(type(o))


def _rdir(fname):
    return f"{RESULTS}/{fname[:-5] if fname.endswith('.json') else fname}_runs"


def load_results(fname):
    """
    Merge of (a) the legacy single JSON file and (b) one small JSON per run in
    <name>_runs/. One-file-per-run means several Colab sessions can write results
    for the same notebook at the same time without overwriting each other.
    """
    data = {}
    f = f'{RESULTS}/{fname}'
    if os.path.exists(f):
        data.update(json.load(open(f)))
    d = _rdir(fname)
    if os.path.isdir(d):
        for g in sorted(os.listdir(d)):
            if g.endswith('.json'):
                try:
                    rec = json.load(open(os.path.join(d, g)))
                    data[rec['__key']] = rec['record']
                except Exception as e:          # half-synced file from another session
                    print(f'  ⚠ skipped unreadable result {g}: {e}')
    return data


def save_result(fname, key, record):
    """One file per run, written atomically (safe for parallel sessions and disconnects)."""
    d = _rdir(fname)
    os.makedirs(d, exist_ok=True)
    safe = hashlib.md5(key.encode()).hexdigest()[:10] + '_' + ''.join(
        c if c.isalnum() or c in '._-' else '_' for c in key)[:150]
    f = os.path.join(d, safe + '.json')
    tmp = f + f'.tmp{os.getpid()}'
    json.dump({'__key': key, 'record': record}, open(tmp, 'w'), indent=1, default=_json_default)
    os.replace(tmp, f)


def save_correct(fname_stub, key, correct):
    """Per-image correctness vectors (needed for McNemar / bootstrap later)."""
    d = f'{RESULTS}/correct/{fname_stub}'
    os.makedirs(d, exist_ok=True)
    np.save(f'{d}/{key}.npy', np.asarray(correct, bool))


def load_correct(fname_stub, key):
    f = f'{RESULTS}/correct/{fname_stub}/{key}.npy'
    return np.load(f) if os.path.exists(f) else None


def env_info():
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        cpu = [l for l in open('/proc/cpuinfo') if l.startswith('model name')]
        info['cpu'] = cpu[0].split(':', 1)[1].strip() if cpu else platform.processor()
        info['cpu_threads'] = os.cpu_count()
    except Exception:
        info['cpu'] = platform.processor()
    return info


# ───────────────────────── fine-tuning ─────────────────────────
class _Head(nn.Module):
    def __init__(self, backbone, dim, n_cls):
        super().__init__()
        self.backbone, self.fc = backbone, nn.Linear(dim, n_cls)
    def forward(self, x):
        return self.fc(_forward_feats(self.backbone, x))


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); C = []
    for x, y, ok in loader:
        with torch.autocast('cuda', enabled=torch.cuda.is_available()):
            p = model(x.to(DEVICE, non_blocking=True)).argmax(1).cpu()
        C.append((p == y)[ok])
    C = torch.cat(C).numpy()
    return 100 * C.mean(), C


def make_loaders(splits, batch, workers=None):
    workers = NUM_WORKERS if workers is None else workers
    pin = torch.cuda.is_available()
    tr = DataLoader(ListDataset(splits['train'], TRAIN_TF), batch_size=batch,
                    shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True,
                    persistent_workers=workers > 0)
    va = DataLoader(ListDataset(splits['val'], EVAL_TF), batch_size=256,
                    num_workers=workers, pin_memory=pin)
    te = DataLoader(ListDataset(splits['test'], EVAL_TF), batch_size=256,
                    num_workers=workers, pin_memory=pin)
    return tr, va, te


CKPT_EVERY_MIN = 10      # mid-run checkpoint interval (minutes) for train_classifier


def train_classifier(model, splits, seed, epochs, lr, wd, batch,
                     teacher=None, T=4.0, alpha=0.5, log_every=5, resume_key=None):
    """
    Generic supervised / KD training loop.
    Selects the epoch with best VAL accuracy and reports TEST accuracy of that
    epoch's weights (test is evaluated exactly once).

    resume_key: if given, the full training state (model, optimizer, scheduler,
    scaler, best weights, RNG states) is saved to Drive at most every
    CKPT_EVERY_MIN minutes and after the last epoch; a re-run with the same key
    continues from the last saved epoch instead of epoch 0. The file is deleted
    when the run finishes. A resumed run is statistically equivalent to an
    uninterrupted one but not bit-identical (data-loader worker order differs).
    """
    set_seed(seed)
    tr, va, te = make_loaders(splits, batch)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    ce = nn.CrossEntropyLoss(reduction='none')
    kl = nn.KLDivLoss(reduction='none')
    if teacher is not None:
        teacher = teacher.to(DEVICE).eval()
    best_val, best_ep, best_sd = -1, -1, None
    start_ep = 0
    ck_f = f'{RESULTS}/_inprogress/{resume_key}.pt' if resume_key else None
    if ck_f and os.path.exists(ck_f):
        ck = torch.load(ck_f, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        best_val, best_ep, best_sd, start_ep = ck['best_val'], ck['best_ep'], ck['best_sd'], ck['epoch']
        torch.set_rng_state(ck['rng_torch']); np.random.set_state(ck['rng_np']); random.setstate(ck['rng_py'])
        if torch.cuda.is_available() and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        print(f'      ↻ resuming {resume_key} from epoch {start_ep}/{epochs} (best_val={best_val:.2f}@{best_ep})')
    t0 = last_save = time.time()
    for ep in range(start_ep, epochs):
        model.train()
        for x, y, ok in tr:
            x, y, ok = x.to(DEVICE, non_blocking=True), y.to(DEVICE), ok.to(DEVICE)
            if not ok.any():
                continue
            with torch.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = ce(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t = teacher(x)
                    lkd = kl(torch.log_softmax(logits.float() / T, 1),
                             torch.softmax(t.float() / T, 1)).sum(1) * T * T
                    loss = alpha * lkd + (1 - alpha) * loss
                loss = (loss * ok).sum() / ok.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        v, _ = evaluate(model, va)
        if v > best_val:
            best_val, best_ep = v, ep + 1
            best_sd = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f'      ep {ep+1:3d}/{epochs}  val={v:.2f}  best_val={best_val:.2f}@{best_ep}'
                  f'  ({(time.time()-t0)/60:.1f} min)')
        if ck_f and (time.time() - last_save >= CKPT_EVERY_MIN * 60 or ep == epochs - 1):
            os.makedirs(os.path.dirname(ck_f), exist_ok=True)
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                        'best_val': best_val, 'best_ep': best_ep, 'best_sd': best_sd,
                        'epoch': ep + 1, 'rng_torch': torch.get_rng_state(),
                        'rng_np': np.random.get_state(), 'rng_py': random.getstate(),
                        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       ck_f + '.tmp')
            os.replace(ck_f + '.tmp', ck_f)          # atomic: a disconnect mid-save never corrupts it
            last_save = time.time()
    model.load_state_dict(best_sd)
    test_acc, correct = evaluate(model, te)
    if ck_f and os.path.exists(ck_f):
        os.remove(ck_f)
    return {'val_acc': best_val, 'best_epoch': best_ep, 'test_acc': test_acc,
            'minutes': (time.time() - t0) / 60}, correct, model


def build_vit_classifier(backbone_name, n_cls):
    bb, dim = load_backbone(backbone_name)
    return _Head(bb, dim, n_cls)


def summarize(vals):
    v = np.asarray(vals, float)
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'n': int(len(v)), 'values': v.tolist()}