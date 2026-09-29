"""Reference retrieval/matching data: downloads a fixed set of real candidate
images around the true location of test_pano.jpg, stores them
losslessly, then records embeddings, DISK features, LightGlue matches (production
settings and pruning disabled) and cv2 RANSAC masks/inlier counts.

Uses the Mapillary token from the app settings (read only); the token is never
written to any output file.
"""
import io
import json
import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image
import kornia.feature as KF

from common import *
import verify_utils
from plonk_core import Predictor
import mapillary_refine as MR
import google_sv_refine as GS
import panoramax_refine as PX

LAT, LON, RAD = 0.0, 0.0, 3.0
N_PER_SOURCE = 10
CAND = os.path.join(REF, 'cands')
os.makedirs(CAND, exist_ok=True)
manifest_path = os.path.join(CAND, 'manifest.json')

token = json.load(open(os.path.join(os.environ['APPDATA'], 'geolocator-gui', 'settings.json')))['mapillaryToken']

if not os.path.exists(manifest_path):
    manifest = []
    np.random.seed(0)
    # Mapillary: keep the raw thumbnail bytes (Python decodes them exactly as it does in production).
    cands = MR.search_nearby_images(LAT, LON, RAD, token, max_images=200)[:N_PER_SOURCE]
    for c in cands:
        b = requests.get(c['thumb_url'], timeout=20).content
        fn = f"mly_{c['id']}.jpg"
        open(os.path.join(CAND, fn), 'wb').write(b)
        manifest.append({'src': 'mapillary', 'file': fn, 'lat': c['lat'], 'lon': c['lon']})
    pans = GS.search_panoramas(LAT, LON, RAD, max_images=200)[:N_PER_SOURCE]
    for p in pans:
        img = GS.download_panorama(p['panoid'])
        if img is None:
            continue
        fn = f"gsv_{p['panoid']}.png"
        img.save(os.path.join(CAND, fn))
        manifest.append({'src': 'google_sv', 'file': fn, 'lat': p['lat'], 'lon': p['lon']})
    px = PX.search_nearby_images(LAT, LON, RAD, max_images=200)[:N_PER_SOURCE]
    for c in px:
        b = requests.get(c['thumb_url'], timeout=20).content
        fn = f"px_{c['id']}.jpg"
        open(os.path.join(CAND, fn), 'wb').write(b)
        manifest.append({'src': 'panoramax', 'file': fn, 'lat': c['lat'], 'lon': c['lon']})
    json.dump(manifest, open(manifest_path, 'w'), indent=1)
manifest = json.load(open(manifest_path))
print(len(manifest), 'candidates')

pred = Predictor('osv5m', 1)
pipe = pred.pipeline


def embed(img):
    return pipe.cond_preprocessing({'img': [img]})['emb'].detach().cpu().numpy()[0]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load(fn):
    return Image.open(os.path.join(CAND, fn)).convert('RGB')


target = Image.open(IMAGES['pano']).convert('RGB')
target_emb = embed(target)
embs = {}
sims = []
for m in manifest:
    e = embed(load(m['file']))
    embs[m['file']] = e
    sims.append({'file': m['file'], 'src': m['src'], 'similarity': cos(target_emb, e)})
np.savez(os.path.join(REF, 'cand_embs.npz'), target=target_emb, **{k.replace('.', '_'): v for k, v in embs.items()})
sims.sort(key=lambda s: -s['similarity'])
json.dump(sims, open(os.path.join(REF, 'retrieval_ranking.json'), 'w'), indent=1)

# --- matching: production settings and no-pruning variant
disk, matcher_prod = verify_utils._get_models()
matcher_np = KF.LightGlueMatcher('disk', {'depth_confidence': -1, 'width_confidence': -1}).to(verify_utils._device).eval()


def run_pair(img1, img2, matcher):
    t1, t2 = verify_utils._prep(img1), verify_utils._prep(img2)
    with torch.no_grad():
        f1 = disk(t1, n=verify_utils.MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
        f2 = disk(t2, n=verify_utils.MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
        if f1.keypoints.shape[0] < 8 or f2.keypoints.shape[0] < 8:
            return f1, f2, None, 0, 0, None
        k1, k2 = f1.keypoints[None], f2.keypoints[None]
        dev = verify_utils._device
        l1 = KF.laf_from_center_scale_ori(k1, torch.ones(1, k1.shape[1], 1, 1, device=dev))
        l2 = KF.laf_from_center_scale_ori(k2, torch.ones(1, k2.shape[1], 1, 1, device=dev))
        _, idxs = matcher(f1.descriptors, f2.descriptors, l1, l2)
    total = idxs.shape[0]
    if total < 8:
        return f1, f2, idxs.cpu().numpy(), 0, total, None
    p1 = f1.keypoints[idxs[:, 0]].cpu().numpy()
    p2 = f2.keypoints[idxs[:, 1]].cpu().numpy()
    _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)
    inl = int(mask.sum()) if mask is not None else 0
    return f1, f2, idxs.cpu().numpy(), inl, total, mask


pairs = []
for m in manifest:
    pairs.append(('pano', m['file'], target, load(m['file'])))
# a positive control: pano vs. a crop-resized version of itself
w, h = target.size
crop = target.crop((int(w * .1), int(h * .1), int(w * .8), int(h * .9))).resize((900, 700))
pairs.append(('pano', 'self_crop', target, crop))
pairs.append(('pano', 'photo2', target, Image.open(IMAGES['photo2']).convert('RGB')))
bk = Image.open(IMAGES['photo2']).convert('RGB')
pairs.append(('photo2', 'self_crop', bk, bk.crop((200, 100, 3200, 2000)).resize((1280, 800))))

report = []
for a, b, i1, i2 in pairs:
    row = {'a': a, 'b': b}
    key = f'{a}__{b}'.replace('.', '_')
    for tag, mt in (('prod', matcher_prod), ('noprune', matcher_np)):
        f1, f2, idxs, inl, total, mask = run_pair(i1, i2, mt)
        row[tag] = {'inliers': inl, 'total': total, 'n_kp1': int(f1.keypoints.shape[0]), 'n_kp2': int(f2.keypoints.shape[0])}
        if tag == 'prod':
            np.savez(os.path.join(REF, f'match_{key}.npz'),
                     kp1=f1.keypoints.cpu().numpy(), kp2=f2.keypoints.cpu().numpy(),
                     desc1=f1.descriptors.cpu().numpy(), desc2=f2.descriptors.cpu().numpy(),
                     det1=f1.detection_scores.cpu().numpy(), det2=f2.detection_scores.cpu().numpy(),
                     idxs=idxs if idxs is not None else np.zeros((0, 2), 'int64'),
                     mask=mask if mask is not None else np.zeros((0, 1), 'uint8'))
    print(row, flush=True)
    report.append(row)
json.dump(report, open(os.path.join(REF, 'match_report.json'), 'w'), indent=1)
