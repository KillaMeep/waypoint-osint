#!/usr/bin/env python3
"""
Geometric match verification via DISK + LightGlue (Kornia) — a lightweight,
no-index-build alternative to netryx-astra's MASt3R stage. Given a short list
of visually-similar candidates (already narrowed by StreetCLIP cosine
similarity in mapillary_refine/google_sv_refine), this re-scores them by
actual feature-point geometric consensus (inlier count after RANSAC), which
is a much stronger signal than embedding similarity alone — two suburban
streets can "look similar" in embedding space while sharing zero real
matchable structure.
"""
import contextlib
import io
import logging
import sys

import cv2
import numpy as np
import torch
import kornia.feature as KF
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_disk = None
_matcher = None

MAX_KEYPOINTS = 2048
RESIZE_MAX_SIDE = 1024  # keep pair-matching fast; full-res panoramas aren't needed for this


def _get_models():
    global _disk, _matcher
    if _disk is None:
        # kornia's LightGlue loader does a raw print() straight to stdout,
        # which would corrupt an NDJSON stdout protocol (as used by the
        # Geolocator GUI's Electron<->Python bridge) — redirect it away.
        with contextlib.redirect_stdout(io.StringIO()):
            _disk = KF.DISK.from_pretrained('depth').to(_device).eval()
            _matcher = KF.LightGlueMatcher('disk').to(_device).eval()
    return _disk, _matcher


def _prep(img: Image.Image) -> torch.Tensor:
    img = img.convert('RGB')
    w, h = img.size
    scale = RESIZE_MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    # DISK expects dimensions divisible by 16
    w2, h2 = img.size
    w2, h2 = (w2 // 16) * 16, (h2 // 16) * 16
    img = img.resize((max(16, w2), max(16, h2)), Image.BILINEAR)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(_device)
    return tensor


def match_pair(img1: Image.Image, img2: Image.Image):
    """Return (inlier_count, total_matches) between two images via DISK
    keypoints + LightGlue matching + RANSAC geometric consensus."""
    disk, matcher = _get_models()

    t1, t2 = _prep(img1), _prep(img2)
    with torch.no_grad():
        feats1 = disk(t1, n=MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
        feats2 = disk(t2, n=MAX_KEYPOINTS, pad_if_not_divisible=True)[0]

        if feats1.keypoints.shape[0] < 8 or feats2.keypoints.shape[0] < 8:
            return 0, 0

        kps1, kps2 = feats1.keypoints[None], feats2.keypoints[None]
        lafs1 = KF.laf_from_center_scale_ori(kps1, torch.ones(1, kps1.shape[1], 1, 1, device=_device))
        lafs2 = KF.laf_from_center_scale_ori(kps2, torch.ones(1, kps2.shape[1], 1, 1, device=_device))

        dists, idxs = matcher(feats1.descriptors, feats2.descriptors, lafs1, lafs2)

    total_matches = idxs.shape[0]
    if total_matches < 8:
        return 0, total_matches

    pts1 = feats1.keypoints[idxs[:, 0]].cpu().numpy()
    pts2 = feats2.keypoints[idxs[:, 1]].cpu().numpy()

    _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, total_matches


def verify_candidates(target_image_path: str, candidates: list, image_key: str = None, image_loader=None, top_k: int = None, on_progress=None):
    """Re-score a list of candidate dicts (each already having a 'similarity'
    from embedding cosine-sim) by geometric match verification. Candidates
    must either carry a PIL image under `image_key`, or `image_loader(cand)`
    must return one (used for e.g. lazily re-downloading a thumbnail).

    Adds 'inliers' and 'total_matches' to each candidate dict, in place, and
    returns candidates re-sorted by inlier count (real geometric consensus is
    NOT overparameterized the way the sun-refine heuristic was — a high
    inlier count is strong evidence, unlike a lucky low-error date/heading fit).

    on_progress(completed, total), if given, is called after each candidate
    is checked — DISK+LightGlue matching is GPU work, not slow per-pair, but
    a batch of 100+ candidates still takes real wall-clock time worth showing."""
    target_img = Image.open(target_image_path).convert('RGB')
    to_check = candidates[:top_k] if top_k else candidates
    total = len(to_check)

    for i, cand in enumerate(to_check, 1):
        try:
            img = cand.get(image_key) if image_key else None
            if img is None and image_loader is not None:
                img = image_loader(cand)
            if img is None:
                cand['inliers'], cand['total_matches'] = 0, 0
            else:
                inliers, total_matches = match_pair(target_img, img)
                cand['inliers'], cand['total_matches'] = inliers, total_matches
        except Exception as e:
            logger.debug(f'Verification failed for a candidate: {e}')
            cand['inliers'], cand['total_matches'] = 0, 0
        if on_progress:
            on_progress(i, total)

    to_check.sort(key=lambda c: c.get('inliers', 0), reverse=True)
    return to_check
