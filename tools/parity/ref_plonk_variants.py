"""Reference outputs for the DINOv2-conditioned PLONK variants (YFCC, iNaturalist):
DINOv2 input tensors and embeddings, fixed-noise samples and their clusters, a
large own-noise run, and DINOv2 embeddings of the retrieval candidates (the
Python engine ranks candidates with the pipeline's own embedder).

Same layout as ref_plonk.py, with the variant as a file-name prefix:
  <v>_<image>_pixel_values.npy, <v>_<image>_emb.npy, <v>_<image>_xN_s<seed>.npy,
  <v>_<image>_samples_s<seed>.npy, <v>_plonk_clusters.json, dinov2_cand_embs.npz
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from common import *
from plonk_core import MODEL_ALIASES, cluster_samples
from plonk import PlonkPipeline

CAND = os.path.join(REF, 'cands')
cand_done = False
for variant in ('yfcc', 'inat'):
    pipe = PlonkPipeline(MODEL_ALIASES[variant])
    ext = pipe.cond_preprocessing
    out = {}
    for name, path in IMAGES.items():
        img = Image.open(path).convert('RGB')
        pv = ext.augmentation(img).unsqueeze(0)
        emb = ext({'img': [img]})['emb'].cpu().numpy()[0]
        np.save(f'{REF}/{variant}_{name}_pixel_values.npy', pv.numpy())
        np.save(f'{REF}/{variant}_{name}_emb.npy', emb)
        for seed in SEEDS:
            g = torch.Generator(device=pipe.device).manual_seed(seed)
            xN = torch.randn(2048, 3, device=pipe.device, generator=g)
            t = time.time()
            ll = pipe(img, batch_size=2048, x_N=xN)
            dt = time.time() - t
            np.save(f'{REF}/{variant}_{name}_xN_s{seed}.npy', xN.cpu().numpy())
            np.save(f'{REF}/{variant}_{name}_samples_s{seed}.npy', ll)
            cl, noise = cluster_samples(ll, 100.0, 0.03, 5)
            out[f'{name}_s{seed}'] = {'clusters': cl, 'noise_frac': noise, 'sec_2048': dt}
            print(variant, name, seed, cl[:2], noise, round(dt, 2), flush=True)

    img = Image.open(IMAGES['pano']).convert('RGB')
    big = np.concatenate([pipe(img, batch_size=8192) for _ in range(3)])
    cl, noise = cluster_samples(big, 100.0, 0.03, 5)
    out['pano_big'] = {'clusters': cl, 'noise_frac': noise}
    print(variant, 'big', cl[:2], noise, flush=True)
    pipe(img, batch_size=64)
    t = time.time()
    pipe(img, batch_size=8192)
    out['throughput_8192'] = 8192 / (time.time() - t)
    save_json(f'{variant}_plonk_clusters.json', out)

    if not cand_done:
        # Both variants use the same DINOv2 (dinov2_vitl14_reg), so once is enough.
        man = json.load(open(os.path.join(CAND, 'manifest.json')))
        target = ext({'img': [Image.open(IMAGES['pano']).convert('RGB')]})['emb'].cpu().numpy()[0]
        embs = {m['file'].replace('.', '_'): ext({'img': [Image.open(os.path.join(CAND, m['file'])).convert('RGB')]})['emb'].cpu().numpy()[0]
                for m in man}
        np.savez(os.path.join(REF, 'dinov2_cand_embs.npz'), target=target, **embs)
        print('candidate embeddings', len(embs), flush=True)
        cand_done = True
    del pipe
    torch.cuda.empty_cache()
