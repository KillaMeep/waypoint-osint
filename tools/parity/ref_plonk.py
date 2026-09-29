"""Reference PLONK outputs: pixel_values, embeddings, fixed-noise samples,
cluster_samples on those exact arrays. Everything Rust must reproduce."""
import time

import numpy as np
import torch
from PIL import Image

from common import *
from plonk_core import Predictor, cluster_samples

pred = Predictor('osv5m', 512)
pipe = pred.pipeline
out = {}
for name, path in IMAGES.items():
    img = Image.open(path).convert('RGB')
    proc = pipe.cond_preprocessing.processor
    print(name, type(proc.image_processor).__name__, flush=True)
    pv = proc(images=[img], return_tensors='pt')['pixel_values']
    emb = pipe.cond_preprocessing({'img': [img]})['emb'].cpu().numpy()[0]
    np.save(f'{REF}/{name}_pixel_values.npy', pv.numpy())
    np.save(f'{REF}/{name}_emb.npy', emb)
    for seed in SEEDS:
        g = torch.Generator(device=pipe.device).manual_seed(seed)
        xN = torch.randn(2048, 3, device=pipe.device, generator=g)
        t = time.time()
        ll = pipe(img, batch_size=2048, x_N=xN)
        dt = time.time() - t
        np.save(f'{REF}/{name}_xN_s{seed}.npy', xN.cpu().numpy())
        np.save(f'{REF}/{name}_samples_s{seed}.npy', ll)
        cl, noise = cluster_samples(ll, 100.0, 0.03, 5)
        out[f'{name}_s{seed}'] = {'clusters': cl, 'noise_frac': noise, 'sec_2048': dt}
        print(name, seed, cl[:2], noise, round(dt, 2), flush=True)

img = Image.open(IMAGES['pano']).convert('RGB')
big = np.concatenate([pipe(img, batch_size=8192) for _ in range(3)])
np.save(f'{REF}/pano_big_samples.npy', big)
cl, noise = cluster_samples(big, 100.0, 0.03, 5)
out['pano_big'] = {'clusters': cl, 'noise_frac': noise}
print(cl, noise)
for bs in (1024, 4096, 8192):
    pipe(img, batch_size=64)
    t = time.time()
    pipe(img, batch_size=bs)
    dt = time.time() - t
    print('throughput', bs, bs / dt)
    out[f'throughput_{bs}'] = bs / dt
save_json('plonk_clusters.json', out)
