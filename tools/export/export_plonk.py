"""Export a PLONK denoiser as ONE Euler step of the Riemannian flow
sampler (plonk/models/samplers/riemannian_flow_sampler.py), so Rust only has
to run the step graph num_steps times:

    x_next = projx(x + dt * net({y: x, gamma, emb}))      projx(v) = v / |v|

Inputs: x[B,3] f32, gamma[1] f32, dt[1] f32, emb[1,1024] f32 (the conditioning
embedding: StreetCLIP CLS for osv5m, see export_clip.py; DINOv2 for yfcc and
inat, see export_dinov2.py). gamma and emb are the same for every row of the batch in
PlonkPipeline, so they go in once and broadcast inside the network instead of
being repeated B times.

Then checks, on the recorded fixed-noise references in tools/parity/ref
(ref_plonk.py for osv5m, ref_plonk_variants.py for the others):
  1. this broadcast torch loop reproduces PlonkPipeline's samples
  2. the ONNX step loop (CPU EP) reproduces the torch loop

  .venv-export\\Scripts\\python.exe tools\\export\\export_plonk.py <out_dir> [osv5m|yfcc|inat]
"""
import os
import sys
import time

import numpy as np
import torch
from plonk.models.pretrained_models import Plonk
from plonk.models.schedulers import SigmoidScheduler

out_dir = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else 'osv5m'
HF_ID = {'osv5m': 'nicolas-dufour/PLONK_OSV_5M', 'yfcc': 'nicolas-dufour/PLONK_YFCC', 'inat': 'nicolas-dufour/PLONK_iNaturalist'}[variant]
REF_PREFIX = '' if variant == 'osv5m' else f'{variant}_'
os.makedirs(out_dir, exist_ok=True)
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parity', 'ref')
NUM_STEPS = 250  # riemannian_flow_sampler default, used by PlonkPipeline

net = Plonk.from_pretrained(HF_ID).eval().cpu().requires_grad_(False)
print('network:', sum(p.numel() for p in net.parameters()) / 1e6, 'M params,',
      len(net.blocks), 'blocks, dim', net.initial_mapper.out_features)


class Step(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, gamma, dt, emb):
        d = self.network({'y': x, 'gamma': gamma, 'emb': emb})
        x = x + dt * d
        return x / x.norm(dim=-1, keepdim=True)


step = Step(net).eval()
path = os.path.join(out_dir, f'plonk_{variant}_step.onnx')
torch.onnx.export(
    step,
    (torch.randn(8, 3), torch.tensor([0.5]), torch.tensor([0.01]), torch.randn(1, 1024)),
    path,
    input_names=['x', 'gamma', 'dt', 'emb'],
    output_names=['x_next'],
    dynamic_axes={'x': {0: 'batch'}, 'x_next': {0: 'batch'}},
    opset_version=17,
    dynamo=False,
)
print('exported', path, round(os.path.getsize(path) / 1e6, 1), 'MB')


def gammas(num_steps=NUM_STEPS):
    # Exactly the sampler's schedule, in float32.
    sched = SigmoidScheduler(-7, 3, 1.0, 1e-9)
    idx = torch.arange(num_steps + 1, dtype=torch.float32)
    return sched(1 - idx / num_steps)


def to_deg(x):
    lat = torch.asin(x[:, 2])
    lon = torch.atan2(x[:, 1], x[:, 0])
    return np.degrees(torch.stack([lat, lon], -1).numpy())


def gc_km(a, b):
    a, b = np.radians(a.astype(np.float64)), np.radians(b.astype(np.float64))
    s = (np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
         + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))


g = gammas()
print('schedule: first', float(g[0]), 'second', float(g[1]), 'last', float(g[-1]))

import onnxruntime as ort
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
step_dev = Step(net.to(dev)).eval()
for name in ('pano', 'photo2'):
    ref = lambda f: os.path.join(REF, REF_PREFIX + f)
    emb = torch.from_numpy(np.load(ref(f'{name}_emb.npy'))).reshape(1, -1)
    for seed in (1, 2, 3):
        xN = torch.from_numpy(np.load(ref(f'{name}_xN_s{seed}.npy')))
        want = np.load(ref(f'{name}_samples_s{seed}.npy'))

        with torch.no_grad():
            x = xN.to(dev)
            for a, b in zip(g[:-1], g[1:]):
                x = step_dev(x, a.reshape(1).to(dev), (b - a).reshape(1).to(dev), emb.to(dev))
        ours_torch = to_deg(x.cpu())

        t = time.time()
        xo = xN.numpy()
        e = emb.numpy()
        for a, b in zip(g[:-1], g[1:]):
            xo = sess.run(None, {'x': xo, 'gamma': a.reshape(1).numpy(), 'dt': (b - a).reshape(1).numpy(), 'emb': e})[0]
        ours_onnx = to_deg(torch.from_numpy(xo))
        dt = time.time() - t

        d1 = gc_km(want, ours_torch)
        d2 = gc_km(ours_torch, ours_onnx)
        print(f'{name} s{seed}: broadcast-torch vs pipeline median {np.median(d1):.4f} km p99 {np.percentile(d1, 99):.3f} km | '
              f'onnx-cpu vs torch median {np.median(d2):.4f} km p99 {np.percentile(d2, 99):.3f} km max {d2.max():.2f} km '
              f'({len(xo)} samples, {dt:.1f}s cpu)')
