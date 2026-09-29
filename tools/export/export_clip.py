"""Export StreetCLIP's vision tower (what PLONK's `cond_preprocessing` uses) to ONNX.

Output `emb` is `last_hidden_state[:, 0]`, the CLS token BEFORE post_layernorm
(NOT pooler_output), exactly as plonk.pipe.StreetClipFeatureExtractor does.
Input `pixel_values` is the CLIPImageProcessor output (1x3x336x336, CLIP mean/std).

  .venv-export\\Scripts\\python.exe tools\\export\\export_clip.py <out_dir>
"""
import os
import sys

import numpy as np
import torch
from transformers import CLIPVisionModel

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

model = CLIPVisionModel.from_pretrained('geolocal/StreetCLIP', attn_implementation='eager').eval().cpu()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).last_hidden_state[:, 0]


w = Wrap(model).eval()
x = torch.randn(2, 3, 336, 336)
path = os.path.join(out_dir, 'streetclip_vision.onnx')
torch.onnx.export(
    w, (x,), path, input_names=['pixel_values'], output_names=['emb'],
    dynamic_axes={'pixel_values': {0: 'batch'}, 'emb': {0: 'batch'}},
    opset_version=17, dynamo=False,
)
print('exported', path, os.path.getsize(path) / 1e6, 'MB')

# quick parity vs torch on reference pixel_values
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parity', 'ref')
import onnxruntime as ort
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
for name in ('pano', 'photo2'):
    pv = np.load(os.path.join(REF, f'{name}_pixel_values.npy'))
    ref = np.load(os.path.join(REF, f'{name}_emb.npy'))
    got = sess.run(None, {'pixel_values': pv})[0][0]
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    print(name, 'cosine vs reference embedding', cos, 'max abs diff', float(np.abs(ref - got).max()))
