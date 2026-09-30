"""Export DINOv2 ViT-L/14 with registers (what plonk.pipe.DinoV2FeatureExtractor
uses to condition PLONK_YFCC and PLONK_iNaturalist) to ONNX.

Output `emb` is the model's default output: the normed CLS token (head is
Identity), 1024 floats. Input `pixel_values` is the extractor's augmentation
output: square centre crop, Pillow bicubic resize to 336, ImageNet mean/std
(1x3x336x336).

The checkpoint was trained at 518 px, so the model resamples its position
embedding (bicubic, antialiased) for every 336 px input. That resample depends
only on the weights, so it is done once here and baked in as a constant.

  .venv-export\\Scripts\\python.exe tools\\export\\export_dinov2.py <out_dir>
"""
import os
import sys

os.environ['XFORMERS_DISABLED'] = '1'  # plain attention, exportable

import numpy as np
import torch

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
SIZE = 336

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').eval().cpu().requires_grad_(False)
n_tokens = 1 + (SIZE // model.patch_size) ** 2
with torch.no_grad():
    pos = model.interpolate_pos_encoding(torch.zeros(1, n_tokens, model.embed_dim), SIZE, SIZE).clone()
model.interpolate_pos_encoding = lambda x, w, h: pos


class Wrap(torch.nn.Module):
    # forward(x, masks=None): export only the image input
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values)


path = os.path.join(out_dir, 'dinov2_vitl14_reg.onnx')
torch.onnx.export(
    Wrap(model).eval(), (torch.randn(1, 3, SIZE, SIZE),), path, input_names=['pixel_values'], output_names=['emb'],
    dynamic_axes={'pixel_values': {0: 'batch'}, 'emb': {0: 'batch'}},
    opset_version=17, dynamo=False,
)
print('exported', path, round(os.path.getsize(path) / 1e6, 1), 'MB')

# Parity vs the recorded PyTorch embeddings (tools/parity/ref_plonk_variants.py).
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parity', 'ref')
import onnxruntime as ort
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
for name in ('pano', 'photo2'):
    pv = np.load(os.path.join(REF, f'yfcc_{name}_pixel_values.npy'))
    ref = np.load(os.path.join(REF, f'yfcc_{name}_emb.npy'))
    got = sess.run(None, {'pixel_values': pv})[0][0]
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    print(name, 'cosine vs reference embedding', cos, 'max abs diff', float(np.abs(ref - got).max()))
