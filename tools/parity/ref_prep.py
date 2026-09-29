"""Reference decoded pixels and the exact tensors the models consume:
  * decoded RGB (PIL) of the test images and a few candidates
  * DISK input (verify_utils._prep) and CLIP pixel_values
  * DISK keypoints/descriptors on the target for the ONNX gate
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from common import *
import verify_utils
from plonk_core import Predictor

pred = Predictor('osv5m', 1)
pipe = pred.pipeline
man = json.load(open(os.path.join(REF, 'cands', 'manifest.json')))
picks = {'pano': IMAGES['pano'], 'photo2': IMAGES['photo2']}
for m in man[:1] + man[10:11] + man[20:21]:
    picks[m['file'].rsplit('.', 1)[0]] = os.path.join(REF, 'cands', m['file'])

disk, _ = verify_utils._get_models()
idx = {}
for name, path in picks.items():
    img = Image.open(path).convert('RGB')
    np.save(f'{REF}/rgb_{name}.npy', np.asarray(img))
    pv = pipe.cond_preprocessing.processor(images=[img], return_tensors='pt')['pixel_values']
    np.save(f'{REF}/pv_{name}.npy', pv.numpy())
    t = verify_utils._prep(img)
    np.save(f'{REF}/disk_in_{name}.npy', t.cpu().numpy())
    with torch.no_grad():
        f = disk(t, n=verify_utils.MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
        hm, desc = disk.heatmap_and_dense_descriptors(t)
    np.savez(f'{REF}/disk_out_{name}.npz', kp=f.keypoints.cpu().numpy(), desc=f.descriptors.cpu().numpy(),
             score=f.detection_scores.cpu().numpy(), heat=hm.cpu().numpy())
    idx[name] = path
    print(name, img.size, t.shape, f.keypoints.shape)
json.dump(idx, open(f'{REF}/prep_index.json', 'w'), indent=1)
