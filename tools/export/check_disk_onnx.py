"""Compare the exported DISK UNet (onnxruntime) with the recorded PyTorch outputs."""
import os
import sys

import numpy as np
import onnxruntime as ort

onnx_dir = sys.argv[1]
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parity', 'ref')
sess = ort.InferenceSession(os.path.join(onnx_dir, 'disk_unet.onnx'), providers=['CPUExecutionProvider'])
for name in ('pano', 'photo2', 'mly_000000000000000'):
    x = np.load(os.path.join(REF, f'disk_in_{name}.npy'))
    z = np.load(os.path.join(REF, f'disk_out_{name}.npz'))
    out = sess.run(None, {'image': x})[0]
    heat = out[:, 128:129]
    print(name, x.shape, 'heatmap max abs diff', float(np.abs(heat - z['heat']).max()))
    kp = z['kp'].astype(int)
    d = out[0, :128][:, kp[:, 1], kp[:, 0]].T
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    print('   descriptor min cosine at keypoints', float((d * z['desc']).sum(1).min()))
