"""Run the exported LightGlue graphs with onnxruntime (same loop as the Rust
code) and compare with kornia's prod matcher on the recorded pairs."""
import os
import sys

import numpy as np
import onnxruntime as ort

onnx_dir = sys.argv[1]
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parity', 'ref')
S = lambda n: ort.InferenceSession(os.path.join(onnx_dir, n), providers=['CPUExecutionProvider'])
front, post, assign = S('lg_front.onnx'), S('lg_post.onnx'), S('lg_assign.onnx')
layers = [S(f'lg_layer{i}.onnx') for i in range(9)]


def norm(k):
    size = k.max(0)
    return ((k - size / 2) / (size.max() / 2))[None].astype(np.float32)


def conf_thr(i):
    return np.float32(min(max(0.8 + 0.1 * np.exp(-4.0 * i / 9), 0), 1))


for name in ('pano__self_crop', 'pano__photo2', 'photo2__self_crop'):
    z = np.load(os.path.join(REF, f'match_{name}.npz'))
    kp0, kp1, de0, de1 = z['kp1'], z['kp2'], z['desc1'][None], z['desc2'][None]
    m, n = len(kp0), len(kp1)
    d0, d1, e0, e1 = front.run(None, {'kpts0': norm(kp0), 'desc0': de0, 'kpts1': norm(kp1), 'desc1': de1})
    ind0, ind1 = np.arange(m), np.arange(n)
    last = 0
    for i in range(9):
        last = i
        d0, d1 = layers[i].run(None, {'desc0': d0, 'desc1': d1, 'enc0': e0, 'enc1': e1})
        if i == 8:
            continue
        t0, t1, m0_, m1_ = post.run(None, {'desc0': d0, 'desc1': d1, 'layer': np.array(i, dtype=np.int64)})
        thr = conf_thr(i)
        ratio = np.float32(1.0) - np.float32((np.concatenate([t0, t1], -1) < thr).sum()) / np.float32(m + n)
        print(f'  layer {i}: ratio {ratio:.4f}, tok0 mean {t0.mean():.4f}')
        if ratio > 0.95:
            break
        if d0.shape[1] > 1536:
            keep = np.where((m0_[0] > 0.01) | (t0[0] <= thr))[0]
            ind0, d0, e0 = ind0[keep], d0[:, keep], e0[:, :, :, keep]
        if d1.shape[1] > 1536:
            keep = np.where((m1_[0] > 0.01) | (t1[0] <= thr))[0]
            ind1, d1, e1 = ind1[keep], d1[:, keep], e1[:, :, :, keep]
    mm, _ = assign.run(None, {'desc0': d0, 'desc1': d1, 'layer': np.array(last, dtype=np.int64)})
    valid = mm[0] > -1
    got = {(int(a), int(b)) for a, b in zip(ind0[np.where(valid)[0]], ind1[mm[0][valid]])}
    want = {tuple(x) for x in z['idxs'].tolist()}
    print(f'{name}: onnx(ort python) {len(got)} kornia {len(want)} common {len(got & want)} (stopped layer {last})')

# debug
z = np.load(os.path.join(REF, 'match_pano__self_crop.npz'))
kp0, kp1, de0, de1 = z['kp1'], z['kp2'], z['desc1'][None], z['desc2'][None]
d0, d1, e0, e1 = front.run(None, {'kpts0': norm(kp0), 'desc0': de0, 'kpts1': norm(kp1), 'desc1': de1})
print('front: d0 sum %.4f d1 sum %.4f e0 sum %.4f e1 sum %.4f' % (d0.sum(), d1.sum(), e0.sum(), e1.sum()), d0.shape, e0.shape)
print('input: kp0[0]=', kp0[0], 'desc0 len', de0.size, 'sum %.4f' % de0.sum(), 'first', de0.reshape(-1)[:3])
