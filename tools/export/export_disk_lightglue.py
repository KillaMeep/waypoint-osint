"""Export DISK (UNet) and LightGlue (as modular graphs) to ONNX.

LightGlue in kornia is adaptive: it stops early once most tokens are
confident and prunes unmatchable keypoints between layers. That control flow
does not survive tracing, so the network is exported as separate graphs and
the (small) adaptive loop is reimplemented in Rust:

  disk_unet.onnx      image[1,3,H,W]                      -> out[1,129,H,W]   (128 descriptor ch + 1 heatmap ch)
  lg_front.onnx       kpts0,desc0,kpts1,desc1 (normalised) -> desc0',desc1',enc0,enc1
  lg_layer{0..8}.onnx desc0,desc1,enc0,enc1               -> desc0,desc1
  lg_post.onnx        desc0,desc1,layer                   -> token0,token1,matchability0,matchability1   (layer 0..7)
  lg_assign.onnx      desc0,desc1,layer                   -> matches0[1,M], mscores0[1,M]  (log-assignment + filter_matches)

`python export_disk_lightglue.py <out_dir>` also runs a torch-side
re-implementation of the loop and compares it with kornia's LightGlue on the
recorded reference pairs.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.feature as KF

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parity', 'ref')

disk = KF.DISK.from_pretrained('depth').eval()
lgm = KF.LightGlueMatcher('disk', {'flash': False}).eval()
lg = lgm.matcher
assert lg.conf.n_layers == 9 and lg.conf.descriptor_dim == 256 and lg.conf.num_heads == 4


class DiskUnet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.unet = d.unet

    def forward(self, x):
        return self.unet(x)


class Front(nn.Module):
    def __init__(self, lg):
        super().__init__()
        self.proj, self.posenc = lg.input_proj, lg.posenc

    def forward(self, k0, d0, k1, d1):
        return self.proj(d0), self.proj(d1), self.posenc(k0), self.posenc(k1)


class Layer(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, d0, d1, e0, e1):
        d0 = self.t.self_attn(d0, e0)
        d1 = self.t.self_attn(d1, e1)
        return self.t.cross_attn(d0, d1)


class Post(nn.Module):
    def __init__(self, lg):
        super().__init__()
        n = lg.conf.n_layers - 1
        self.register_buffer('wt', torch.stack([lg.token_confidence[i].token[0].weight[0] for i in range(n)]))
        self.register_buffer('bt', torch.stack([lg.token_confidence[i].token[0].bias[0] for i in range(n)]))
        self.register_buffer('wm', torch.stack([lg.log_assignment[i].matchability.weight[0] for i in range(n)]))
        self.register_buffer('bm', torch.stack([lg.log_assignment[i].matchability.bias[0] for i in range(n)]))

    def forward(self, d0, d1, layer):
        wt, bt, wm, bm = self.wt[layer], self.bt[layer], self.wm[layer], self.bm[layer]
        tok = lambda d: torch.sigmoid((d * wt).sum(-1) + bt)
        mat = lambda d: torch.sigmoid((d * wm).sum(-1) + bm)
        return tok(d0), tok(d1), mat(d0), mat(d1)


class Assign(nn.Module):
    def __init__(self, lg, th=0.1):
        super().__init__()
        n = lg.conf.n_layers
        self.th = th
        st = lambda f: torch.stack([f(lg.log_assignment[i]) for i in range(n)])
        self.register_buffer('wp', st(lambda m: m.final_proj.weight))
        self.register_buffer('bp', st(lambda m: m.final_proj.bias))
        self.register_buffer('wm', st(lambda m: m.matchability.weight[0]))
        self.register_buffer('bm', st(lambda m: m.matchability.bias[0]))

    def forward(self, d0, d1, layer):
        wp, bp, wm, bm = self.wp[layer], self.bp[layer], self.wm[layer], self.bm[layer]
        m0, m1 = d0 @ wp.T + bp, d1 @ wp.T + bp
        d = m0.shape[-1]
        m0, m1 = m0 / d ** 0.25, m1 / d ** 0.25
        sim = torch.einsum('bmd,bnd->bmn', m0, m1)
        z0 = ((d0 * wm).sum(-1) + bm).unsqueeze(-1)
        z1 = ((d1 * wm).sum(-1) + bm).unsqueeze(-1)
        certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        s0 = F.log_softmax(sim, 2)
        s1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores = s0 + s1 + certainties  # (b, m, n): the [:m,:n] block of the log assignment
        max0v, max0i = scores.max(2)
        max1v, max1i = scores.max(1)
        idx0 = torch.arange(max0i.shape[1], device=max0i.device)[None]
        idx1 = torch.arange(max1i.shape[1], device=max1i.device)[None]
        mutual0 = idx0 == max1i.gather(1, max0i)
        mutual1 = idx1 == max0i.gather(1, max1i)
        e0 = max0v.exp()
        zero = torch.zeros_like(e0)
        ms0 = torch.where(mutual0, e0, zero)
        valid0 = mutual0 & (ms0 > self.th)
        m0i = torch.where(valid0, max0i, torch.full_like(max0i, -1))
        return m0i, ms0


def export(mod, args, name, inputs, outputs, dyn):
    path = os.path.join(out_dir, name)
    torch.onnx.export(mod, args, path, input_names=inputs, output_names=outputs, dynamic_axes=dyn, opset_version=17, dynamo=False)
    print('exported', name, round(os.path.getsize(path) / 1e6, 2), 'MB', flush=True)
    return path


export(DiskUnet(disk).eval(), (torch.rand(1, 3, 128, 192),), 'disk_unet.onnx', ['image'], ['out'],
       {'image': {2: 'h', 3: 'w'}, 'out': {2: 'h', 3: 'w'}})
M, N = 300, 280
k0, k1 = torch.rand(1, M, 2) * 2 - 1, torch.rand(1, N, 2) * 2 - 1
d0, d1 = torch.randn(1, M, 128), torch.randn(1, N, 128)
front = Front(lg).eval()
export(front, (k0, d0, k1, d1), 'lg_front.onnx', ['kpts0', 'desc0', 'kpts1', 'desc1'], ['desc0o', 'desc1o', 'enc0', 'enc1'],
       {'kpts0': {1: 'm'}, 'desc0': {1: 'm'}, 'kpts1': {1: 'n'}, 'desc1': {1: 'n'}, 'desc0o': {1: 'm'}, 'desc1o': {1: 'n'}, 'enc0': {3: 'm'}, 'enc1': {3: 'n'}})
p0, p1, e0, e1 = front(k0, d0, k1, d1)
for i in range(9):
    export(Layer(lg.transformers[i]).eval(), (p0, p1, e0, e1), f'lg_layer{i}.onnx', ['desc0', 'desc1', 'enc0', 'enc1'], ['desc0o', 'desc1o'],
           {'desc0': {1: 'm'}, 'desc1': {1: 'n'}, 'enc0': {3: 'm'}, 'enc1': {3: 'n'}, 'desc0o': {1: 'm'}, 'desc1o': {1: 'n'}})
post, assign = Post(lg).eval(), Assign(lg).eval()
lay = torch.tensor(3)
export(post, (p0, p1, lay), 'lg_post.onnx', ['desc0', 'desc1', 'layer'], ['tok0', 'tok1', 'mat0', 'mat1'],
       {'desc0': {1: 'm'}, 'desc1': {1: 'n'}, 'tok0': {1: 'm'}, 'tok1': {1: 'n'}, 'mat0': {1: 'm'}, 'mat1': {1: 'n'}})
export(assign, (p0, p1, lay), 'lg_assign.onnx', ['desc0', 'desc1', 'layer'], ['matches0', 'mscores0'],
       {'desc0': {1: 'm'}, 'desc1': {1: 'n'}, 'matches0': {1: 'm'}, 'mscores0': {1: 'm'}})


# ---- torch re-implementation of kornia's adaptive loop using the modular pieces --------------------
def norm_kp(k):
    size = k.max(dim=1)[0].squeeze(0)  # (x_max, y_max), as LightGlueMatcher.forward derives it
    shift = size / 2
    scale = size.max() / 2
    return (k - shift[None, None]) / scale


def modular_match(kp0, ds0, kp1, ds1, depth_conf=0.95, width_conf=0.99, prune_th=1536):
    m, n = kp0.shape[1], kp1.shape[1]
    d0, d1, e0, e1 = front(norm_kp(kp0), ds0, norm_kp(kp1), ds1)
    ind0 = torch.arange(m)[None]
    ind1 = torch.arange(n)[None]
    i = 0
    for i in range(9):
        d0, d1 = Layer(lg.transformers[i])(d0, d1, e0, e1)
        if i == 8:
            continue
        t0, t1, mt0, mt1 = post(d0, d1, torch.tensor(i))
        thr = lg.confidence_thresholds[i]
        conf = torch.cat([t0, t1], -1)
        ratio = 1.0 - (conf < thr).float().sum() / (m + n)
        if ratio > depth_conf:
            break
        if d0.shape[-2] > prune_th:
            keep = (mt0 > (1 - width_conf)) | (t0 <= thr)
            k = torch.where(keep)[1]
            ind0, d0, e0 = ind0.index_select(1, k), d0.index_select(1, k), e0.index_select(-2, k)
        if d1.shape[-2] > prune_th:
            keep = (mt1 > (1 - width_conf)) | (t1 <= thr)
            k = torch.where(keep)[1]
            ind1, d1, e1 = ind1.index_select(1, k), d1.index_select(1, k), e1.index_select(-2, k)
    m0, _ = assign(d0, d1, torch.tensor(i))
    valid = m0[0] > -1
    a = ind0[0][torch.where(valid)[0]]
    b = ind1[0][m0[0][valid]]
    return torch.stack([a, b], -1), i


with torch.no_grad():
    for name in ('pano__self_crop',):
        z = np.load(os.path.join(REF, f'match_{name}.npz'))
        kp0, kp1 = torch.from_numpy(z['kp1'])[None], torch.from_numpy(z['kp2'])[None]
        de0, de1 = torch.from_numpy(z['desc1'])[None], torch.from_numpy(z['desc2'])[None]
        got, stop = modular_match(kp0, de0, kp1, de1)
        want = z['idxs']
        gs, ws = {tuple(x) for x in got.tolist()}, {tuple(x) for x in want.tolist()}
        print(f'{name}: modular {len(gs)} matches (stopped at layer {stop}), kornia prod {len(ws)}, common {len(gs & ws)}')
