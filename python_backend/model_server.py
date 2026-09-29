#!/usr/bin/env python3
"""Long-lived model server: the only Python the Rust backend still needs.

Speaks NDJSON over stdin/stdout, one request per line, one response per line:

  {"id": 1, "op": "load", "model": "osv5m"}
  {"id": 2, "op": "sample", "image": "C:/x.jpg", "batch_size": 512, "seed": 7}   (seed optional)
  {"id": 3, "op": "embed", "images": ["a.png", "b.jpg"]}
  {"id": 4, "op": "match", "a": "target.jpg", "b": "cand.png"}
  {"id": 5, "op": "shutdown"}

Responses carry {"id": N, "ok": true, ...} or {"id": N, "ok": false, "error": "..."}.
Float arrays travel as base64 little-endian float32.

The neural pieces exposed here are exactly the ones Waypoint has not yet moved
to Rust: PLONK sampling, PLONK's own image embedder, and DISK + LightGlue
keypoint matching. Everything else (HTTP, clustering, sun math, RANSAC) is Rust.
"""
import base64
import json
import os
import sys

# The protocol owns the real stdout. Anything a library prints goes to stderr.
_proto = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', newline='\n')
sys.stdout = sys.stderr

import logging

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

_pipeline = None
_target_cache = {}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype='<f4').tobytes()).decode('ascii')


def _open(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def op_load(req):
    global _pipeline
    from plonk import PlonkPipeline
    from plonk_core import MODEL_ALIASES
    model = req.get('model', 'osv5m')
    _pipeline = PlonkPipeline(MODEL_ALIASES.get(model, model))
    return {}


def op_sample(req):
    import torch
    img = _open(req['image'])
    n = int(req['batch_size'])
    seed = req.get('seed')
    gen = None
    if seed is not None:
        gen = torch.Generator(device=_pipeline.device).manual_seed(int(seed))
    out = _pipeline(img, batch_size=n, generator=gen)
    return {'n': int(out.shape[0]), 'f32': _b64(out)}


def op_embed(req):
    embs = []
    for p in req['images']:
        e = _pipeline.cond_preprocessing({'img': [_open(p)]})['emb'].detach().cpu().numpy()[0]
        embs.append(e)
    arr = np.stack(embs) if embs else np.zeros((0, 0), 'float32')
    return {'n': int(arr.shape[0]), 'dim': int(arr.shape[1]) if arr.ndim == 2 and arr.shape[0] else 0, 'f32': _b64(arr)}


def op_match(req):
    import verify_utils
    key = req['a']
    if key not in _target_cache:
        _target_cache.clear()
        _target_cache[key] = _open(key)
    pts1, pts2, n1, n2 = verify_utils.match_points(_target_cache[key], _open(req['b']))
    return {'n_kp1': n1, 'n_kp2': n2, 'm': int(pts1.shape[0]), 'pts1': _b64(pts1), 'pts2': _b64(pts2)}


OPS = {'load': op_load, 'sample': op_sample, 'embed': op_embed, 'match': op_match}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get('id')
            if req.get('op') == 'shutdown':
                _proto.write(json.dumps({'id': rid, 'ok': True}) + '\n')
                _proto.flush()
                return
            res = OPS[req['op']](req)
            res.update({'id': rid, 'ok': True})
        except Exception as e:  # report and keep serving
            logging.exception('request failed')
            res = {'id': rid, 'ok': False, 'error': f'{type(e).__name__}: {e}'}
        _proto.write(json.dumps(res) + '\n')
        _proto.flush()


if __name__ == '__main__':
    main()
