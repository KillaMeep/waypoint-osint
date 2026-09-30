"""Shared helpers for the parity harness. The reference outputs come from the
original PyTorch pipeline, kept in reference_pipeline/. Run with a Python 3.11
environment that has torch, torchvision and reference_pipeline/requirements_base.txt,
e.g. (PowerShell):

  $env:PYTHONDONTWRITEBYTECODE=1
  & <venv>\\Scripts\\python.exe tools\\parity\\ref_plonk.py
"""
import os, sys, json
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, 'reference_pipeline'))  # the original PyTorch pipeline
REF = os.path.join(HERE, 'ref')
os.makedirs(REF, exist_ok=True)
IMAGES = {
    'pano': 'test_pano.jpg',
    'photo2': 'test_photo2.jpg',
}
SEEDS = [1, 2, 3]


def save_json(name, obj):
    with open(os.path.join(REF, name), 'w') as f:
        json.dump(obj, f, indent=1, default=str)
