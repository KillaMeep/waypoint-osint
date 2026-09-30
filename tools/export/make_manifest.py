"""Write src-tauri/models.json: name, size and SHA-256 of every exported ONNX
file. The app embeds this manifest and refuses any download that doesn't
match it, so whoever hosts the files only serves bytes.

  python tools/export/make_manifest.py <onnx_dir>

Export the files first (all with the .venv-export interpreter):
  export_clip.py <dir>, export_disk_lightglue.py <dir>, export_dinov2.py <dir>,
  export_plonk.py <dir> osv5m, export_plonk.py <dir> yfcc, export_plonk.py <dir> inat

First-run setup downloads every file listed, all three PLONK models included.
Then upload every file listed in the manifest to the model host (see
MODEL_BASE_URL in src-tauri/src/lib.rs), e.g.
  huggingface-cli upload <user>/waypoint-models <dir> . --include "*.onnx"
"""
import hashlib
import json
import os
import sys

FILES = [
    'streetclip_vision.onnx',
    'disk_unet.onnx',
    'lg_front.onnx',
    *[f'lg_layer{i}.onnx' for i in range(9)],
    'lg_post.onnx',
    'lg_assign.onnx',
    'plonk_osv5m_step.onnx',
    'dinov2_vitl14_reg.onnx',  # YFCC and iNaturalist encoder
    'plonk_yfcc_step.onnx',
    'plonk_inat_step.onnx',
]

src = sys.argv[1]
out = []
for name in FILES:
    path = os.path.join(src, name)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    out.append({'name': name, 'bytes': os.path.getsize(path), 'sha256': h.hexdigest()})
    print(f'{name:28s} {out[-1]["bytes"] / 1e6:8.1f} MB  {out[-1]["sha256"][:16]}')

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src-tauri', 'models.json')
with open(dest, 'w', newline='\n') as f:
    json.dump({'files': out}, f, indent=2)
    f.write('\n')
print('total', round(sum(o['bytes'] for o in out) / 1e9, 2), 'GB ->', os.path.normpath(dest))
