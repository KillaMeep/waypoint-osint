"""Copy the small subset of reference data that the Rust unit tests need into
waypoint-core/tests/fixtures (so `cargo test` works without the full ref/ set)."""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'ref')
DST = os.path.abspath(os.path.join(HERE, '..', '..', 'waypoint-core', 'tests', 'fixtures'))
os.makedirs(DST, exist_ok=True)

for f in ('sun_ref.json', 'plonk_clusters.json', 'pano_samples_s1.npy', 'pano_samples_s2.npy'):
    shutil.copy(os.path.join(REF, f), os.path.join(DST, f))

for name in ('pano__self_crop',):
    z = np.load(os.path.join(REF, f'match_{name}.npz'))
    np.savez_compressed(os.path.join(DST, f'match_{name}.npz'), kp1=z['kp1'], kp2=z['kp2'], idxs=z['idxs'], mask=z['mask'])

# trim the astral table's payload: keep as is (it is a few hundred KB)
for f in os.listdir(DST):
    print(f, os.path.getsize(os.path.join(DST, f)))
