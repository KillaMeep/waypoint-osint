# Rust port: parity report

How closely the Rust backend (`waypoint-core`) and the ONNX models reproduce the original
PyTorch pipeline. The reference outputs come from `tools/parity/ref_*.py`, run read-only against
the original Python environment. The test images are `test_pano.jpg` (a street panorama) and
`test_photo2.jpg`.

Machine: RTX 5080, Windows 11. ONNX Runtime 1.24.4 with DirectML 1.15.4.

## How to rerun

```powershell
# deterministic suites (fixtures committed in waypoint-core/tests/fixtures)
cd waypoint-core; cargo test --release

# model suites: need the runtime DLL, the exported models and tools/parity/ref
$env:WAYPOINT_ORT_DLL = "<dir>\onnxruntime.dll"; $env:WAYPOINT_ONNX_DIR = "<onnx dir>"
cargo test --release --test onnx_parity  -- --ignored --nocapture
cargo test --release --test plonk_parity -- --ignored --nocapture --test-threads 1
# the other PLONK models (references: tools/parity/ref_plonk_variants.py)
$env:WAYPOINT_PLONK = "yfcc"   # or "inat"
cargo test --release --test plonk_parity -- --ignored --nocapture --test-threads 1
```

## Phase 1: algorithms (no neural networks)

| Check | Result |
|---|---|
| astral sunrise, sunset and azimuth, 920 rows (103 polar) | 0 µs time difference, 2.8e-14° azimuth difference, same polar failures |
| Sun best-match, 96 cases | 96/96 identical |
| Sun image statistics (shadow offset, confidence) | pano -18.83 vs -18.86, conf 0.568 vs 0.573; photo2 -0.378 vs -0.393, conf 0.793 vs 0.791 (JPEG decode differences) |
| DBSCAN clustering, 5 sample sets | same counts, weights and noise fraction; centres within 1e-4° |
| RANSAC fundamental-matrix inliers vs OpenCV | 1251 vs 1246, 1109 vs 1119, 14 vs 16 (OpenCV's random sampling can't be reproduced) |

One known difference: where two road headings fit the sun equally well, Python's set iteration
order picks one and Rust picks the lowest.

## Phase 2: StreetCLIP, DISK, LightGlue (ONNX)

| Gate | Result |
|---|---|
| Embedding, same pixels as PyTorch | cosine 1.000000 |
| Embedding, Rust decode + resize, 2 test images + 21 real candidates | min cosine 0.999619, mean 0.999862 (gate 0.999) |
| Retrieval ranking of the 21 candidates | same top 1; same top 5 with places 4 and 5 swapped |
| LightGlue on the same DISK features | 1264/1264, 46/46, 1122/1122 identical match pairs, on both the CPU and DirectML providers |
| DISK keypoints vs kornia | 96.9–100% at the same positions (100% on lossless PNG) |
| Inliers, full Rust path vs PyTorch, control pair | 1240/1253 vs 1246/1264 (0.5% difference) |
| Inliers, 23 real pairs | 0–2 inliers apart on counts of 8–15 (same table on CPU and DirectML) |
| LightGlue speed per pair | 28–200 ms on DirectML, 450–560 ms on CPU |

The app runs every model, LightGlue included, on DirectML, and falls back to the CPU provider when
DirectML is unavailable. The gates above were run on both configurations.

## Phase 3: PLONK (ONNX step graph + Rust flow sampler)

The exported graph is one Euler step: `x + dt * net(x, gamma, emb)`, projected back onto the
sphere. The 250-step loop, the sigmoid(-7, 3) schedule and the noise are Rust.

| Gate | Result |
|---|---|
| Export check (Python): ONNX step loop vs PyTorch, 6 runs x 2048 samples | median 0.6 m, p99 4–6 m, worst 60 m |
| Same noise + reference embedding (Rust, DirectML) vs PyTorch samples | median 1.1 m, p99 6–10 m, worst 93 m |
| Clusters from those samples, 6 runs | identical: same counts, weights and noise fraction; centres within 1 m |
| Full Rust path (Rust decode, ONNX StreetCLIP, ONNX PLONK), same noise | top cluster 0.3–2.4 km from PyTorch's (limit 5 km); minor clusters 0.1–7.2 km (limit 2 standard errors, 10.6–15 km) |
| Own Gaussian noise, 3 x 8192 samples vs PyTorch's 3 x 8192 | top cluster 1.06 km apart; weight 0.935 vs 0.938 |
| Speed, batch 8192 | 1299 samples/s (DirectML) vs 1235 samples/s (PyTorch CUDA) |

About the full-path gate: the remaining gap comes from the embedding (cosine 0.9998, from
decoding the JPEG in Rust instead of Pillow). For scale, PyTorch's own top cluster moves about
9 km between two seeds. A cluster of ~100 samples with a ~100 km spread has a standard error
near 8 km, so minor clusters are held to two standard errors and the top cluster to a flat 5 km.

## Phase 4: YFCC and iNaturalist (DINOv2 + their step graphs)

PLONK_YFCC and PLONK_iNaturalist condition on DINOv2 ViT-L/14 with registers instead of
StreetCLIP. DINOv2 is exported with its 336 px position embedding resampled once at export time
(the model does the same resample on every call). The Python engine also ranks retrieval
candidates with this encoder, so the native engine does too.

| Gate | YFCC | iNaturalist |
|---|---|---|
| Export check (Python): ONNX step loop vs PyTorch, 6 runs | median 0.4–0.8 m, p99 9–18 m | median 0.8–1.1 m, p99 6–20 m |
| Same noise + reference embedding (Rust), 6 runs, DirectML and CPU | median 0.8–1.6 m; identical clusters | median 1.1–1.6 m; identical clusters |
| Own noise, 3 x 8192 vs PyTorch's 3 x 8192 | top 3 clusters within 0.6–2.8 km | top 3 within 1.3–6.8 km, once matched by position |
| Full Rust path, same noise | clusters survive: >= 93% of each top cluster's samples | >= 95.5% |
| Speed, batch 8192, DirectML (PyTorch CUDA) | 1301/s (939/s) | 4000/s (3078/s) |
| Speed, CPU | 67–73/s | 181–223/s |

DINOv2 alone:

| Gate | Result |
|---|---|
| ONNX vs PyTorch, same pixels | cosine 1.000000 |
| Rust crop + resize on Pillow's decode vs torchvision | bit-exact (max diff 0.000000) |
| Rust decode + preprocessing, 2 test images + 21 candidates | min cosine 0.999761, mean 0.999947; same retrieval top 1, 5/5 of top 5 |

Three gates were loosened after they failed on these models. Each failure came from the
clustering, not the port:

- **Clusters are matched by position, not rank.** iNaturalist's third and fourth clusters (New York
  1368 samples, Dallas 1359) are a near-tie, and their order flips between runs.
- **The full-path check compares membership.** With the same noise, sample i is the same draw on
  both sides. YFCC's Philadelphia and Richmond clusters are joined by a thin bridge of samples.
  PyTorch merges them with seed 1 (1191 samples) and splits them with seeds 2 and 3 (968 + 162).
  Rust splits them with seed 1. The check now follows each cluster's samples: at least 90% must
  land in at most two clusters on the other side. The worst case is 93.4%, from edge samples
  crossing DBSCAN's density threshold (OSV-5M already loses up to 3% this way).
- **Per-sample drift is a sanity bound (median < 5 km).** iNaturalist's output on a street photo
  is diffuse (60% of samples are noise), so a 0.99997-cosine embedding moves samples by 3.5 km
  median. Its clusters still land within 1–3 km of PyTorch's. The strict gate is on the input:
  embedding cosine >= 0.999.

The OSV-5M gates were rerun with the same code: unchanged results.

In the app (scratch data folder, local mirror; at the time YFCC and iNaturalist were separate
downloads, since folded into first-run setup): YFCC downloads (1.36 GB) and is selected.
iNaturalist then fetches only its 37 MB step graph and reuses DINOv2. A YFCC Locate on `test_pano.jpg`, 2 x 4096 samples, exits 0 in 114 s.
The log reads "ONNX Runtime ready on DirectMl ... + PLONK YFCC", and the top cluster is
(39.76, -75.36), weight 0.578 (PyTorch: (39.77, -75.33), weight 0.565). A run with an
uninstalled model selected stops with an error that names the model.
`waypoint-cli --model inat` runs natively as well.

## End to end (in the app, native engine, no Python present)

- Setup from an empty data folder: runtime from NuGet plus 1.41 GB of models, every file
  SHA-256-checked, then calibration. 33 s with the models on a local mirror.
- `DirectML.dll` loads from the app's own `ort` folder (1.15.4), not from System32.
- Locate on `test_pano.jpg`, 2 x 4096 samples: sampling 8.7 s with live progress, then sun
  refine, Street View and Panoramax. Top candidate (41.135, -73.110), weight 0.94. PyTorch gives
  (41.12, -73.11), weight 0.94.
- Stop takes effect in about 0.3 s during sampling and 0.9 s during retrieval. Closing the app
  leaves no processes.
- The Python engine still works through the same shell (Locate exit 0). An existing Python
  install upgrades from Settings in one click.
- Fallbacks: when the model host is unreachable, first-run setup installs Python instead. When a
  download fails its hash check, setup stops with Retry and starts no Python install. When the
  native engine is installed but fails to load, a run falls back to an installed Python engine
  (checked in the app and in `waypoint-cli`).

## CPU only (`WAYPOINT_ACCEL=cpu`)

Machine: Ryzen 7 9800X3D (8 cores, 16 threads). Same ONNX files and runtime as above.

| Check | Result |
|---|---|
| PLONK fixed-noise gate, 6 runs | identical clusters to PyTorch (same counts, weights, noise); median 0.8–1.1 m per sample |
| Embedding, DISK, LightGlue gates | same results as on DirectML (cosine min 0.999619; LightGlue 1264/46/1122 identical; control inliers 0.48%) |
| PLONK sampling speed | 67–74 samples/s at batches 256–4096 (flat); calibration measures 71/s, so the app defaults to 704 samples per run |
| PyTorch on the same CPU | 27–33 samples/s. The native engine is about 2.2x faster |
| Per-image costs | embedding ~370 ms, DISK 400–590 ms, LightGlue 370–490 ms per pair |

Full Locate on `test_pano.jpg` with the CPU defaults (704 x 3 samples, Mapillary on): 334 s, exit 0.
Sampling took 30 s. Each retrieval source spends most of its time embedding its 100–150 candidates
(~0.37 s each, about 150 s in all), plus ~13 s verifying 15 pairs. The same run on DirectML takes
about 170 s, almost all of it network. Top cluster (41.09, -73.18), weight 0.937; matches found on all
three sources.

The embedding step reports no progress events, so on CPU the progress bar pauses for up to a
minute per source while it runs.
