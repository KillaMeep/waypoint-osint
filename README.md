<div align="center">

<img src="resources/icon.png" width="112" alt="Waypoint logo" />

# Waypoint

**Free, open-source image geolocation. Figure out where a photo was taken.**

![License](https://img.shields.io/badge/license-MIT-2f6feb)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D6)
![Tauri](https://img.shields.io/badge/Tauri-2-24C8DB?logo=tauri&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-1.90%2B-B7410E?logo=rust&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-DirectML-005CED?logo=onnx&logoColor=white)

</div>

<p align="center">
  <img src="assets/screenshot.png" width="920" alt="Waypoint results view: a candidate on an interactive map with ranked street-level matches" />
</p>

## What it is

Waypoint takes a single outdoor photo and estimates **where on Earth it was taken**. It generates a
coarse guess with a neural network. Then it checks the guess against the sun's position. Finally,
it confirms the location by matching the photo to real street-level imagery. Every candidate shows
up on an interactive map with a confidence score you can drill into.

Use Waypoint for OSINT research, verification, journalism, and geolocation CTFs. Everything runs
locally. Only the imagery lookups in the refinement stages send data from your machine.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/pipeline-light.svg">
  <img src="assets/pipeline-light.svg" alt="Pipeline: photo, PLONK coarse locate, sun and season plausibility, retrieval refinement, geometric verification, ranked candidates on the map">
</picture>

1. **Coarse localization.** [PLONK](https://github.com/nicolas-dufour/plonk), a generative model, samples a spread of plausible locations. Waypoint clusters the locations into weighted candidates.
2. **Sun / season plausibility.** Waypoint checks shadow direction and lighting against the sun's position for the implied place, date, and time. Guesses that do not match lose rank.
3. **Retrieval refinement.** Waypoint compares each candidate against nearby real photos from **Mapillary**, **Google Street View**, and **Panoramax**.
4. **Geometric verification.** Waypoint confirms promising matches with local-feature matching (**DISK + LightGlue**). The inlier count shows how strong a match is.
5. **Refine.** Pick any candidate or match. Waypoint then runs an exhaustive search of a small radius around it.

## Features

- **Interactive map** (Leaflet + OpenStreetMap) with candidate and per-source match pins, auto-fit to the results.
- **Confidence, uncertainty, and evidence.** Cluster weight, sample spread, and the sun-plausibility verdict, all surfaced per candidate.
- **One-click refine.** Send any result into a focused, exhaustively-verified re-search.
- **One map, two modes.** Locate finds candidates. Refine searches a small radius around any point you pick: from a candidate, from a match, or by clicking the map.
- **Drop a photo anywhere** on the window. The photo stays in view, with a full-size viewer for side-by-side comparison.
- **Light and dark theme**, a Stop button for long runs, and a log drawer when you want the raw events.
- **No paid APIs.** Google Street View, Panoramax, and OpenStreetMap need no keys. Mapillary is optional and free.
- **Three PLONK models.** OSV-5M for street scenes, YFCC for general photos, iNaturalist for nature shots. Switch in Settings.
- **Zero manual setup.** The app downloads its own inference engine and models on first launch. No Python required.
- **Any GPU, or none.** Inference runs through DirectML on NVIDIA, AMD and Intel GPUs, with a CPU fallback.

## Download

Download `Waypoint_<version>_x64-setup.exe` from the [latest release](https://github.com/KillaMeep/waypoint-osint/releases/latest)
and run it. It installs for your user account (no admin prompt) into `%LOCALAPPDATA%\Waypoint`, with a
Start menu shortcut, and uninstalls from **Settings → Apps**.

Waypoint keeps itself up to date. A few seconds after launch it checks the latest release; when a
newer build exists, it offers **Install and restart**. Every update is signed, and Waypoint refuses
one whose signature doesn't match. **Settings → Updates** shows your version and checks on demand.

You need:

- **Windows 10 / 11 (x64).** _(Waypoint does not yet support macOS or Linux.)_ WebView2 ships with both.
- **Microsoft Visual C++ 2015–2022 Redistributable (x64)**, which ONNX Runtime needs. Most machines already have it; if setup reports that the inference engine could not start, [install it](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
- **~3 GB free disk** for the engine and models, and an **internet connection** for first-run setup and the imagery lookups.
- **Optional: a DirectX 12 GPU** (NVIDIA, AMD or Intel) for faster inference, used through DirectML. Otherwise it runs on the CPU.

## Run from source

Also needs **[Node.js](https://nodejs.org/) 18+**, **[Rust](https://rustup.rs/) 1.90+** and the
**Visual Studio C++ Build Tools** (see the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)).

```bash
git clone https://github.com/KillaMeep/waypoint-osint.git
cd waypoint-osint
npm install
npm start
```

### First launch (automatic, one-time)

On the very first run, Waypoint downloads its **inference engine**:

- [ONNX Runtime](https://onnxruntime.ai/) and DirectML, from their official NuGet packages.
- The exported models (about 2.8 GB) from [the model host](https://huggingface.co/killameep/waypoint-models):
  all three PLONK models, StreetCLIP, DINOv2, DISK and LightGlue.

Waypoint checks every file against a pinned SHA-256 hash before it uses the file. Then it measures
how fast your machine samples, to pick a default for **Samples**. Everything lands in the app's own
data folder (`%APPDATA%\geolocator-gui`).

If a download or the speed test fails, setup shows the error with **Retry**. Files that already
passed their hash check are kept. To wipe everything and set up again, use **Settings → Delete
downloads**. That also removes the Python runtime older versions of Waypoint installed.

### Choosing a model

**Settings → Model** picks the PLONK model that predicts the location:

| Model | Trained on | Image encoder |
|-------|------------|---------------|
| OSV-5M (default) | Street-level photos | StreetCLIP |
| YFCC | General Flickr photos: landmarks, landscapes, indoor and tourist shots | DINOv2 |
| iNaturalist | Nature photos: plants, animals, wild outdoor scenes | DINOv2 |

Setup installs all three. The retrieval stages rank street-level imagery with the chosen model's
encoder, as the original PyTorch pipeline does. An install from an older version runs setup again on
launch, which fetches only the missing files.

### Optional: Mapillary token

The Mapillary refinement stage needs a free API token.

1. Open **Settings**.
2. Paste a token from [mapillary.com/dashboard/developers](https://www.mapillary.com/dashboard/developers).
3. Save.

Without a token, Waypoint skips the Mapillary stage. Google Street View, Panoramax, and
OpenStreetMap work with no keys at all.

## Building

```bash
npm run dist        # standalone exe: src-tauri/target/release/waypoint.exe
npm run installer   # signed NSIS installer: src-tauri/target/release/bundle/nsis/
```

The frontend is embedded in the exe; the engine and models download on first launch. The installer
build also signs the updater artifact, so it needs the private key in `TAURI_SIGNING_PRIVATE_KEY`
(and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD=""`). The public key is in `src-tauri/tauri.conf.json`.

Every push to `main` builds the installer in GitHub Actions (after the Rust test suite passes) and
publishes it as release `v<major>.<minor>.<run number>`, together with the `latest.json` manifest
the in-app updater reads. The workflow keeps the two newest releases.

### Publishing the models

The app downloads the ONNX models from `MODEL_BASE_URL` in `src-tauri/src/lib.rs`, and checks them
against `src-tauri/models.json`. To produce and publish them:

1. Export the models with a Python environment that has `torch`, `diff-plonk`, `kornia`,
   `transformers`, `onnx` and `onnxruntime`:
   `tools/export/export_clip.py`, `tools/export/export_disk_lightglue.py`, `tools/export/export_dinov2.py`,
   and `tools/export/export_plonk.py` once per model (`osv5m`, `yfcc`, `inat`). Each takes the output
   folder first.
2. Run `python tools/export/make_manifest.py <folder>` to rewrite `src-tauri/models.json`.
3. Upload every `.onnx` file to the host. Check each model's license before you redistribute it.

For testing, set `WAYPOINT_MODEL_URL` to another host (a local HTTP server works), and
`WAYPOINT_DATA_DIR` to a scratch folder.

### Parity with the original PyTorch pipeline

Waypoint started as a Python + PyTorch app; that pipeline is kept in
`tools/parity/reference_pipeline/`, and `tools/parity/` records reference outputs from it. The ignored tests in
`waypoint-core/tests/` compare the Rust port against them. See
[`tools/parity/REPORT.md`](tools/parity/REPORT.md) for the numbers.

## Performance

Measured on a Ryzen 7 9800X3D with an RTX 5080 (details in [`tools/parity/REPORT.md`](tools/parity/REPORT.md)):

| | Waypoint (Rust + ONNX) | Original PyTorch pipeline |
|---|---|---|
| Install size (all three models) | ~2.6 GB | ~9.8 GB |
| PLONK sampling, GPU (OSV-5M) | 1299 samples/s (DirectML) | 1235 samples/s (CUDA) |
| PLONK sampling, CPU (OSV-5M) | ~70 samples/s | 27–33 samples/s |
| Non-NVIDIA GPUs | Used, through DirectML | CPU only |

With the same input, the native engine reproduces PyTorch's samples to about a metre and gives
the same clusters.

## Under the hood

| Layer | Tech |
|-------|------|
| Desktop shell | Tauri 2 (Rust) + WebView2 |
| Pipeline | `waypoint-core` (Rust): orchestration, clustering, sun math, RANSAC, imagery clients |
| Inference | ONNX Runtime (DirectML, CPU fallback), in-process |
| Install & updates | NSIS installer (per user), signed updates via the Tauri updater and GitHub releases |
| Coarse geolocation | PLONK (OSV-5M, YFCC or iNaturalist), its flow sampler reimplemented in Rust |
| Image embedding | StreetCLIP (OSV-5M) · DINOv2 (YFCC, iNaturalist) |
| Sun/season check | Port of `astral` (sun position vs. OpenStreetMap road bearings) |
| Street-level imagery | Mapillary API · Google Street View · Panoramax |
| Feature matching | DISK + LightGlue, LightGlue's adaptive loop reimplemented in Rust |
| Geocoding / roads | Nominatim · OSM Overpass |
| Map UI | Leaflet + OpenStreetMap tiles |

## Ethical use

Waypoint is a research and verification tool. Use it responsibly and lawfully: for OSINT research,
journalism, imagery verification, and CTFs. **Don't use it to stalk, harass, or endanger anyone.**
Estimates are probabilistic, not proof. Always corroborate before acting on a result.

## License

[MIT](LICENSE). Free to use, modify, and distribute.

<div align="center">
<sub>Built on the shoulders of PLONK, StreetCLIP, DINOv2, DISK, LightGlue, ONNX Runtime, Tauri, Leaflet, OpenStreetMap, Mapillary, and Panoramax.</sub>
</div>
