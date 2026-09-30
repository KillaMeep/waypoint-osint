---
license: other
license_name: mixed
license_link: https://huggingface.co/killameep/waypoint-models/blob/main/README.md
tags:
- onnx
- geolocation
- plonk
---

# Waypoint models

ONNX exports of the networks that [Waypoint](https://github.com/KillaMeep/waypoint-osint) runs through
ONNX Runtime, an image geolocation desktop app. The app downloads these files on first launch and checks
each against the SHA-256 below before using it.

**These are converted copies of other people's models. Each file keeps the license of the model it
came from.** The conversion changes the format only: the weights are the originals, exported with
`torch.onnx` by the scripts in `tools/export/` of the Waypoint repository, and checked against the
PyTorch originals (see `tools/parity/REPORT.md` there).

## License notice

- **`streetclip_vision.onnx` is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**,
  inherited from StreetCLIP. **Non-commercial use only.** Changes: the vision tower alone, exported to ONNX,
  with the CLS token (before the post-layernorm) as output.
- DINOv2, DISK and LightGlue files: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
  Changes: exported to ONNX; DINOv2's position embedding resampled to 336 px at export time; LightGlue
  split into one graph per stage.
- PLONK files: [MIT](https://github.com/nicolas-dufour/plonk/blob/main/LICENSE), Copyright (c) Nicolas Dufour.
  Changes: one step of the Riemannian flow sampler exported to ONNX.

Because of StreetCLIP, the default model set as a whole can be used for non-commercial purposes only.

## Files

| File | What | From | License | Size | SHA-256 |
|------|------|------|---------|------|---------|
| `streetclip_vision.onnx` | StreetCLIP vision tower | [geolocal/StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) | CC BY-NC 4.0 | 1214.3 MB | `f28cb3ff3f2ed69d43f31d8eea00498f48180c4078a09aa910ed039aba6ffa5d` |
| `disk_unet.onnx` | DISK keypoint network | [cvlab-epfl/disk](https://github.com/cvlab-epfl/disk) (`depth-save.pth`) | Apache 2.0 | 4.4 MB | `47ebf56f396249791915a101fba2f7346ea9ed36d60144b016af96c61dd7f98f` |
| `lg_front.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 0.1 MB | `3d26471ae3f835297a27d2fa54ae9db79720e119514d76e1261bd6b2462fe483` |
| `lg_layer0.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `418492373bda83d36252712f78cb1db5a1a118a8329358b4829b13f8b87f7179` |
| `lg_layer1.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `f6a2c9ac52cad88b15e9961829521bd1c44f43228f31a196b6cbf58d4af676eb` |
| `lg_layer2.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `3b0e41ee55d9b7a0ea95e884c6aeef0a0695cc96afc721383f26c63682c6f50c` |
| `lg_layer3.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `68c00b6bcbecb95627eb0caf9d0bd739fa706517c8ceb6d7b66b26047a3ebda8` |
| `lg_layer4.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `45f11d7d36a9835ae6b3815c375f86a3bd63a06b9e332699467f0a45e7a1e37e` |
| `lg_layer5.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `06208eafee116630fd98bfaa8ec97742788eb77be16a2eabc7ee3dd39c0fce4e` |
| `lg_layer6.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `ae8364c15dbb16befba2ec535258a0b618810f8b31c16d7b1e79117e270add96` |
| `lg_layer7.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `daa33b674cf7e50e041441213290592bb834b37240bdd17bf6dc7e88dc0a4a78` |
| `lg_layer8.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 5.1 MB | `169d0ca06bbc102c26b9ac1c9d62835f1c764299bd95e8f7a06249f30736d1de` |
| `lg_post.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 0.0 MB | `8930e67cdc4d0d212a860bf80ca10bd7726d83e52f512121cdef4d1a52c045b8` |
| `lg_assign.onnx` | LightGlue (DISK weights), one graph of the adaptive loop | [cvg/LightGlue](https://github.com/cvg/LightGlue) (`disk_lightglue`) | Apache 2.0 | 2.4 MB | `ddace2667d842a69725c8e6112b3fd1111f42cd27b51885205ae8fcea1390880` |
| `plonk_osv5m_step.onnx` | PLONK OSV-5M, one flow step | [nicolas-dufour/PLONK_OSV_5M](https://huggingface.co/nicolas-dufour/PLONK_OSV_5M) | MIT | 144.3 MB | `d0c5a8e14d0bf284cca9ff7d6e1074202a09c3a85b74bac77b9edd29590745d8` |
| `dinov2_vitl14_reg.onnx` | DINOv2 ViT-L/14 with registers | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache 2.0 | 1214.5 MB | `a52dddb65b9b4133a10c201a2408340bab150acc738ad641f5c8d1708ca6c031` |
| `plonk_yfcc_step.onnx` | PLONK YFCC, one flow step | [nicolas-dufour/PLONK_YFCC](https://huggingface.co/nicolas-dufour/PLONK_YFCC) | MIT | 144.3 MB | `7208345304e378431462146dacc84899b95c9d3ca13f6e790842ce511dd13af7` |
| `plonk_inat_step.onnx` | PLONK iNaturalist, one flow step | [nicolas-dufour/PLONK_iNaturalist](https://huggingface.co/nicolas-dufour/PLONK_iNaturalist) | MIT | 36.7 MB | `04a00b7fd0db6c0aecebeb744dc39ea13be94830017bff2ddd0780317331966d` |

## Credits

- StreetCLIP: Haas, Alberti, Skreta. *Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization*, arXiv:2302.00275.
- PLONK: Dufour, Picard, Kalogeiton, Landrieu. *Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation*, CVPR 2025, arXiv:2412.06781.
- DINOv2: Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193; registers: Darcet et al., arXiv:2309.16588.
- DISK: Tyszkiewicz, Fua, Trulls. *DISK: Learning local features with policy gradient*, NeurIPS 2020.
- LightGlue: Lindenberger, Sarlin, Pollefeys. *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023.
