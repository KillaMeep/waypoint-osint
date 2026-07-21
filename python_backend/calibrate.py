#!/usr/bin/env python3
"""One-shot sampling throughput calibration, run once during setup so the GUI
can size a hardware-appropriate default for the Samples field.

VRAM turned out to be a poor proxy for this: PLONK's coarse-locate sampler
computes the image embedding once and just repeats it across batch_size (see
plonk.pipe.PlonkPipeline.__call__), so a 4096-sample batch only uses a few GB
regardless of card. What actually varies across hardware is compute
throughput, so this loads the real model and times a real batch to measure
it directly instead of guessing from VRAM or GPU name.
"""
import json
import os
import tempfile
import time

from PIL import Image

from plonk_core import Predictor

CALIBRATION_BATCH = 256


def main():
    predictor = Predictor('osv5m', CALIBRATION_BATCH)

    fd, tmp_path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        Image.new('RGB', (512, 512), color=(120, 120, 120)).save(tmp_path)
        start = time.time()
        predictor.sample(tmp_path)
        elapsed = time.time() - start
    finally:
        os.unlink(tmp_path)

    samples_per_sec = CALIBRATION_BATCH / elapsed if elapsed > 0 else 0
    print(json.dumps({'event': 'calibration', 'samples_per_sec': samples_per_sec, 'elapsed': round(elapsed, 2)}), flush=True)


if __name__ == '__main__':
    main()
