#!/usr/bin/env python3
"""One-shot sampling throughput calibration, run once during setup so the GUI
can size a hardware-appropriate default for the Samples field.

VRAM turned out to be a poor proxy for this: PLONK's coarse-locate sampler
computes the image embedding once and just repeats it across batch_size (see
plonk.pipe.PlonkPipeline.__call__), so a 4096-sample batch only uses a few GB
regardless of card. What actually varies across hardware is compute
throughput, so this loads the real model and times real batches to measure
it directly instead of guessing from VRAM or GPU name.

A single small timed batch isn't enough either: measured back-to-back on the
same loaded model, throughput itself scales with batch size (a roughly-fixed
per-call cost -- ODE solver step count, CUDA kernel launch overhead -- gets
amortized better at larger batches), so timing only a small batch
systematically undercounts what a fast GPU can actually sustain at the
larger sizes we'd recommend. Escalating through increasing batch sizes on
the same predictor and stopping once a batch takes long enough to be a
reliable near-plateau read gives strong hardware a realistic number while
letting weak/CPU hardware bail out early instead of paying for a batch it
was never going to reach anyway.
"""
import json
import os
import tempfile
import time

from PIL import Image

from plonk_core import Predictor

WARMUP_BATCH = 64
CANDIDATE_BATCHES = [256, 1024, 4096]
TIME_BUDGET_SECONDS = 3.0  # stop escalating once a batch takes at least this long


def main():
    # num_samples here only sizes Predictor's own convenience wrapper; every
    # call below goes straight through predictor.pipeline with an explicit
    # batch_size instead, so one Predictor instance covers the warm-up and
    # every candidate batch.
    predictor = Predictor('osv5m', CANDIDATE_BATCHES[-1])

    fd, tmp_path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        Image.new('RGB', (512, 512), color=(120, 120, 120)).save(tmp_path)
        image = Image.open(tmp_path).convert('RGB')

        # Untimed warm-up: the first-ever call pays for cuDNN algorithm
        # autotuning, CUDA kernel JIT/launch setup, and torchdiffeq's first
        # Python-level trace through the ODE solver. None of that reflects
        # steady-state per-sample throughput, so it must not be in a timed
        # window.
        predictor.pipeline(image, batch_size=WARMUP_BATCH)

        batch_size, elapsed = CANDIDATE_BATCHES[0], None
        for batch_size in CANDIDATE_BATCHES:
            start = time.time()
            predictor.pipeline(image, batch_size=batch_size)
            elapsed = time.time() - start
            if elapsed >= TIME_BUDGET_SECONDS:
                break
    finally:
        os.unlink(tmp_path)

    samples_per_sec = batch_size / elapsed if elapsed > 0 else 0
    print(json.dumps({
        'event': 'calibration',
        'samples_per_sec': samples_per_sec,
        'elapsed': round(elapsed, 2),
        'batch_size': batch_size,
    }), flush=True)


if __name__ == '__main__':
    main()
