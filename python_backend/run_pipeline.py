#!/usr/bin/env python3
"""
Geolocator GUI pipeline orchestrator.

Run standalone (spawned by the Electron main process); emits one JSON object
per line to stdout (NDJSON) so the GUI can show live step-by-step progress
instead of waiting on one big blocking call. Two modes:

  one   --image_path X [--mapillary_token T]
        Full pipeline: PLONK coarse locate -> sun/season refinement ->
        Mapillary/Google SV/Panoramax retrieval -> DISK+LightGlue geometric
        verification. Mirrors the "distinctive urban scene -> PLONK gets the
        metro right -> retrieval-refinement narrows within a few miles ->
        zoom-in verification" pipeline validated in the OSINT toolkit.

  point --image_path X --lat LAT --lon LON [--radius_km R]
        Zoom-in mode: skip PLONK, run retrieval+verification directly around
        an explicit point (typically the best match from a prior "one" run).

All stdout lines are JSON: {"event": "...", ...}. Anything unparseable on
stdout is a bug, not log noise. All logging goes to stderr instead.
"""
import argparse
import json
import logging
import sys
import time

import numpy as np

# Route all logging to stderr; stdout is reserved for the NDJSON protocol.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)


def emit(event: str, **kwargs):
    payload = {'event': event, 'ts': time.time(), **kwargs}
    print(json.dumps(payload, default=str), flush=True)


def _progress_emitter(stage):
    """Returns an on_progress(phase, completed, total) callback that emits
    NDJSON 'progress' events. The slow parts of each retrieval stage
    (downloading dozens-to-hundreds of images, then verifying them) are where
    a real percentage + ETA matters most, not a static spinner."""
    def _cb(phase, completed, total):
        emit('progress', stage=stage, phase=phase, completed=completed, total=total)
    return _cb


def run_retrieval_stages(pipeline, image_path, cluster, args, stage_prefix=''):
    """Run Mapillary + Google SV + Panoramax retrieval-refinement + geometric
    verification against a single candidate cluster/point. Emits progress
    events as each source completes."""
    verify_top_n = args.verify_top_n

    if args.mapillary_token:
        from mapillary_refine import refine_with_retrieval
        stage = f'{stage_prefix}mapillary'
        emit('stage_start', stage=stage)
        try:
            cluster['mapillary_matches'] = refine_with_retrieval(
                pipeline, image_path, cluster, args.mapillary_token,
                radius_km=args.radius_km, max_images=args.max_images,
                verify=not args.no_verify, verify_top_n=verify_top_n,
                on_progress=_progress_emitter(stage),
            )
        except Exception as e:
            logger.warning(f'Mapillary stage failed: {e}')
            cluster['mapillary_matches'] = []
        emit('stage_done', stage=stage, matches=cluster.get('mapillary_matches', []))
    else:
        emit('stage_skipped', stage=f'{stage_prefix}mapillary', reason='no token configured')

    from google_sv_refine import refine_with_google_sv
    stage = f'{stage_prefix}google_sv'
    emit('stage_start', stage=stage)
    try:
        cluster['google_sv_matches'] = refine_with_google_sv(
            pipeline, image_path, cluster,
            radius_km=args.radius_km, max_images=args.max_images,
            verify=not args.no_verify, verify_top_n=verify_top_n,
            on_progress=_progress_emitter(stage),
        )
    except Exception as e:
        logger.warning(f'Google SV stage failed: {e}')
        cluster['google_sv_matches'] = []
    emit('stage_done', stage=stage, matches=cluster.get('google_sv_matches', []))

    from panoramax_refine import refine_with_panoramax
    stage = f'{stage_prefix}panoramax'
    emit('stage_start', stage=stage)
    try:
        cluster['panoramax_matches'] = refine_with_panoramax(
            pipeline, image_path, cluster,
            radius_km=args.radius_km, max_images=args.max_images,
            verify=not args.no_verify, verify_top_n=verify_top_n,
            on_progress=_progress_emitter(stage),
        )
    except Exception as e:
        logger.warning(f'Panoramax stage failed: {e}')
        cluster['panoramax_matches'] = []
    emit('stage_done', stage=stage, matches=cluster.get('panoramax_matches', []))

    return cluster


def run_one(args):
    from plonk_core import Predictor

    emit('stage_start', stage='model_load')
    predictor = Predictor(args.model, args.num_samples)
    emit('stage_done', stage='model_load')

    emit('stage_start', stage='plonk_sampling')
    result = predictor.predict(args.image_path, args.cluster_radius_km, args.top_k, not args.no_geocode, args.num_runs)
    emit('stage_done', stage='plonk_sampling', clusters=result['clusters'], noise_frac=result['noise_frac'])

    if not result['clusters']:
        emit('done', result=result)
        return

    if not args.no_sun_refine:
        from sun_refine import refine_clusters
        emit('stage_start', stage='sun_refine')
        try:
            result['clusters'], sun_meta = refine_clusters(result['clusters'], args.image_path, args.fov)
        except Exception as e:
            logger.warning(f'Sun refine failed: {e}')
            sun_meta = None
        emit('stage_done', stage='sun_refine', clusters=result['clusters'], meta=sun_meta)

    for i, cluster in enumerate(result['clusters'][:args.retrieval_top_clusters]):
        emit('cluster_selected', index=i, lat=cluster['lat'], lon=cluster['lon'], weight=cluster['weight'])
        run_retrieval_stages(predictor.pipeline, args.image_path, cluster, args, stage_prefix=f'c{i}_')

    emit('done', result=result)


def run_point(args):
    from plonk_core import Predictor

    emit('stage_start', stage='model_load')
    predictor = Predictor(args.model, num_samples=1)
    emit('stage_done', stage='model_load')

    cluster = {'lat': args.lat, 'lon': args.lon, 'lat_std': 0, 'lon_std': 0, 'weight': 1.0, 'count': 0}
    emit('cluster_selected', index=0, lat=cluster['lat'], lon=cluster['lon'], weight=1.0)

    run_retrieval_stages(predictor.pipeline, args.image_path, cluster, args, stage_prefix='zoom_')

    emit('done', result={'clusters': [cluster], 'num_samples': 1, 'noise_frac': 0.0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['one', 'point'])
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--model', default='osv5m')
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--cluster_radius_km', type=float, default=100.0)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--no_geocode', action='store_true')
    parser.add_argument('--no_sun_refine', action='store_true')
    parser.add_argument('--no_verify', action='store_true')
    parser.add_argument('--verify_top_n', type=int, default=15)
    parser.add_argument('--mapillary_token', default=None)
    parser.add_argument('--radius_km', type=float, default=None)
    parser.add_argument('--max_images', type=int, default=150)
    parser.add_argument('--retrieval_top_clusters', type=int, default=1)
    parser.add_argument('--fov', type=float, default=65.0)
    parser.add_argument('--lat', type=float, default=None)
    parser.add_argument('--lon', type=float, default=None)
    args = parser.parse_args()

    try:
        if args.mode == 'one':
            run_one(args)
        else:
            if args.lat is None or args.lon is None:
                raise ValueError('point mode requires --lat and --lon')
            run_point(args)
    except Exception as e:
        logger.exception('Pipeline failed')
        emit('error', message=str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
