#!/usr/bin/env python3
"""
Retrieval-based geolocation refinement (PIGEON-style): once PLONK has narrowed
a photo down to a candidate area, pull real street-level photos from that area
via the Mapillary API and rank them by visual similarity to the target photo,
using PLONK's own embedder (StreetCLIP/DINOv2) so the comparison happens in
the same feature space the model was trained on.

This is a fundamentally different, stronger signal than PLONK's own generative
sampling: instead of the model guessing again, it's comparing against actual
nearby photos. A high-similarity match is meaningful evidence; PLONK's cluster
weight narrows *where* to look, this narrows *which exact spot*.

Requires a free Mapillary API token, passed in by the GUI via --mapillary_token
(set it in the app's Settings screen). Without a token this stage is skipped.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

MAPILLARY_IMAGES_URL = 'https://graph.mapillary.com/images'
BBOX_DEG = 0.008  # Mapillary bbox search requires bbox < 0.01 deg square
MAX_WORKERS = 20  # Mapillary's rate limit (10k/min) has huge headroom for this
SEARCH_WORKERS = 256  # tile-scan requests are tiny (a few small JSON records each) and
                       # purely round-trip-bound, not bandwidth-bound. 128 workers already
                       # got a ~1.5k-tile scan down to ~30s (~50 req/s) — still well under
                       # Mapillary's 10k/min (~166 req/s) limit, so there's headroom to push
                       # further. Stopped short of 4x/512 here: if per-request latency drops
                       # as concurrency rises, sustained throughput could approach the rate
                       # limit itself and start tripping 429s, which would net lose time via
                       # retries/backoff rather than gain it.

# One pooled session per process, reused across every tile query in a run:
# requests.get() with no session opens a fresh TCP+TLS connection per call,
# and with 1000+ tiles that handshake overhead dwarfs the actual response
# time. A shared Session's connection pool lets urllib3 keep connections
# alive and reuse them across threads instead.
_search_session = requests.Session()
_search_session.mount('https://', requests.adapters.HTTPAdapter(
    pool_connections=SEARCH_WORKERS, pool_maxsize=SEARCH_WORKERS))


def cluster_radius_km(cluster: dict, min_km: float = 3.0, max_km: float = 15.0) -> float:
    """Size the search radius from PLONK's own reported spread for this cluster,
    instead of a fixed guess — a tight cluster gets a tight search, a wide/uncertain
    cluster gets a wider one (capped, since cost scales with area)."""
    lat_km = cluster.get('lat_std', 0) * 111.0
    lon_km = cluster.get('lon_std', 0) * 111.0 * max(np.cos(np.radians(cluster['lat'])), 0.1)
    spread_km = float(np.hypot(lat_km, lon_km))
    return float(np.clip(spread_km, min_km, max_km))


def _bbox_tiles(lat: float, lon: float, radius_km: float):
    """Cover a circular search area with a grid of small bbox tiles (Mapillary
    caps bbox queries to <0.01 degrees square, so a wide radius needs tiling)."""
    lat_span = radius_km / 111.0
    lon_span = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.1))

    n_lat = max(1, int(np.ceil((2 * lat_span) / BBOX_DEG)))
    n_lon = max(1, int(np.ceil((2 * lon_span) / BBOX_DEG)))

    tiles = []
    for i in range(n_lat):
        for j in range(n_lon):
            tile_lat = lat - lat_span + i * BBOX_DEG
            tile_lon = lon - lon_span + j * BBOX_DEG
            tiles.append((tile_lat, tile_lon, tile_lat + BBOX_DEG, tile_lon + BBOX_DEG))
    return tiles


def _query_tile(tile, token: str, per_tile_limit: int):
    min_lat, min_lon, max_lat, max_lon = tile
    params = {
        'access_token': token,
        'fields': 'id,thumb_1024_url,computed_geometry',
        'bbox': f'{min_lon},{min_lat},{max_lon},{max_lat}',
        'limit': per_tile_limit,
    }
    try:
        resp = _search_session.get(MAPILLARY_IMAGES_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get('data', [])
    except Exception as e:
        logger.debug(f'Mapillary tile query failed: {e}')
        return []


def search_nearby_images(lat: float, lon: float, radius_km: float, token: str,
                          per_tile_limit: int = 5, max_images: int = 150, on_progress=None):
    """Query Mapillary for real street-level photos within radius_km of a point.
    Tile queries run concurrently — Mapillary's rate limit (10k/min) has plenty
    of headroom, and serial per-tile requests don't scale to wider radii.

    on_progress(completed, total), if given, is called after each tile finishes
    — with wide radii this can be 1000+ tiles and take a while, so it's worth
    reporting alongside the download/verify phases rather than leaving the UI
    blank until the scan finishes."""
    tiles = _bbox_tiles(lat, lon, radius_km)
    logger.info(f'Mapillary: scanning {len(tiles)} tiles within {radius_km:.1f}km of ({lat:.4f},{lon:.4f}) ({SEARCH_WORKERS} concurrent)...')

    seen_ids = set()
    results = []
    total_tiles = len(tiles)
    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
        futures = [pool.submit(_query_tile, t, token, per_tile_limit) for t in tiles]
        for i, future in enumerate(as_completed(futures), 1):
            if on_progress:
                on_progress(i, total_tiles)
            for item in future.result():
                if item['id'] in seen_ids:
                    continue
                seen_ids.add(item['id'])
                geom = item.get('computed_geometry', {}).get('coordinates')
                if not geom or not item.get('thumb_1024_url'):
                    continue
                results.append({
                    'id': item['id'],
                    'lon': geom[0],
                    'lat': geom[1],
                    'thumb_url': item['thumb_1024_url'],
                })

    logger.info(f'Mapillary: found {len(results)} candidate images')
    return results[:max_images]


def _download_image(url: str):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert('RGB')


def _download_all(candidates: list, on_progress=None):
    """Download thumbnails concurrently; embedding itself stays sequential
    (GPU-bound, and PLONK's embedder doesn't actually batch internally).

    on_progress(completed, total), if given, is called after each download
    completes — this is the slow part of the stage (network-bound, can take
    minutes for 150 images), so it's the most useful place for a UI to show
    real percentage progress rather than an indefinite spinner."""
    images = {}
    total = len(candidates)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_image, c['thumb_url']): c['id'] for c in candidates}
        for i, future in enumerate(as_completed(futures), 1):
            cand_id = futures[future]
            try:
                images[cand_id] = future.result()
            except Exception as e:
                logger.debug(f'Skipping candidate {cand_id}: {e}')
            if on_progress:
                on_progress(i, total)
    return images


def _embed(pipeline, image: Image.Image):
    emb = pipeline.cond_preprocessing({'img': [image]})['emb']
    return emb.detach().cpu().numpy()[0]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def refine_with_retrieval(pipeline, image_path: str, cluster: dict, token: str,
                           radius_km: float = None, max_images: int = 150, top_k: int = 5,
                           verify: bool = True, verify_top_n: int = 15, on_progress=None):
    """Find the most visually similar real photos near a candidate cluster.
    Returns a list of top matches (each with similarity, coords, thumb/page URLs).

    radius_km: if None, auto-sized from the cluster's own reported spread
    (cluster_radius_km) instead of a fixed guess — searching a fixed 3km
    radius around a cluster centroid that's itself only accurate to tens of
    km means the true spot may not even be inside the search area.

    verify: if True, re-scores the top verify_top_n by-similarity candidates
    with DISK+LightGlue geometric matching (see verify_utils) — embedding
    cosine similarity alone can't reliably tell "same street" from "similar-
    looking street," geometric inlier count can.

    on_progress(phase, completed, total), if given, is called during the
    slow parts of this stage ('search', 'download', and 'verify') for a UI
    progress bar."""
    if radius_km is None:
        radius_km = cluster_radius_km(cluster)

    search_cb = (lambda i, n: on_progress('search', i, n)) if on_progress else None
    candidates = search_nearby_images(cluster['lat'], cluster['lon'], radius_km, token,
                                       max_images=max_images, on_progress=search_cb)
    if not candidates:
        return []

    logger.info(f'Mapillary: downloading {len(candidates)} images ({MAX_WORKERS} concurrent)...')
    download_cb = (lambda i, n: on_progress('download', i, n)) if on_progress else None
    images = _download_all(candidates, on_progress=download_cb)

    target_img = Image.open(image_path).convert('RGB')
    target_emb = _embed(pipeline, target_img)

    scored = []
    for cand in candidates:
        img = images.get(cand['id'])
        if img is None:
            continue
        emb = _embed(pipeline, img)
        sim = _cosine_sim(target_emb, emb)
        scored.append({
            'similarity': round(sim, 4),
            'lat': cand['lat'],
            'lon': cand['lon'],
            'mapillary_url': f"https://www.mapillary.com/app/?pKey={cand['id']}&focus=photo",
            '_id': cand['id'],
        })

    scored.sort(key=lambda c: c['similarity'], reverse=True)

    if verify and scored:
        from verify_utils import verify_candidates
        top = scored[:verify_top_n]
        for c in top:
            c['_img'] = images.get(c['_id'])
        logger.info(f'Mapillary: geometrically verifying top {len(top)} candidates (DISK+LightGlue)...')
        verify_cb = (lambda i, n: on_progress('verify', i, n)) if on_progress else None
        verify_candidates(image_path, top, image_key='_img', on_progress=verify_cb)
        for c in top:
            c.pop('_img', None)
        rest = scored[verify_top_n:]
        scored = top + rest

    for c in scored:
        c.pop('_id', None)

    return scored[:top_k]
