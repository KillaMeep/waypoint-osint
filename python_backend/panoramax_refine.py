#!/usr/bin/env python3
"""
Panoramax retrieval-refinement — free, open, no API key (STAC-based street-level
imagery, https://panoramax.fr). Same retrieval-refinement pattern as
mapillary_refine/google_sv_refine: pull real nearby photos, rank by StreetCLIP
embedding similarity, then geometrically verify the top candidates.

Unlike Mapillary, a single bbox query covers the whole search radius directly
(no need to tile — Panoramax's STAC /search endpoint doesn't cap bbox size the
way Mapillary's Images API does).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

PANORAMAX_SEARCH_URL = 'https://api.panoramax.xyz/api/search'
MAX_WORKERS = 20


def _bbox_for_radius(lat: float, lon: float, radius_km: float):
    lat_span = radius_km / 111.0
    lon_span = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.1))
    return (lon - lon_span, lat - lat_span, lon + lon_span, lat + lat_span)


def search_nearby_images(lat: float, lon: float, radius_km: float, max_images: int = 150):
    min_lon, min_lat, max_lon, max_lat = _bbox_for_radius(lat, lon, radius_km)
    try:
        resp = requests.get(PANORAMAX_SEARCH_URL, params={
            'bbox': f'{min_lon},{min_lat},{max_lon},{max_lat}',
            'limit': min(max_images, 100),  # STAC search page size cap on this instance
        }, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f'Panoramax search failed: {e}')
        return []

    results = []
    for feat in data.get('features', []):
        coords = feat.get('geometry', {}).get('coordinates')
        thumb = feat.get('assets', {}).get('sd', {}).get('href') or feat.get('assets', {}).get('thumb', {}).get('href')
        if not coords or not thumb:
            continue
        results.append({
            'id': feat['id'],
            'lon': coords[0],
            'lat': coords[1],
            'thumb_url': thumb,
        })

    logger.info(f'Panoramax: found {len(results)} candidate images')
    return results[:max_images]


def _download_image(url: str):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert('RGB')


def _download_all(candidates: list, on_progress=None):
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


def refine_with_panoramax(pipeline, image_path: str, cluster: dict,
                           radius_km: float = None, max_images: int = 150, top_k: int = 5,
                           verify: bool = True, verify_top_n: int = 15, on_progress=None):
    if radius_km is None:
        from mapillary_refine import cluster_radius_km
        radius_km = cluster_radius_km(cluster)

    candidates = search_nearby_images(cluster['lat'], cluster['lon'], radius_km, max_images=max_images)
    if not candidates:
        return []

    logger.info(f'Panoramax: downloading {len(candidates)} images ({MAX_WORKERS} concurrent)...')
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
            'panoramax_url': f"https://api.panoramax.xyz/#focus=pic&pic={cand['id']}",
            '_id': cand['id'],
        })

    scored.sort(key=lambda c: c['similarity'], reverse=True)

    if verify and scored:
        from verify_utils import verify_candidates
        top = scored[:verify_top_n]
        for c in top:
            c['_img'] = images.get(c['_id'])
        logger.info(f'Panoramax: geometrically verifying top {len(top)} candidates (DISK+LightGlue)...')
        verify_cb = (lambda i, n: on_progress('verify', i, n)) if on_progress else None
        verify_candidates(image_path, top, image_key='_img', on_progress=verify_cb)
        for c in top:
            c.pop('_img', None)
        scored = top + scored[verify_top_n:]

    for c in scored:
        c.pop('_id', None)

    return scored[:top_k]
