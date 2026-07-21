#!/usr/bin/env python3
"""
Google Street View retrieval-refinement — free, no API key, no billing account.

Reuses the technique from netryx-astra-v2's own indexer (this toolkit's
tools/netryx-astra-v2, see its test_super.py): Google Maps' internal,
undocumented GeoPhotoService.SingleImageSearch endpoint returns nearby
panorama IDs + exact coordinates for free, and streetviewpixels-pa.googleapis.com
serves the raw panorama tiles directly, as long as the request carries an
Origin/Referer matching Google Maps itself (no cookies/session/key needed).

Unlike Mapillary/KartaView (crowdsourced, patchy suburban coverage), Google
actually drove and photographed nearly every residential street, so this is
the fix for the "moderate similarity, wrong match" problem seen in suburban
testing. Also unlike Mapillary's blind area-grid tiling, panoramas only exist
along roads — so this samples points along real OSM road geometry near the
candidate (reusing sun_refine's Overpass query) instead of scanning the whole
2D area, which is both cheaper and far more likely to hit real coverage.
"""
import io
import itertools
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

import overpass_utils

logger = logging.getLogger(__name__)

PANOIDS_URL = ("https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
               "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}!2d50!3m10!2m2!1sen!2sGB"
               "!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5")
TILE_URL = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom=2&nbt=1&fover=2"
TILE_HEADERS = {
    "origin": "https://www.google.com",
    "referer": "https://www.google.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
}
IMGX, IMGY = 4, 2  # panorama tile grid at zoom=2 (matches netryx-astra-v2)
TILE_SIZE = 512
MAX_WORKERS = 20
ROAD_SAMPLE_SPACING_M = 60  # ~ the panoid search endpoint's own 50m implicit radius


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(a))


def get_road_sample_points(lat: float, lon: float, radius_km: float, spacing_m: float = ROAD_SAMPLE_SPACING_M, max_points: int = 400):
    """Fetch nearby road geometry from OpenStreetMap and sample points along it
    every spacing_m meters — panoramas only exist along roads, so this is a far
    smaller and more useful search set than scanning the whole 2D area.

    A wide-radius `around` query can be too expensive for the free Overpass
    instances (504 Gateway Timeout) regardless of retries/mirrors — retrying
    the identical query just repeats the same timeout, so on failure this
    shrinks the radius instead and tries again, down to a floor."""
    data = None
    tried_radius = radius_km
    for _ in range(3):
        radius_m = int(tried_radius * 1000)
        overpass_ql = f"""
        [out:json][timeout:25];
        way(around:{radius_m},{lat},{lon})[highway];
        out geom;
        """
        data = overpass_utils.query(overpass_ql)
        if data is not None:
            if tried_radius < radius_km:
                logger.info(f'Google SV: road query succeeded at reduced radius {tried_radius:.1f}km (original {radius_km:.1f}km was too heavy)')
            break
        if tried_radius <= 1.0:
            break
        tried_radius = max(1.0, tried_radius / 2)
        logger.debug(f'Google SV: road query timed out, retrying at smaller radius {tried_radius:.1f}km')

    if data is None:
        return []

    points = []
    for element in data.get('elements', []):
        geom = element.get('geometry', [])
        for i in range(len(geom) - 1):
            lat1, lon1 = geom[i]['lat'], geom[i]['lon']
            lat2, lon2 = geom[i + 1]['lat'], geom[i + 1]['lon']
            seg_len_m = _haversine_m(lat1, lon1, lat2, lon2)
            n_samples = max(1, int(seg_len_m // spacing_m))
            for t in np.linspace(0, 1, n_samples, endpoint=False):
                points.append((lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1)))

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = [points[i] for i in idx]
    return points


def _query_panoids(lat: float, lon: float):
    try:
        resp = requests.get(PANOIDS_URL.format(lat=lat, lon=lon), timeout=10)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return []

    matches = re.findall(r'"([A-Za-z0-9_-]{22})"', text)
    out = []
    for panoid in matches:
        latlon = re.findall(r'"' + re.escape(panoid) + r'".+?\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)', text)
        if latlon:
            plat, plon = map(float, latlon[0])
        else:
            plat, plon = lat, lon
        out.append({'panoid': panoid, 'lat': plat, 'lon': plon})
    return out


def search_panoramas(lat: float, lon: float, radius_km: float, max_images: int = 150):
    """Find nearby Street View panorama IDs by sampling points along real roads."""
    sample_points = get_road_sample_points(lat, lon, radius_km)
    if not sample_points:
        logger.warning('Google SV: no road geometry found nearby, cannot sample panoids.')
        return []

    logger.info(f'Google SV: probing {len(sample_points)} road-sampled points within {radius_km:.1f}km ({MAX_WORKERS} concurrent)...')

    seen_ids = set()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_query_panoids, plat, plon) for plat, plon in sample_points]
        for future in as_completed(futures):
            for p in future.result():
                if p['panoid'] in seen_ids:
                    continue
                seen_ids.add(p['panoid'])
                results.append(p)

    logger.info(f'Google SV: found {len(results)} distinct panoramas')
    return results[:max_images]


def _download_tile(x, y, panoid):
    url = TILE_URL.format(panoid=panoid, x=x, y=y)
    for _ in range(2):
        try:
            resp = requests.get(url, headers=TILE_HEADERS, timeout=10)
            if resp.status_code == 200:
                return x, y, resp.content
        except Exception:
            continue
    return x, y, None


def download_panorama(panoid: str):
    """Download and stitch a panorama's tiles into a single PIL Image."""
    coords = list(itertools.product(range(IMGX), range(IMGY)))
    tiles = {}
    with ThreadPoolExecutor(max_workers=IMGX * IMGY) as pool:
        futures = [pool.submit(_download_tile, x, y, panoid) for x, y in coords]
        for future in as_completed(futures):
            x, y, data = future.result()
            if data:
                tiles[(x, y)] = data

    if not tiles:
        return None

    pano = Image.new('RGB', (IMGX * TILE_SIZE, IMGY * TILE_SIZE))
    for (x, y), data in tiles.items():
        try:
            tile_img = Image.open(io.BytesIO(data)).convert('RGB')
            pano.paste(tile_img, (x * TILE_SIZE, y * TILE_SIZE))
        except Exception:
            continue
    return pano


def _download_all_panoramas(panoids: list, on_progress=None):
    images = {}
    total = len(panoids)
    with ThreadPoolExecutor(max_workers=8) as pool:  # each panorama itself spawns 8 tile downloads
        futures = {pool.submit(download_panorama, p['panoid']): p['panoid'] for p in panoids}
        for i, future in enumerate(as_completed(futures), 1):
            pid = futures[future]
            try:
                img = future.result()
                if img is not None:
                    images[pid] = img
            except Exception as e:
                logger.debug(f'Skipping panorama {pid}: {e}')
            if on_progress:
                on_progress(i, total)
    return images


def _embed(pipeline, image: Image.Image):
    emb = pipeline.cond_preprocessing({'img': [image]})['emb']
    return emb.detach().cpu().numpy()[0]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def refine_with_google_sv(pipeline, image_path: str, cluster: dict,
                           radius_km: float = None, max_images: int = 150, top_k: int = 5,
                           verify: bool = True, verify_top_n: int = 15, on_progress=None):
    """Same retrieval-refinement idea as mapillary_refine, but sourcing real
    photos from Google's own Street View coverage instead of crowdsourced
    platforms — fixes the suburban coverage gap Mapillary has.

    verify: if True, re-scores the top verify_top_n by-similarity candidates
    with DISK+LightGlue geometric matching (see verify_utils).

    on_progress(phase, completed, total), if given, is called during the
    'download' and 'verify' phases for a UI progress bar."""
    if radius_km is None:
        from mapillary_refine import cluster_radius_km
        # Cap lower than Mapillary's own sizing: roads (and therefore panoramas)
        # exist everywhere, so a wide radius buys little here while making the
        # Overpass road-geometry query heavy enough to trip its rate limit.
        radius_km = cluster_radius_km(cluster, max_km=8.0)

    panoids = search_panoramas(cluster['lat'], cluster['lon'], radius_km, max_images=max_images)
    if not panoids:
        return []

    logger.info(f'Google SV: downloading+stitching {len(panoids)} panoramas...')
    download_cb = (lambda i, n: on_progress('download', i, n)) if on_progress else None
    images = _download_all_panoramas(panoids, on_progress=download_cb)
    if not images:
        return []

    target_img = Image.open(image_path).convert('RGB')
    target_emb = _embed(pipeline, target_img)

    scored = []
    for p in panoids:
        img = images.get(p['panoid'])
        if img is None:
            continue
        emb = _embed(pipeline, img)
        sim = _cosine_sim(target_emb, emb)
        scored.append({
            'similarity': round(sim, 4),
            'lat': p['lat'],
            'lon': p['lon'],
            'street_view_url': f"https://www.google.com/maps?q=&layer=c&cbll={p['lat']},{p['lon']}",
            '_panoid': p['panoid'],
        })

    scored.sort(key=lambda c: c['similarity'], reverse=True)

    if verify and scored:
        from verify_utils import verify_candidates
        top = scored[:verify_top_n]
        for c in top:
            c['_img'] = images.get(c['_panoid'])
        logger.info(f'Google SV: geometrically verifying top {len(top)} candidates (DISK+LightGlue)...')
        verify_cb = (lambda i, n: on_progress('verify', i, n)) if on_progress else None
        verify_candidates(image_path, top, image_key='_img', on_progress=verify_cb)
        for c in top:
            c.pop('_img', None)
        rest = scored[verify_top_n:]
        scored = top + rest

    for c in scored:
        c.pop('_panoid', None)

    return scored[:top_k]
