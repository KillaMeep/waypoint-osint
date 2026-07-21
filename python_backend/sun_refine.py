#!/usr/bin/env python3
"""
Fully automated sun-position refinement for PLONK candidate clusters.
No timestamp input required — everything is estimated from the image itself:

1. Season estimate: vegetation/ground color (green/brown/white) in the lower
   portion of the frame narrows the plausible time-of-year window.
2. Golden-hour check: overall warmth/sky-brightness pattern flags whether this
   looks like a sunrise/sunset shot (the case sun-azimuth analysis works for).
3. Sun bearing in frame: brightest+warmest region in the sky, mapped to an
   angular offset from camera-forward via an assumed field of view.
4. For each PLONK candidate location, sweep sunrise/sunset azimuths across the
   estimated season window and test against nearby road bearings (assuming
   the photographer was standing on/facing along a street) pulled from
   OpenStreetMap. The (date, sunrise/sunset, camera-heading) combo with the
   lowest angular error becomes that candidate's plausibility score.

This is a plausibility re-ranker for PLONK's existing candidates, not an
independent solver — treat close scores as "can't distinguish," and a
missing/low-confidence season or golden-hour read means the sun evidence for
that candidate is weak and shouldn't override PLONK's own cluster weight.
"""
import logging
from datetime import date, timedelta

import numpy as np
import requests
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from astral import Observer
from astral.sun import sun as sun_events, azimuth as sun_azimuth_at

import overpass_utils

logger = logging.getLogger(__name__)

# Representative sample dates for each hemisphere season, as (month, day) pairs.
# Northern hemisphere; flipped 6 months for southern latitudes.
SEASON_MONTHS_N = {
    'green': list(range(4, 10)),     # Apr-Sep: spring/summer growth
    'brown': [9, 10, 11, 12, 1],     # Sep-Jan: fall senescence / dormancy
    'snow': [11, 12, 1, 2, 3],       # Nov-Mar: snow cover plausible
    'unknown': list(range(1, 13)),
}


def _sample_dates(months: list, year: int = 2024, step_days: int = 6):
    """All dates in `year` whose month is in `months`, sparsely sampled."""
    dates = []
    for m in months:
        d = date(year, m, 1)
        while d.month == m:
            dates.append(d)
            d += timedelta(days=step_days)
    return dates


def estimate_season(image_path: str, lat_hint: float = 40.0):
    """Classify ground/vegetation color in the lower frame into green/brown/snow.
    Returns (label, sample_dates, confidence)."""
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    ground = arr[int(h * 0.55):, :, :]  # lower ~45% of frame: grass/ground/foreground

    hsv = rgb_to_hsv(ground)
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]

    vegetal_mask = sat > 0.15  # exclude near-gray pavement/concrete
    if vegetal_mask.sum() < 50:
        return 'unknown', _sample_dates(SEASON_MONTHS_N['unknown']), 0.0

    mean_hue = float(np.mean(hue[vegetal_mask]))
    mean_sat = float(np.mean(sat[vegetal_mask]))
    mean_val = float(np.mean(val[vegetal_mask]))
    white_frac = float(np.mean((sat < 0.1) & (val > 0.75)))

    if white_frac > 0.4:
        label, months = 'snow', SEASON_MONTHS_N['snow']
        confidence = min(white_frac, 0.8)
    elif 70 <= mean_hue <= 170 and mean_sat > 0.2:
        label, months = 'green', SEASON_MONTHS_N['green']
        confidence = min(mean_sat, 0.8)
    elif 20 <= mean_hue < 70:
        label, months = 'brown', SEASON_MONTHS_N['brown']
        confidence = min(mean_sat + (1 - mean_val) * 0.3, 0.8)
    else:
        label, months = 'unknown', SEASON_MONTHS_N['unknown']
        confidence = 0.0

    if lat_hint < 0:  # southern hemisphere: shift season window by 6 months
        months = [((m - 1 + 6) % 12) + 1 for m in months]

    return label, _sample_dates(months), round(confidence, 2)


def estimate_golden_hour(image_path: str):
    """Heuristic: is this plausibly a sunrise/sunset shot? Compares warm-hued
    sky fraction against the whole-sky brightness. Returns (is_golden, confidence)."""
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    sky = arr[: int(h * 0.6), :, :]

    r, g, b = sky[..., 0], sky[..., 1], sky[..., 2]
    warm_frac = float(np.mean((r - b) > 0.08))
    blue_frac = float(np.mean((b - r) > 0.08))

    is_golden = warm_frac > 0.25 and warm_frac > blue_frac
    confidence = round(min(warm_frac, 0.9), 2) if is_golden else round(min(blue_frac, 0.9), 2)
    return is_golden, confidence


def estimate_sun_bearing_offset(image_path: str, fov_deg: float = 65.0):
    """Estimate the sun/glow's horizontal angular offset from camera-forward."""
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    sky = arr[: int(h * 0.6), :, :]

    r, g, b = sky[..., 0], sky[..., 1], sky[..., 2]
    brightness = (r + g + b) / 3.0
    warmth = np.clip(r - b, 0, 255)
    score = brightness * 0.5 + warmth * 0.5

    threshold = np.percentile(score, 99)
    mask = score >= threshold
    if not mask.any():
        return None, 0.0

    _, xs = np.nonzero(mask)
    centroid_x = float(np.mean(xs))
    offset_deg = ((centroid_x / w) - 0.5) * fov_deg

    concentration = mask.sum() / mask.size
    confidence = float(np.clip(1.0 - concentration * 20, 0.1, 1.0))
    return offset_deg, confidence


def get_nearby_road_bearings(lat: float, lon: float, radius_m: int = 300):
    overpass_ql = f"""
    [out:json][timeout:15];
    way(around:{radius_m},{lat},{lon})[highway];
    out geom;
    """
    data = overpass_utils.query(overpass_ql)
    if data is None:
        return []

    bearings = []
    for element in data.get('elements', []):
        geom = element.get('geometry', [])
        for i in range(len(geom) - 1):
            lat1, lon1 = geom[i]['lat'], geom[i]['lon']
            lat2, lon2 = geom[i + 1]['lat'], geom[i + 1]['lon']
            bearings.append(_bearing(lat1, lon1, lat2, lon2) % 180)
    return bearings


def _bearing(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _signed_diff(a: float, b: float) -> float:
    return ((a - b + 180) % 360) - 180


def best_match_for_candidate(lat: float, lon: float, sample_dates: list, road_bearings: list, observed_offset_deg: float):
    """Sweep sunrise/sunset azimuths over sample_dates and road-facing hypotheses;
    return the best (lowest-error) explanation for the observed sun bearing."""
    if not road_bearings:
        return None

    observer = Observer(latitude=lat, longitude=lon)
    headings = set()
    for b in set(round(b) for b in road_bearings):
        headings.add(b)
        headings.add((b + 180) % 360)

    best = None
    for d in sample_dates:
        try:
            events = sun_events(observer, date=d)
        except Exception:
            continue
        for event_name in ('sunrise', 'sunset'):
            t = events.get(event_name)
            if t is None:
                continue
            az = sun_azimuth_at(observer, t)
            for heading in headings:
                expected_offset = _signed_diff(az, heading)
                error = abs(_signed_diff(expected_offset, observed_offset_deg))
                if best is None or error < best['error']:
                    best = {
                        'error': round(error, 1),
                        'date': d.isoformat(),
                        'event': event_name,
                        'sun_azimuth': round(az, 1),
                        'camera_heading': heading,
                    }
    return best


def refine_clusters(clusters: list, image_path: str, fov_deg: float = 65.0):
    lat_hint = clusters[0]['lat'] if clusters else 40.0
    season_label, sample_dates, season_conf = estimate_season(image_path, lat_hint)
    is_golden, golden_conf = estimate_golden_hour(image_path)
    offset_deg, offset_conf = estimate_sun_bearing_offset(image_path, fov_deg)

    logger.info(f'Season estimate: {season_label} (confidence {season_conf}) -> {len(sample_dates)} sample dates')
    logger.info(f'Golden-hour estimate: {is_golden} (confidence {golden_conf})')

    if offset_deg is None:
        logger.warning('No bright sky region detected — skipping sun-based refinement.')
        return clusters, None
    if not is_golden:
        logger.warning('Image does not look like a sunrise/sunset shot — sun-bearing refinement is unreliable here, skipping.')
        return clusters, None

    logger.info(f'Estimated sun/glow bearing offset from center: {offset_deg:+.1f}° (confidence {offset_conf:.2f}, assumed FOV {fov_deg}°)')

    # NOTE: we deliberately do NOT re-sort clusters by sun_match_error. With a
    # multi-month date sweep x every nearby road bearing (and its reciprocal)
    # as free parameters, this search is overparameterized enough that it will
    # usually find *some* combo that fits near-perfectly, even for a wrong
    # candidate — a low error is weak confirming evidence. A HIGH error is the
    # informative case: it means no plausible date/heading explains the
    # observed sun position there, which is real evidence against that
    # candidate. Keep PLONK's own weight as the primary ranking signal.
    for cluster in clusters:
        road_bearings = get_nearby_road_bearings(cluster['lat'], cluster['lon'])
        n_distinct_bearings = len(set(round(b) for b in road_bearings))
        match = best_match_for_candidate(cluster['lat'], cluster['lon'], sample_dates, road_bearings, offset_deg)
        if match:
            match['n_road_bearings'] = n_distinct_bearings
            match['reliability'] = 'low (many road orientations nearby, fit is easy)' if n_distinct_bearings > 4 else 'moderate (few road orientations, fit is more constrained)'
        cluster['sun_evidence'] = match if match else {'error': None, 'note': 'no usable road data nearby'}

    meta = {
        'season': season_label,
        'season_confidence': season_conf,
        'golden_hour': is_golden,
        'golden_hour_confidence': golden_conf,
        'observed_offset_deg': round(offset_deg, 1),
        'offset_confidence': round(offset_conf, 2),
    }
    return clusters, meta
