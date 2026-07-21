#!/usr/bin/env python3
"""Shared, resilient Overpass API querying — retries with backoff and falls
back across public mirrors, since the free overpass-api.de instance rate-limits
(429) or times out (504) under back-to-back queries, which multiple refinement
stages in this toolchain (sun_refine, google_sv_refine) both hit per run."""
import logging
import time

import requests

logger = logging.getLogger(__name__)

MIRRORS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.openstreetmap.ru/api/interpreter',
]


def query(overpass_ql: str, timeout: int = 30, max_retries: int = 3):
    """POST an Overpass QL query, retrying with backoff and rotating mirrors
    on rate-limit/server errors. Returns the parsed JSON dict, or None."""
    for attempt in range(max_retries + 1):
        mirror = MIRRORS[attempt % len(MIRRORS)]
        try:
            resp = requests.post(mirror, data={'data': overpass_ql},
                                  headers={'User-Agent': 'osint-toolkit-plonk/1.0'}, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.debug(f'Overpass {mirror} returned {resp.status_code}, retrying...')
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f'Overpass query to {mirror} failed: {e}')
            time.sleep(2 * (attempt + 1))
            continue

    logger.warning('Overpass query failed after retries across all mirrors.')
    return None
