#!/usr/bin/env python3
"""
Core PLONK prediction + clustering, extracted from the OSINT toolkit's
tools/PLONK/infer.py for reuse in the Geolocator GUI backend.
"""
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    'osv5m': 'nicolas-dufour/PLONK_OSV_5M',
    'yfcc': 'nicolas-dufour/PLONK_YFCC',
    'inat': 'nicolas-dufour/PLONK_iNaturalist',
}

EARTH_RADIUS_KM = 6371.0


def reverse_geocode(lat: float, lon: float):
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent='geolocator-gui')
        loc = geolocator.reverse((lat, lon), language='en', timeout=10)
        return loc.address if loc else None
    except Exception as e:
        logger.warning(f'Reverse geocoding failed: {e}')
        return None


def cluster_samples(coords: np.ndarray, cluster_radius_km: float = 100.0, min_cluster_frac: float = 0.03, top_k: int = 5):
    """Group sampled lat/lon points into geographic clusters via DBSCAN (haversine metric)."""
    from sklearn.cluster import DBSCAN

    n = coords.shape[0]
    min_samples = max(3, int(n * min_cluster_frac))
    coords_rad = np.radians(coords)
    eps_rad = cluster_radius_km / EARTH_RADIUS_KM

    labels = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine').fit_predict(coords_rad)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        members = coords[labels == label]
        clusters.append({
            'lat': float(np.mean(members[:, 0])),
            'lon': float(np.mean(members[:, 1])),
            'lat_std': float(np.std(members[:, 0])),
            'lon_std': float(np.std(members[:, 1])),
            'count': int(members.shape[0]),
            'weight': round(float(members.shape[0]) / n, 3),
        })

    clusters.sort(key=lambda c: c['count'], reverse=True)
    noise_frac = round(float(np.sum(labels == -1)) / n, 3)
    return clusters[:top_k], noise_frac


class Predictor:
    def __init__(self, model_alias: str, num_samples: int):
        from plonk import PlonkPipeline
        model_path = MODEL_ALIASES.get(model_alias, model_alias)
        logger.info(f'Loading {model_path} (first run downloads weights from Hugging Face)...')
        self.pipeline = PlonkPipeline(model_path)
        self.num_samples = num_samples
        logger.info('Model ready.')

    def sample(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        return self.pipeline(image, batch_size=self.num_samples)

    def predict(self, image_path: str, cluster_radius_km: float, top_k: int, geocode: bool, num_runs: int = 1):
        coords = np.concatenate([self.sample(image_path) for _ in range(num_runs)], axis=0)
        total_samples = self.num_samples * num_runs
        clusters, noise_frac = cluster_samples(coords, cluster_radius_km=cluster_radius_km, top_k=top_k)

        for c in clusters:
            c['address'] = reverse_geocode(c['lat'], c['lon']) if geocode else None

        return {
            'num_samples': total_samples,
            'num_runs': num_runs,
            'clusters': clusters,
            'noise_frac': noise_frac,
        }
