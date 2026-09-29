"""Reference outputs for the deterministic sun/HSV/astral logic."""
import datetime as dt
import json

import numpy as np
from astral import Observer
from astral.sun import sun as sun_events, azimuth as az_at

from common import *
import sun_refine as SR

out = {'images': {}, 'astral': [], 'best_match': []}

for name, path in IMAGES.items():
    lab, dates, conf = SR.estimate_season(path, 40.0)
    gold, gconf = SR.estimate_golden_hour(path)
    off, oconf = SR.estimate_sun_bearing_offset(path, 65.0)
    lab_s, dates_s, conf_s = SR.estimate_season(path, -33.0)
    out['images'][name] = {
        'season': lab, 'n_dates': len(dates), 'dates': [d.isoformat() for d in dates], 'season_conf': conf,
        'season_south': lab_s, 'season_south_conf': conf_s,
        'golden': bool(gold), 'golden_conf': gconf, 'offset': off, 'offset_conf': oconf,
    }

places = [(51.5, -0.1), (-33.9, 151.2), (35.7, 139.7), (64.1, -21.9), (69.6, 18.9),
          (-54.8, -68.3), (0.0, 0.0), (1.3, 103.8), (21.3, -157.8), (-17.7, -149.4), (51.5, 179.9),
          (64.8, -147.7), (-77.8, 166.7), (37.0, -122.0), (40.7, -74.0), (-1.3, 36.8), (55.7, 37.6),
          (89.9, 10.0), (-45.0, 170.5), (60.0, -179.0), (48.9, 2.3)]
dates = []
for m in range(1, 13):
    dates += [dt.date(2024, m, 1), dt.date(2024, m, 15), dt.date(2024, m, 28)]
dates += [dt.date(2024, 6, 21), dt.date(2024, 12, 21), dt.date(2024, 3, 20), dt.date(2024, 9, 22)]
for lat, lon in places:
    obs = Observer(latitude=lat, longitude=lon)
    for d in dates:
        try:
            ev = sun_events(obs, date=d)
            row = {'lat': lat, 'lon': lon, 'date': d.isoformat(),
                   **{k: v.isoformat() for k, v in ev.items()},
                   'az_sunrise': az_at(obs, ev['sunrise']), 'az_sunset': az_at(obs, ev['sunset'])}
        except ValueError as e:
            row = {'lat': lat, 'lon': lon, 'date': d.isoformat(), 'error': str(e)}
        out['astral'].append(row)

# best_match_for_candidate on fixed synthetic bearings
rng = np.random.default_rng(5)
for lat, lon in [(51.5, -0.1), (-33.9, 151.2), (64.1, -21.9)]:
    bearings = list((rng.random(int(rng.integers(1, 80))) * 180).round(3))
    for label in ('green', 'brown', 'snow', 'unknown'):
        sd = SR._sample_dates(SR.SEASON_MONTHS_N[label])
        for offset in (-20.0, -0.4, 13.3, 31.0):
            r = SR.best_match_for_candidate(lat, lon, sd, bearings, offset)
            out['best_match'].append({'lat': lat, 'lon': lon, 'season': label, 'offset': offset,
                                      'bearings': bearings, 'result': r})
json.dump(out, open(os.path.join(REF, 'sun_ref.json'), 'w'), indent=1, default=float)
print('astral rows', len(out['astral']), 'errors', sum('error' in r for r in out['astral']))
print(json.dumps(out['images'], default=float)[:1500])
