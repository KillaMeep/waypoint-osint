//! Port of the parts of the `astral` library (v3.2, `astral.sun`) that
//! `sun_refine.py` uses: `sun()` (its success/failure and sunrise/sunset
//! instants) and `azimuth()`. The formulas and their quirks (two-pass
//! transit iteration, +-1 day re-search, integer-second truncation in the
//! azimuth call) are copied so results agree with astral to the microsecond.

const SUN_APPARENT_RADIUS: f64 = 32.0 / (60.0 * 2.0);
const US_PER_DAY: i64 = 86_400_000_000;

/// A calendar date as days since 1970-01-01.
pub type Day = i64;

pub fn days_from_civil(y: i32, m: u32, d: u32) -> Day {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn civil_from_days(z: Day) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y } as i32, m, d)
}

/// Julian day for the start of `day`, plus `secs` whole seconds into it
/// (`astral.julian.julianday`, which truncates to whole seconds).
fn julianday(day: Day, secs: i64) -> f64 {
    let (mut year, mut month, d) = civil_from_days(day);
    let day_fraction = secs as f64 / 86_400.0;
    if month <= 2 {
        year -= 1;
        month += 12;
    }
    let a = (year as f64 / 100.0).trunc() as i64;
    let b = 2 - a + (a as f64 / 4.0).trunc() as i64;
    (365.25 * (year as f64 + 4716.0)).trunc() + (30.6001 * (month as f64 + 1.0)).trunc() + d as f64 + day_fraction + b as f64
        - 1524.5
}

fn juliancentury(jd: f64) -> f64 {
    (jd - 2_451_545.0) / 36_525.0
}

fn geom_mean_long_sun(jc: f64) -> f64 {
    (280.46646 + jc * (36000.76983 + 0.0003032 * jc)).rem_euclid(360.0)
}
fn geom_mean_anomaly_sun(jc: f64) -> f64 {
    357.52911 + jc * (35999.05029 - 0.0001537 * jc)
}
fn eccentric_location_earth_orbit(jc: f64) -> f64 {
    0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
}
fn sun_eq_of_center(jc: f64) -> f64 {
    let m = geom_mean_anomaly_sun(jc);
    let mrad = m.to_radians();
    let sinm = mrad.sin();
    let sin2m = (mrad + mrad).sin();
    let sin3m = (mrad + mrad + mrad).sin();
    sinm * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + sin2m * (0.019993 - 0.000101 * jc) + sin3m * 0.000289
}
fn sun_true_long(jc: f64) -> f64 {
    geom_mean_long_sun(jc) + sun_eq_of_center(jc)
}
fn sun_apparent_long(jc: f64) -> f64 {
    let true_long = sun_true_long(jc);
    let omega = 125.04 - 1934.136 * jc;
    true_long - 0.00569 - 0.00478 * omega.to_radians().sin()
}
fn mean_obliquity_of_ecliptic(jc: f64) -> f64 {
    let seconds = 21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813));
    23.0 + (26.0 + (seconds / 60.0)) / 60.0
}
fn obliquity_correction(jc: f64) -> f64 {
    let e0 = mean_obliquity_of_ecliptic(jc);
    let omega = 125.04 - 1934.136 * jc;
    e0 + 0.00256 * omega.to_radians().cos()
}
fn sun_declination(jc: f64) -> f64 {
    let e = obliquity_correction(jc);
    let lambd = sun_apparent_long(jc);
    (e.to_radians().sin() * lambd.to_radians().sin()).asin().to_degrees()
}
fn var_y(jc: f64) -> f64 {
    let y = (obliquity_correction(jc).to_radians() / 2.0).tan();
    y * y
}
fn eq_of_time(jc: f64) -> f64 {
    let l0 = geom_mean_long_sun(jc);
    let e = eccentric_location_earth_orbit(jc);
    let m = geom_mean_anomaly_sun(jc);
    let y = var_y(jc);
    let sin2l0 = (2.0 * l0.to_radians()).sin();
    let sinm = m.to_radians().sin();
    let cos2l0 = (2.0 * l0.to_radians()).cos();
    let sin4l0 = (4.0 * l0.to_radians()).sin();
    let sin2m = (2.0 * m.to_radians()).sin();
    let etime = y * sin2l0 - 2.0 * e * sinm + 4.0 * e * y * sinm * cos2l0 - 0.5 * y * y * sin4l0 - 1.25 * e * e * sin2m;
    etime.to_degrees() * 4.0
}

fn refraction_at_zenith(zenith: f64) -> f64 {
    let elevation = 90.0 - zenith;
    if elevation >= 85.0 {
        return 0.0;
    }
    let te = elevation.to_radians().tan();
    let corr = if elevation > 5.0 {
        58.1 / te - 0.07 / (te * te * te) + 0.000086 / (te * te * te * te * te)
    } else if elevation > -0.575 {
        let step1 = -12.79 + elevation * 0.711;
        let step2 = 103.4 + elevation * step1;
        let step3 = -518.2 + elevation * step2;
        1735.0 + elevation * step3
    } else {
        -20.774 / te
    };
    corr / 3600.0
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Rising,
    Setting,
}

/// `hour_angle`: None where Python's `math.acos` would raise a domain error.
fn hour_angle(latitude: f64, declination: f64, zenith: f64, direction: Direction) -> Option<f64> {
    let (lat, dec, z) = (latitude.to_radians(), declination.to_radians(), zenith.to_radians());
    let h = (z.cos() - lat.sin() * dec.sin()) / (lat.cos() * dec.cos());
    if !(-1.0..=1.0).contains(&h) {
        return None;
    }
    let ha = h.acos();
    Some(if direction == Direction::Setting { -ha } else { ha })
}

fn minutes_to_us(minutes: f64) -> i64 {
    // datetime.timedelta(days=d, seconds=s, microseconds=us) built with
    // truncating int() conversions, exactly like astral.
    let d = (minutes / 1440.0).trunc();
    let mut minutes = minutes - d * 1440.0;
    minutes *= 60.0;
    let s = minutes.trunc();
    let sfrac = minutes - s;
    let us = (sfrac * 1_000_000.0).trunc();
    d as i64 * US_PER_DAY + s as i64 * 1_000_000 + us as i64
}

/// Microseconds since the Unix epoch (UTC) at which the sun crosses `zenith`.
/// `None` where astral raises ValueError (sun never reaches that zenith).
pub fn time_of_transit(latitude: f64, longitude: f64, day: Day, zenith: f64, direction: Direction) -> Option<i64> {
    let latitude = latitude.clamp(-89.8, 89.8);
    let refraction = refraction_at_zenith(zenith);
    let jd = julianday(day, 0);
    let mut adjustment = 0.0;
    let mut time_utc = 0.0;
    for _ in 0..2 {
        let jc = juliancentury(jd + adjustment);
        let declination = sun_declination(jc);
        let ha = hour_angle(latitude, declination, zenith + refraction, direction)?;
        let delta = -longitude - ha.to_degrees();
        let eqtime = eq_of_time(jc);
        let mut offset = delta * 4.0 - eqtime;
        if offset < -720.0 {
            offset += 1440.0;
        }
        time_utc = 720.0 + offset;
        adjustment = time_utc / 1440.0;
    }
    Some(day * US_PER_DAY + minutes_to_us(time_utc))
}

fn date_of(us: i64) -> Day {
    us.div_euclid(US_PER_DAY)
}

/// astral's dawn/sunrise/sunset/dusk: compute for `day`, and when the result
/// lands on another UTC date, retry on the adjacent day; still off -> error.
fn on_date(latitude: f64, longitude: f64, day: Day, zenith: f64, direction: Direction) -> Option<i64> {
    let tot = time_of_transit(latitude, longitude, day, zenith, direction)?;
    let tot_date = date_of(tot);
    if tot_date == day {
        return Some(tot);
    }
    let new_day = if tot_date < day { day + 1 } else { day - 1 };
    let tot = time_of_transit(latitude, longitude, new_day, zenith, direction)?;
    if date_of(tot) != day {
        return None;
    }
    Some(tot)
}

/// Sunrise and sunset instants (microseconds since epoch, UTC) for `day`, or
/// `None` where `astral.sun.sun()` raises ValueError. `sun()` also computes
/// civil dawn and dusk, so polar cases where only those fail are skipped too.
pub fn sun_rise_set(latitude: f64, longitude: f64, day: Day) -> Option<(i64, i64)> {
    let _dawn = on_date(latitude, longitude, day, 96.0, Direction::Rising)?;
    let rise = on_date(latitude, longitude, day, 90.0 + SUN_APPARENT_RADIUS, Direction::Rising)?;
    let set = on_date(latitude, longitude, day, 90.0 + SUN_APPARENT_RADIUS, Direction::Setting)?;
    let _dusk = on_date(latitude, longitude, day, 96.0, Direction::Setting)?;
    Some((rise, set))
}

/// `astral.sun.azimuth(observer, dt)` for a UTC instant given in microseconds
/// since the epoch (sub-second part ignored, as astral does).
pub fn azimuth(latitude: f64, longitude: f64, instant_us: i64) -> f64 {
    let latitude = latitude.clamp(-89.8, 89.8);
    let day = date_of(instant_us);
    let sod = instant_us.rem_euclid(US_PER_DAY) / 1_000_000; // whole seconds into the day
    let (hour, minute, second) = (sod / 3600, (sod % 3600) / 60, sod % 60);

    let jd = julianday(day, sod);
    let t = juliancentury(jd);
    let declination = sun_declination(t);
    let eqtime = eq_of_time(t);

    let solar_time_fix = eqtime + 4.0 * longitude + 60.0 * 0.0;
    let mut true_solar_time = hour as f64 * 60.0 + minute as f64 + second as f64 / 60.0 + solar_time_fix;
    while true_solar_time > 1440.0 {
        true_solar_time -= 1440.0;
    }
    let mut hourangle = true_solar_time / 4.0 - 180.0;
    if hourangle < -180.0 {
        hourangle += 360.0;
    }

    let ch = hourangle.to_radians().cos();
    let cl = latitude.to_radians().cos();
    let sl = latitude.to_radians().sin();
    let sd = declination.to_radians().sin();
    let cd = declination.to_radians().cos();
    let csz = (cl * cd * ch + sl * sd).clamp(-1.0, 1.0);
    let zenith = csz.acos().to_degrees();

    let az_denom = cl * zenith.to_radians().sin();
    let mut azimuth;
    if az_denom.abs() > 0.001 {
        let az_rad = ((sl * zenith.to_radians().cos()) - sd) / az_denom;
        let az_rad = az_rad.clamp(-1.0, 1.0);
        azimuth = 180.0 - az_rad.acos().to_degrees();
        if hourangle > 0.0 {
            azimuth = -azimuth;
        }
    } else if latitude > 0.0 {
        azimuth = 180.0;
    } else {
        azimuth = 0.0;
    }
    if azimuth < 0.0 {
        azimuth += 360.0;
    }
    azimuth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_roundtrip() {
        for d in [-800_000i64, -1, 0, 1, 19_000, 20_000, 60_000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        assert_eq!(days_from_civil(2024, 2, 29), 19_782);
    }
}
