//! HTTP plumbing shared by the imagery sources, plus the Overpass and
//! Nominatim clients. Request shapes mirror the Python originals
//! (`overpass_utils.py`, geopy's Nominatim geocoder, the `requests` calls).

use std::io::Read;
use std::time::Duration;

use serde_json::Value;

use crate::util::{Cancel, Error, Result};

/// User agent `requests` sends when a call sets none; used wherever the
/// Python code relied on that default.
pub const PY_REQUESTS_UA: &str = "python-requests/2.34.2";

const OVERPASS_MIRRORS: [&str; 3] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
];

/// A pooled agent. `requests.Session` keeps one connection per worker
/// alive; ureq needs the per-host idle limit raised to do the same.
pub fn agent(pool: usize, timeout_secs: u64) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .max_idle_connections(pool.max(8))
        .max_idle_connections_per_host(pool.max(8))
        .timeout_connect(Duration::from_secs(timeout_secs))
        .timeout_read(Duration::from_secs(timeout_secs))
        .timeout_write(Duration::from_secs(timeout_secs))
        .build()
}

/// GET a URL and return the body bytes; any non-2xx status is an error
/// (`raise_for_status`).
pub fn get_bytes(agent: &ureq::Agent, url: &str, headers: &[(&str, &str)]) -> std::result::Result<Vec<u8>, String> {
    let mut req = agent.get(url);
    // ureq keeps both if a differently-cased duplicate is added, and Google's
    // tile server answers 403 to a doubled User-Agent, so set exactly one.
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("user-agent")) {
        req = req.set("User-Agent", PY_REQUESTS_UA);
    }
    for (k, v) in headers {
        req = req.set(k, v);
    }
    let resp = req.call().map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Same, decoded as JSON.
pub fn get_json(agent: &ureq::Agent, url: &str, query: &[(&str, &str)]) -> std::result::Result<Value, String> {
    let mut req = agent.get(url).set("User-Agent", PY_REQUESTS_UA).set("Accept", "*/*");
    for (k, v) in query {
        req = req.query(k, v);
    }
    let resp = req.call().map_err(|e| e.to_string())?;
    serde_json::from_reader(resp.into_reader()).map_err(|e| e.to_string())
}

/// POST an Overpass QL query, retrying with backoff and rotating mirrors on
/// rate-limit/server errors (port of `overpass_utils.query`). Ok(None) when
/// every attempt failed.
pub fn overpass_query(ql: &str, timeout_secs: u64, max_retries: usize, cancel: &Cancel, log: &dyn Fn(&str)) -> Result<Option<Value>> {
    let agent = agent(1, timeout_secs);
    for attempt in 0..=max_retries {
        cancel.check()?;
        let mirror = OVERPASS_MIRRORS[attempt % OVERPASS_MIRRORS.len()];
        let res = agent
            .post(mirror)
            .set("User-Agent", "osint-toolkit-plonk/1.0")
            .send_form(&[("data", ql)]);
        let wait = Duration::from_secs(2 * (attempt as u64 + 1));
        match res {
            Ok(resp) => match serde_json::from_reader::<_, Value>(resp.into_reader()) {
                Ok(v) => return Ok(Some(v)),
                Err(e) => log(&format!("Overpass query to {mirror} failed: {e}")),
            },
            Err(ureq::Error::Status(code, _)) => {
                log(&format!("Overpass {mirror} returned {code}, retrying..."));
            }
            Err(e) => log(&format!("Overpass query to {mirror} failed: {e}")),
        }
        cancel.sleep(wait)?;
    }
    log("Overpass query failed after retries across all mirrors.");
    Ok(None)
}

/// Reverse geocode through Nominatim the way geopy does
/// (`Nominatim(user_agent='geolocator-gui').reverse((lat, lon), language='en', timeout=10)`).
pub fn reverse_geocode(lat: f64, lon: f64, log: &dyn Fn(&str)) -> Option<String> {
    let agent = agent(1, 10);
    let res = agent
        .get("https://nominatim.openstreetmap.org/reverse")
        .set("User-Agent", "geolocator-gui")
        .set("Accept", "*/*")
        .query("lat", &lat.to_string())
        .query("lon", &lon.to_string())
        .query("format", "json")
        .query("accept-language", "en")
        .query("addressdetails", "1")
        .call();
    match res {
        Ok(resp) => match serde_json::from_reader::<_, Value>(resp.into_reader()) {
            Ok(v) => v.get("display_name").and_then(Value::as_str).map(str::to_string),
            Err(e) => {
                log(&format!("Reverse geocoding failed: {e}"));
                None
            }
        },
        Err(e) => {
            log(&format!("Reverse geocoding failed: {e}"));
            None
        }
    }
}

pub fn err(msg: impl Into<String>) -> Error {
    Error::Msg(msg.into())
}
