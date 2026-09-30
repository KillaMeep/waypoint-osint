//! Network probes (ignored by default): `cargo test --test net_probe -- --ignored --nocapture`

use waypoint_core::net;

#[test]
#[ignore]
fn gsv_tile_probe() {
    let agent = net::agent(4, 10);
    let Ok(panoid) = std::env::var("WAYPOINT_TEST_PANOID") else { eprintln!("skipped: set WAYPOINT_TEST_PANOID"); return };
    let url = format!("https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x=0&y=0&zoom=2&nbt=1&fover=2");
    let hdrs = [
        ("origin", "https://www.google.com"),
        ("referer", "https://www.google.com/"),
        ("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"),
    ];
    let mut req = agent.get(&url);
    for (k, v) in hdrs {
        req = req.set(k, v);
    }
    match req.call().map(|r| r.status()) {
        Ok(s) => println!("plain request status {s}"),
        Err(e) => println!("plain request error: {e}"),
    }
    let mut req = agent.get(&url).set("Accept", "*/*").set("Accept-Encoding", "gzip, deflate");
    for (k, v) in hdrs {
        req = req.set(k, v);
    }
    match req.call().map(|r| r.status()) {
        Ok(s) => println!("with accept status {s}"),
        Err(e) => println!("with accept error: {e}"),
    }
    match net::get_bytes(&agent, &url, &hdrs) {
        Ok(b) => println!("tile ok: {} bytes, head {:02x?}", b.len(), &b[..b.len().min(8)]),
        Err(e) => println!("tile error: {e}"),
    }
}
