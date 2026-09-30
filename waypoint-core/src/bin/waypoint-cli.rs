//! Command-line driver: `waypoint-cli one|point --image_path X ...`.
//! Prints the same NDJSON events the desktop app consumes, on stdout.
//! Logs go to stderr. Same flags as the original `run_pipeline.py`. The
//! runtime and models come from the app's data folder (WAYPOINT_DATA_DIR, or
//! WAYPOINT_ORT_DLL / WAYPOINT_ONNX_DIR).

use std::io::Write;
use std::path::PathBuf;

use serde_json::Value;
use waypoint_core::{run, Cancel, Mode, RunArgs, Sink};

struct Stdout;

impl Sink for Stdout {
    fn event(&self, ev: Value) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{}", ev);
        let _ = out.flush();
    }
    fn log(&self, line: &str) {
        eprintln!("{}", line.trim_end());
    }
}

fn main() {
    let mut argv = std::env::args().skip(1);
    let mode = match argv.next().as_deref() {
        Some("one") => Mode::One,
        Some("point") => Mode::Point,
        _ => {
            eprintln!("usage: waypoint-cli one|point --image_path X [options]");
            std::process::exit(2);
        }
    };
    let mut a = RunArgs::new(mode, "");
    while let Some(flag) = argv.next() {
        let mut val = || argv.next().unwrap_or_else(|| { eprintln!("missing value for {flag}"); std::process::exit(2) });
        match flag.as_str() {
            "--image_path" => a.image_path = val().into(),
            "--model" => a.model = val(),
            "--num_samples" => a.num_samples = val().parse().expect("num_samples"),
            "--num_runs" => a.num_runs = val().parse().expect("num_runs"),
            "--cluster_radius_km" => a.cluster_radius_km = val().parse().expect("cluster_radius_km"),
            "--top_k" => a.top_k = val().parse().expect("top_k"),
            "--no_geocode" => a.no_geocode = true,
            "--no_sun_refine" => a.no_sun_refine = true,
            "--no_verify" => a.no_verify = true,
            "--verify_top_n" => a.verify_top_n = val().parse().expect("verify_top_n"),
            "--mapillary_token" => a.mapillary_token = Some(val()),
            "--radius_km" => a.radius_km = Some(val().parse().expect("radius_km")),
            "--max_images" => a.max_images = val().parse().expect("max_images"),
            "--retrieval_top_clusters" => a.retrieval_top_clusters = val().parse().expect("retrieval_top_clusters"),
            "--fov" => a.fov = val().parse().expect("fov"),
            "--lat" => a.lat = Some(val().parse().expect("lat")),
            "--lon" => a.lon = Some(val().parse().expect("lon")),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let sink = Stdout;
    let data_dir = std::env::var_os("WAYPOINT_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("geolocator-gui"));
    let Some(loc) = waypoint_core::native::locate(&data_dir).filter(|l| l.has(&a.model)) else {
        let msg = format!("ONNX runtime or the files of model '{}' not found", a.model);
        sink.event(serde_json::json!({ "event": "error", "message": msg }));
        std::process::exit(1);
    };
    let code = match waypoint_core::native::open(&loc, &a.model, &|s| sink.log(s)) {
        Ok(models) => run(&a, &models, &sink, &Cancel::new()),
        Err(e) => {
            sink.event(serde_json::json!({ "event": "error", "message": format!("the inference engine could not start: {e}") }));
            1
        }
    };
    std::process::exit(code);
}
