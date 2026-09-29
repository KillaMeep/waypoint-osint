//! Command-line driver: `waypoint-cli one|point --image_path X ...`.
//! Prints the same NDJSON events the desktop app consumes, on stdout.
//! Logs go to stderr. Same flags as the old `run_pipeline.py`, plus:
//!   --python <exe>       Python interpreter for the model server (env WAYPOINT_PYTHON)
//!   --backend <dir>      folder containing model_server.py (default ./python_backend)
//!   --engine auto|native|py   retrieval networks: ONNX Runtime when installed (auto), or force one
//!   --seed N             (unused by the Python server path, reserved)

use std::io::Write;
use std::path::PathBuf;

use serde_json::Value;
use waypoint_core::pyserver::PyModels;
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
    let mut python: Option<PathBuf> = std::env::var_os("WAYPOINT_PYTHON").map(PathBuf::from);
    let mut backend = PathBuf::from("python_backend");
    let mut engine = String::from("auto"); // auto | native | py
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
            "--python" => python = Some(val().into()),
            "--engine" => engine = val(),
            "--backend" => backend = val().into(),
            "--seed" => {
                let _ = val();
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let python = python.unwrap_or_else(|| {
        PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("geolocator-gui/python-runtime/venv/Scripts/python.exe")
    });
    let sink = Stdout;
    let models = match PyModels::spawn(&python, &backend.join("model_server.py"), &backend, |s| eprint!("{s}")) {
        Ok(m) => m,
        Err(e) => {
            sink.event(serde_json::json!({ "event": "error", "message": e.to_string() }));
            std::process::exit(1);
        }
    };
    let data_dir = std::env::var_os("WAYPOINT_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("geolocator-gui"));
    let native = match engine.as_str() {
        "py" => None,
        _ => match waypoint_core::native::locate(&data_dir) {
            Some(loc) => match waypoint_core::native::open(&loc, models.clone(), &|s| sink.log(s)) {
                Ok(n) => Some(n),
                Err(e) => {
                    sink.log(&format!("native models unavailable ({e}); using the Python model server"));
                    None
                }
            },
            None => {
                if engine == "native" {
                    sink.event(serde_json::json!({ "event": "error", "message": "ONNX models / runtime not found" }));
                    std::process::exit(1);
                }
                None
            }
        },
    };
    let code = match &native {
        Some(n) => run(&a, n, &sink, &Cancel::new()),
        None => run(&a, &*models, &sink, &Cancel::new()),
    };
    drop(native);
    models.shutdown();
    std::process::exit(code);
}
