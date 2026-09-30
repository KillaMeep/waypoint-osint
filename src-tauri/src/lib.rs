//! Waypoint desktop shell.
//!
//! The Rust side owns everything the webview can't do on its own: installing
//! the inference engine, running the pipeline (`waypoint-core`), settings, and
//! native dialogs.
//!
//! Inference runs in-process on ONNX Runtime (DirectML, CPU fallback) with the
//! exported PLONK, StreetCLIP, DINOv2, DISK and LightGlue models. First-run
//! setup downloads the runtime from NuGet and the models from MODEL_BASE_URL.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::thread;

use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use tauri_plugin_updater::UpdaterExt;
use sha2::{Digest, Sha256};
use waypoint_core::native;
use waypoint_core::onnx::{init_runtime, Accel};
use waypoint_core::plonk::{self, PlonkStep};
use waypoint_core::{run as core_run, Cancel, Mode, RunArgs as CoreArgs, Sink};

// Aim the default Samples value at ~10 s of sampling for one run.
const CALIBRATION_TARGET_SECONDS: f64 = 10.0;
const IMAGE_EXTS: [&str; 5] = ["jpg", "jpeg", "png", "bmp", "webp"];

// Native engine files. Every one is checked against a pinned SHA-256 before
// use, so the hosts only serve bytes. WAYPOINT_MODEL_URL overrides the model
// host (e.g. a local mirror); tools/export/make_manifest.py writes models.json.
const MODEL_BASE_URL: &str = "https://huggingface.co/killameep/waypoint-models/resolve/main";
const MODEL_MANIFEST: &str = include_str!("../models.json");

/// A DLL taken from an official NuGet package.
struct NugetFile {
    package: &'static str,
    version: &'static str,
    package_bytes: u64,
    member: &'static str,
    file: &'static str,
    sha256: &'static str,
}

const RUNTIME_FILES: [NugetFile; 2] = [
    NugetFile {
        package: "microsoft.ml.onnxruntime.directml",
        version: "1.24.4",
        package_bytes: 12_458_649,
        member: "runtimes/win-x64/native/onnxruntime.dll",
        file: "onnxruntime.dll",
        sha256: "e7eedec6a6f26dc39dc948276a75ef6d2bee3fff944d874ceed0bbd3b97bff40",
    },
    NugetFile {
        package: "microsoft.ai.directml",
        version: "1.15.4",
        package_bytes: 202_292_617,
        member: "bin/x64-win/DirectML.dll",
        file: "DirectML.dll",
        sha256: "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1",
    },
];

// Folder name kept from the Electron build (its `userData` dir), so an
// existing install keeps its settings.
const DATA_DIR_NAME: &str = "geolocator-gui";

struct Paths {
    data: PathBuf,
}

impl Paths {
    fn ort_dir(&self) -> PathBuf { self.data.join("ort") }
    fn onnx_dir(&self) -> PathBuf { self.data.join("onnx") }
    fn settings(&self) -> PathBuf { self.data.join("settings.json") }
    /// Left by versions that ran a Python engine; only "Delete downloads" touches them.
    fn legacy_python(&self) -> [PathBuf; 2] { [self.data.join("python-runtime"), self.data.join("backend")] }
}

struct AppState {
    paths: Paths,
    active_run: Mutex<Option<ActiveRun>>,
    image: Mutex<Option<PathBuf>>,
}

/// The pipeline run in flight.
struct ActiveRun {
    cancel: Cancel,
}

/// Forwards backend events to the webview.
struct TauriSink {
    app: AppHandle,
}

impl Sink for TauriSink {
    fn event(&self, ev: Value) {
        let _ = self.app.emit("pipeline-event", ev);
    }
    fn log(&self, line: &str) {
        let _ = self.app.emit("pipeline-event", json!({ "event": "log", "message": line }));
    }
}

// ---------------------------------------------------------------- processes

fn hide_console(cmd: &mut Command) -> &mut Command {
    // A GUI-subsystem parent gets a console window flashed up for every
    // console child unless it is told not to.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

// ---------------------------------------------------------------- settings

fn read_settings(p: &Paths) -> Map<String, Value> {
    fs::read_to_string(p.settings())
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

/// Shallow-merge `patch` into the stored settings (same semantics as before:
/// saving the token never drops the calibration data, and vice versa).
fn merge_settings(p: &Paths, patch: Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut settings = read_settings(p);
    settings.extend(patch);
    fs::create_dir_all(&p.data).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&Value::Object(settings.clone())).map_err(|e| e.to_string())?;
    fs::write(p.settings(), text).map_err(|e| e.to_string())?;
    Ok(settings)
}

#[tauri::command]
fn settings_get(state: State<'_, AppState>) -> Value {
    Value::Object(read_settings(&state.paths))
}

#[tauri::command]
fn settings_set(state: State<'_, AppState>, settings: Map<String, Value>) -> Result<Value, String> {
    merge_settings(&state.paths, settings).map(Value::Object)
}

// ---------------------------------------------------------------- environment

/// Every model file the native engine installs (src-tauri/models.json).
fn manifest_models() -> Result<Vec<Value>, String> {
    let manifest: Value = serde_json::from_str(MODEL_MANIFEST).map_err(|e| e.to_string())?;
    Ok(manifest["files"].as_array().ok_or_else(|| "bad model manifest".to_string())?.clone())
}

#[tauri::command]
fn env_status(state: State<'_, AppState>) -> Value {
    let loc = native::locate(&state.paths.data);
    let models: Vec<Value> = plonk::VARIANTS
        .iter()
        .map(|v| json!({ "key": v.key, "label": v.label, "native": loc.as_ref().is_some_and(|l| l.has(v.key)) }))
        .collect();
    json!({
        // The native engine counts as installed once every PLONK model is: an
        // older install with only OSV-5M is offered the (incremental) install.
        "native": loc.as_ref().is_some_and(|l| plonk::VARIANTS.iter().all(|v| l.has(v.key))),
        "models": models,
    })
}

fn query_gpu() -> Value {
    let out = hide_console(
        Command::new("nvidia-smi").args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]),
    )
    .output();
    // nvidia-smi missing or failing means no usable NVIDIA GPU/driver.
    let Ok(out) = out else { return json!({ "hasGpu": false }) };
    if !out.status.success() {
        return json!({ "hasGpu": false });
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.lines().next().unwrap_or("").split(',').map(str::trim);
    let name = parts.next().filter(|s| !s.is_empty());
    let vram: Option<u64> = parts.next().and_then(|s| s.parse().ok());
    json!({ "hasGpu": true, "gpuName": name, "vramMb": vram })
}

fn sha256_file(path: &Path) -> Option<String> {
    let mut f = fs::File::open(path).ok()?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Some(format!("{:x}", h.finalize()))
}

/// Stream `url` into `dest`, returning its SHA-256. `on_bytes` gets each chunk size.
fn fetch(url: &str, dest: &Path, on_bytes: &mut dyn FnMut(u64)) -> Result<String, String> {
    let resp = ureq::get(url).call().map_err(|e| format!("Download failed: {e}"))?;
    let mut reader = resp.into_reader();
    let mut file = fs::File::create(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Download failed: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        h.update(&buf[..n]);
        on_bytes(n as u64);
    }
    file.flush().map_err(|e| e.to_string())?;
    Ok(format!("{:x}", h.finalize()))
}

/// Download the native engine: ONNX Runtime + DirectML from NuGet and the
/// exported models (all three PLONK models) from the model host. Files already
/// present with the right hash are kept. `progress(done, total, status)` (bytes)
/// drives the setup screen.
fn install_native(state: &AppState, progress: &dyn Fn(u64, u64, &str), log: &dyn Fn(&str)) -> Result<(), String> {
    let p = &state.paths;
    let models = manifest_models()?;
    let base = std::env::var("WAYPOINT_MODEL_URL").unwrap_or_else(|_| MODEL_BASE_URL.to_string());
    let base = base.trim_end_matches('/');

    // Nothing else is worth fetching if the model host isn't there.
    let first = models[0]["name"].as_str().unwrap_or_default();
    ureq::head(&format!("{base}/{first}"))
        .call()
        .map_err(|e| format!("The model files are not available at {base} ({e}). Check your internet connection and retry."))?;

    let total: u64 = RUNTIME_FILES.iter().map(|f| f.package_bytes).sum::<u64>()
        + models.iter().map(|m| m["bytes"].as_u64().unwrap_or(0)).sum::<u64>();
    let done = std::cell::Cell::new(0u64);
    let last = std::cell::Cell::new(-1i64);
    let tick = |n: u64, status: &str| {
        done.set(done.get() + n);
        let permille = (done.get() * 1000 / total.max(1)) as i64;
        if permille != last.get() {
            last.set(permille);
            progress(done.get(), total, status);
        }
    };

    fs::create_dir_all(p.ort_dir()).map_err(|e| e.to_string())?;
    fs::create_dir_all(p.onnx_dir()).map_err(|e| e.to_string())?;

    for f in &RUNTIME_FILES {
        let dest = p.ort_dir().join(f.file);
        if sha256_file(&dest).as_deref() == Some(f.sha256) {
            tick(f.package_bytes, "Downloading ONNX Runtime");
            continue;
        }
        log(&format!("Downloading {} {} from NuGet...", f.package, f.version));
        let url = format!("https://api.nuget.org/v3-flatcontainer/{0}/{1}/{0}.{1}.nupkg", f.package, f.version);
        let pkg = p.ort_dir().join(format!("{}.nupkg.part", f.package));
        fetch(&url, &pkg, &mut |n| tick(n, "Downloading ONNX Runtime"))?;
        // Scoped so the archive is closed before the package file is deleted.
        let bytes = {
            let mut zip = zip::ZipArchive::new(fs::File::open(&pkg).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            let mut member = zip.by_name(f.member).map_err(|e| format!("{} not in {}: {e}", f.member, f.package))?;
            let mut v = Vec::new();
            member.read_to_end(&mut v).map_err(|e| e.to_string())?;
            v
        };
        let _ = fs::remove_file(&pkg);
        let got = format!("{:x}", Sha256::digest(&bytes));
        if got != f.sha256 {
            return Err(format!("{} from {} {} has an unexpected hash ({got})", f.file, f.package, f.version));
        }
        fs::write(&dest, bytes).map_err(|e| e.to_string())?;
    }

    for m in &models {
        let name = m["name"].as_str().ok_or_else(|| "bad model manifest".to_string())?;
        let (bytes, want) = (m["bytes"].as_u64().unwrap_or(0), m["sha256"].as_str().unwrap_or_default());
        let dest = p.onnx_dir().join(name);
        if fs::metadata(&dest).is_ok_and(|md| md.len() == bytes) && sha256_file(&dest).as_deref() == Some(want) {
            tick(bytes, "Downloading models");
            continue;
        }
        log(&format!("Downloading {name} ({:.0} MB)...", bytes as f64 / 1e6));
        let part = p.onnx_dir().join(format!("{name}.part"));
        let got = fetch(&format!("{base}/{name}"), &part, &mut |n| tick(n, "Downloading models"))?;
        if got != want {
            let _ = fs::remove_file(&part);
            return Err(format!("{name} failed its integrity check (sha256 {got}, expected {want})"));
        }
        fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load the native engine and time its PLONK sampler (DirectML, else CPU).
/// Returns (samples/s, accelerator used). Any failure here means the engine
/// can't run on this machine.
fn native_calibration(state: &AppState, log: &dyn Fn(&str)) -> Result<(f64, &'static str), String> {
    let p = &state.paths;
    let unusable = |e: String| format!("The inference engine could not start ({e})");
    init_runtime(&p.ort_dir().join("onnxruntime.dll")).map_err(|e| {
        unusable(format!("{e}. ONNX Runtime needs the Microsoft Visual C++ 2015-2022 Redistributable (x64)"))
    })?;
    // Calibrated on the default model; YFCC is the same size, iNaturalist is smaller.
    let v = plonk::variant("osv5m").expect("osv5m");
    let (step, accel) = match PlonkStep::load(&p.onnx_dir(), v, Accel::DirectMl) {
        Ok(s) => (s, "DirectML"),
        Err(e) => {
            log(&format!("DirectML unavailable ({e}); using the CPU"));
            (PlonkStep::load(&p.onnx_dir(), v, Accel::Cpu).map_err(|e| unusable(e.to_string()))?, "CPU")
        }
    };
    let sps = step.throughput().map_err(|e| unusable(e.to_string()))?;
    Ok((sps, accel))
}

fn setup_native(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();
    let p = &state.paths;
    let send = |v: Value| { let _ = app.emit("env-progress", v); };
    let log = |m: &str| send(json!({ "event": "log", "message": m.trim() }));
    // Downloads fill the bar to 92%; calibration takes the rest.
    let progress = |done: u64, total: u64, status: &str| {
        let pct = 1.0 + done as f64 / total.max(1) as f64 * 91.0;
        send(json!({ "event": "progress", "pct": pct, "status": status, "done": done, "total": total }))
    };

    send(json!({ "event": "stage_start", "stage": "native_env" }));
    install_native(&state, &progress, &log)?;
    send(json!({ "event": "stage_done", "stage": "native_env" }));

    send(json!({ "event": "stage_start", "stage": "calibrate" }));
    send(json!({ "event": "progress", "pct": 94.0, "status": "Measuring sampling speed" }));
    let (sps, accel) = native_calibration(&state, &log)?;
    let recommended = samples_from_throughput(sps);
    // hasGpu / gpuName describe an NVIDIA card (nvidia-smi); `accel` is what
    // the native engine actually runs on (DirectML covers any DX12 GPU).
    let mut hardware = query_gpu().as_object().cloned().unwrap_or_default();
    hardware.insert("accel".into(), json!(accel));
    hardware.insert("samplesPerSec".into(), json!(sps));
    hardware.insert("recommendedSamples".into(), json!(recommended));
    send(json!({ "event": "stage_done", "stage": "calibrate", "samples_per_sec": sps, "recommended_samples": recommended }));
    merge_settings(p, Map::from_iter([("hardware".to_string(), Value::Object(hardware))]))?;
    send(json!({ "event": "setup_complete" }));
    Ok(())
}

fn samples_from_throughput(samples_per_sec: f64) -> u64 {
    if !(samples_per_sec > 0.0) {
        return 512;
    }
    let raw = samples_per_sec * CALIBRATION_TARGET_SECONDS;
    ((raw / 64.0).round() as u64 * 64).clamp(256, 8192)
}

#[tauri::command]
async fn env_setup(app: AppHandle) -> Value {
    let handle = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || setup_native(&handle))
        .await
        .unwrap_or_else(|e| Err(e.to_string()));
    match result {
        Ok(()) => json!({ "ok": true }),
        Err(e) => {
            let _ = app.emit("env-progress", json!({ "event": "error", "message": e }));
            json!({ "ok": false, "error": e })
        }
    }
}

/// Delete every downloaded runtime and model (and a Python engine left by an
/// older version).
#[tauri::command]
fn env_purge(state: State<'_, AppState>) -> Value {
    let p = &state.paths;
    let mut errors = vec![];
    let [py_runtime, py_backend] = p.legacy_python();
    for dir in [p.onnx_dir(), p.ort_dir(), py_runtime, py_backend] {
        match fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            // onnxruntime.dll stays locked while it's loaded in this process.
            Err(e) => errors.push(format!("{}: {e}", dir.display())),
        }
    }
    if errors.is_empty() {
        json!({ "ok": true })
    } else {
        json!({ "ok": false, "error": format!("{}. Restart Waypoint and purge again.", errors.join("; ")) })
    }
}

// ---------------------------------------------------------------- updates

/// The updater, pointed at the release manifest in tauri.conf.json.
/// WAYPOINT_UPDATE_URL points it at another manifest (HTTPS only, unless the
/// build's config sets plugins.updater.dangerousInsecureTransportProtocol).
fn updater(app: &AppHandle) -> Result<tauri_plugin_updater::Updater, String> {
    let mut b = app.updater_builder();
    if let Ok(url) = std::env::var("WAYPOINT_UPDATE_URL") {
        let parsed = url.parse().map_err(|e| format!("WAYPOINT_UPDATE_URL: {e}"))?;
        b = b.endpoints(vec![parsed]).map_err(|e| e.to_string())?;
    }
    b.build().map_err(|e| e.to_string())
}

/// Is a newer release published? `auto` is the startup check, skipped in
/// debug builds so development runs don't nag.
#[tauri::command]
async fn update_check(app: AppHandle, auto: Option<bool>) -> Value {
    let current = app.package_info().version.to_string();
    if auto.unwrap_or(false) && cfg!(debug_assertions) {
        return json!({ "ok": true, "available": false, "current": current, "skipped": true });
    }
    let found = match updater(&app) {
        Ok(u) => u.check().await.map_err(|e| e.to_string()),
        Err(e) => Err(e),
    };
    match found {
        Ok(Some(u)) => json!({
            "ok": true, "available": true, "current": current,
            "version": u.version, "notes": u.body, "date": u.date.map(|d| d.to_string()),
        }),
        Ok(None) => json!({ "ok": true, "available": false, "current": current }),
        Err(e) => json!({ "ok": false, "current": current, "error": e }),
    }
}

/// Download the newest release (signature-checked against the public key in
/// tauri.conf.json), run its installer and restart. Progress goes out as
/// `update-progress` events: { done, total } in bytes.
#[tauri::command]
async fn update_install(app: AppHandle) -> Value {
    if app.state::<AppState>().active_run.lock().unwrap().is_some() {
        return json!({ "ok": false, "error": "Stop the running pipeline first." });
    }
    let result: Result<(), String> = async {
        let update = updater(&app)?.check().await.map_err(|e| e.to_string())?.ok_or("No update is available.")?;
        let (mut done, mut last_pct) = (0u64, u64::MAX);
        let events = app.clone();
        update
            .download_and_install(
                |chunk, total| {
                    done += chunk as u64;
                    let pct = total.map_or(0, |t| done * 100 / t.max(1));
                    if pct != last_pct {
                        last_pct = pct;
                        let _ = events.emit("update-progress", json!({ "done": done, "total": total }));
                    }
                },
                || {},
            )
            .await
            .map_err(|e| e.to_string())
    }
    .await;
    match result {
        // On Windows the installer takes over and restarts Waypoint; elsewhere, restart here.
        Ok(()) => app.restart(),
        Err(e) => json!({ "ok": false, "error": e }),
    }
}

#[tauri::command]
fn app_version(app: AppHandle) -> String {
    app.package_info().version.to_string()
}

// ---------------------------------------------------------------- images

fn load_image_inner(state: &AppState, path: PathBuf) -> Result<Value, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if !IMAGE_EXTS.contains(&ext.as_str()) {
        return Err(format!("Unsupported file type: .{ext}"));
    }
    let bytes = fs::read(&path).map_err(|e| format!("Could not read {}: {e}", path.display()))?;
    let mime = match ext.as_str() {
        "png" => "image/png",
        "bmp" => "image/bmp",
        "webp" => "image/webp",
        _ => "image/jpeg",
    };
    let data_url = format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&bytes));
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let out = json!({ "path": path.to_string_lossy(), "name": name, "bytes": bytes.len(), "dataUrl": data_url });
    *state.image.lock().unwrap() = Some(path);
    Ok(out)
}

#[tauri::command]
async fn select_image(app: AppHandle) -> Result<Option<Value>, String> {
    let picker = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        picker.dialog().file().add_filter("Images", &IMAGE_EXTS).blocking_pick_file()
    })
    .await
    .map_err(|e| e.to_string())?;
    let Some(picked) = picked else { return Ok(None) };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    load_image_inner(&app.state::<AppState>(), path).map(Some)
}

/// Used for files dropped onto the window.
#[tauri::command]
fn load_image(state: State<'_, AppState>, path: String) -> Result<Value, String> {
    load_image_inner(&state, PathBuf::from(path))
}

#[tauri::command]
fn open_external(app: AppHandle, url: String) -> Result<(), String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("Only http(s) links can be opened".into());
    }
    app.opener().open_url(url, None::<&str>).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- pipeline

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunArgs {
    num_samples: Option<u32>,
    num_runs: Option<u32>,
    retrieval_top_clusters: Option<u32>,
    lat: Option<f64>,
    lon: Option<f64>,
    radius_km: Option<f64>,
    max_images: Option<u32>,
}

/// Returns at spawn time. The webview releases its busy state on the
/// `exit` event, which is always sent once the run has ended, whether it
/// finished, failed or was cancelled.
fn run_pipeline(app: AppHandle, mode: Mode, args: RunArgs) -> Result<u32, String> {
    let state = app.state::<AppState>();
    let mut active = state.active_run.lock().unwrap();
    if active.is_some() {
        return Err("A pipeline run is already in progress".into());
    }
    // Always the image the user picked through the app, never a path from the page.
    let image = state.image.lock().unwrap().clone().ok_or("No image selected")?;
    let settings = read_settings(&state.paths);
    let model = settings.get("plonkModel").and_then(Value::as_str).and_then(plonk::variant).unwrap_or(&plonk::VARIANTS[0]);
    let Some(loc) = native::locate(&state.paths.data) else {
        return Err("The inference engine is not installed. Restart Waypoint to run setup.".into());
    };
    if !loc.has(model.key) {
        return Err(format!("The {} model is not installed. Restart Waypoint to run setup.", model.label));
    }

    let mut ra = CoreArgs::new(mode.clone(), image);
    ra.model = model.key.into();
    ra.mapillary_token = settings
        .get("mapillaryToken")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .map(str::to_string);
    match mode {
        Mode::Point => {
            ra.lat = args.lat;
            ra.lon = args.lon;
            ra.radius_km = args.radius_km;
            if let Some(v) = args.max_images { ra.max_images = v as usize; }
        }
        Mode::One => {
            if let Some(v) = args.num_samples { ra.num_samples = v as usize; }
            if let Some(v) = args.num_runs { ra.num_runs = v as usize; }
            if let Some(v) = args.retrieval_top_clusters { ra.retrieval_top_clusters = v as usize; }
        }
    }

    let cancel = Cancel::new();
    *active = Some(ActiveRun { cancel: cancel.clone() });
    drop(active);

    thread::spawn(move || {
        let sink = TauriSink { app: app.clone() };
        let code = match native::open(&loc, model.key, &|s| sink.log(s)) {
            Ok(models) => core_run(&ra, &models, &sink, &cancel),
            Err(e) => {
                sink.event(json!({ "event": "error", "message": format!("The inference engine could not start: {e}") }));
                1
            }
        };
        *app.state::<AppState>().active_run.lock().unwrap() = None;
        let _ = app.emit("pipeline-event", json!({ "event": "exit", "code": code }));
    });
    Ok(0)
}

#[tauri::command]
fn run_one(app: AppHandle, args: RunArgs) -> Result<u32, String> {
    run_pipeline(app, Mode::One, args)
}

#[tauri::command]
fn run_point(app: AppHandle, args: RunArgs) -> Result<u32, String> {
    run_pipeline(app, Mode::Point, args)
}

#[tauri::command]
fn cancel_run(state: State<'_, AppState>) {
    if let Some(run) = state.active_run.lock().unwrap().as_ref() {
        run.cancel.cancel();
    }
}

// ---------------------------------------------------------------- startup

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let data = match std::env::var_os("WAYPOINT_DATA_DIR") {
                Some(dir) => PathBuf::from(dir),
                None => app.path().data_dir()?.join(DATA_DIR_NAME),
            };
            let paths = Paths { data };
            app.manage(AppState {
                paths,
                active_run: Mutex::default(),
                image: Mutex::default(),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            settings_get,
            settings_set,
            env_status,
            env_setup,
            env_purge,
            select_image,
            load_image,
            open_external,
            update_check,
            update_install,
            app_version,
            run_one,
            run_point,
            cancel_run,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Waypoint");
}
