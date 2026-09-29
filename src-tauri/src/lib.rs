//! Waypoint desktop shell.
//!
//! The Rust side owns everything the webview can't do on its own: the
//! portable Python runtime (uv + a managed interpreter + PyTorch), spawning the
//! NDJSON-emitting pipeline scripts, settings, and native dialogs. The
//! geolocation work itself stays in `python_backend/` (PLONK, kornia, ...),
//! which is embedded in the binary and unpacked next to the runtime at launch
//! so the release build is a single portable exe.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use include_dir::{include_dir, Dir};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use waypoint_core::pyserver::PyModels;
use waypoint_core::{run as core_run, Cancel, Mode, RunArgs as CoreArgs, Sink};

static BACKEND: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../python_backend");

// uv manages the interpreter itself, so no system Python is needed.
const UV_VERSION: &str = "0.11.30";
const PYTHON_VERSION: &str = "3.11";
const CUDA_TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
// Aim the default Samples value at ~10 s of sampling for one run.
const CALIBRATION_TARGET_SECONDS: f64 = 10.0;
const IMAGE_EXTS: [&str; 5] = ["jpg", "jpeg", "png", "bmp", "webp"];

// Folder name kept from the Electron build (its `userData` dir), so an
// existing install keeps its multi-GB runtime and settings after the port.
const DATA_DIR_NAME: &str = "geolocator-gui";

struct Paths {
    data: PathBuf,
}

impl Paths {
    fn runtime(&self) -> PathBuf { self.data.join("python-runtime") }
    fn uv_dir(&self) -> PathBuf { self.runtime().join("uv") }
    fn uv_exe(&self) -> PathBuf { self.uv_dir().join("uv.exe") }
    fn uv_zip(&self) -> PathBuf { self.runtime().join("uv.zip") }
    // Keeps the managed interpreter inside the app's data dir instead of the
    // user's global uv cache.
    fn uv_python_dir(&self) -> PathBuf { self.runtime().join("uv-python") }
    fn venv(&self) -> PathBuf { self.runtime().join("venv") }
    // WAYPOINT_PYTHON lets a developer point at another interpreter (read-only use).
    fn python_exe(&self) -> PathBuf {
        match std::env::var_os("WAYPOINT_PYTHON") {
            Some(p) => PathBuf::from(p),
            None => self.venv().join("Scripts").join("python.exe"),
        }
    }
    fn deps_marker(&self) -> PathBuf { self.runtime().join(".deps_installed") }
    fn settings(&self) -> PathBuf { self.data.join("settings.json") }
    fn backend(&self) -> PathBuf { self.data.join("backend") }
}

struct AppState {
    paths: Paths,
    children: Mutex<HashSet<u32>>,
    active_run: Mutex<Option<ActiveRun>>,
    image: Mutex<Option<PathBuf>>,
}

/// The pipeline run in flight: its cancel flag, and the model server it talks to.
struct ActiveRun {
    cancel: Cancel,
    server_pid: Arc<Mutex<Option<u32>>>,
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

impl AppState {
    fn track(&self, pid: u32) { self.children.lock().unwrap().insert(pid); }
    fn untrack(&self, pid: u32) { self.children.lock().unwrap().remove(&pid); }
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

// Killing only the python PID would orphan any workers it started; taskkill
// /T takes down the whole tree.
fn kill_tree(pid: u32) {
    #[cfg(windows)]
    {
        let _ = hide_console(Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"])).status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
}

fn pump<R: Read + Send + 'static>(mut reader: R, sink: impl Fn(String) + Send + 'static) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => sink(String::from_utf8_lossy(&buf[..n]).into_owned()),
            }
        }
    })
}

/// Run a tool to completion, streaming its stdout and stderr to `log`.
fn run_command(state: &AppState, exe: &Path, args: &[String], log: &dyn Fn(&str)) -> Result<(), String> {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .env("UV_PYTHON_INSTALL_DIR", state.paths.uv_python_dir())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = hide_console(&mut cmd)
        .spawn()
        .map_err(|e| format!("failed to start {}: {e}", exe.display()))?;
    state.track(child.id());

    // Both pipes drain on their own threads; a full stderr pipe would
    // otherwise stall uv/pip mid-install.
    let (tx, rx) = mpsc::channel::<String>();
    let tx2 = tx.clone();
    let out = pump(child.stdout.take().unwrap(), move |s| { let _ = tx.send(s); });
    let err = pump(child.stderr.take().unwrap(), move |s| { let _ = tx2.send(s); });
    for msg in rx {
        log(&msg);
    }
    let _ = out.join();
    let _ = err.join();

    let status = child.wait().map_err(|e| e.to_string());
    state.untrack(child.id());
    let status = status?;
    if status.success() {
        Ok(())
    } else {
        let code = status.code().map_or_else(|| "?".to_string(), |c| c.to_string());
        Err(format!("{} {} exited with code {code}", exe.display(), args.join(" ")))
    }
}

struct BackendProc {
    child: Child,
    readers: Vec<JoinHandle<()>>,
}

impl BackendProc {
    /// Waits for exit, then for the readers, so every event is delivered
    /// before the caller reports the process as finished.
    fn wait(mut self) -> Option<i32> {
        let code = self.child.wait().ok().and_then(|s| s.code());
        for r in self.readers {
            let _ = r.join();
        }
        code
    }
}

/// Spawn a backend script that prints one JSON object per stdout line.
fn spawn_backend(
    state: &AppState,
    script: &str,
    args: &[String],
    on_event: impl Fn(Value) + Send + 'static,
    on_log: impl Fn(String) + Send + Sync + 'static,
) -> Result<BackendProc, String> {
    let p = &state.paths;
    if !p.python_exe().exists() {
        return Err("Python runtime is not installed. Run first-time setup.".into());
    }
    let backend = p.backend();
    let mut cmd = Command::new(p.python_exe());
    cmd.arg(backend.join(script))
        .args(args)
        .current_dir(&backend)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = hide_console(&mut cmd)
        .spawn()
        .map_err(|e| format!("failed to start python: {e}"))?;
    state.track(child.id());

    let on_log = Arc::new(on_log);
    let log_out = on_log.clone();
    let stdout = child.stdout.take().unwrap();
    let out = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&line);
                    let text = text.trim();
                    if text.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(text) {
                        Ok(v) => on_event(v),
                        Err(_) => log_out(format!("[unparsed stdout] {text}")),
                    }
                }
            }
        }
    });
    let err = pump(child.stderr.take().unwrap(), move |s| on_log(s));
    Ok(BackendProc { child, readers: vec![out, err] })
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

#[tauri::command]
fn env_status(state: State<'_, AppState>) -> Value {
    json!({
        "pythonInstalled": state.paths.python_exe().exists(),
        "depsInstalled": state.paths.deps_marker().exists(),
    })
}

fn download(url: &str, dest: &Path, log: &dyn Fn(&str)) -> Result<(), String> {
    let resp = ureq::get(url).call().map_err(|e| format!("Download failed: {e}"))?;
    let total: u64 = resp.header("Content-Length").and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut file = fs::File::create(dest).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 64 * 1024];
    let (mut received, mut last_pct) = (0u64, u64::MAX);
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Download failed: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        received += n as u64;
        if total > 0 {
            let pct = received * 100 / total;
            if pct != last_pct {
                last_pct = pct;
                log(&format!("Downloading uv... {pct}%"));
            }
        }
    }
    Ok(())
}

fn ensure_uv(state: &AppState, log: &dyn Fn(&str)) -> Result<(), String> {
    let p = &state.paths;
    if p.uv_exe().exists() {
        return Ok(());
    }
    fs::create_dir_all(p.runtime()).map_err(|e| e.to_string())?;
    log("Downloading uv (Python package/interpreter manager)...");
    let url = format!("https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip");
    download(&url, &p.uv_zip(), log)?;

    log("Extracting uv...");
    fs::create_dir_all(p.uv_dir()).map_err(|e| e.to_string())?;
    let zip = fs::File::open(p.uv_zip()).map_err(|e| e.to_string())?;
    zip::ZipArchive::new(zip)
        .and_then(|mut a| a.extract(p.uv_dir()))
        .map_err(|e| format!("Extracting uv failed: {e}"))?;
    let _ = fs::remove_file(p.uv_zip());
    Ok(())
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

fn install_env(state: &AppState, log: &dyn Fn(&str)) -> Result<Value, String> {
    let p = &state.paths;
    ensure_uv(state, log)?;
    let uv = p.uv_exe();
    let s = |x: &Path| x.to_string_lossy().into_owned();

    log(&format!("Installing Python {PYTHON_VERSION} (managed by uv, self-contained)..."));
    run_command(state, &uv, &["python".into(), "install".into(), PYTHON_VERSION.into()], log)?;

    log("Creating virtual environment...");
    run_command(state, &uv, &["venv".into(), s(&p.venv()), "--python".into(), PYTHON_VERSION.into(), "--clear".into()], log)?;

    log("Checking for an NVIDIA GPU...");
    let gpu = query_gpu();
    let has_gpu = gpu["hasGpu"].as_bool().unwrap_or(false);
    if has_gpu {
        let name = gpu["gpuName"].as_str().unwrap_or("unknown model");
        let vram = gpu["vramMb"].as_u64().map(|mb| format!(", {:.1}GB", mb as f64 / 1024.0)).unwrap_or_default();
        log(&format!("NVIDIA GPU detected ({name}{vram}), installing CUDA-enabled torch."));
    } else {
        log("No NVIDIA GPU detected, installing CPU-only torch (inference will be slower).");
    }

    let mut args: Vec<String> = vec!["pip".into(), "install".into(), "--python".into(), s(&p.python_exe()), "torch".into(), "torchvision".into()];
    if has_gpu {
        args.extend(["--index-url".into(), CUDA_TORCH_INDEX.into()]);
    }
    run_command(state, &uv, &args, log)?;
    Ok(gpu)
}

fn samples_from_throughput(samples_per_sec: f64) -> u64 {
    if !(samples_per_sec > 0.0) {
        return 512;
    }
    let raw = samples_per_sec * CALIBRATION_TARGET_SECONDS;
    ((raw / 64.0).round() as u64 * 64).clamp(256, 8192)
}

fn run_calibration(app: &AppHandle) -> Result<f64, String> {
    let state = app.state::<AppState>();
    let result: Arc<Mutex<Option<f64>>> = Arc::default();
    let sink = result.clone();
    let log_app = app.clone();
    let proc = spawn_backend(
        &state,
        "calibrate.py",
        &[],
        move |evt| {
            if evt["event"] == "calibration" {
                *sink.lock().unwrap() = evt["samples_per_sec"].as_f64();
            }
        },
        move |msg| { let _ = log_app.emit("env-progress", json!({ "event": "log", "message": msg.trim() })); },
    )?;
    let pid = proc.child.id();
    let code = proc.wait();
    state.untrack(pid);
    let measured = *result.lock().unwrap();
    match (code, measured) {
        (Some(0), Some(sps)) => Ok(sps),
        _ => Err(format!("calibration process exited with code {}", code.map_or("?".into(), |c| c.to_string()))),
    }
}

fn setup_env(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();
    let p = &state.paths;
    let send = |v: Value| { let _ = app.emit("env-progress", v); };
    let log = |m: &str| send(json!({ "event": "log", "message": m.trim() }));

    send(json!({ "event": "stage_start", "stage": "python_env" }));
    let gpu = install_env(&state, &log)?;
    merge_settings(p, Map::from_iter([("hardware".to_string(), gpu.clone())]))?;
    send(json!({
        "event": "stage_done", "stage": "python_env",
        "gpu_detected": gpu["hasGpu"], "gpu_name": gpu["gpuName"], "gpu_vram_mb": gpu["vramMb"],
    }));

    send(json!({ "event": "stage_start", "stage": "pipeline_deps" }));
    log("Installing pipeline dependencies...");
    let reqs = p.backend().join("requirements_base.txt");
    let args: Vec<String> = vec![
        "pip".into(), "install".into(), "--python".into(),
        p.python_exe().to_string_lossy().into_owned(), "-r".into(), reqs.to_string_lossy().into_owned(),
    ];
    run_command(&state, &p.uv_exe(), &args, &log)?;
    send(json!({ "event": "stage_done", "stage": "pipeline_deps" }));

    // Loads the real model and times a real batch, so the Samples default
    // reflects measured throughput on this machine (see calibrate.py).
    send(json!({ "event": "stage_start", "stage": "calibrate" }));
    match run_calibration(app) {
        Ok(sps) => {
            let recommended = samples_from_throughput(sps);
            let mut hardware = gpu.as_object().cloned().unwrap_or_default();
            hardware.insert("samplesPerSec".into(), json!(sps));
            hardware.insert("recommendedSamples".into(), json!(recommended));
            merge_settings(p, Map::from_iter([("hardware".to_string(), Value::Object(hardware))]))?;
            send(json!({ "event": "stage_done", "stage": "calibrate", "samples_per_sec": sps, "recommended_samples": recommended }));
        }
        Err(e) => {
            log(&format!("Calibration skipped: {e}"));
            send(json!({ "event": "stage_done", "stage": "calibrate" }));
        }
    }

    fs::create_dir_all(p.runtime()).map_err(|e| e.to_string())?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    fs::write(p.deps_marker(), stamp.to_string()).map_err(|e| e.to_string())?;
    send(json!({ "event": "setup_complete" }));
    Ok(())
}

#[tauri::command]
async fn env_setup(app: AppHandle) -> Value {
    let handle = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || setup_env(&handle))
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

#[tauri::command]
fn env_purge(state: State<'_, AppState>) -> Value {
    let runtime = state.paths.runtime();
    match fs::remove_dir_all(&runtime) {
        Ok(()) => json!({ "ok": true }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({ "ok": true }),
        Err(e) => json!({ "ok": false, "error": e.to_string() }),
    }
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
    let python = state.paths.python_exe();
    if !python.exists() {
        return Err("Python runtime is not installed. Run first-time setup.".into());
    }

    let mut ra = CoreArgs::new(mode.clone(), image);
    ra.mapillary_token = read_settings(&state.paths)
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
    let server_pid: Arc<Mutex<Option<u32>>> = Arc::default();
    *active = Some(ActiveRun { cancel: cancel.clone(), server_pid: server_pid.clone() });
    drop(active);

    let script = state.paths.backend().join("model_server.py");
    let cwd = state.paths.backend();
    thread::spawn(move || {
        let sink = TauriSink { app: app.clone() };
        let log_app = app.clone();
        let code = match PyModels::spawn(&python, &script, &cwd, move |s| {
            let _ = log_app.emit("pipeline-event", json!({ "event": "log", "message": s }));
        }) {
            Ok(models) => {
                let pid = models.pid();
                *server_pid.lock().unwrap() = Some(pid);
                app.state::<AppState>().track(pid);
                let code = core_run(&ra, &*models, &sink, &cancel);
                models.shutdown();
                app.state::<AppState>().untrack(pid);
                code
            }
            Err(e) => {
                sink.event(json!({ "event": "error", "message": e.to_string() }));
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
        // Killing the model server unblocks any request the backend is waiting on.
        if let Some(pid) = *run.server_pid.lock().unwrap() {
            kill_tree(pid);
        }
    }
}

// ---------------------------------------------------------------- startup

/// Unpack the embedded backend, rewriting only files whose bytes changed.
fn extract_dir(dir: &Dir, root: &Path) -> std::io::Result<()> {
    for file in dir.files() {
        let dest = root.join(file.path());
        if file.path().components().any(|c| c.as_os_str() == "__pycache__") {
            continue;
        }
        if fs::read(&dest).map(|cur| cur == file.contents()).unwrap_or(false) {
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest, file.contents())?;
    }
    for sub in dir.dirs() {
        if sub.path().file_name().is_some_and(|n| n == "__pycache__") {
            continue;
        }
        extract_dir(sub, root)?;
    }
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let data = match std::env::var_os("WAYPOINT_DATA_DIR") {
                Some(dir) => PathBuf::from(dir),
                None => app.path().data_dir()?.join(DATA_DIR_NAME),
            };
            let paths = Paths { data };
            extract_dir(&BACKEND, &paths.backend())?;
            app.manage(AppState {
                paths,
                children: Mutex::default(),
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
            run_one,
            run_point,
            cancel_run,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Waypoint")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                let pids: Vec<u32> = app.state::<AppState>().children.lock().unwrap().iter().copied().collect();
                for pid in pids {
                    kill_tree(pid);
                }
            }
        });
}
