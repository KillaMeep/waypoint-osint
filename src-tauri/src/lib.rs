//! Waypoint desktop shell.
//!
//! The Rust side owns everything the webview can't do on its own: installing
//! the inference engine, running the pipeline (`waypoint-core`), settings, and
//! native dialogs.
//!
//! Two engines, picked per run:
//! * native: ONNX Runtime (DirectML, CPU fallback) with the exported PLONK,
//!   StreetCLIP, DISK and LightGlue models. No Python. First-run setup
//!   downloads the runtime from NuGet and the models from MODEL_BASE_URL.
//! * Python: the uv-managed runtime with PyTorch, serving the networks through
//!   `python_backend/model_server.py`. Used when the native files are absent
//!   (older installs, or when the model host can't be reached during setup).
//!   The backend scripts are embedded and unpacked at launch.

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
use sha2::{Digest, Sha256};
use waypoint_core::native;
use waypoint_core::onnx::{init_runtime, Accel};
use waypoint_core::plonk::{self, PlonkStep};
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
    fn ort_dir(&self) -> PathBuf { self.data.join("ort") }
    fn onnx_dir(&self) -> PathBuf { self.data.join("onnx") }
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
        "pythonInstalled": state.paths.python_exe().exists(),
        "depsInstalled": state.paths.deps_marker().exists(),
        // The native engine counts as installed once every PLONK model is: an
        // older install with only OSV-5M is offered the (incremental) install.
        "native": loc.as_ref().is_some_and(|l| plonk::VARIANTS.iter().all(|v| l.has(v.key))),
        "models": models,
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

/// Why the native engine could not be installed.
enum NativeError {
    /// The model host can't be reached, or the engine can't run on this
    /// machine. First-run setup falls back to the Python runtime.
    Unavailable(String),
    /// A download or integrity check failed part-way. Retrying is the fix;
    /// starting a multi-GB Python install instead would not be.
    Failed(String),
}

impl From<String> for NativeError {
    fn from(e: String) -> Self {
        NativeError::Failed(e)
    }
}

/// Download the native engine: ONNX Runtime + DirectML from NuGet and the
/// exported models (all three PLONK models) from the model host. Files already
/// present with the right hash are kept. `progress(done, total, status)` (bytes)
/// drives the setup screen.
fn install_native(state: &AppState, progress: &dyn Fn(u64, u64, &str), log: &dyn Fn(&str)) -> Result<(), NativeError> {
    let p = &state.paths;
    let models = manifest_models()?;
    let base = std::env::var("WAYPOINT_MODEL_URL").unwrap_or_else(|_| MODEL_BASE_URL.to_string());
    let base = base.trim_end_matches('/');

    // Nothing else is worth fetching if the model host isn't there.
    let first = models[0]["name"].as_str().unwrap_or_default();
    ureq::head(&format!("{base}/{first}"))
        .call()
        .map_err(|e| NativeError::Unavailable(format!("the model files are not available at {base} ({e})")))?;

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
            return Err(format!("{} from {} {} has an unexpected hash ({got})", f.file, f.package, f.version).into());
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
            return Err(format!("{name} failed its integrity check (sha256 {got}, expected {want})").into());
        }
        fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load the native engine and time its PLONK sampler (DirectML, else CPU).
/// Returns (samples/s, accelerator used). Any failure here means the engine
/// can't run on this machine.
fn native_calibration(state: &AppState, log: &dyn Fn(&str)) -> Result<(f64, &'static str), NativeError> {
    let p = &state.paths;
    let unusable = |e: String| NativeError::Unavailable(format!("the native engine could not start ({e})"));
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

fn setup_native(app: &AppHandle) -> Result<(), NativeError> {
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

/// First-run setup: the native engine when its files can be fetched, else the
/// Python runtime. `native_only` (the Settings upgrade button) never falls back.
fn setup_env(app: &AppHandle, native_only: bool) -> Result<(), String> {
    match setup_native(app) {
        Ok(()) => Ok(()),
        Err(NativeError::Failed(e)) => Err(e),
        Err(NativeError::Unavailable(e)) if native_only => Err(e),
        Err(NativeError::Unavailable(e)) => {
            let msg = format!("Native engine unavailable: {e}. Installing the Python runtime instead.");
            let _ = app.emit("env-progress", json!({ "event": "log", "message": msg }));
            setup_python(app)
        }
    }
}

fn setup_python(app: &AppHandle) -> Result<(), String> {
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
async fn env_setup(app: AppHandle, native_only: Option<bool>) -> Value {
    let handle = app.clone();
    let native_only = native_only.unwrap_or(false);
    let result = tauri::async_runtime::spawn_blocking(move || setup_env(&handle, native_only))
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

/// Delete every downloaded runtime and model (Python and native).
#[tauri::command]
fn env_purge(state: State<'_, AppState>) -> Value {
    let p = &state.paths;
    let mut errors = vec![];
    for dir in [p.runtime(), p.onnx_dir(), p.ort_dir()] {
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
    // The native engine runs a model whose files are all installed, with no
    // Python; otherwise the whole run goes through the Python model server.
    let engine = native::locate(&state.paths.data);
    let has_engine = engine.is_some();
    let native_loc = engine.filter(|l| l.has(model.key));
    let python = state.paths.python_exe();
    if native_loc.is_none() && !python.exists() {
        return Err(if has_engine {
            format!("The {} model is not installed. Open Settings and install the native engine, or pick another model.", model.label)
        } else {
            "No inference engine is installed. Run first-time setup.".into()
        });
    }
    let needs_python = native_loc.is_none();

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
    let server_pid: Arc<Mutex<Option<u32>>> = Arc::default();
    *active = Some(ActiveRun { cancel: cancel.clone(), server_pid: server_pid.clone() });
    drop(active);

    let script = state.paths.backend().join("model_server.py");
    let cwd = state.paths.backend();
    thread::spawn(move || {
        let sink = TauriSink { app: app.clone() };
        let spawn_py = || {
            let log_app = app.clone();
            match PyModels::spawn(&python, &script, &cwd, move |s| {
                let _ = log_app.emit("pipeline-event", json!({ "event": "log", "message": s }));
            }) {
                Ok(m) => {
                    let pid = m.pid();
                    *server_pid.lock().unwrap() = Some(pid);
                    app.state::<AppState>().track(pid);
                    Some(m)
                }
                Err(e) => {
                    sink.log(&format!("could not start the Python engine: {e}"));
                    None
                }
            }
        };
        let mut py = if needs_python { spawn_py() } else { None };
        let native = native_loc.and_then(|loc| {
            native::open(&loc, model.key, &|s| sink.log(s)).map_err(|e| sink.log(&format!("native engine failed to load: {e}"))).ok()
        });
        // A native engine that won't load falls back to an installed Python engine.
        if native.is_none() && py.is_none() && python.exists() {
            sink.log("falling back to the Python engine");
            py = spawn_py();
        }
        let code = match (&native, &py) {
            (Some(n), _) => core_run(&ra, n, &sink, &cancel),
            (None, Some(m)) => core_run(&ra, &**m, &sink, &cancel),
            (None, None) => {
                sink.event(json!({ "event": "error", "message": "No inference engine could be started. See the log." }));
                1
            }
        };
        drop(native);
        if let Some(m) = py {
            let pid = m.pid();
            m.shutdown();
            app.state::<AppState>().untrack(pid);
        }
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
