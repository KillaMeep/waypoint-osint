//! Client for `python_backend/model_server.py`: a long-lived Python process
//! that answers NDJSON requests for the three neural operations.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine;
use image::RgbImage;
use serde_json::{json, Value};

use crate::models::{Models, PairMatches};
use crate::util::{Error, Result};

#[cfg(windows)]
pub fn hide_console(cmd: &mut Command) -> &mut Command {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000) // CREATE_NO_WINDOW
}
#[cfg(not(windows))]
pub fn hide_console(cmd: &mut Command) -> &mut Command {
    cmd
}

/// Kill a process and everything it spawned.
pub fn kill_tree(pid: u32) {
    #[cfg(windows)]
    {
        let _ = hide_console(Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null())).status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
}

struct Io {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

pub struct PyModels {
    child: Mutex<Child>,
    pid: u32,
    io: Mutex<Io>,
    next_id: AtomicU64,
    tmp: PathBuf,
    tmp_seq: AtomicU64,
}

fn decode_f32(b64: &str) -> Result<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|e| Error::Msg(format!("bad base64 from model server: {e}")))?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

impl PyModels {
    /// Start the server. `on_log` receives its stderr (library chatter, warnings).
    pub fn spawn(python: &Path, script: &Path, cwd: &Path, on_log: impl Fn(String) + Send + 'static) -> Result<Arc<Self>> {
        let mut cmd = Command::new(python);
        cmd.arg(script)
            .current_dir(cwd)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = hide_console(&mut cmd).spawn().map_err(|e| Error::Msg(format!("failed to start python model server: {e}")))?;
        let pid = child.id();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut stderr = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
                on_log(String::from_utf8_lossy(&buf[..n]).into_owned());
            }
        });
        let tmp = std::env::temp_dir().join(format!("waypoint-{}-{}", std::process::id(), pid));
        std::fs::create_dir_all(&tmp)?;
        Ok(Arc::new(PyModels {
            child: Mutex::new(child),
            pid,
            io: Mutex::new(Io { stdin, stdout }),
            next_id: AtomicU64::new(1),
            tmp,
            tmp_seq: AtomicU64::new(0),
        }))
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    fn request(&self, mut req: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        req["id"] = json!(id);
        let mut io = self.io.lock().unwrap();
        let line = serde_json::to_string(&req)? + "\n";
        io.stdin.write_all(line.as_bytes()).and_then(|_| io.stdin.flush()).map_err(|e| Error::Msg(format!("model server is gone: {e}")))?;
        loop {
            let mut buf = String::new();
            let n = io.stdout.read_line(&mut buf).map_err(|e| Error::Msg(format!("model server read failed: {e}")))?;
            if n == 0 {
                return Err(Error::Msg("model server exited unexpectedly".into()));
            }
            let Ok(v) = serde_json::from_str::<Value>(buf.trim()) else { continue };
            if v["id"].as_u64() != Some(id) {
                continue;
            }
            if v["ok"].as_bool() == Some(true) {
                return Ok(v);
            }
            return Err(Error::Msg(v["error"].as_str().unwrap_or("model server error").to_string()));
        }
    }

    fn stash(&self, img: &RgbImage) -> Result<PathBuf> {
        let p = self.tmp.join(format!("img{}.bmp", self.tmp_seq.fetch_add(1, Ordering::SeqCst)));
        img.save_with_format(&p, image::ImageFormat::Bmp).map_err(|e| Error::Msg(format!("could not stage image for the model server: {e}")))?;
        Ok(p)
    }

    pub fn shutdown(&self) {
        let _ = self.io.lock().map(|mut io| {
            let _ = io.stdin.write_all(b"{\"id\":0,\"op\":\"shutdown\"}\n");
            let _ = io.stdin.flush();
        });
        kill_tree(self.pid);
        let _ = self.child.lock().map(|mut c| c.wait());
        let _ = std::fs::remove_dir_all(&self.tmp);
    }
}

impl Drop for PyModels {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Models for PyModels {
    fn load(&self, model: &str) -> Result<()> {
        self.request(json!({ "op": "load", "model": model })).map(|_| ())
    }

    fn sample(&self, path: &Path, batch: usize, seed: Option<u64>) -> Result<Vec<[f32; 2]>> {
        let mut req = json!({ "op": "sample", "image": path.to_string_lossy(), "batch_size": batch });
        if let Some(s) = seed {
            req["seed"] = json!(s);
        }
        let v = self.request(req)?;
        let flat = decode_f32(v["f32"].as_str().unwrap_or(""))?;
        Ok(flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
    }

    fn embed_path(&self, path: &Path) -> Result<Vec<f32>> {
        let v = self.request(json!({ "op": "embed", "images": [path.to_string_lossy()] }))?;
        decode_f32(v["f32"].as_str().unwrap_or(""))
    }

    fn embed(&self, images: &[&RgbImage]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(images.len());
        for chunk in images.chunks(16) {
            let paths: Vec<PathBuf> = chunk.iter().map(|i| self.stash(i)).collect::<Result<_>>()?;
            let names: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
            let res = self.request(json!({ "op": "embed", "images": names }));
            for p in &paths {
                let _ = std::fs::remove_file(p);
            }
            let v = res?;
            let dim = v["dim"].as_u64().unwrap_or(0) as usize;
            let flat = decode_f32(v["f32"].as_str().unwrap_or(""))?;
            if dim == 0 || flat.len() != dim * chunk.len() {
                return Err(Error::Msg("model server returned a malformed embedding batch".into()));
            }
            out.extend(flat.chunks_exact(dim).map(|c| c.to_vec()));
        }
        Ok(out)
    }

    fn match_pair(&self, target: &Path, candidate: &RgbImage) -> Result<PairMatches> {
        let p = self.stash(candidate)?;
        let res = self.request(json!({ "op": "match", "a": target.to_string_lossy(), "b": p.to_string_lossy() }));
        let _ = std::fs::remove_file(&p);
        let v = res?;
        let pair = |k: &str| -> Result<Vec<[f32; 2]>> { Ok(decode_f32(v[k].as_str().unwrap_or(""))?.chunks_exact(2).map(|c| [c[0], c[1]]).collect()) };
        Ok(PairMatches {
            pts1: pair("pts1")?,
            pts2: pair("pts2")?,
            n_kp1: v["n_kp1"].as_u64().unwrap_or(0) as usize,
            n_kp2: v["n_kp2"].as_u64().unwrap_or(0) as usize,
        })
    }
}
