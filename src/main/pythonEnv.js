'use strict';
const fs = require('fs');
const path = require('path');
const https = require('https');
const { spawn } = require('child_process');
const extractZip = require('extract-zip');

// uv manages the Python interpreter itself (no system Python required) and
// is much faster than pip — same tool already used throughout this OSINT
// toolkit's own setup (F:\!!OSINT\tools\PLONK). This replaces an earlier
// approach that manually downloaded the Windows embeddable Python zip and
// bootstrapped pip by hand; uv does all of that more reliably in one tool.
const UV_VERSION = '0.11.30';
const UV_ZIP_URL = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-pc-windows-msvc.zip`;
const PYTHON_VERSION = '3.11';
const CUDA_TORCH_INDEX = 'https://download.pytorch.org/whl/cu128';

function paths(resourcesDir) {
  const uvDir = path.join(resourcesDir, 'uv');
  const venvDir = path.join(resourcesDir, 'venv');
  return {
    uvDir,
    uvExe: path.join(uvDir, 'uv.exe'),
    uvZip: path.join(resourcesDir, 'uv.zip'),
    uvPythonInstallDir: path.join(resourcesDir, 'uv-python'), // keeps the managed Python fully inside the app's own data dir, not the user's global uv cache
    venvDir,
    pythonExe: path.join(venvDir, 'Scripts', 'python.exe'),
  };
}

function isInstalled(resourcesDir) {
  return fs.existsSync(paths(resourcesDir).pythonExe);
}

function downloadFile(url, destPath, onProgress) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    const request = (u) => {
      https.get(u, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          request(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`Download failed: ${res.statusCode} ${u}`));
          return;
        }
        const total = parseInt(res.headers['content-length'] || '0', 10);
        let received = 0;
        res.on('data', (chunk) => {
          received += chunk.length;
          if (onProgress && total) onProgress(received / total);
        });
        res.pipe(file);
        file.on('finish', () => file.close(() => resolve()));
      }).on('error', reject);
    };
    request(url);
  });
}

function runCommand(exe, args, onLog, extraEnv) {
  return new Promise((resolve, reject) => {
    const proc = spawn(exe, args, { windowsHide: true, env: { ...process.env, ...(extraEnv || {}) } });
    proc.stdout.on('data', (d) => onLog(d.toString()));
    proc.stderr.on('data', (d) => onLog(d.toString()));
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${exe} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

async function ensureUv(resourcesDir, onLog) {
  const p = paths(resourcesDir);
  if (fs.existsSync(p.uvExe)) return;

  fs.mkdirSync(resourcesDir, { recursive: true });
  onLog('Downloading uv (Python package/interpreter manager)...');
  await downloadFile(UV_ZIP_URL, p.uvZip, (frac) => onLog(`Downloading uv... ${Math.round(frac * 100)}%`));

  onLog('Extracting uv...');
  fs.mkdirSync(p.uvDir, { recursive: true });
  await extractZip(p.uvZip, { dir: p.uvDir });
  fs.unlinkSync(p.uvZip);
}

function detectGpu(onLog) {
  return new Promise((resolve) => {
    const proc = spawn('nvidia-smi', [], { windowsHide: true });
    proc.on('error', () => resolve(false)); // nvidia-smi not on PATH -> no NVIDIA GPU/driver
    proc.on('close', (code) => resolve(code === 0));
  });
}

async function installEnv(resourcesDir, onLog) {
  const p = paths(resourcesDir);
  await ensureUv(resourcesDir, onLog);

  const uvEnv = { UV_PYTHON_INSTALL_DIR: p.uvPythonInstallDir };

  onLog(`Installing Python ${PYTHON_VERSION} (managed by uv, self-contained)...`);
  await runCommand(p.uvExe, ['python', 'install', PYTHON_VERSION], onLog, uvEnv);

  onLog('Creating virtual environment...');
  await runCommand(p.uvExe, ['venv', p.venvDir, '--python', PYTHON_VERSION, '--clear'], onLog, uvEnv);

  onLog('Checking for an NVIDIA GPU...');
  const hasGpu = await detectGpu(onLog);
  onLog(hasGpu ? 'NVIDIA GPU detected — installing CUDA-enabled torch.' : 'No NVIDIA GPU detected — installing CPU-only torch (inference will be slower).');

  const torchArgs = ['pip', 'install', '--python', p.pythonExe, 'torch', 'torchvision'];
  if (hasGpu) torchArgs.push('--index-url', CUDA_TORCH_INDEX);
  await runCommand(p.uvExe, torchArgs, onLog, uvEnv);

  return hasGpu;
}

async function installRequirements(resourcesDir, requirementsFile, onLog) {
  const p = paths(resourcesDir);
  onLog('Installing pipeline dependencies...');
  await runCommand(p.uvExe, ['pip', 'install', '--python', p.pythonExe, '-r', requirementsFile], onLog,
    { UV_PYTHON_INSTALL_DIR: p.uvPythonInstallDir });
}

/**
 * Spawn a backend script (run_pipeline.py) that emits NDJSON on stdout.
 * Calls onEvent(parsedJsonObject) per line; stderr is forwarded to onLog.
 */
function spawnBackendScript(resourcesDir, backendDir, scriptName, args, onEvent, onLog) {
  const p = paths(resourcesDir);
  const scriptPath = path.join(backendDir, scriptName);
  const proc = spawn(p.pythonExe, [scriptPath, ...args], {
    windowsHide: true,
    cwd: backendDir,
  });

  let buffer = '';
  proc.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      try {
        onEvent(JSON.parse(line));
      } catch (e) {
        onLog(`[unparsed stdout] ${line}`);
      }
    }
  });
  proc.stderr.on('data', (d) => onLog(d.toString()));

  return proc;
}

module.exports = {
  paths,
  isInstalled,
  installEnv,
  installRequirements,
  spawnBackendScript,
};
