'use strict';
const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const pythonEnv = require('./pythonEnv');

// Installed at runtime into userData (writable even when the app itself is a
// read-only portable .exe mount) rather than inside the packaged app's own
// resources folder.
const RESOURCES_DIR = () => path.join(app.getPath('userData'), 'python-runtime');
const BACKEND_DIR = () => {
  // Dev: python_backend/ sits next to package.json. Packaged: extraResources.
  const devPath = path.join(__dirname, '..', '..', 'python_backend');
  if (fs.existsSync(devPath)) return devPath;
  return path.join(process.resourcesPath, 'python_backend');
};
const DEPS_MARKER = () => path.join(RESOURCES_DIR(), '.deps_installed');
const SETTINGS_FILE = () => path.join(app.getPath('userData'), 'settings.json');
const ICON_FILE = () => {
  const devPath = path.join(__dirname, '..', '..', 'resources', 'icon.png');
  if (fs.existsSync(devPath)) return devPath;
  return path.join(process.resourcesPath, 'resources', 'icon.png');
};

let mainWindow;
const activeProcesses = new Set();

// child.kill() on Windows (TerminateProcess) only kills the target PID, not
// any helper/worker processes it may have spawned. Those get orphaned and
// keep running invisibly after the app quits. `taskkill /T` kills the whole
// process tree rooted at that PID instead.
function killProcessTree(proc) {
  if (process.platform === 'win32' && proc.pid) {
    execFile('taskkill', ['/PID', String(proc.pid), '/T', '/F'], () => {});
  } else {
    try { proc.kill(); } catch (e) { /* already dead */ }
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 900,
    minHeight: 640,
    title: 'Waypoint',
    icon: ICON_FILE(),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  // No application menu (File / Edit / View / Window / Help).
  Menu.setApplicationMenu(null);
  createWindow();
});

app.on('window-all-closed', () => {
  for (const proc of activeProcesses) killProcessTree(proc);
  if (process.platform !== 'darwin') app.quit();
});

// ---- Settings (Mapillary token etc.) ----

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(SETTINGS_FILE(), 'utf8'));
  } catch (e) {
    return {};
  }
}

function writeSettings(settings) {
  fs.mkdirSync(path.dirname(SETTINGS_FILE()), { recursive: true });
  fs.writeFileSync(SETTINGS_FILE(), JSON.stringify(settings, null, 2), 'utf8');
}

ipcMain.handle('settings:get', () => readSettings());
ipcMain.handle('settings:set', (_evt, settings) => {
  writeSettings({ ...readSettings(), ...settings });
  return readSettings();
});

// ---- Environment setup (uv-managed Python + deps) ----

// Runs calibrate.py (loads the real model, times a real batch) once during
// setup so the Samples field default reflects actual measured throughput on
// this machine instead of a VRAM/GPU-name guess -- see calibrate.py for why.
function runCalibration(resourcesDir, log) {
  return new Promise((resolve, reject) => {
    let result = null;
    const proc = pythonEnv.spawnBackendScript(
      resourcesDir, BACKEND_DIR(), 'calibrate.py', [],
      (evt) => { if (evt.event === 'calibration') result = evt; },
      log,
    );
    proc.on('close', (code) => {
      if (code === 0 && result) resolve(result);
      else reject(new Error(`calibration process exited with code ${code}`));
    });
  });
}

// Targets ~10s of sampling for a single run at the recommended default,
// rounded to the field's step size and clamped to a sane range.
const CALIBRATION_TARGET_SECONDS = 10;
function samplesFromThroughput(samplesPerSec) {
  if (!(samplesPerSec > 0)) return 512;
  const raw = samplesPerSec * CALIBRATION_TARGET_SECONDS;
  return Math.max(256, Math.min(8192, Math.round(raw / 64) * 64));
}

ipcMain.handle('env:status', () => {
  const resourcesDir = RESOURCES_DIR();
  return {
    pythonInstalled: pythonEnv.isInstalled(resourcesDir),
    depsInstalled: fs.existsSync(DEPS_MARKER()),
  };
});

ipcMain.handle('env:setup', async (event) => {
  const resourcesDir = RESOURCES_DIR();
  const send = (payload) => event.sender.send('env:progress', payload);
  const log = (message) => send({ event: 'log', message: message.trim ? message.trim() : message });

  try {
    send({ event: 'stage_start', stage: 'python_env' });
    const gpuInfo = await pythonEnv.installEnv(resourcesDir, log);
    writeSettings({ ...readSettings(), hardware: gpuInfo });
    send({ event: 'stage_done', stage: 'python_env', gpu_detected: gpuInfo.hasGpu, gpu_name: gpuInfo.gpuName, gpu_vram_mb: gpuInfo.vramMb });

    send({ event: 'stage_start', stage: 'pipeline_deps' });
    const requirementsFile = path.join(BACKEND_DIR(), 'requirements_base.txt');
    await pythonEnv.installRequirements(resourcesDir, requirementsFile, log);
    send({ event: 'stage_done', stage: 'pipeline_deps' });

    send({ event: 'stage_start', stage: 'calibrate' });
    try {
      const calib = await runCalibration(resourcesDir, log);
      const recommendedSamples = samplesFromThroughput(calib.samples_per_sec);
      writeSettings({ ...readSettings(), hardware: { ...gpuInfo, samplesPerSec: calib.samples_per_sec, recommendedSamples } });
      send({ event: 'stage_done', stage: 'calibrate', samples_per_sec: calib.samples_per_sec, recommended_samples: recommendedSamples });
    } catch (e) {
      log(`Calibration skipped: ${e.message}`);
      send({ event: 'stage_done', stage: 'calibrate' });
    }

    fs.mkdirSync(RESOURCES_DIR(), { recursive: true });
    fs.writeFileSync(DEPS_MARKER(), new Date().toISOString(), 'utf8');
    send({ event: 'setup_complete' });
    return { ok: true };
  } catch (e) {
    send({ event: 'error', message: e.message });
    return { ok: false, error: e.message };
  }
});

ipcMain.handle('env:purge', async () => {
  const resourcesDir = RESOURCES_DIR();
  try {
    if (fs.existsSync(resourcesDir)) {
      fs.rmSync(resourcesDir, { recursive: true, force: true });
    }
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

// ---- Pipeline execution ----

function runPipeline(event, mode, args) {
  const resourcesDir = RESOURCES_DIR();
  const send = (payload) => event.sender.send('pipeline:event', payload);

  const argv = [mode, '--image_path', args.imagePath];
  if (args.mapillaryToken) argv.push('--mapillary_token', args.mapillaryToken);
  if (mode === 'point') {
    argv.push('--lat', String(args.lat), '--lon', String(args.lon));
    if (args.radiusKm) argv.push('--radius_km', String(args.radiusKm));
    if (args.maxImages) argv.push('--max_images', String(args.maxImages));
  } else {
    if (args.numSamples) argv.push('--num_samples', String(args.numSamples));
    if (args.numRuns) argv.push('--num_runs', String(args.numRuns));
    if (args.retrievalTopClusters) argv.push('--retrieval_top_clusters', String(args.retrievalTopClusters));
  }

  const proc = pythonEnv.spawnBackendScript(
    resourcesDir, BACKEND_DIR(), 'run_pipeline.py', argv,
    (evt) => send(evt),
    (log) => send({ event: 'log', message: log }),
  );
  activeProcesses.add(proc);
  // The renderer gates its run buttons on this terminal signal (not on the
  // invoke() promise, which resolves at spawn time). Fires on normal exit
  // AND on crash, so the UI never stays stuck "busy".
  proc.on('close', (code) => {
    activeProcesses.delete(proc);
    send({ event: 'exit', code });
  });
  return proc.pid;
}

ipcMain.handle('pipeline:runOne', (event, args) => {
  const settings = readSettings();
  return runPipeline(event, 'one', { ...args, mapillaryToken: settings.mapillaryToken });
});

ipcMain.handle('pipeline:runPoint', (event, args) => {
  const settings = readSettings();
  return runPipeline(event, 'point', { ...args, mapillaryToken: settings.mapillaryToken });
});

ipcMain.handle('dialog:selectImage', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp'] }],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});
