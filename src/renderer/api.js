'use strict';

/* Bridge between the page and the Tauri (Rust) shell. renderer.js only talks
   to window.api, so the shell stays swappable. */
(function () {
  const tauri = window.__TAURI__;
  if (!tauri) return; // e.g. a test harness that installs its own window.api

  const { invoke } = tauri.core;
  const { listen } = tauri.event;

  window.api = {
    openExternal: (url) => invoke('open_external', { url }),
    getSettings: () => invoke('settings_get'),
    setSettings: (settings) => invoke('settings_set', { settings }),

    getEnvStatus: () => invoke('env_status'),
    // { nativeOnly: true } installs only the native engine (no Python fallback).
    runEnvSetup: (opts) => invoke('env_setup', { nativeOnly: !!(opts && opts.nativeOnly) }),
    purgeEnv: () => invoke('env_purge'),
    onEnvProgress: (cb) => listen('env-progress', (e) => cb(e.payload)),

    // Both resolve to { path, name, bytes, dataUrl } (selectImage: or null).
    selectImage: () => invoke('select_image'),
    loadImage: (path) => invoke('load_image', { path }),
    onFileDrag: (cb) => {
      listen('tauri://drag-enter', () => cb('enter'));
      listen('tauri://drag-leave', () => cb('leave'));
      listen('tauri://drag-drop', (e) => cb('drop', (e.payload && e.payload.paths) || []));
    },

    runOne: (args) => invoke('run_one', { args }),
    runPoint: (args) => invoke('run_point', { args }),
    cancelRun: () => invoke('cancel_run'),
    onPipelineEvent: (cb) => listen('pipeline-event', (e) => cb(e.payload)),
  };
})();
