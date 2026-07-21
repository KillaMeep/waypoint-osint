'use strict';
const { contextBridge, ipcRenderer, shell } = require('electron');

contextBridge.exposeInMainWorld('api', {
  openExternal: (url) => shell.openExternal(url),
  getSettings: () => ipcRenderer.invoke('settings:get'),
  setSettings: (settings) => ipcRenderer.invoke('settings:set', settings),

  getEnvStatus: () => ipcRenderer.invoke('env:status'),
  runEnvSetup: () => ipcRenderer.invoke('env:setup'),
  purgeEnv: () => ipcRenderer.invoke('env:purge'),
  onEnvProgress: (callback) => ipcRenderer.on('env:progress', (_evt, payload) => callback(payload)),

  selectImage: () => ipcRenderer.invoke('dialog:selectImage'),

  runOne: (args) => ipcRenderer.invoke('pipeline:runOne', args),
  runPoint: (args) => ipcRenderer.invoke('pipeline:runPoint', args),
  onPipelineEvent: (callback) => ipcRenderer.on('pipeline:event', (_evt, payload) => callback(payload)),
});
