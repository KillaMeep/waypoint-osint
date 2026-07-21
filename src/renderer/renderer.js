'use strict';

/* ============================================================
   Geolocator renderer: tabbed views (Analyze / Results / Refine)
   The pipeline event contract is fixed by run_pipeline.py.
   ============================================================ */

const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// How many top clusters the backend refines with retrieval imagery.
// Kept in sync with the --retrieval_top_clusters we pass to runOne so the
// step list can show the whole plan up front instead of growing mid-run.
const RETRIEVAL_TOP = 1;

const STAGE_LABELS = {
  model_load: 'Load PLONK model',
  plonk_sampling: 'PLONK coarse locate',
  sun_refine: 'Sun / season plausibility',
  mapillary: 'Mapillary refinement',
  google_sv: 'Google Street View refinement',
  panoramax: 'Panoramax refinement',
};
const SOURCES = {
  mapillary_matches:  { label: 'Mapillary',          color: () => cssVar('--src-mly'), urlKey: 'mapillary_url' },
  google_sv_matches:  { label: 'Google Street View', color: () => cssVar('--src-gsv'), urlKey: 'street_view_url' },
  panoramax_matches:  { label: 'Panoramax',          color: () => cssVar('--src-pmx'), urlKey: 'panoramax_url' },
};
const STAGE_TO_MATCHKEY = { mapillary: 'mapillary_matches', google_sv: 'google_sv_matches', panoramax: 'panoramax_matches' };

let selectedImagePath = null;
let currentClusterIndex = 0;
let resultLatLngs = [];   // candidate + match points, for fitting the results map
let refineLatLngs = [];   // refine point + zoom matches, for fitting the refine map
let resultsFitted = false;
let pipelineBusy = false;  // a backend process is running, block launching another
let activeRunKind = null;  // 'full' | 'refine'

// Only one pipeline may run at a time (each loads the model into VRAM).
function setBusy(busy, owner) {
  pipelineBusy = busy;
  const r1 = $('runOneBtn'), rz = $('runZoomBtn');
  if (busy) {
    r1.disabled = true; rz.disabled = true;
    (owner === 'refine' ? rz : r1).classList.add('busy');
  } else {
    r1.classList.remove('busy'); rz.classList.remove('busy');
    r1.disabled = !selectedImagePath;
    const lat = parseFloat($('zoomLat').value), lon = parseFloat($('zoomLon').value);
    rz.disabled = !(isFinite(lat) && isFinite(lon));
  }
}

function stageInfo(stage) {
  let prefix = '';
  let bare = stage;
  const cm = bare.match(/^c(\d+)_/);
  if (cm) { prefix = `Candidate ${parseInt(cm[1], 10) + 1} · `; bare = bare.slice(cm[0].length); }
  else if (bare.startsWith('zoom_')) { bare = bare.slice(5); }
  return prefix + (STAGE_LABELS[bare] || bare);
}

/* ============================================================ Theme */
(function initTheme() {
  const saved = localStorage.getItem('geo-theme');
  if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved);
})();
$('themeBtn').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme')
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('geo-theme', next);
});

/* ============================================================ Screens */
function showScreen(id) {
  for (const el of document.querySelectorAll('.screen')) el.classList.add('hidden');
  $(id).classList.remove('hidden');
  if (id === 'mainScreen') refreshActiveMap();
}

/* ============================================================ Tabbed views */
function showView(name) {
  for (const v of document.querySelectorAll('.view')) v.classList.toggle('is-active', v.id === `view-${name}`);
  for (const t of document.querySelectorAll('.tab')) t.classList.toggle('is-active', t.dataset.view === name);
  refreshActiveMap();
}
function refreshActiveMap() {
  const active = document.querySelector('.view.is-active');
  if (!active) return;
  if (active.id === 'view-results' && maps.results) {
    setTimeout(() => {
      maps.results.invalidateSize();
      if (!resultsFitted && resultLatLngs.length) { fitBounds(maps.results, resultLatLngs); resultsFitted = true; }
    }, 40);
  } else if (active.id === 'view-refine' && maps.refine) {
    setTimeout(() => { maps.refine.invalidateSize(); if (refineLatLngs.length) fitBounds(maps.refine, refineLatLngs); }, 40);
  }
}
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => { if (!tab.disabled) showView(tab.dataset.view); });
});
function setTabDone(name, done) {
  const t = document.querySelector(`.tab[data-view="${name}"]`);
  if (t) t.classList.toggle('is-done', !!done);
}

/* ============================================================ First-run setup */
const setupLog = $('setupLog');
let depsCrawl = null;
function setupAppend(text) { setupLog.textContent += text.replace(/\s+$/, '') + '\n'; setupLog.scrollTop = setupLog.scrollHeight; }
function setupBar(pct, status) {
  $('setupBar').style.width = `${clamp(pct, 0, 100)}%`;
  if (status) $('setupStatus').textContent = status;
  $('setupPct').textContent = pct >= 100 ? 'Done' : `${Math.round(pct)}%`;
}
function stopCrawl() { if (depsCrawl) { clearInterval(depsCrawl); depsCrawl = null; } }
function setupPhase(msg) {
  const dl = msg.match(/Downloading uv\.\.\.\s*(\d+)%/);
  if (dl) { setupBar(parseInt(dl[1], 10) * 0.25, 'Downloading Python manager'); return; }
  if (/Downloading uv \(/.test(msg))            return setupBar(2,  'Downloading Python manager');
  if (/Extracting uv/.test(msg))                return setupBar(27, 'Extracting Python manager');
  if (/Installing Python/.test(msg))            return setupBar(42, 'Installing Python runtime');
  if (/Creating virtual environment/.test(msg)) return setupBar(55, 'Creating virtual environment');
  if (/Checking for an NVIDIA GPU/.test(msg))   return setupBar(60, 'Detecting GPU');
  if (/GPU detected/.test(msg))                 return setupBar(66, 'Installing PyTorch (GPU)');
  if (/No NVIDIA GPU/.test(msg))                return setupBar(66, 'Installing PyTorch (CPU)');
  if (/Installing pipeline dependencies/.test(msg)) {
    setupBar(78, 'Installing dependencies');
    stopCrawl();
    depsCrawl = setInterval(() => { const w = parseFloat($('setupBar').style.width) || 78; if (w < 95) setupBar(w + 0.4); }, 700);
  }
}
$('setupStartBtn').addEventListener('click', async () => {
  const btn = $('setupStartBtn');
  btn.disabled = true; btn.classList.add('busy');
  setupBar(1, 'Starting…');
  const result = await window.api.runEnvSetup();
  btn.classList.remove('busy'); stopCrawl();
  if (result.ok) { setupBar(100, 'Setup complete'); setTimeout(async () => { showScreen('mainScreen'); await applyRecommendedSamples(); }, 700); }
  else { setupAppend(`Setup failed: ${result.error}`); $('setupStatus').textContent = 'Setup failed, see details.'; btn.disabled = false; }
});
window.api.onEnvProgress((payload) => {
  if (payload.event === 'log') {
    setupAppend(payload.message);
    for (const line of String(payload.message).split('\n')) if (line.trim()) setupPhase(line);
  } else if (payload.event === 'stage_start') {
    setupAppend(`--- ${payload.stage} ---`);
    if (payload.stage === 'calibrate') { stopCrawl(); setupBar(96, 'Benchmarking sampling speed'); }
  } else if (payload.event === 'stage_done') {
    setupAppend(`${payload.stage} done.`);
    if (payload.stage === 'python_env' && 'gpu_detected' in payload)
      setupAppend(payload.gpu_detected ? 'NVIDIA GPU detected.' : 'No GPU detected, CPU mode (slower).');
    if (payload.stage === 'calibrate' && payload.samples_per_sec)
      setupAppend(`Measured ${payload.samples_per_sec.toFixed(0)} samples/sec, default Samples set to ${payload.recommended_samples}.`);
  }
});

/* ============================================================ Settings */
$('settingsBtn').addEventListener('click', async () => {
  const settings = await window.api.getSettings();
  $('mapillaryTokenInput').value = settings.mapillaryToken || '';
  $('settingsSaved').textContent = '';
  showScreen('settingsScreen');
});
$('settingsCloseBtn').addEventListener('click', () => showScreen('mainScreen'));
$('tokenReveal').addEventListener('click', () => {
  const inp = $('mapillaryTokenInput');
  const show = inp.type === 'password';
  inp.type = show ? 'text' : 'password';
  $('tokenReveal').textContent = show ? 'Hide' : 'Show';
});
$('settingsSaveBtn').addEventListener('click', async () => {
  await window.api.setSettings({ mapillaryToken: $('mapillaryTokenInput').value.trim() });
  const note = $('settingsSaved');
  note.textContent = 'Saved ✓';
  setTimeout(() => { if (note.textContent === 'Saved ✓') note.textContent = ''; }, 2000);
});
$('mapillaryLink').addEventListener('click', (e) => { e.preventDefault(); window.api.openExternal('https://mapillary.com/dashboard/developers'); });
$('purgeEnvBtn').addEventListener('click', async () => {
  const confirmed = confirm(
    'This deletes the downloaded Python runtime, uv, and all installed packages (including PyTorch). ' +
    'You will need to redo first-run setup before running the pipeline again. Continue?');
  if (!confirmed) return;
  const btn = $('purgeEnvBtn'); const status = $('purgeStatus');
  btn.disabled = true; status.textContent = 'Purging…';
  const result = await window.api.purgeEnv();
  status.textContent = result.ok ? 'Done. Restart the app to run first-run setup again.' : `Failed: ${result.error}`;
  if (result.ok) resetSteps();
  btn.disabled = false;
});

/* ============================================================ Image selection */
async function pickImage() {
  const filePath = await window.api.selectImage();
  if (!filePath) return;
  selectedImagePath = filePath;
  const img = $('imagePreview');
  img.src = `file://${filePath.replace(/\\/g, '/')}`;
  img.classList.remove('hidden');
  $('imageEmpty').classList.add('hidden');
  $('runOneBtn').disabled = false;
}
$('selectImageBtn').addEventListener('click', pickImage);
$('imageDrop').addEventListener('click', pickImage);
$('imageDrop').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickImage(); } });

/* ============================================================ Maps (Leaflet, one per split view) */
const maps = { results: null, refine: null };
const layers = { results: {}, refine: {} };

function ensureMap(which, elId) {
  if (maps[which]) return maps[which];
  const m = L.map(elId, { zoomControl: true, attributionControl: true, worldCopyJump: true }).setView([20, 0], 2);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(m);
  layers[which] = {
    candidates: L.layerGroup().addTo(m),
    matches: L.layerGroup().addTo(m),
    point: L.layerGroup().addTo(m),
  };
  maps[which] = m;
  setTimeout(() => m.invalidateSize(), 40);
  return m;
}
function candIcon(rank, lead) {
  const bg = lead ? cssVar('--accent') : '#64748b';
  return L.divIcon({ className: '', iconSize: [26, 26], iconAnchor: [13, 26],
    html: `<div class="pin pin-cand" style="background:${bg}"><span>${rank}</span></div>` });
}
function matchIcon(color) {
  return L.divIcon({ className: '', iconSize: [14, 14], iconAnchor: [7, 14],
    html: `<div class="pin pin-match" style="background:${color}"></div>` });
}
function fitBounds(m, latlngs) {
  if (!latlngs.length) return;
  m.fitBounds(L.latLngBounds(latlngs).pad(0.35), { maxZoom: 13, animate: false });
}
function addCandidateMarker(i, c) {
  const m = L.marker([c.lat, c.lon], { icon: candIcon(i + 1, i === 0), zIndexOffset: 1000 - i })
    .bindTooltip(`Candidate ${i + 1}: ${(c.weight * 100).toFixed(0)}%`, { direction: 'top', offset: [0, -22] });
  m.on('click', () => refineCandidate(i, c));
  m.addTo(layers.results.candidates);
  resultLatLngs.push([c.lat, c.lon]);
}
function addMatchMarker(m, sourceKey, isZoom) {
  const which = isZoom ? 'refine' : 'results';
  (isZoom ? refineLatLngs : resultLatLngs).push([m.lat, m.lon]);
  if (!maps[which]) return;
  const color = isZoom ? '#a855f7' : SOURCES[sourceKey].color();
  L.marker([m.lat, m.lon], { icon: matchIcon(color) })
    .bindTooltip(`${SOURCES[sourceKey].label} · sim ${m.similarity.toFixed(3)}`, { direction: 'top', offset: [0, -10] })
    .on('click', () => fillRefine(m.lat, m.lon, 1))
    .addTo(layers[which].matches);
}

/* ============================================================ Steps + overall progress */
const stepList = $('stepList');
const stepEls = {};
const overallStages = { total: new Set(), completed: new Set() };
const progressStartTimes = {};

function resetSteps() {
  stepList.innerHTML = '';
  for (const k of Object.keys(stepEls)) delete stepEls[k];
  for (const k of Object.keys(progressStartTimes)) delete progressStartTimes[k];
  overallStages.total.clear(); overallStages.completed.clear();
  updateOverallProgress(true);
  for (const s of ['model_load', 'plonk_sampling', 'sun_refine']) ensureStep(s, 'pending');
  // Seed the retrieval steps for the clusters that will be refined, so the
  // whole plan is visible from the start rather than appearing mid-run.
  for (let i = 0; i < RETRIEVAL_TOP; i++)
    for (const s of ['mapillary', 'google_sv', 'panoramax']) ensureStep(`c${i}_${s}`, 'pending');
  updateOverallProgress(true); // seeding flips the pill off "Idle"; restore it until a run starts
}
const STEP_ICONS = {
  pending: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
  active:  '<svg viewBox="0 0 20 20"><circle class="spin-ring" cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-dasharray="30 14" stroke-linecap="round"/></svg>',
  done:    '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="9" fill="currentColor" opacity=".16"/><path d="M6 10.5l2.6 2.6L14.5 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  failed:  '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="9" fill="currentColor" opacity=".16"/><path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  skipped: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="9" fill="currentColor" opacity=".16"/><path d="M6 10h8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
};
function ensureStep(stage, initialState, listEl) {
  const list = listEl || stepList;
  const registry = list === stepList ? stepEls : (list.__els || (list.__els = {}));
  if (registry[stage]) return registry[stage];
  const li = document.createElement('li');
  li.className = `step ${initialState || 'pending'}`;
  li.innerHTML = `
    <span class="step-ico">${STEP_ICONS[initialState || 'pending']}</span>
    <span class="step-name">${stageInfo(stage)}</span>
    <span class="step-meta"></span>`;
  list.appendChild(li);
  registry[stage] = li;
  if (list === stepList) { overallStages.total.add(stage); updateOverallProgress(); }
  return li;
}
function stepListFor(stage) { return stage.startsWith('zoom_') ? $('zoomSteps') : stepList; }
function markStep(stage, state, metaText) {
  const li = ensureStep(stage, undefined, stepListFor(stage));
  li.className = `step ${state}`;
  li.querySelector('.step-ico').innerHTML = STEP_ICONS[state] || '';
  if (metaText !== undefined) li.querySelector('.step-meta').textContent = metaText;
  const isMain = !stage.startsWith('zoom_');
  if (isMain && (state === 'done' || state === 'failed' || state === 'skipped')) {
    overallStages.completed.add(stage);
    if (metaText === undefined) li.querySelector('.step-meta').textContent = '';
    updateOverallProgress();
  }
}
function updateOverallProgress(idle) {
  const total = Math.max(overallStages.total.size, 1);
  const done = overallStages.completed.size;
  const pct = Math.round((done / total) * 100);
  $('overallProgressBar').style.width = `${pct}%`;
  const pill = $('overallProgressLabel');
  if (idle) { pill.textContent = 'Idle'; pill.className = 'pill pill-muted'; return; }
  if (done >= total) { pill.textContent = 'Complete'; pill.className = 'pill pill-done'; }
  else { pill.textContent = `Running · ${done}/${total}`; pill.className = 'pill pill-run'; }
}
function formatEta(sec) {
  if (!isFinite(sec) || sec < 0) return '';
  if (sec < 1) return '<1s';
  if (sec < 60) return `${Math.round(sec)}s`;
  return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
}
function updateStepProgress(stage, phase, completed, total) {
  const li = ensureStep(stage, undefined, stepListFor(stage));
  if (!li.classList.contains('active')) markStep(stage, 'active');
  const key = `${stage}:${phase}`;
  if (!(key in progressStartTimes) || progressStartTimes[key].startCompleted > completed)
    progressStartTimes[key] = { startTs: Date.now(), startCompleted: completed };
  const { startTs, startCompleted } = progressStartTimes[key];
  const elapsed = (Date.now() - startTs) / 1000;
  const doneSince = completed - startCompleted;
  const rate = doneSince > 0 ? doneSince / elapsed : 0;
  const eta = rate > 0 ? (total - completed) / rate : NaN;
  const etaText = formatEta(eta) ? ` · ETA ${formatEta(eta)}` : '';
  li.querySelector('.step-meta').textContent = `${phase} ${completed}/${total}${etaText}`;
}

/* ============================================================ Developer log */
const pipelineLog = $('pipelineLog');
const zoomLog = $('zoomLog');
function appendPipelineLog(text) {
  pipelineLog.textContent += text + '\n';
  pipelineLog.scrollTop = pipelineLog.scrollHeight;
  // Mirror into the Refine view's log so raw progress is visible there too.
  if (activeRunKind === 'refine') { zoomLog.textContent += text + '\n'; zoomLog.scrollTop = zoomLog.scrollHeight; }
}

/* ============================================================ Candidate cards */
const clusterResults = $('clusterResults');
function spreadKm(c) {
  if (c.lat_std == null || c.lon_std == null) return null;
  const latK = c.lat_std * 111;
  const lonK = c.lon_std * 111 * Math.cos((c.lat || 0) * Math.PI / 180);
  return Math.sqrt(latK * latK + lonK * lonK);
}
function copyBtn(text) {
  return `<button class="copy-btn" data-copy="${text}" title="Copy coordinates">
    <svg viewBox="0 0 24 24" width="14" height="14"><rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M5 15V6a2 2 0 0 1 2-2h9" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>
  </button>`;
}
function renderCandidateCard(i, c) {
  let card = clusterResults.querySelector(`[data-cluster="${i}"]`);
  const coords = `${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}`;
  const conf = (c.weight * 100).toFixed(0);
  if (!card) {
    card = document.createElement('div');
    card.className = 'candidate' + (i === 0 ? ' lead' : '');
    card.dataset.cluster = i;
    card.innerHTML = `
      <div class="cand-head">
        <div class="cand-rank">${i + 1}</div>
        <div class="cand-main">
          <div class="cand-coords"><span class="coord-text">${coords}</span>${copyBtn(coords)}</div>
          <div class="cand-addr"></div>
          <div class="cand-meta-row"></div>
        </div>
        <div class="cand-conf">
          <span class="conf-val">${conf}%</span>
          <span class="conf-label">Confidence</span>
          <span class="track conf-bar"><span class="progress-fill" style="width:${conf}%"></span></span>
        </div>
      </div>
      <div class="sources"></div>
      <div class="cand-foot"><button class="btn btn-secondary btn-sm refine-area">Refine this area →</button></div>`;
    card.querySelector('.refine-area').addEventListener('click', () => refineCandidate(i, c));
    clusterResults.appendChild(card);
  }
  updateCandidateMeta(card, c);
  return card;
}
function updateCandidateMeta(card, c) {
  const addr = card.querySelector('.cand-addr');
  if (c.address) { addr.textContent = c.address; addr.title = c.address; }
  const chips = [];
  const sp = spreadKm(c);
  if (sp != null) chips.push(`<span class="chip" title="Spread of coarse samples">± ${sp < 1 ? (sp * 1000).toFixed(0) + ' m' : sp.toFixed(1) + ' km'}</span>`);
  if (c.count != null) chips.push(`<span class="chip" title="Coarse samples in this cluster">${c.count} samples</span>`);
  const ev = c.sun_evidence;
  if (ev) {
    if (ev.error == null) chips.push(`<span class="chip" title="${ev.note || 'no sun fit'}">☀ n/a</span>`);
    else {
      const rel = String(ev.reliability || '').split(' ')[0] || 'checked';
      const cls = rel === 'high' ? 'good' : rel === 'low' ? 'warn' : '';
      chips.push(`<span class="chip ${cls}" title="Sun/road bearing fit: ${ev.reliability || ''} (error ${ev.error}°, ${ev.event || ''} ${ev.date || ''})">☀ sun ${rel}</span>`);
    }
  }
  card.querySelector('.cand-meta-row').innerHTML = chips.join('');
}

/* ---------- Send a result to the Refine view ---------- */
function refineCandidate(i, c) {
  const sp = spreadKm(c);
  const radius = sp != null ? clamp(sp, 0.5, 20) : 2;
  fillRefine(c.lat, c.lon, radius);
}
function fillRefine(lat, lon, radiusKm) {
  $('zoomLat').value = Number(lat).toFixed(6);
  $('zoomLon').value = Number(lon).toFixed(6);
  if (radiusKm != null) $('zoomRadius').value = Number(radiusKm).toFixed(radiusKm < 1 ? 2 : 1);
  $('runZoomBtn').disabled = pipelineBusy; // don't allow a second launch mid-run
  showView('refine');
  const m = ensureMap('refine', 'refineMap');
  layers.refine.point.clearLayers();
  refineLatLngs = [[lat, lon]];
  L.marker([lat, lon], { icon: candIcon('★', true) }).addTo(layers.refine.point);
  const r = parseFloat($('zoomRadius').value);
  if (isFinite(r)) L.circle([lat, lon], { radius: r * 1000, color: cssVar('--accent'), weight: 1.5, fillOpacity: 0.06 }).addTo(layers.refine.point);
  setTimeout(() => { m.invalidateSize(); m.setView([lat, lon], r > 6 ? 10 : r > 2 ? 12 : 13, { animate: false }); }, 60);
}

/* ---------- Matches ---------- */
function renderMatches(container, sourceKey, matches, isZoom) {
  const meta = SOURCES[sourceKey];
  let group = container.querySelector(`details.source[data-source="${sourceKey}"]`);
  if (!group) {
    group = document.createElement('details');
    group.className = 'source';
    group.dataset.source = sourceKey;
    group.style.setProperty('--src-color', meta.color());
    group.innerHTML = `
      <summary>
        <span class="src-name">${meta.label}</span>
        <span class="source-count">0</span>
        <svg class="caret" viewBox="0 0 24 24" width="16" height="16"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </summary>
      <div class="matches"></div>`;
    container.appendChild(group);
  }
  const list = group.querySelector('.matches');
  for (const m of matches) {
    const url = m[meta.urlKey] || m.mapillary_url || m.street_view_url || m.panoramax_url || '';
    const simPct = clamp(Math.round(m.similarity * 100), 2, 100);
    const hasInliers = 'inliers' in m;
    const weak = hasInliers && m.total_matches > 0 && m.inliers / m.total_matches < 0.4;
    const row = document.createElement('div');
    row.className = 'match-row';
    row.innerHTML = `
      <span class="match-sim"><span class="sim-meter"><i style="width:${simPct}%"></i></span>${m.similarity.toFixed(3)}</span>
      <span class="match-inliers ${weak ? 'weak' : ''}">${hasInliers ? `${m.inliers}/${m.total_matches}` : ''}</span>
      <span class="match-coords">${m.lat.toFixed(5)}, ${m.lon.toFixed(5)}</span>
      <span class="match-actions">
        ${url ? '<button class="link-btn view">View</button>' : ''}
        <button class="link-btn primary refine">Refine</button>
      </span>`;
    if (url) row.querySelector('.view').addEventListener('click', () => window.api.openExternal(url));
    row.querySelector('.refine').addEventListener('click', () => fillRefine(m.lat, m.lon, 1));
    list.appendChild(row);
    addMatchMarker(m, sourceKey, isZoom);
  }
  const count = list.children.length;
  group.querySelector('.source-count').textContent = count;
  if (count === 0) { group.classList.add('empty-src'); group.open = false; }
  else { group.classList.remove('empty-src'); group.open = true; }
}

/* ============================================================ Run full pipeline */
$('runOneBtn').addEventListener('click', async () => {
  if (!selectedImagePath || pipelineBusy) return;
  activeRunKind = 'full';
  clusterResults.innerHTML = '';
  pipelineLog.textContent = '';
  resultLatLngs = []; resultsFitted = false;
  $('resultsEmpty').classList.add('hidden');
  $('resultsBody').classList.remove('hidden');
  $('noiseBadge').classList.add('hidden');
  ensureMap('results', 'resultsMap');
  layers.results.candidates.clearLayers(); layers.results.matches.clearLayers();
  resetSteps(); updateOverallProgress();
  setBusy(true, 'analyze');
  setTabDone('analyze', false);

  const numSamples = parseInt($('numSamples').value, 10);
  const numRuns = parseInt($('numRuns').value, 10);
  // Resolves at spawn time; the run buttons stay locked until the 'exit' event.
  await window.api.runOne({ imagePath: selectedImagePath, numSamples, numRuns, retrievalTopClusters: RETRIEVAL_TOP });
});

window.api.onPipelineEvent((payload) => {
  appendPipelineLog(JSON.stringify(payload));
  switch (payload.event) {
    case 'stage_start':
      markStep(payload.stage, 'active');
      break;
    case 'stage_skipped':
      markStep(payload.stage, 'skipped', payload.reason || 'skipped');
      break;
    case 'stage_failed':
    case 'error':
      if (payload.stage) markStep(payload.stage, 'failed');
      if (payload.message) appendPipelineLog(`ERROR: ${payload.message}`);
      break;
    case 'progress':
      updateStepProgress(payload.stage, payload.phase, payload.completed, payload.total);
      break;
    case 'stage_done': {
      markStep(payload.stage, 'done');
      if (payload.stage === 'plonk_sampling' && payload.clusters) {
        payload.clusters.forEach((c, i) => { renderCandidateCard(i, c); addCandidateMarker(i, c); });
        // If fewer candidates than we seeded steps for, retire the surplus.
        for (let i = payload.clusters.length; i < RETRIEVAL_TOP; i++)
          for (const s of ['mapillary', 'google_sv', 'panoramax']) markStep(`c${i}_${s}`, 'skipped', 'no candidate');
        if (payload.clusters.length) {
          const tab = document.querySelector('.tab[data-view="results"]');
          tab.disabled = false;
          $('tabResultsCount').textContent = payload.clusters.length;
        }
        if (typeof payload.noise_frac === 'number') {
          const nb = $('noiseBadge');
          nb.textContent = `noise ${(payload.noise_frac * 100).toFixed(0)}%`;
          nb.className = 'pill map-badge ' + (payload.noise_frac > 0.25 ? 'pill-run' : 'pill-muted');
          nb.classList.remove('hidden');
        }
      }
      if (payload.stage === 'sun_refine' && payload.clusters) {
        payload.clusters.forEach((c, i) => {
          const card = clusterResults.querySelector(`[data-cluster="${i}"]`);
          if (card) updateCandidateMeta(card, c);
        });
      }
      const bare = payload.stage
        .replace(/^c(\d+)_/, (_, i) => { currentClusterIndex = parseInt(i, 10); return ''; })
        .replace(/^zoom_/, '');
      if (STAGE_TO_MATCHKEY[bare] && payload.matches) {
        const isZoom = payload.stage.startsWith('zoom_');
        const container = isZoom
          ? $('zoomResults')
          : (clusterResults.querySelector(`[data-cluster="${currentClusterIndex}"] .sources`) || clusterResults);
        renderMatches(container, STAGE_TO_MATCHKEY[bare], payload.matches, isZoom);
      }
      break;
    }
    case 'done':
      if (activeRunKind === 'refine') {
        // Fit the refine map to the point + all zoom matches.
        if (maps.refine && refineLatLngs.length) fitBounds(maps.refine, refineLatLngs);
      } else {
        updateOverallProgress();
        setTabDone('analyze', true);
        if (clusterResults.children.length) {
          resultsFitted = false;   // refit now that matches exist; refreshActiveMap
          showView('results');     // fits AFTER invalidateSize so the container has size
        }
      }
      break;
    case 'exit':
      // Definitive terminal signal (normal exit or crash). Release the lock.
      if (activeRunKind === 'refine' && maps.refine && refineLatLngs.length) fitBounds(maps.refine, refineLatLngs);
      setBusy(false);
      activeRunKind = null;
      break;
    default: break;
  }
});

/* ============================================================ Refine (zoom-in) */
$('runZoomBtn').addEventListener('click', async () => {
  if (!selectedImagePath || pipelineBusy) return;
  const lat = parseFloat($('zoomLat').value);
  const lon = parseFloat($('zoomLon').value);
  const radiusKm = parseFloat($('zoomRadius').value);
  if (!(isFinite(lat) && isFinite(lon))) return;

  activeRunKind = 'refine';
  $('zoomResults').innerHTML = '';
  $('zoomSteps').innerHTML = '';
  $('zoomLog').textContent = '';
  const zl = $('zoomSteps'); zl.__els = {};
  ensureMap('refine', 'refineMap');
  layers.refine.matches.clearLayers();
  refineLatLngs = [[lat, lon]];
  setBusy(true, 'refine');
  if (maps.refine) maps.refine.setView([lat, lon], 13, { animate: true });

  // Resolves at spawn time; the run buttons stay locked until the 'exit' event.
  await window.api.runPoint({ imagePath: selectedImagePath, lat, lon, radiusKm, maxImages: 300 });
});

/* ============================================================ Delegated: copy coordinates */
function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText)
    return navigator.clipboard.writeText(text).then(() => true, () => fallbackCopy(text));
  return Promise.resolve(fallbackCopy(text));
}
function fallbackCopy(text) {
  try {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    const ok = document.execCommand('copy'); document.body.removeChild(ta); return ok;
  } catch { return false; }
}
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.copy-btn');
  if (!btn) return;
  copyText(btn.dataset.copy).then((ok) => {
    if (!ok) return;
    const svg = btn.innerHTML; btn.textContent = '✓';
    setTimeout(() => { btn.innerHTML = svg; }, 1200);
  });
});

/* ============================================================ Startup */

// Setup times a real calibration batch (see calibrate.py) and stores the
// result as hardware.recommendedSamples -- that's the real signal, since
// VRAM turned out to be a poor proxy (a 4096-sample batch only uses a few GB
// regardless of card; PLONK's sampler operates on 3D coordinates, not
// images, so compute throughput varies far more across cards than memory
// use does). The VRAM-tier fallback below only applies to settings written
// before calibration existed, until the user re-runs setup.
function recommendedSamples(hardware) {
  if (!hardware || !hardware.hasGpu) return 512;
  if (hardware.recommendedSamples) return hardware.recommendedSamples;
  const vramGb = (hardware.vramMb || 0) / 1024;
  if (vramGb >= 16) return 4096;
  if (vramGb >= 12) return 2048;
  if (vramGb >= 8) return 1024;
  return 768;
}
async function applyRecommendedSamples() {
  const settings = await window.api.getSettings();
  if (settings.hardware) $('numSamples').value = recommendedSamples(settings.hardware);
}

(async () => {
  const status = await window.api.getEnvStatus();
  resetSteps();
  if (status.pythonInstalled && status.depsInstalled) { showScreen('mainScreen'); await applyRecommendedSamples(); }
  else showScreen('setupScreen');
})();
