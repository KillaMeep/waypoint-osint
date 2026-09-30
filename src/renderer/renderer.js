'use strict';

/* ============================================================
   Waypoint renderer: one persistent map, a sidebar with two modes
   (Locate / Refine). The pipeline event contract is fixed by
   waypoint-core's pipeline (after the original run_pipeline.py); the shell
   bridge lives in api.js.
   ============================================================ */

const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name, cls = 'i') => `<svg class="${cls}"><use href="#i-${name}"/></svg>`;

// How many top clusters the backend refines with retrieval imagery. Kept in
// sync with --retrieval_top_clusters so the step list shows the whole plan
// up front instead of growing mid-run.
const RETRIEVAL_TOP = 1;

const STAGE_LABELS = {
  model_load: 'Load model',
  plonk_sampling: 'Coarse locate',
  sun_refine: 'Sun and season check',
  mapillary: 'Mapillary search',
  google_sv: 'Street View search',
  panoramax: 'Panoramax search',
};
const SOURCES = {
  mapillary_matches: { label: 'Mapillary',   color: '--src-mly', urlKey: 'mapillary_url' },
  google_sv_matches: { label: 'Street View', color: '--src-gsv', urlKey: 'street_view_url' },
  panoramax_matches: { label: 'Panoramax',   color: '--src-pmx', urlKey: 'panoramax_url' },
};
const SOURCE_KEYS = Object.keys(SOURCES);
const STAGE_TO_MATCHKEY = { mapillary: 'mapillary_matches', google_sv: 'google_sv_matches', panoramax: 'panoramax_matches' };

const STEP_ICONS = {
  pending: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
  active:  '<svg viewBox="0 0 20 20"><circle class="spin-ring" cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="30 14" stroke-linecap="round"/></svg>',
  done:    '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="currentColor" opacity=".16"/><path d="M6.5 10.3l2.3 2.3 4.7-4.9" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  failed:  '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="currentColor" opacity=".16"/><path d="M7.5 7.5l5 5M12.5 7.5l-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  skipped: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="currentColor" opacity=".16"/><path d="M7 10h6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
};

const state = {
  mode: 'locate',
  image: null,            // { path, name, bytes, dataUrl }
  busy: false,
  runKind: null,          // 'full' | 'refine'
  runFailed: false,
  cancelled: false,
  clusterIndex: 0,
  hiddenSources: new Set(),
  fit: { locate: [], refine: [] },
};

/* ============================================================ Small UI helpers */
let toastTimer = null;
function toast(msg, isError) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), isError ? 5000 : 2200);
}
function fmtBytes(n) {
  if (!n) return '';
  return n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`;
}
function fmtCoords(lat, lon, dp = 5) { return `${Number(lat).toFixed(dp)}, ${Number(lon).toFixed(dp)}`; }
function formatEta(sec) {
  if (!isFinite(sec) || sec < 0) return '';
  if (sec < 1) return '<1s';
  if (sec < 60) return `${Math.round(sec)}s`;
  return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
}

/* ============================================================ Theme */
(function initTheme() {
  try {
    const saved = localStorage.getItem('geo-theme');
    if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved);
  } catch { /* storage unavailable: follow the system theme */ }
})();
$('themeBtn').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme')
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('geo-theme', next); } catch { /* not persisted */ }
  restyleMarkers();
});

/* ============================================================ Screens + modes */
function showScreen(id) {
  for (const s of ['setupScreen', 'mainScreen']) $(s).classList.toggle('hidden', s !== id);
  if (id === 'mainScreen') setTimeout(() => map.invalidateSize(), 30);
}

function setMode(mode) {
  state.mode = mode;
  for (const b of document.querySelectorAll('.mode')) b.classList.toggle('is-active', b.dataset.mode === mode);
  $('panel-locate').classList.toggle('is-active', mode === 'locate');
  $('panel-refine').classList.toggle('is-active', mode === 'refine');
  const showResults = mode === 'locate';
  toggleLayer(layers.results.cands, showResults);
  toggleLayer(layers.refine.point, !showResults);
  applySourceVisibility();
  $('map').classList.toggle('picking', mode === 'refine');
  updateMapHint();
  fitMode();
}
document.querySelectorAll('.mode').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

function updateMapHint() {
  $('mapHint').classList.toggle('hidden', !(state.mode === 'refine' && !state.busy));
}

/* ============================================================ Map */
const map = L.map('map', { zoomControl: false, worldCopyJump: true, attributionControl: true }).setView([25, 5], 2);
L.control.zoom({ position: 'topright' }).addTo(map);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);
map.attributionControl.setPrefix(false);

const sourceGroups = () => Object.fromEntries(SOURCE_KEYS.map((k) => [k, L.layerGroup()]));
const layers = {
  results: { cands: L.layerGroup().addTo(map), matches: sourceGroups() },
  refine:  { point: L.layerGroup(), matches: sourceGroups() },
};
const candMarkers = [];

function toggleLayer(layer, on) {
  if (on && !map.hasLayer(layer)) layer.addTo(map);
  if (!on && map.hasLayer(layer)) map.removeLayer(layer);
}
function applySourceVisibility() {
  for (const k of SOURCE_KEYS) {
    const visible = !state.hiddenSources.has(k);
    toggleLayer(layers.results.matches[k], visible && state.mode === 'locate');
    toggleLayer(layers.refine.matches[k], visible && state.mode === 'refine');
  }
}
document.querySelectorAll('.legend-item').forEach((btn) => {
  btn.addEventListener('click', () => {
    const k = btn.dataset.src;
    if (state.hiddenSources.has(k)) state.hiddenSources.delete(k); else state.hiddenSources.add(k);
    btn.classList.toggle('off', state.hiddenSources.has(k));
    applySourceVisibility();
  });
});

function fitMode() {
  const pts = state.fit[state.mode];
  if (!pts.length) return;
  if (pts.length === 1) { map.setView(pts[0], Math.max(map.getZoom(), 12)); return; }
  map.fitBounds(L.latLngBounds(pts).pad(0.25), { maxZoom: 14 });
}
$('fitBtn').addEventListener('click', fitMode);

function candIcon(rank, lead, focus) {
  const size = lead ? 32 : 28;
  return L.divIcon({
    className: '', iconSize: [size, size], iconAnchor: [size / 2, size / 2],
    html: `<div class="pin pin-cand${lead ? ' lead' : ''}${focus ? ' focus' : ''}">${rank}</div>`,
  });
}
function matchIcon(sourceKey, hot) {
  return L.divIcon({
    className: '', iconSize: [12, 12], iconAnchor: [6, 6],
    html: `<div class="pin pin-match${hot ? ' hot' : ''}" style="background:${cssVar(SOURCES[sourceKey].color)}"></div>`,
  });
}
// Source colors are resolved from CSS at creation time; refresh on theme flip.
const matchMarkers = [];
function restyleMarkers() {
  for (const { marker, sourceKey } of matchMarkers) marker.setIcon(matchIcon(sourceKey));
}

applySourceVisibility();

map.on('click', (e) => {
  if (state.mode !== 'refine' || state.busy) return;
  const { lat, lng } = e.latlng.wrap();
  setRefinePoint(lat, lng, null, false);
});

/* ============================================================ Image selection */
function setImage(img) {
  state.image = img;
  $('photoImg').src = img.dataUrl;
  $('lightboxImg').src = img.dataUrl;
  $('photo').classList.remove('is-empty');
  $('photo').setAttribute('aria-label', 'View photo full size');
  $('photoName').textContent = [img.name, fmtBytes(img.bytes)].filter(Boolean).join('  ·  ');
  $('photoName').title = img.path || '';
  $('photoMeta').classList.remove('hidden');
  refreshRunButtons();
}
async function pickImage() {
  try {
    const img = await window.api.selectImage();
    if (img) setImage(img);
  } catch (e) { toast(String(e), true); }
}
$('photo').addEventListener('click', (e) => {
  if (e.target.closest('.photo-zoom')) return;
  if (state.image) openLightbox(); else pickImage();
});
$('photo').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); state.image ? openLightbox() : pickImage(); }
});
$('photoZoom').addEventListener('click', (e) => { e.stopPropagation(); openLightbox(); });
$('photoReplace').addEventListener('click', pickImage);

function openLightbox() { if (state.image) $('lightbox').classList.remove('hidden'); }
$('lightbox').addEventListener('click', () => $('lightbox').classList.add('hidden'));
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  $('lightbox').classList.add('hidden');
  $('logDrawer').classList.add('hidden'); $('logBtn').classList.remove('is-on');
});

if (window.api.onFileDrag) {
  window.api.onFileDrag(async (kind, paths) => {
    $('dropOverlay').classList.toggle('hidden', kind !== 'enter');
    if (kind !== 'drop' || !paths.length) return;
    try { setImage(await window.api.loadImage(paths[0])); }
    catch (e) { toast(String(e), true); }
  });
}

/* ============================================================ Busy / run state */
function refreshRunButtons() {
  $('runOneBtn').disabled = state.busy || !state.image;
  const lat = parseFloat($('zoomLat').value), lon = parseFloat($('zoomLon').value);
  $('runZoomBtn').disabled = state.busy || !state.image || !(isFinite(lat) && isFinite(lon));
  $('refineHint').textContent = state.image
    ? 'Refine compares against the photo loaded in Locate.'
    : 'Load a photo in Locate first. Refine compares against it.';
}
function setBusy(busy, kind) {
  state.busy = busy;
  $('runOneBtn').classList.toggle('busy', busy && kind === 'full');
  $('runZoomBtn').classList.toggle('busy', busy && kind === 'refine');
  $('cancelBtn').classList.toggle('hidden', !busy);
  refreshRunButtons();
  updateMapHint();
}
function setRunState(text, cls) {
  const el = $('runState');
  el.textContent = text;
  el.className = 'run-state' + (cls ? ` ${cls}` : '');
}
$('cancelBtn').addEventListener('click', () => {
  state.cancelled = true;
  setRunState('Stopping…', 'running');
  window.api.cancelRun();
});

/* ============================================================ Steps */
const progressStart = {};

function stageInfo(stage) {
  let group = '';
  let bare = stage;
  const cm = bare.match(/^c(\d+)_/);
  if (cm) { group = RETRIEVAL_TOP > 1 ? `#${parseInt(cm[1], 10) + 1} ` : ''; bare = bare.slice(cm[0].length); }
  else if (bare.startsWith('zoom_')) bare = bare.slice(5);
  return { group, name: STAGE_LABELS[bare] || bare };
}
// Point mode emits a bare 'model_load' too, so route by the active run.
const listFor = (stage) => (stage.startsWith('zoom_') || state.runKind === 'refine' ? $('zoomSteps') : $('stepList'));

function ensureStep(stage, initial = 'pending') {
  const list = listFor(stage);
  let li = list.querySelector(`[data-stage="${CSS.escape(stage)}"]`);
  if (li) return li;
  const { group, name } = stageInfo(stage);
  li = document.createElement('li');
  li.className = `step ${initial}`;
  li.dataset.stage = stage;
  li.innerHTML = `<span class="step-ico">${STEP_ICONS[initial]}</span>
    <span class="step-name">${group ? `<span class="step-group">${esc(group)}</span>` : ''}${esc(name)}</span>
    <span class="step-meta"></span>`;
  list.appendChild(li);
  return li;
}
function markStep(stage, st, meta) {
  const li = ensureStep(stage);
  li.className = `step ${st}`;
  li.querySelector('.step-ico').innerHTML = STEP_ICONS[st] || '';
  if (meta !== undefined) li.querySelector('.step-meta').textContent = meta;
  else if (st !== 'active') li.querySelector('.step-meta').textContent = '';
  updateOverall();
}
function updateStepProgress(stage, phase, completed, total) {
  const li = ensureStep(stage);
  if (!li.classList.contains('active')) markStep(stage, 'active');
  const key = `${stage}:${phase}`;
  if (!progressStart[key] || progressStart[key].from > completed) progressStart[key] = { ts: Date.now(), from: completed };
  const { ts, from } = progressStart[key];
  const rate = (completed - from) / ((Date.now() - ts) / 1000);
  const eta = rate > 0 ? formatEta((total - completed) / rate) : '';
  li.querySelector('.step-meta').textContent = `${completed}/${total}${eta ? ` · ${eta}` : ''}`;
  li.querySelector('.step-meta').title = phase;
}
function resetSteps(kind) {
  for (const k of Object.keys(progressStart)) delete progressStart[k];
  if (kind === 'refine') {
    $('zoomSteps').innerHTML = '';
    for (const s of ['model_load', 'zoom_mapillary', 'zoom_google_sv', 'zoom_panoramax']) ensureStep(s);
    return;
  }
  $('stepList').innerHTML = '';
  for (const s of ['model_load', 'plonk_sampling', 'sun_refine']) ensureStep(s);
  for (let i = 0; i < RETRIEVAL_TOP; i++)
    for (const s of ['mapillary', 'google_sv', 'panoramax']) ensureStep(`c${i}_${s}`);
}
function updateOverall() {
  if (!state.busy) return;
  const list = state.runKind === 'refine' ? $('zoomSteps') : $('stepList');
  const steps = list.querySelectorAll('.step');
  const done = list.querySelectorAll('.step.done, .step.failed, .step.skipped').length;
  const total = Math.max(steps.length, 1);
  $('overallBar').classList.remove('fade');
  $('overallBar').style.width = `${Math.round((done / total) * 100)}%`;
  if (!state.cancelled) setRunState(`${state.runKind === 'refine' ? 'Refining' : 'Locating'} · ${done} of ${total}`, 'running');
}
function finishRun(code) {
  const list = state.runKind === 'refine' ? $('zoomSteps') : $('stepList');
  const failed = state.runFailed || (code !== 0 && code !== null && !state.cancelled) || (code === null && !state.cancelled);
  for (const li of list.querySelectorAll('.step.active, .step.pending')) {
    if (state.cancelled) markStep(li.dataset.stage, 'skipped', 'stopped');
    else if (failed && li.classList.contains('active')) markStep(li.dataset.stage, 'failed');
  }
  $('overallBar').style.width = '100%';
  setTimeout(() => { $('overallBar').classList.add('fade'); }, 500);
  setTimeout(() => { if (!state.busy) $('overallBar').style.width = '0'; }, 1000);
  if (state.cancelled) setRunState('Stopped', 'failed');
  else if (failed) setRunState('Run failed. Open the log for details', 'failed');
  else setRunState(state.runKind === 'refine' ? 'Refine complete' : 'Locate complete', 'done');
}

/* ============================================================ Log drawer */
const pipelineLog = $('pipelineLog');
function appendLog(text) {
  const atBottom = pipelineLog.scrollTop + pipelineLog.clientHeight >= pipelineLog.scrollHeight - 8;
  pipelineLog.textContent += text.replace(/\s+$/, '') + '\n';
  if (atBottom) pipelineLog.scrollTop = pipelineLog.scrollHeight;
}
$('logBtn').addEventListener('click', () => {
  const open = $('logDrawer').classList.toggle('hidden') === false;
  $('logBtn').classList.toggle('is-on', open);
  if (open) pipelineLog.scrollTop = pipelineLog.scrollHeight;
});
$('logClose').addEventListener('click', () => { $('logDrawer').classList.add('hidden'); $('logBtn').classList.remove('is-on'); });

/* ============================================================ Candidates */
const clusterResults = $('clusterResults');
const clusters = [];

function spreadKm(c) {
  if (c.lat_std == null || c.lon_std == null) return null;
  const latK = c.lat_std * 111;
  const lonK = c.lon_std * 111 * Math.cos((c.lat || 0) * Math.PI / 180);
  return Math.sqrt(latK * latK + lonK * lonK);
}
const copyBtn = (text) => `<button class="copy-btn" data-copy="${esc(text)}" title="Copy coordinates">${icon('copy')}</button>`;

function renderCandidateCard(i, c) {
  clusters[i] = c;
  let card = clusterResults.querySelector(`[data-cluster="${i}"]`);
  const coords = fmtCoords(c.lat, c.lon);
  const conf = Math.round(c.weight * 100);
  if (!card) {
    card = document.createElement('article');
    card.className = 'cand' + (i === 0 ? ' lead' : '');
    card.dataset.cluster = i;
    card.innerHTML = `
      <div class="cand-head">
        <div class="cand-rank">${i + 1}</div>
        <div class="cand-main">
          <div class="cand-coords"><span>${esc(coords)}</span>${copyBtn(coords)}</div>
          <div class="cand-addr"></div>
          <div class="cand-tags"></div>
        </div>
        <div class="cand-conf" title="Share of the coarse samples in this cluster">
          <div class="conf-val">${conf}<small>%</small></div>
          <span class="conf-bar"><i style="width:${clamp(conf, 2, 100)}%"></i></span>
        </div>
      </div>
      <div class="cand-actions">
        <button class="btn btn-quiet btn-xs refine-area">${icon('target')}Refine this area</button>
      </div>
      <div class="sources"></div>`;
    card.querySelector('.cand-head').addEventListener('click', (e) => { if (!e.target.closest('.copy-btn')) focusCandidate(i, true); });
    card.querySelector('.refine-area').addEventListener('click', () => refineCandidate(clusters[i]));
    clusterResults.appendChild(card);
  }
  updateCandidateMeta(card, c);
  return card;
}
function updateCandidateMeta(card, c) {
  const addr = card.querySelector('.cand-addr');
  if (c.address) { addr.textContent = c.address; addr.title = c.address; }
  const tags = [];
  const sp = spreadKm(c);
  if (sp != null) tags.push(`<span class="tag" title="Spread of the coarse samples in this cluster">± ${sp < 1 ? `${(sp * 1000).toFixed(0)} m` : `${sp.toFixed(1)} km`}</span>`);
  if (c.count != null) tags.push(`<span class="tag" title="Coarse samples in this cluster">${Number(c.count)} samples</span>`);
  const ev = c.sun_evidence;
  if (ev) {
    if (ev.error == null) tags.push(`<span class="tag" title="${esc(ev.note || 'No sun fit')}">${icon('sun')}n/a</span>`);
    else {
      const rel = String(ev.reliability || '').split(' ')[0] || 'checked';
      const cls = rel === 'high' ? 'good' : rel === 'low' ? 'warn' : '';
      const tip = `Sun and road bearing fit: ${ev.reliability || ''} (error ${ev.error}°, ${ev.event || ''} ${ev.date || ''})`;
      tags.push(`<span class="tag ${cls}" title="${esc(tip)}">${icon('sun')}sun ${esc(rel)}</span>`);
    }
  }
  card.querySelector('.cand-tags').innerHTML = tags.join('');
  card.querySelectorAll('.tag .i').forEach((s) => { s.style.width = '12px'; s.style.height = '12px'; });
}
function addCandidateMarker(i, c) {
  const m = L.marker([c.lat, c.lon], { icon: candIcon(i + 1, i === 0), zIndexOffset: 1000 - i, riseOnHover: true })
    .bindTooltip(`Candidate ${i + 1} · ${Math.round(c.weight * 100)}%`, { direction: 'top', offset: [0, -16] })
    .on('click', () => focusCandidate(i, false))
    .addTo(layers.results.cands);
  candMarkers[i] = m;
  state.fit.locate.push([c.lat, c.lon]);
}
function focusCandidate(i, fromList) {
  clusterResults.querySelectorAll('.cand').forEach((el) => el.classList.toggle('is-focus', Number(el.dataset.cluster) === i));
  candMarkers.forEach((m, j) => m && m.setIcon(candIcon(j + 1, j === 0, j === i)));
  const c = clusters[i];
  if (!c) return;
  if (fromList) map.flyTo([c.lat, c.lon], Math.max(map.getZoom(), 11), { duration: .6 });
  else clusterResults.querySelector(`[data-cluster="${i}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ============================================================ Matches */
function renderMatches(container, sourceKey, matches, isZoom) {
  const meta = SOURCES[sourceKey];
  let group = container.querySelector(`details.source[data-source="${sourceKey}"]`);
  if (!group) {
    group = document.createElement('details');
    group.className = 'source';
    group.dataset.source = sourceKey;
    group.style.setProperty('--c', `var(${meta.color})`);
    group.innerHTML = `<summary><span>${meta.label}</span><span class="src-count">0</span>${icon('chevron', 'caret')}</summary><div class="matches"></div>`;
    // Keep source order stable no matter which stage finishes first.
    const after = SOURCE_KEYS.slice(SOURCE_KEYS.indexOf(sourceKey) + 1)
      .map((k) => container.querySelector(`details.source[data-source="${k}"]`)).find(Boolean);
    container.insertBefore(group, after || null);
    group.querySelector('summary').addEventListener('click', (e) => { if (group.classList.contains('is-empty')) e.preventDefault(); });
  }
  const list = group.querySelector('.matches');
  const layer = (isZoom ? layers.refine : layers.results).matches[sourceKey];
  for (const m of matches) {
    const url = m[meta.urlKey] || m.mapillary_url || m.street_view_url || m.panoramax_url || '';
    const hasInliers = 'inliers' in m;
    const weak = hasInliers && m.total_matches > 0 && m.inliers / m.total_matches < 0.4;
    const row = document.createElement('div');
    row.className = 'match';
    row.innerHTML = `
      <div class="match-score" title="Retrieval similarity">
        <span class="meter"><i style="width:${clamp(Math.round(m.similarity * 100), 2, 100)}%"></i></span>
        <span>${m.similarity.toFixed(3)}</span>
        <span class="match-coord" title="${esc(fmtCoords(m.lat, m.lon))}">${esc(fmtCoords(m.lat, m.lon))}</span>
      </div>
      <span class="match-inl ${weak ? 'weak' : ''}" title="Geometric inliers / total keypoint matches">${hasInliers ? `${m.inliers}/${m.total_matches}` : ''}</span>
      <span class="match-actions">
        ${url ? `<button class="link-btn view" title="Open this image in ${meta.label}">${icon('external')}</button>` : ''}
        <button class="link-btn accent refine" title="Refine around this match">${icon('target')}</button>
      </span>`;
    const marker = L.marker([m.lat, m.lon], { icon: matchIcon(sourceKey) })
      .bindTooltip(`${meta.label} · ${m.similarity.toFixed(3)}${hasInliers ? ` · ${m.inliers} inliers` : ''}`, { direction: 'top', offset: [0, -8] })
      .on('click', () => { group.open = true; row.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); flashRow(row); })
      .on('mouseover', () => row.classList.add('is-hot'))
      .on('mouseout', () => row.classList.remove('is-hot'))
      .addTo(layer);
    matchMarkers.push({ marker, sourceKey });
    row.addEventListener('mouseenter', () => marker.setIcon(matchIcon(sourceKey, true)));
    row.addEventListener('mouseleave', () => marker.setIcon(matchIcon(sourceKey)));
    row.addEventListener('click', (e) => { if (!e.target.closest('button')) map.panTo([m.lat, m.lon]); });
    if (url) row.querySelector('.view').addEventListener('click', () => window.api.openExternal(url));
    row.querySelector('.refine').addEventListener('click', () => setRefinePoint(m.lat, m.lon, 1, true));
    list.appendChild(row);
    state.fit[isZoom ? 'refine' : 'locate'].push([m.lat, m.lon]);
  }
  const count = list.children.length;
  group.querySelector('.src-count').textContent = count;
  group.classList.toggle('is-empty', count === 0);
  group.open = count > 0;
}
function flashRow(row) {
  row.classList.add('is-hot');
  setTimeout(() => row.classList.remove('is-hot'), 1200);
}

/* ============================================================ Refine point */
function refineCandidate(c) {
  const sp = spreadKm(c);
  setRefinePoint(c.lat, c.lon, sp != null ? clamp(sp, 0.5, 20) : 2, true);
}
function setRefinePoint(lat, lon, radiusKm, switchMode) {
  $('zoomLat').value = Number(lat).toFixed(6);
  $('zoomLon').value = Number(lon).toFixed(6);
  if (radiusKm != null) $('zoomRadius').value = Number(radiusKm).toFixed(radiusKm < 1 ? 2 : 1);
  for (const id of ['zoomLat', 'zoomLon']) { $(id).classList.remove('flash'); void $(id).offsetWidth; $(id).classList.add('flash'); }
  if (switchMode && state.mode !== 'refine') setMode('refine');
  drawRefinePoint(true);
  refreshRunButtons();
}
function drawRefinePoint(fly) {
  const lat = parseFloat($('zoomLat').value), lon = parseFloat($('zoomLon').value);
  const r = parseFloat($('zoomRadius').value);
  layers.refine.point.clearLayers();
  if (!(isFinite(lat) && isFinite(lon))) return;
  L.marker([lat, lon], { icon: L.divIcon({ className: '', iconSize: [20, 20], iconAnchor: [10, 10], html: '<div class="pin pin-point"></div>' }), zIndexOffset: 2000 })
    .addTo(layers.refine.point);
  if (isFinite(r) && r > 0) {
    const circle = L.circle([lat, lon], { radius: r * 1000, color: cssVar('--accent'), weight: 1.5, dashArray: '5 5', fillOpacity: 0.06, interactive: false })
      .addTo(layers.refine.point);
    if (fly) map.flyToBounds(circle.getBounds().pad(0.15), { duration: .6, maxZoom: 16 });
  }
  if (!state.busy || state.runKind !== 'refine') state.fit.refine = [[lat, lon]];
}
let redrawTimer = null;
for (const id of ['zoomLat', 'zoomLon', 'zoomRadius']) {
  $(id).addEventListener('input', () => {
    refreshRunButtons();
    clearTimeout(redrawTimer);
    redrawTimer = setTimeout(() => drawRefinePoint(true), 400);
  });
}

/* ============================================================ Runs */
function startRun(kind) {
  state.runKind = kind;
  state.runFailed = false;
  state.cancelled = false;
  pipelineLog.textContent = '';
  setBusy(true, kind);
  resetSteps(kind);
  updateOverall();
}
function runFailedToStart(e) {
  setBusy(false);
  setRunState('Could not start', 'failed');
  toast(String(e), true);
}

$('runOneBtn').addEventListener('click', async () => {
  if (!state.image || state.busy) return;
  clusterResults.innerHTML = '';
  clusters.length = 0;
  candMarkers.length = 0;
  layers.results.cands.clearLayers();
  for (const k of SOURCE_KEYS) layers.results.matches[k].clearLayers();
  state.fit.locate = [];
  $('stepsBlock').classList.remove('hidden');
  $('resultsBlock').classList.add('hidden');
  $('noiseBadge').classList.add('hidden');
  startRun('full');
  try {
    // Resolves at spawn time; the buttons stay locked until the 'exit' event.
    await window.api.runOne({
      numSamples: parseInt($('numSamples').value, 10),
      numRuns: parseInt($('numRuns').value, 10),
      retrievalTopClusters: RETRIEVAL_TOP,
    });
  } catch (e) { runFailedToStart(e); }
});

$('runZoomBtn').addEventListener('click', async () => {
  if (!state.image || state.busy) return;
  const lat = parseFloat($('zoomLat').value), lon = parseFloat($('zoomLon').value);
  const radiusKm = parseFloat($('zoomRadius').value);
  if (!(isFinite(lat) && isFinite(lon))) return;
  $('zoomResults').innerHTML = '';
  for (const k of SOURCE_KEYS) layers.refine.matches[k].clearLayers();
  state.fit.refine = [[lat, lon]];
  $('zoomStepsBlock').classList.remove('hidden');
  $('zoomResultsBlock').classList.add('hidden');
  startRun('refine');
  try {
    await window.api.runPoint({ lat, lon, radiusKm, maxImages: 300 });
  } catch (e) { runFailedToStart(e); }
});

window.api.onPipelineEvent((p) => {
  if (p.event === 'log') { appendLog(String(p.message)); return; }
  appendLog(JSON.stringify(p));
  switch (p.event) {
    case 'stage_start':
      markStep(p.stage, 'active');
      break;
    case 'stage_skipped':
      markStep(p.stage, 'skipped', p.reason || 'skipped');
      break;
    case 'stage_failed':
    case 'error':
      if (p.stage) markStep(p.stage, 'failed');
      else state.runFailed = true;
      if (p.message) appendLog(`ERROR: ${p.message}`);
      break;
    case 'progress':
      updateStepProgress(p.stage, p.phase, p.completed, p.total);
      break;
    case 'stage_done':
      onStageDone(p);
      break;
    case 'done':
      if (state.runKind === 'full') {
        if (clusters.length) { focusCandidate(0, false); if (state.mode === 'locate') fitMode(); }
      } else if (state.mode === 'refine') fitMode();
      break;
    case 'exit':
      finishRun(p.code);
      setBusy(false);
      state.runKind = null;
      break;
    default: break;
  }
});

function onStageDone(p) {
  markStep(p.stage, 'done');
  if (p.stage === 'plonk_sampling' && p.clusters) {
    p.clusters.forEach((c, i) => { renderCandidateCard(i, c); addCandidateMarker(i, c); });
    // Fewer candidates than seeded steps: retire the surplus.
    for (let i = p.clusters.length; i < RETRIEVAL_TOP; i++)
      for (const s of ['mapillary', 'google_sv', 'panoramax']) markStep(`c${i}_${s}`, 'skipped', 'no candidate');
    if (p.clusters.length) {
      $('resultsBlock').classList.remove('hidden');
      if (state.mode === 'locate') fitMode();
    }
    if (typeof p.noise_frac === 'number') {
      const nb = $('noiseBadge');
      nb.textContent = `${Math.round(p.noise_frac * 100)}% noise`;
      nb.className = 'tag' + (p.noise_frac > 0.25 ? ' warn' : '');
    }
  }
  if (p.stage === 'sun_refine' && p.clusters) {
    p.clusters.forEach((c, i) => {
      const card = clusterResults.querySelector(`[data-cluster="${i}"]`);
      if (card) { clusters[i] = { ...clusters[i], ...c }; updateCandidateMeta(card, c); }
    });
  }
  const bare = p.stage
    .replace(/^c(\d+)_/, (_, i) => { state.clusterIndex = parseInt(i, 10); return ''; })
    .replace(/^zoom_/, '');
  if (STAGE_TO_MATCHKEY[bare] && p.matches) {
    const isZoom = p.stage.startsWith('zoom_');
    const container = isZoom
      ? $('zoomResults')
      : (clusterResults.querySelector(`[data-cluster="${state.clusterIndex}"] .sources`) || clusterResults);
    if (isZoom) $('zoomResultsBlock').classList.remove('hidden');
    renderMatches(container, STAGE_TO_MATCHKEY[bare], p.matches, isZoom);
  }
}

/* ============================================================ Copy coordinates */
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; }
  catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      const ok = document.execCommand('copy'); ta.remove(); return ok;
    } catch { return false; }
  }
}
document.addEventListener('click', async (e) => {
  const btn = e.target.closest('.copy-btn');
  if (!btn) return;
  e.stopPropagation();
  if (!(await copyText(btn.dataset.copy))) { toast('Could not copy to the clipboard', true); return; }
  btn.classList.add('ok');
  btn.innerHTML = icon('check');
  setTimeout(() => { btn.classList.remove('ok'); btn.innerHTML = icon('copy'); }, 1200);
});

/* ============================================================ Settings */
const settingsDialog = $('settingsDialog');
$('settingsBtn').addEventListener('click', async () => {
  const s = await window.api.getSettings();
  $('mapillaryTokenInput').value = s.mapillaryToken || '';
  $('mapillaryTokenInput').type = 'password';
  $('tokenReveal').textContent = 'Show';
  $('settingsSaved').textContent = '';
  $('purgeStatus').textContent = '';
  resetPurgeButton();
  const env = await window.api.getEnvStatus();
  renderModels(s, env);
  $('engineInfo').textContent = 'ONNX Runtime with DirectML (CPU fallback).';
  $('versionInfo').textContent = `Waypoint ${await window.api.appVersion()}`;
  $('updateStatus').textContent = '';
  const hw = s.hardware;
  // Setup records what the engine runs on (accel); settings written by older
  // versions only know about NVIDIA cards (hasGpu / gpuName from nvidia-smi).
  const device = !hw ? null
    : hw.accel === 'CPU' ? 'CPU (no DirectML GPU)'
    : hw.accel === 'DirectML' ? (hw.gpuName ? `${hw.gpuName} via DirectML` : 'GPU via DirectML')
    : hw.hasGpu ? (hw.gpuName || 'NVIDIA GPU')
    : 'CPU (no NVIDIA GPU found)';
  $('hardwareInfo').textContent = !hw ? 'Unknown until setup runs.'
    : [device, hw.vramMb ? `${(hw.vramMb / 1024).toFixed(0)} GB VRAM` : null,
      hw.samplesPerSec ? `${Math.round(hw.samplesPerSec)} samples/s measured` : null].filter(Boolean).join('  ·  ');
  settingsDialog.showModal();
});
$('tokenReveal').addEventListener('click', () => {
  const inp = $('mapillaryTokenInput');
  const show = inp.type === 'password';
  inp.type = show ? 'text' : 'password';
  $('tokenReveal').textContent = show ? 'Hide' : 'Show';
});
$('settingsSaveBtn').addEventListener('click', async () => {
  await window.api.setSettings({ mapillaryToken: $('mapillaryTokenInput').value.trim() });
  const note = $('settingsSaved');
  note.textContent = 'Saved';
  setTimeout(() => { if (note.textContent === 'Saved') note.textContent = ''; }, 2000);
});
$('mapillaryLink').addEventListener('click', (e) => { e.preventDefault(); window.api.openExternal('https://mapillary.com/dashboard/developers'); });

/* ---------- Model (PLONK variant) */
const MODEL_INFO = {
  osv5m: 'Street-level photos (OpenStreetView-5M). The default, and the best fit for street scenes.',
  yfcc: 'General photos from Flickr (YFCC100M): landmarks, landscapes, indoor and tourist shots.',
  inat: 'Nature photos from iNaturalist: plants, animals and wild outdoor scenes.',
};

// Rows from env.models ({ key, label, native }).
function renderModels(settings, env) {
  const current = settings.plonkModel || 'osv5m';
  $('modelList').innerHTML = (env.models || []).map((m) => {
    const usable = m.native;
    const status = m.native ? 'Installed' : 'Not installed';
    return `<label class="model-row${usable ? '' : ' is-off'}">
      <input type="radio" name="plonkModel" value="${m.key}" ${m.key === current ? 'checked' : ''} ${usable ? '' : 'disabled'} />
      <span class="model-text"><b>${esc(m.label)}</b><span class="note">${esc(MODEL_INFO[m.key] || '')}</span></span>
      <span class="model-side"><span class="note">${esc(status)}</span></span>
    </label>`;
  }).join('');
}
$('modelList').addEventListener('change', async (e) => {
  if (e.target.name !== 'plonkModel') return;
  await window.api.setSettings({ plonkModel: e.target.value });
  toast(`Model: ${e.target.closest('.model-row').querySelector('b').textContent}`);
});

// Two-step confirm inside the dialog; no native confirm() popup.
let purgeArmed = null;
function resetPurgeButton() {
  clearTimeout(purgeArmed); purgeArmed = null;
  $('purgeEnvBtn').textContent = 'Delete downloads';
}
$('purgeEnvBtn').addEventListener('click', async () => {
  const btn = $('purgeEnvBtn');
  if (state.busy) { $('purgeStatus').textContent = 'Stop the running pipeline first.'; return; }
  if (!purgeArmed) {
    btn.textContent = 'Click again to delete';
    purgeArmed = setTimeout(resetPurgeButton, 4000);
    return;
  }
  resetPurgeButton();
  btn.disabled = true;
  $('purgeStatus').textContent = 'Deleting…';
  const result = await window.api.purgeEnv();
  $('purgeStatus').textContent = result.ok ? 'Done. Restart Waypoint to run setup again.' : `Failed: ${result.error}`;
  btn.disabled = false;
});

/* ============================================================ First-run setup */
const setupLog = $('setupLog');
function setupAppend(text) { setupLog.textContent += String(text).replace(/\s+$/, '') + '\n'; setupLog.scrollTop = setupLog.scrollHeight; }

// Steps shown beside the dial.
const SETUP_STEPS = [
  ['runtime', 'ONNX Runtime', 'Inference runtime and DirectML, from NuGet'],
  ['models', 'Models', 'PLONK (3 models), StreetCLIP, DINOv2, DISK, LightGlue'],
  ['speed', 'Speed test', 'Picks a default for Samples'],
];
let setupSteps = SETUP_STEPS;
let stepStates = {};
function renderSteps() {
  $('setupSteps').innerHTML = setupSteps.map(([key, name, sub]) => `<li data-state="${stepStates[key] || 'pending'}">
    <span class="step-dot">${icon('check', 'i i-check')}${icon('close', 'i i-x')}</span>
    <span><div class="step-name">${esc(name)}</div><div class="step-sub">${esc(sub)}</div></span>
  </li>`).join('');
}
function useSteps(list) { setupSteps = list; stepStates = {}; renderSteps(); }
// Mark `key` active and every step before it done.
function setStep(key) {
  const idx = setupSteps.findIndex(([k]) => k === key);
  if (idx < 0 || stepStates[key] === 'active') return;
  setupSteps.forEach(([k], i) => { if (i < idx) stepStates[k] = 'done'; });
  stepStates[key] = 'active';
  renderSteps();
}
function endSteps(ok) {
  for (const [k] of setupSteps) {
    if (ok) stepStates[k] = 'done';
    else if (stepStates[k] === 'active') stepStates[k] = 'failed';
  }
  renderSteps();
}

const RING = 2 * Math.PI * 52;
let setupPct = 0;
function setupBar(pct, status) {
  pct = clamp(pct, 0, 100);
  setupPct = pct;
  $('setupRing').style.strokeDashoffset = String(RING * (1 - pct / 100));
  $('setupDial').setAttribute('aria-valuenow', String(Math.round(pct)));
  if (status) $('setupStatus').textContent = status;
  $('setupPct').textContent = `${Math.floor(pct)}%`;
}
function setDial(state) { $('setupDial').dataset.state = state; }

// Bytes, rate and time left for the native downloads (rate smoothed).
const fmtSize = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${Math.round(n / 1e6)} MB`);
let rate = { t: 0, done: 0, bps: 0 };
function setupBytes(done, total) {
  const now = performance.now();
  if (rate.t && done > rate.done) {
    const inst = ((done - rate.done) * 1000) / (now - rate.t);
    rate.bps = rate.bps ? rate.bps * 0.85 + inst * 0.15 : inst;
  }
  rate = { t: now, done, bps: rate.bps };
  const parts = [`${fmtSize(done)} of ${fmtSize(total)}`];
  if (rate.bps > 0 && done < total) {
    parts.push(`${(rate.bps / 1e6).toFixed(1)} MB/s`);
    const eta = formatEta((total - done) / rate.bps);
    if (eta) parts.push(`${eta} left`);
  }
  $('setupDetail').textContent = parts.join(' · ');
}
$('setupStartBtn').addEventListener('click', async () => {
  const btn = $('setupStartBtn');
  btn.disabled = true;
  btn.classList.add('hidden'); // the dial and steps show progress; back again as Retry on failure
  useSteps(SETUP_STEPS);
  rate = { t: 0, done: 0, bps: 0 };
  $('setupDetail').textContent = '';
  setDial('running');
  setupBar(1, 'Starting…');
  const result = await window.api.runEnvSetup();
  if (result.ok) {
    setupBar(100, 'Setup complete');
    endSteps(true);
    setDial('done');
    $('setupPct').textContent = 'Done';
    setTimeout(async () => { showScreen('mainScreen'); await applyRecommendedSamples(); scheduleUpdateCheck(); }, 1400);
  } else {
    setupAppend(`Setup failed: ${result.error}`);
    endSteps(false);
    setDial('error');
    $('setupStatus').textContent = 'Setup failed';
    $('setupDetail').textContent = 'See the installer output below.';
    document.querySelector('#setupScreen .disclosure').open = true;
    btn.querySelector('.btn-label').textContent = 'Retry setup';
    btn.disabled = false;
    btn.classList.remove('hidden');
  }
});
window.api.onEnvProgress((p) => {
  if (p.event === 'progress') {
    setupBar(p.pct, p.status);
    if (/ONNX Runtime/.test(p.status)) setStep('runtime');
    else if (/models/i.test(p.status)) setStep('models');
    if (p.total) setupBytes(p.done, p.total);
  } else if (p.event === 'log') {
    setupAppend(p.message);
  } else if (p.event === 'stage_start') {
    setupAppend(`--- ${p.stage} ---`);
    if (p.stage === 'calibrate') { setStep('speed'); $('setupDetail').textContent = 'Timing real sampling batches'; }
  } else if (p.event === 'stage_done') {
    setupAppend(`${p.stage} done.`);
    if (p.stage === 'calibrate' && p.samples_per_sec) {
      setupAppend(`Measured ${p.samples_per_sec.toFixed(0)} samples/sec, default Samples set to ${p.recommended_samples}.`);
      $('setupDetail').textContent = `${p.samples_per_sec.toFixed(0)} samples/s · default Samples ${p.recommended_samples}`;
    }
  }
});

/* ============================================================ Updates */
// The Rust side checks the signed release manifest (latest.json on the newest
// GitHub release). The card offers the update; "Later" hides it for that version.
let pendingUpdate = null;
function showUpdate(u) {
  pendingUpdate = u;
  $('updateTitle').textContent = `Waypoint ${u.version} is available`;
  $('updateSub').textContent = `You have ${u.current}. It installs in a few seconds, then Waypoint restarts.`;
  $('updateProgress').classList.add('hidden');
  $('updateInstallBtn').disabled = false;
  $('updateLaterBtn').disabled = false;
  $('updateCard').classList.remove('hidden');
}
async function checkForUpdates(auto) {
  const r = await window.api.checkUpdate(auto);
  if (r.ok && r.available) {
    const s = await window.api.getSettings();
    if (!auto || s.updateDismissed !== r.version) showUpdate(r);
  }
  return r;
}
function scheduleUpdateCheck() { setTimeout(() => checkForUpdates(true).catch(() => {}), 4000); }
$('updateLaterBtn').addEventListener('click', async () => {
  $('updateCard').classList.add('hidden');
  if (pendingUpdate) await window.api.setSettings({ updateDismissed: pendingUpdate.version });
});
$('updateInstallBtn').addEventListener('click', async () => {
  if (state.busy) { toast('Stop the running pipeline first.', true); return; }
  $('updateInstallBtn').disabled = true;
  $('updateLaterBtn').disabled = true;
  $('updateTitle').textContent = `Downloading Waypoint ${pendingUpdate ? pendingUpdate.version : ''}…`;
  $('updateSub').textContent = 'Waypoint restarts when it is done.';
  $('updateBar').style.width = '0%';
  $('updateProgress').classList.remove('hidden');
  const r = await window.api.installUpdate(); // on success the app restarts before this returns
  if (!r.ok) {
    toast(`Update failed: ${r.error}`, true);
    if (pendingUpdate) showUpdate(pendingUpdate);
  }
});
window.api.onUpdateProgress((p) => {
  if (p.total) {
    $('updateBar').style.width = `${clamp((p.done / p.total) * 100, 0, 100)}%`;
    $('updateSub').textContent = `${fmtSize(p.done)} of ${fmtSize(p.total)}`;
  }
});
$('updateCheckBtn').addEventListener('click', async () => {
  const btn = $('updateCheckBtn');
  btn.disabled = true;
  $('updateStatus').textContent = 'Checking…';
  const r = await checkForUpdates(false);
  btn.disabled = false;
  $('updateStatus').textContent = !r.ok ? `Could not check: ${r.error}`
    : r.available ? `Version ${r.version} is available.` : 'You have the latest version.';
  if (r.ok && r.available) settingsDialog.close();
});

/* ============================================================ Startup */

// Setup times real sampling batches (plonk::throughput) and stores
// hardware.recommendedSamples. VRAM turned out to be a poor proxy
// for throughput; the VRAM tiers below only cover settings written before
// calibration existed, until the user re-runs setup.
function recommendedSamples(hw) {
  if (hw && hw.recommendedSamples) return hw.recommendedSamples;
  if (!hw || !hw.hasGpu) return 512;
  const vramGb = (hw.vramMb || 0) / 1024;
  if (vramGb >= 16) return 4096;
  if (vramGb >= 12) return 2048;
  if (vramGb >= 8) return 1024;
  return 768;
}
async function applyRecommendedSamples() {
  const s = await window.api.getSettings();
  if (s.hardware) $('numSamples').value = recommendedSamples(s.hardware);
}

(async () => {
  renderSteps();
  resetSteps('full');
  refreshRunButtons();
  const status = await window.api.getEnvStatus();
  if (status.native) { showScreen('mainScreen'); await applyRecommendedSamples(); scheduleUpdateCheck(); }
  else showScreen('setupScreen');
})();
