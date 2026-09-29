// Drive the real app over WebView2 CDP (launch it with
//   WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9333).
//   node e2e.mjs one  <image> <out.ndjson> [numSamples] [numRuns]
//   node e2e.mjs point <image> <out.ndjson> <lat> <lon> <radiusKm>
//   node e2e.mjs cancel <image> <out.ndjson>      (starts a run, cancels after 20 s)
import fs from 'node:fs';

const [, , mode, image, outFile, a1, a2, a3] = process.argv;
const targets = await (await fetch('http://127.0.0.1:9333/json')).json();
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
};
await new Promise((r) => (ws.onopen = r));
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result?.result?.value;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

await ev(`(async () => {
  window.__ev = [];
  window.api.onPipelineEvent((p) => window.__ev.push(p));
  setImage(await window.api.loadImage(${JSON.stringify(image)}));
  return true;
})()`);

if (mode === 'one' || mode === 'cancel') {
  await ev(`(() => { if (${JSON.stringify(a1 || '')}) $('numSamples').value = ${JSON.stringify(a1 || '')}; if (${JSON.stringify(a2 || '')}) $('numRuns').value = ${JSON.stringify(a2 || '')}; $('runOneBtn').click(); return true; })()`);
} else {
  await ev(`(() => { setMode('refine'); $('zoomLat').value = ${JSON.stringify(a1)}; $('zoomLon').value = ${JSON.stringify(a2)}; $('zoomRadius').value = ${JSON.stringify(a3)}; $('zoomLat').dispatchEvent(new Event('input')); $('runZoomBtn').click(); return true; })()`);
}
const t0 = Date.now();
if (mode === 'cancel') {
  await sleep(20000);
  await ev(`(() => { const b = document.getElementById('cancelBtn') || [...document.querySelectorAll('button')].find(x => /stop|cancel/i.test(x.textContent + x.id)); b.click(); return b.id; })()`);
}
for (;;) {
  await sleep(2000);
  const done = await ev(`window.__ev.some((e) => e.event === 'exit')`);
  if (done) break;
  if (Date.now() - t0 > 25 * 60 * 1000) throw new Error('timeout');
}
const events = await ev(`window.__ev`);
fs.writeFileSync(outFile, events.map((e) => JSON.stringify(e)).join('\n') + '\n');
const counts = {};
for (const e of events) counts[e.event] = (counts[e.event] || 0) + 1;
console.log('events', JSON.stringify(counts), 'elapsed', ((Date.now() - t0) / 1000).toFixed(1) + 's');
console.log('state', JSON.stringify(await ev(`({busy: state.busy, runState: document.getElementById('runState')?.textContent})`)));
ws.close();
