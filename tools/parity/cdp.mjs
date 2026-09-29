// Minimal CDP driver: node cdp.mjs "<js expression>" [screenshot.png]
const [,, expr, shot] = process.argv;
const targets = await (await fetch('http://127.0.0.1:9333/json')).json();
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
await new Promise((r) => (ws.onopen = r));
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
if (expr) {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  console.log(JSON.stringify(r.result?.result?.value ?? r.result?.exceptionDetails ?? r, null, 1));
}
if (shot) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  (await import('node:fs')).writeFileSync(shot, Buffer.from(r.result.data, 'base64'));
  console.log('saved', shot);
}
ws.close();
