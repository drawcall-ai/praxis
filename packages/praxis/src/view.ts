import { createServer as createHttpServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { createServer } from 'vite';
import { ViteNodeServer } from 'vite-node/server';
import { ViteNodeRunner } from 'vite-node/client';
import { config as loadEnv } from 'dotenv';
import { generateText } from './generate.js';
import type { ModelConfig, ModelDefinition } from './types.js';

interface ViewOptions {
  definitionPath: string;
  configPath: string;
  port: number;
}

export async function startViewServer(options: ViewOptions) {
  const { definitionPath, configPath, port } = options;

  // Load env
  let dir = resolve(dirname(definitionPath));
  for (let i = 0; i < 10; i++) {
    loadEnv({ path: resolve(dir, '.env') });
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  // Load config
  const raw = await readFile(configPath, 'utf-8');
  const config: ModelConfig = JSON.parse(raw);

  // Load definition via vite-node
  const viteServer = await createServer({
    optimizeDeps: { disabled: true },
    logLevel: 'silent',
  });
  await viteServer.pluginContainer.buildStart({});
  const nodeServer = new ViteNodeServer(viteServer);
  const runner = new ViteNodeRunner({
    root: viteServer.config.root,
    fetchModule(id) { return nodeServer.fetchModule(id); },
    resolveId(id, importer) { return nodeServer.resolveId(id, importer); },
  });
  const mod = await runner.executeFile(definitionPath);
  const definition: ModelDefinition = (mod.default && typeof mod.default === 'object' && 'input' in mod.default)
    ? mod.default
    : mod;

  const server = createHttpServer(async (req, res) => {
    const url = new URL(req.url ?? '/', `http://localhost:${port}`);

    if (url.pathname === '/api/config') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(config));
      return;
    }

    if (url.pathname === '/api/schema') {
      const inputFields: Record<string, { type: string; description?: string; enumValues?: string[] }> = {};
      for (const [key, zodType] of Object.entries(definition.input.shape)) {
        const zt = zodType as any;
        const type = zt.def?.type ?? 'string';
        const desc = zt.description;
        const enumValues = type === 'enum' && zt.def?.entries ? Object.keys(zt.def.entries) : undefined;
        inputFields[key] = { type, description: desc, enumValues };
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ inputFields, hasMetric: !!definition.metric }));
      return;
    }

    if (url.pathname === '/api/run' && req.method === 'POST') {
      let body = '';
      req.on('data', (chunk) => { body += chunk; });
      req.on('end', async () => {
        try {
          const input = JSON.parse(body);
          const { output, score } = await generateText({ definition, input, config });
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ output, score }));
        } catch (err: unknown) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
        }
      });
      return;
    }

    if (url.pathname === '/' || url.pathname === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(buildHTML());
      return;
    }

    res.writeHead(404);
    res.end('Not found');
  });

  server.listen(port, () => {
    console.log(`\n  \x1b[1mpraxis view\x1b[0m  \x1b[2m→\x1b[0m  http://localhost:${port}\n`);
  });

  return server;
}

function buildHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Praxis</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0a;
    --surface: #141414;
    --surface-2: #1c1c1c;
    --border: #262626;
    --border-light: #333;
    --text: #e5e5e5;
    --text-dim: #737373;
    --text-muted: #525252;
    --accent: #3b82f6;
    --accent-dim: #1e3a5f;
    --green: #22c55e;
    --green-dim: #14532d;
    --red: #ef4444;
    --red-dim: #450a0a;
    --orange: #f59e0b;
    --radius: 8px;
    --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  .container {
    max-width: 960px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  header {
    margin-bottom: 48px;
  }

  header h1 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  header .model-name {
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
  }

  /* Stats grid */
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1px;
    background: var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 48px;
  }

  .stat {
    background: var(--surface);
    padding: 20px 24px;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }

  .stat-value {
    font-size: 20px;
    font-weight: 600;
    font-family: var(--mono);
  }

  .stat-value.good { color: var(--green); }
  .stat-value.warn { color: var(--orange); }
  .stat-value.bad { color: var(--red); }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
  }

  .tab {
    padding: 12px 20px;
    font-size: 14px;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
    font-family: inherit;
  }

  .tab:hover { color: var(--text); }
  .tab.active { color: var(--text); border-bottom-color: var(--text); }

  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* Instruction */
  .instruction {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 32px;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    color: var(--text-dim);
  }

  .instruction h2 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-top: 16px;
    margin-bottom: 4px;
  }

  .instruction h2:first-child { margin-top: 0; }

  /* Eval table */
  .eval-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  .eval-table th {
    text-align: left;
    padding: 10px 16px;
    color: var(--text-dim);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .eval-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    font-family: var(--mono);
    font-size: 12px;
  }

  .eval-table tr:last-child td { border-bottom: none; }

  .eval-table tr:hover td { background: var(--surface); }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: var(--mono);
  }

  .badge.pass { background: var(--green-dim); color: var(--green); }
  .badge.fail { background: var(--red-dim); color: var(--red); }
  .badge.partial { background: #422006; color: var(--orange); }

  .cell-value {
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cell-value:hover {
    white-space: normal;
    word-break: break-word;
  }

  /* Playground */
  .playground {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }

  .playground-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-dim);
  }

  .playground-body {
    padding: 24px;
  }

  .field {
    margin-bottom: 20px;
  }

  .field label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }

  .field label .desc {
    text-transform: none;
    letter-spacing: normal;
    color: var(--text-muted);
    font-weight: 400;
  }

  .field textarea, .field input, .field select {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    color: var(--text);
    font-size: 14px;
    font-family: var(--mono);
    outline: none;
    transition: border-color 0.15s;
  }

  .field textarea:focus, .field input:focus, .field select:focus {
    border-color: var(--border-light);
  }

  .field textarea { resize: vertical; min-height: 80px; }

  .run-btn {
    background: var(--text);
    color: var(--bg);
    border: none;
    padding: 10px 24px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .run-btn:hover { opacity: 0.85; }
  .run-btn:disabled { opacity: 0.3; cursor: not-allowed; }

  .result-area {
    margin-top: 24px;
    display: none;
  }

  .result-area.visible { display: block; }

  .result-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
    font-family: var(--mono);
    font-size: 13px;
    white-space: pre-wrap;
    line-height: 1.6;
  }

  .result-score {
    margin-top: 12px;
    font-size: 12px;
    color: var(--text-dim);
  }

  .result-score span {
    font-family: var(--mono);
    font-weight: 600;
  }

  .error-box {
    background: var(--red-dim);
    border: 1px solid #991b1b;
    color: var(--red);
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 13px;
    font-family: var(--mono);
  }

  .empty-state {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-muted);
    font-size: 14px;
  }

  .empty-state p { margin-bottom: 4px; }

  .section-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-dim);
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  @media (max-width: 640px) {
    .container { padding: 24px 16px; }
    .stats { grid-template-columns: 1fr 1fr; }
    .cell-value { max-width: 140px; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Praxis</h1>
    <div class="model-name" id="model-name">Loading...</div>
  </header>

  <div class="stats" id="stats"></div>

  <div class="tabs">
    <button class="tab active" data-tab="eval">Eval Runs</button>
    <button class="tab" data-tab="playground">Playground</button>
    <button class="tab" data-tab="prompt">Optimized Prompt</button>
  </div>

  <div class="tab-content active" id="tab-eval">
    <div id="eval-content"></div>
  </div>

  <div class="tab-content" id="tab-playground">
    <div class="playground">
      <div class="playground-header">Manual Test</div>
      <div class="playground-body">
        <form id="run-form"></form>
        <div class="result-area" id="result-area">
          <div class="section-title">Output</div>
          <div id="result-box"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="tab-content" id="tab-prompt">
    <div class="instruction" id="instruction"></div>
  </div>
</div>

<script>
(async function() {
  const [configRes, schemaRes] = await Promise.all([
    fetch('/api/config').then(r => r.json()),
    fetch('/api/schema').then(r => r.json()),
  ]);

  const config = configRes;
  const schema = schemaRes;
  const opt = config.optimization;

  // Header
  document.getElementById('model-name').textContent = config.model;

  // Stats
  const statsEl = document.getElementById('stats');
  const addStat = (label, value, cls) => {
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = '<div class="stat-label">' + label + '</div><div class="stat-value ' + (cls||'') + '">' + value + '</div>';
    statsEl.appendChild(d);
  };

  addStat('Optimizer', opt.optimizer.toUpperCase());

  if (typeof opt.bestScore === 'number') {
    const cls = opt.bestScore >= 0.8 ? 'good' : opt.bestScore >= 0.5 ? 'warn' : 'bad';
    addStat('Best Score', opt.bestScore.toFixed(2), cls);
  } else if (typeof opt.bestScore === 'object') {
    for (const [k, v] of Object.entries(opt.bestScore)) {
      const cls = v >= 0.8 ? 'good' : v >= 0.5 ? 'warn' : 'bad';
      addStat(k, v.toFixed(2), cls);
    }
  }

  addStat('Demos', opt.demos.length);

  if (opt.stats?.convergenceInfo?.converged != null) {
    addStat('Converged', opt.stats.convergenceInfo.converged ? 'Yes' : 'No',
            opt.stats.convergenceInfo.converged ? 'good' : '');
  }

  if (config.teacher) addStat('Teacher', config.teacher.split('/').pop());

  // Instruction tab
  const instrEl = document.getElementById('instruction');
  const instrText = opt.instruction || '(no instruction)';
  instrEl.innerHTML = instrText
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^- (.+)$/gm, '&bull; $1');

  // Eval runs tab
  const evalEl = document.getElementById('eval-content');
  const runs = opt.evalRuns || [];

  if (runs.length === 0) {
    evalEl.innerHTML = '<div class="empty-state"><p>No eval runs recorded yet.</p><p style="color:var(--text-muted)">Re-train the model to capture per-example evaluation traces.</p></div>';
  } else {
    const outputKeys = Object.keys(runs[0].expectedOutput || runs[0].modelOutput || {});
    const inputKeys = Object.keys(runs[0].input || {});

    let html = '<table class="eval-table"><thead><tr><th>#</th>';
    for (const k of inputKeys) html += '<th>' + k + '</th>';
    html += '<th>Expected</th><th>Predicted</th><th>Score</th></tr></thead><tbody>';

    runs.forEach((run, i) => {
      const score = run.score;
      let scoreStr, badgeCls;
      if (typeof score === 'number') {
        scoreStr = score.toFixed(2);
        badgeCls = score >= 1 ? 'pass' : score > 0 ? 'partial' : 'fail';
      } else {
        const vals = Object.values(score);
        const avg = vals.reduce((a,b) => a+b, 0) / vals.length;
        scoreStr = Object.entries(score).map(([k,v]) => k + ':' + v.toFixed(1)).join(' ');
        badgeCls = avg >= 1 ? 'pass' : avg > 0 ? 'partial' : 'fail';
      }

      const expected = outputKeys.map(k => run.expectedOutput[k]).join(', ');
      const predicted = outputKeys.map(k => run.modelOutput[k]).join(', ');
      const inputVals = inputKeys.map(k => run.input[k]);

      html += '<tr><td>' + (i+1) + '</td>';
      for (const v of inputVals) html += '<td><div class="cell-value" title="' + String(v).replace(/"/g,'&quot;') + '">' + String(v) + '</div></td>';
      html += '<td><div class="cell-value">' + expected + '</div></td>';
      html += '<td><div class="cell-value">' + predicted + '</div></td>';
      html += '<td><span class="badge ' + badgeCls + '">' + scoreStr + '</span></td>';
      html += '</tr>';
    });

    html += '</tbody></table>';
    evalEl.innerHTML = html;
  }

  // Playground tab
  const form = document.getElementById('run-form');
  let fieldsHTML = '';

  for (const [name, field] of Object.entries(schema.inputFields)) {
    const desc = field.description ? ' <span class="desc">' + field.description + '</span>' : '';
    fieldsHTML += '<div class="field"><label>' + name + desc + '</label>';

    if (field.enumValues) {
      fieldsHTML += '<select name="' + name + '">';
      for (const v of field.enumValues) fieldsHTML += '<option value="' + v + '">' + v + '</option>';
      fieldsHTML += '</select>';
    } else if (field.type === 'number') {
      fieldsHTML += '<input type="number" step="any" name="' + name + '">';
    } else if (field.type === 'boolean') {
      fieldsHTML += '<select name="' + name + '"><option value="true">true</option><option value="false">false</option></select>';
    } else {
      fieldsHTML += '<textarea name="' + name + '" rows="3"></textarea>';
    }

    fieldsHTML += '</div>';
  }

  fieldsHTML += '<button type="submit" class="run-btn" id="run-btn">Run</button>';
  form.innerHTML = fieldsHTML;

  const resultArea = document.getElementById('result-area');
  const resultBox = document.getElementById('result-box');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    resultArea.classList.add('visible');
    resultBox.innerHTML = '<div style="color:var(--text-muted)">Generating...</div>';

    const formData = new FormData(form);
    const input = {};
    for (const [name, field] of Object.entries(schema.inputFields)) {
      const val = formData.get(name);
      if (field.type === 'number') input[name] = parseFloat(val);
      else if (field.type === 'boolean') input[name] = val === 'true';
      else input[name] = val;
    }

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      });
      const data = await res.json();

      if (data.error) {
        resultBox.innerHTML = '<div class="error-box">' + data.error + '</div>';
      } else {
        let html = '<div class="result-box">' + JSON.stringify(data.output, null, 2) + '</div>';
        if (data.score != null) {
          const scoreStr = typeof data.score === 'number'
            ? data.score.toFixed(2)
            : JSON.stringify(data.score);
          html += '<div class="result-score">Score: <span>' + scoreStr + '</span></div>';
        }
        resultBox.innerHTML = html;
      }
    } catch (err) {
      resultBox.innerHTML = '<div class="error-box">' + err.message + '</div>';
    }

    btn.disabled = false;
    btn.textContent = 'Run';
  });

  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
  });
})();
</script>
</body>
</html>`;
}
