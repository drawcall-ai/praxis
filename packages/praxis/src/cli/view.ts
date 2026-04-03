import { createServer as createHttpServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { createServer } from 'vite';
import { ViteNodeServer } from 'vite-node/server';
import { ViteNodeRunner } from 'vite-node/client';
import { config as loadEnv } from 'dotenv';
import { generateText } from '../generate.js';
import type { ModelConfig, ModelDefinition } from '../types.js';
import { DEFAULT_CONFIG, bold, dim, red, resolveDefinitionPaths } from './utils.js';

interface ViewEntry {
  definitionPath: string;
  configPath: string;
}

interface LoadedModel {
  name: string;
  definition: ModelDefinition;
  config?: ModelConfig;
}

// ── CLI handler ────────────────────────────────────────────────────────

export async function handleView(opts: { definition?: string; config?: string; port: string }) {
  if (opts.config && !opts.definition) {
    const paths = await resolveDefinitionPaths();
    if (paths.length > 1) {
      console.error(`\n  ${bold('-c')} cannot be used with multiple definitions. Pass ${bold('-d')} to select one.\n`);
      process.exit(1);
    }
  }

  const definitionPaths = await resolveDefinitionPaths(opts.definition);

  const entries: ViewEntry[] = definitionPaths.map((defPath) => ({
    definitionPath: defPath,
    configPath: opts.config ? resolve(opts.config) : resolve(dirname(defPath), DEFAULT_CONFIG),
  }));

  await startViewServer(entries, parseInt(opts.port, 10));
}

// ── View server ────────────────────────────────────────────────────────

async function startViewServer(entries: ViewEntry[], port: number) {
  // Load env from first entry's directory
  let dir = resolve(dirname(entries[0].definitionPath));
  for (let i = 0; i < 10; i++) {
    loadEnv({ path: resolve(dir, '.env') });
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  // Load all models
  const viteServer = await createServer({
    root: dirname(entries[0].definitionPath),
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

  const models: LoadedModel[] = [];

  for (const entry of entries) {
    try {
      const mod = await runner.executeFile(entry.definitionPath);
      const definition: ModelDefinition = (mod.default && typeof mod.default === 'object' && 'input' in mod.default)
        ? mod.default
        : mod;

      // Validate minimum requirements
      if (!definition.input || !definition.output || !definition.student) {
        console.error(`  ${red('✗')} ${dim(basename(dirname(entry.definitionPath)))} — invalid definition, skipping`);
        continue;
      }

      let config: ModelConfig | undefined;
      try {
        const raw = await readFile(entry.configPath, 'utf-8');
        config = JSON.parse(raw);
      } catch { /* no config yet — untrained model */ }

      const defDir = basename(dirname(entry.definitionPath));
      models.push({ name: defDir, definition, config });
    } catch (err) {
      const label = basename(dirname(entry.definitionPath));
      const msg = err instanceof Error ? err.message.split('\n')[0] : String(err);
      console.error(`  ${red('✗')} ${dim(label)} — ${msg}`);
    }
  }

  await viteServer.close();

  if (models.length === 0) {
    console.error('\n  No valid definitions found.\n');
    process.exit(1);
  }

  const server = createHttpServer(async (req, res) => {
    const url = new URL(req.url ?? '/', `http://localhost:${port}`);
    const modelIdx = Math.min(parseInt(url.searchParams.get('model') ?? '0', 10), models.length - 1);
    const model = models[Math.max(0, modelIdx)];

    if (url.pathname === '/api/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(models.map((m, i) => ({
        name: m.definition.name ?? m.name,
        index: i,
        trained: !!m.config,
        student: m.definition.student,
      }))));
      return;
    }

    if (url.pathname === '/api/config') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(model.config ?? null));
      return;
    }

    if (url.pathname === '/api/schema') {
      const inputFields: Record<string, { type: string; description?: string; enumValues?: string[] }> = {};
      for (const [key, zodType] of Object.entries(model.definition.input.shape)) {
        const zt = zodType as any;
        const type = zt.def?.type ?? 'string';
        const desc = zt.description;
        const enumValues = type === 'enum' && zt.def?.entries ? Object.keys(zt.def.entries) : undefined;
        inputFields[key] = { type, description: desc, enumValues };
      }

      const outputFields: Record<string, { type: string; description?: string; enumValues?: string[] }> = {};
      for (const [key, zodType] of Object.entries(model.definition.output.shape)) {
        const zt = zodType as any;
        const type = zt.def?.type ?? 'string';
        const desc = zt.description;
        const enumValues = type === 'enum' && zt.def?.entries ? Object.keys(zt.def.entries) : undefined;
        outputFields[key] = { type, description: desc, enumValues };
      }

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        inputFields,
        outputFields,
        hasMetric: !!model.definition.metric,
        student: model.definition.student,
        name: model.definition.name ?? null,
        description: model.definition.description ?? null,
        teacher: model.definition.teacher ?? null,
        version: model.definition.version ?? null,
      }));
      return;
    }

    if (url.pathname === '/api/run' && req.method === 'POST') {
      let body = '';
      req.on('data', (chunk) => { body += chunk; });
      req.on('end', async () => {
        try {
          const input = JSON.parse(body);
          const { output, score } = await generateText({ definition: model.definition, input, ...(model.config ? { config: model.config } : {}) });
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
    margin-bottom: 32px;
  }

  header h1 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .model-desc {
    font-size: 14px;
    color: var(--text-dim);
    line-height: 1.5;
    margin-bottom: 24px;
  }

  /* Model tabs */
  .model-tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 32px;
  }

  .model-tab {
    position: relative;
    padding: 12px 20px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    border: none;
    background: none;
    font-family: inherit;
    transition: color 0.15s;
    white-space: nowrap;
  }

  .model-tab:hover { color: var(--text-dim); }

  .model-tab.active {
    color: var(--text);
  }

  .model-tab.active::after {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: -2px;
    height: 2px;
    background: var(--text);
  }

  .model-tab .badge-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-left: 8px;
    vertical-align: middle;
  }

  .model-tab .badge-dot.trained { background: var(--green); }
  .model-tab .badge-dot.untrained { background: var(--text-muted); }

  /* Stats grid */
  .stats {
    display: flex;
    gap: 1px;
    background: var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 32px;
  }

  .stat {
    background: var(--surface);
    padding: 14px 20px;
    flex: 1;
    min-width: 0;
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
  }

  .stat-value {
    font-size: 14px;
    font-weight: 600;
    font-family: var(--mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .stat-value.good { color: var(--green); }
  .stat-value.warn { color: var(--orange); }
  .stat-value.bad { color: var(--red); }
  .stat-value.muted { color: var(--text-muted); }

  /* Section tabs */
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
    vertical-align: middle;
    font-family: var(--mono);
    font-size: 12px;
  }

  .eval-table tr:last-child td { border-bottom: none; }

  .eval-table tbody tr { cursor: pointer; transition: background 0.1s; }
  .eval-table tbody tr:hover td { background: var(--surface-2); }

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

  .cell-truncate {
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Detail modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    z-index: 1000;
    display: none;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.15s;
  }

  .modal-overlay.visible { opacity: 1; }

  .modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    width: calc(100vw - 48px);
    height: calc(100vh - 48px);
    overflow-y: auto;
    transform: translateY(8px);
    transition: transform 0.15s;
    display: flex;
    flex-direction: column;
  }

  .modal-overlay.visible .modal { transform: translateY(0); }

  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
  }

  .modal-header h3 {
    font-size: 14px;
    font-weight: 600;
  }

  .modal-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: color 0.15s;
  }

  .modal-close:hover { color: var(--text); }

  .modal-body { padding: 24px; flex: 1; overflow-y: auto; }

  .modal-section {
    margin-bottom: 24px;
  }

  .modal-section:last-child { margin-bottom: 0; }

  .modal-section-title {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
  }

  .modal-fields {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }

  .modal-field-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-dim);
    margin-bottom: 4px;
  }

  .modal-field-value {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
  }

  .modal-diff {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .modal-diff-col h4 {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }

  .modal-diff-col .modal-field-value.match { border-color: #14532d; }
  .modal-diff-col .modal-field-value.mismatch { border-color: #450a0a; }

  .modal-footer {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  .modal-footer .try-btn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-dim);
    cursor: pointer;
    padding: 8px 16px;
    font-size: 12px;
    font-family: inherit;
    transition: all 0.15s;
  }

  .modal-footer .try-btn:hover {
    color: var(--text);
    border-color: var(--border-light);
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
    .stats { flex-wrap: wrap; }
    .stat { min-width: 45%; }
    .cell-value { max-width: 140px; }
    .model-tabs { overflow-x: auto; }
  }
  .lucide { width: 14px; height: 14px; stroke-width: 2; vertical-align: -2px; }

  .stat-label .lucide { width: 12px; height: 12px; vertical-align: -1px; opacity: 0.6; }

  .tab .lucide { width: 14px; height: 14px; vertical-align: -2px; margin-right: 4px; }

  .run-btn .lucide { width: 14px; height: 14px; vertical-align: -2px; margin-right: 2px; }

  .try-btn .lucide { width: 12px; height: 12px; vertical-align: -1px; }

  @keyframes spin { to { transform: rotate(360deg); } }
  .spin { animation: spin 1s linear infinite; }
</style>
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"><\/script>
</head>
<body>
<div class="container">
  <header>
    <h1>Praxis</h1>
  </header>

  <div class="model-tabs" id="model-tabs" style="display:none"></div>

  <div class="model-desc" id="model-desc" style="display:none"></div>

  <div class="stats" id="stats"></div>

  <div class="tabs" id="section-tabs">
    <button class="tab active" data-tab="playground"><i data-lucide="play"></i> Playground</button>
    <button class="tab" data-tab="eval"><i data-lucide="list"></i> Eval Runs</button>
    <button class="tab" data-tab="prompt"><i data-lucide="pen-line"></i> Optimized Prompt</button>
  </div>

  <div class="tab-content active" id="tab-playground">
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

  <div class="tab-content" id="tab-eval">
    <div id="eval-content"></div>
  </div>

  <div class="modal-overlay" id="eval-modal">
    <div class="modal">
      <div class="modal-header">
        <h3 id="modal-title">Eval Run</h3>
        <button class="modal-close" id="modal-close"><i data-lucide="x"></i></button>
      </div>
      <div class="modal-body" id="modal-body"></div>
      <div class="modal-footer" id="modal-footer"></div>
    </div>
  </div>

  <div class="tab-content" id="tab-prompt">
    <div class="instruction" id="instruction"></div>
  </div>
</div>

<script>
(async function() {
  let currentModel = 0;

  const modelsRes = await fetch('/api/models');
  const models = await modelsRes.json();

  // Model tabs (shown when multiple models)
  const modelTabsEl = document.getElementById('model-tabs');
  if (models.length > 1) {
    modelTabsEl.style.display = '';
    for (const m of models) {
      const btn = document.createElement('button');
      btn.className = 'model-tab' + (m.index === 0 ? ' active' : '');
      btn.dataset.modelIdx = m.index;
      const dotCls = m.trained ? 'trained' : 'untrained';
      btn.innerHTML = m.name + '<span class="badge-dot ' + dotCls + '"></span>';
      btn.addEventListener('click', () => {
        currentModel = m.index;
        modelTabsEl.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        loadModel(currentModel);
      });
      modelTabsEl.appendChild(btn);
    }
  }

  async function loadModel(idx) {
    const [configRes, schemaRes] = await Promise.all([
      fetch('/api/config?model=' + idx).then(r => r.json()),
      fetch('/api/schema?model=' + idx).then(r => r.json()),
    ]);
    renderModel(configRes, schemaRes, idx);
  }

  function renderModel(config, schema, idx) {
    const trained = !!config;
    const opt = config?.optimization;

    // Name in header
    const headerEl = document.querySelector('header h1');
    if (schema.name) {
      headerEl.textContent = schema.name;
    } else {
      headerEl.textContent = 'Praxis';
    }

    // Description
    const descEl = document.getElementById('model-desc');
    if (schema.description) {
      descEl.textContent = schema.description;
      descEl.style.display = '';
    } else {
      descEl.style.display = 'none';
    }

    // Stats — consistent single row for all models
    const statsEl = document.getElementById('stats');
    statsEl.innerHTML = '';
    const addStat = (icon, label, value, cls) => {
      const d = document.createElement('div');
      d.className = 'stat';
      d.innerHTML = '<div class="stat-label"><i data-lucide="' + icon + '"></i> ' + label + '</div><div class="stat-value ' + (cls||'') + '" title="' + value + '">' + value + '</div>';
      statsEl.appendChild(d);
    };

    // Always: student, status
    const shortStudent = schema.student.includes('/') ? schema.student.split('/').pop() : schema.student;
    addStat('zap', 'Student', shortStudent);
    addStat('circle', 'Status', trained ? 'Trained' : 'Untrained', trained ? 'good' : 'muted');

    // Always: teacher if present
    if (schema.teacher) {
      const shortTeacher = schema.teacher.includes('/') ? schema.teacher.split('/').pop() : schema.teacher;
      addStat('graduation-cap', 'Teacher', shortTeacher);
    }

    // Trained-only: scores
    if (trained) {
      // Best score
      if (opt.bestScore && typeof opt.bestScore === 'object') {
        const vals = Object.values(opt.bestScore);
        const avg = vals.length > 0 ? vals.reduce((a,b) => a+b, 0) / vals.length : 0;
        const cls = avg >= 0.8 ? 'good' : avg >= 0.5 ? 'warn' : 'bad';
        const label = Object.entries(opt.bestScore).map(([k,v]) => k + ': ' + v.toFixed(2)).join(', ');
        addStat('target', 'Score', avg.toFixed(2), cls);
      }

      // Test score (computed from eval runs)
      const evalRuns = opt.evalRuns || [];
      if (evalRuns.length > 0) {
        const perRun = evalRuns.map(r => {
          if (!r.score || typeof r.score !== 'object') return 0;
          const vals = Object.values(r.score);
          return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
        });
        const testAvg = perRun.reduce((a, b) => a + b, 0) / perRun.length;
        const cls = testAvg >= 0.8 ? 'good' : testAvg >= 0.5 ? 'warn' : 'bad';
        addStat('flask-conical', 'Test', testAvg.toFixed(2), cls);
      }

      // Reasoning effort
      if (opt.reasoningEffort != null) {
        addStat('brain', 'Reasoning', opt.reasoningEffort);
      }

      // Stats info
      if (opt.stats?.agentSteps) {
        addStat('settings', 'Steps', String(opt.stats.agentSteps));
      }
    }

    // Show/hide trained-only section tabs
    document.querySelectorAll('[data-tab="eval"], [data-tab="prompt"]').forEach(el => {
      el.style.display = trained ? '' : 'none';
    });

    // If currently on a hidden tab, switch to playground
    const activeTab = document.querySelector('.tab.active');
    if (activeTab && activeTab.style.display === 'none') {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.querySelector('[data-tab="playground"]').classList.add('active');
      document.getElementById('tab-playground').classList.add('active');
    }

    // Instruction tab
    const instrEl = document.getElementById('instruction');
    const instrText = opt?.instruction || '(no instruction)';
    instrEl.innerHTML = instrText
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^- (.+)$/gm, '&bull; $1');

    // Eval runs tab
    const evalEl = document.getElementById('eval-content');
    const runs = opt?.evalRuns || [];

    // Store runs globally for modal access
    window.__evalRuns = runs;

    if (runs.length === 0) {
      evalEl.innerHTML = '<div class="empty-state"><p>No eval runs recorded yet.</p><p style="color:var(--text-muted)">Train the model to capture per-example evaluation traces.</p></div>';
    } else {
      const inputKeys = Object.keys(runs[0].input || {});
      // Pick first input key as summary column
      const summaryKey = inputKeys[0] || 'input';

      let html = '<table class="eval-table"><thead><tr>';
      html += '<th>#</th>';
      html += '<th>' + summaryKey + '</th>';
      html += '<th>Score</th>';
      html += '</tr></thead><tbody>';

      runs.forEach((run, i) => {
        const score = run.score || {};
        const vals = typeof score === 'object' ? Object.values(score) : [score];
        const avg = vals.length > 0 ? vals.reduce((a,b) => a+b, 0) / vals.length : 0;
        const scoreStr = typeof score === 'object'
          ? Object.entries(score).map(([k,v]) => k + ':' + (typeof v === 'number' ? v.toFixed(1) : v)).join(' ')
          : String(score);
        const badgeCls = avg >= 1 ? 'pass' : avg > 0 ? 'partial' : 'fail';

        const summaryVal = String(run.input[summaryKey] ?? '');

        html += '<tr data-run-idx="' + i + '">';
        html += '<td>' + (i+1) + '</td>';
        html += '<td><div class="cell-truncate">' + summaryVal.replace(/</g,'&lt;') + '</div></td>';
        html += '<td><span class="badge ' + badgeCls + '">' + scoreStr + '</span></td>';
        html += '</tr>';
      });

      html += '</tbody></table>';
      evalEl.innerHTML = html;

      // Attach click handlers to each row
      evalEl.querySelectorAll('tr[data-run-idx]').forEach(row => {
        row.addEventListener('click', () => {
          openModal(parseInt(row.dataset.runIdx, 10));
        });
      });
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

    fieldsHTML += '<button type="submit" class="run-btn" id="run-btn"><i data-lucide="play"></i> Run</button>';
    form.innerHTML = fieldsHTML;

    const resultArea = document.getElementById('result-area');
    resultArea.classList.remove('visible');
    const resultBox = document.getElementById('result-box');

    form.onsubmit = async (e) => {
      e.preventDefault();
      const btn = document.getElementById('run-btn');
      btn.disabled = true;
      btn.innerHTML = '<i data-lucide="loader-2" class="lucide spin"></i> Running\u2026';
      lucide.createIcons({ nodes: [btn] });
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
        const res = await fetch('/api/run?model=' + currentModel, {
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
      btn.innerHTML = '<i data-lucide="play"></i> Run';
      lucide.createIcons({ nodes: [btn] });
    };

    // Render all Lucide icons in the page
    lucide.createIcons();
  }

  // Eval detail modal
  const modalOverlay = document.getElementById('eval-modal');
  const modalBody = document.getElementById('modal-body');
  const modalFooter = document.getElementById('modal-footer');
  const modalTitle = document.getElementById('modal-title');

  function openModal(idx) {
    const run = window.__evalRuns[idx];
    if (!run) return;

    modalTitle.textContent = 'Eval Run #' + (idx + 1);

    const fmt = (v) => {
      if (v == null) return '';
      if (typeof v === 'object') return JSON.stringify(v, null, 2);
      return String(v);
    };

    let html = '';

    // Input section
    html += '<div class="modal-section"><div class="modal-section-title">Input</div><div class="modal-fields">';
    for (const [k, v] of Object.entries(run.input)) {
      html += '<div><div class="modal-field-label">' + k + '</div><div class="modal-field-value">' + fmt(v).replace(/</g,'&lt;') + '</div></div>';
    }
    html += '</div></div>';

    // Expected Output section
    if (run.expectedOutput && Object.keys(run.expectedOutput).length > 0) {
      html += '<div class="modal-section"><div class="modal-section-title">Expected Output</div><div class="modal-fields">';
      for (const [k, v] of Object.entries(run.expectedOutput)) {
        html += '<div><div class="modal-field-label">' + k + '</div><div class="modal-field-value">' + fmt(v).replace(/</g,'&lt;') + '</div></div>';
      }
      html += '</div></div>';
    }

    // Model Output section
    if (run.modelOutput && Object.keys(run.modelOutput).length > 0) {
      html += '<div class="modal-section"><div class="modal-section-title">Model Output</div><div class="modal-fields">';
      for (const [k, v] of Object.entries(run.modelOutput)) {
        html += '<div><div class="modal-field-label">' + k + '</div><div class="modal-field-value">' + fmt(v).replace(/</g,'&lt;') + '</div></div>';
      }
      html += '</div></div>';
    }

    // Score
    const score = run.score || {};
    const scoreStr = typeof score === 'object'
      ? Object.entries(score).map(([k,v]) => k + ': ' + (typeof v === 'number' ? v.toFixed(2) : v)).join(', ')
      : String(score);
    html += '<div class="modal-section" style="margin-top:24px"><div class="modal-section-title">Score</div>';
    html += '<div class="modal-field-value">' + scoreStr + '</div></div>';

    modalBody.innerHTML = html;

    // Footer with Try in Playground button
    modalFooter.innerHTML = '<button class="try-btn" id="modal-try-btn"><i data-lucide="play"></i> Try in Playground</button>';
    document.getElementById('modal-try-btn').addEventListener('click', () => {
      const form = document.getElementById('run-form');
      for (const [name, value] of Object.entries(run.input)) {
        const el = form.querySelector('[name="' + name + '"]');
        if (el) el.value = String(value);
      }
      closeModal();
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.querySelector('[data-tab="playground"]').classList.add('active');
      document.getElementById('tab-playground').classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    modalOverlay.style.display = 'flex';
    requestAnimationFrame(() => { modalOverlay.classList.add('visible'); });
    lucide.createIcons({ nodes: [modalOverlay] });
  }

  function closeModal() {
    modalOverlay.classList.remove('visible');
    setTimeout(() => { modalOverlay.style.display = 'none'; }, 150);
  }

  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeModal();
  });
  document.getElementById('modal-close').addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  // Row click handlers are attached in renderModel after table is built

  // Section tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
  });

  // Initial load
  await loadModel(0);
})();
</script>
</body>
</html>`;
}
