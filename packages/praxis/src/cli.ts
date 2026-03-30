import { resolve, dirname } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { config as loadEnv } from 'dotenv';
import type { ModelConfig, ModelDefinition, TrainOptions } from './types.js';

const DEFAULT_CONFIG = 'model.config.json';
const DEFAULT_DEFINITION = 'model.definition.ts';

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;

function loadEnvUp(from: string) {
  let dir = resolve(from);
  for (let i = 0; i < 10; i++) {
    loadEnv({ path: resolve(dir, '.env') });
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
}

async function main() {
  loadEnvUp(process.cwd());
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    printUsage();
    process.exit(0);
  }

  const command = args[0];

  if (command === 'train') await handleTrain(args.slice(1));
  else if (command === 'run') await handleRun(args.slice(1));
  else if (command === 'validate') await handleValidate(args.slice(1));
  else {
    console.error(`Unknown command: ${command}`);
    printUsage();
    process.exit(1);
  }
}

function requireEnvKey(): string {
  const key = process.env.OPENROUTER_KEY ?? process.env.OPENROUTER_API_KEY;
  if (!key) {
    console.error(`\n  ${bold('OPENROUTER_KEY')} is not set.\n`);
    console.error(`  Add it to your ${dim('.env')} file:`);
    console.error(`  OPENROUTER_KEY=sk-or-...\n`);
    process.exit(1);
  }
  return key;
}

interface ZodField {
  def: { type: string; entries?: Record<string, string> };
  description?: string;
}

function formatZodSchema(definition: ModelDefinition): string {
  const lines: string[] = [];

  lines.push(`  ${dim('input')}`);
  for (const [key, zodType] of Object.entries(definition.input.shape)) {
    const zt = zodType as unknown as ZodField;
    const type = zt.def.type;
    const desc = zt.description ? dim(` — ${zt.description}`) : '';
    lines.push(`    ${cyan(key)} ${dim(type)}${desc}`);
  }

  lines.push(`  ${dim('output')}`);
  for (const [key, zodType] of Object.entries(definition.output.shape)) {
    const zt = zodType as unknown as ZodField;
    let type = zt.def.type;
    if (type === 'enum' && zt.def.entries) type = Object.keys(zt.def.entries).join(' | ');
    const desc = zt.description ? dim(` — ${zt.description}`) : '';
    lines.push(`    ${cyan(key)} ${dim(type)}${desc}`);
  }

  return lines.join('\n');
}

// ── Train ────────────────────────────────────────────────────────────

async function handleTrain(args: string[]) {
  const definitionPath = args[0] && !args[0].startsWith('-') ? args[0] : DEFAULT_DEFINITION;

  const options: TrainOptions = {
    definitionPath: resolve(definitionPath),
    output: resolve(flagValue(args, '--output', '-o') ?? DEFAULT_CONFIG),
    optimizer: (flagValue(args, '--optimizer') ?? 'auto') as TrainOptions['optimizer'],
    split: parseFloat(flagValue(args, '--split') ?? '0.7'),
  };

  loadEnv({ path: resolve(dirname(options.definitionPath), '.env') });
  requireEnvKey();

  const definition = await loadDefinition(options.definitionPath);
  validateDefinition(definition, true);

  console.log('');
  const teacherLabel = definition.teacher ? ` · teacher: ${definition.teacher}` : '';
  console.log(`  ${bold(definition.model)} ${dim(`${options.optimizer.toUpperCase()} · ${definition.examples.length} examples · ${options.split}/${(1 - options.split).toFixed(1)} split${teacherLabel}`)}`);
  console.log('');
  console.log(formatZodSchema(definition));
  console.log('');

  const { train } = await import('./train.js');
  const { config, testScore } = await train(definition, options);

  await writeFile(options.output, JSON.stringify(config, null, 2));

  console.log('');
  console.log(`  ${green('✓')} ${bold(options.output)}`);
  if (testScore !== null) {
    console.log(`  ${green('✓')} test score ${bold(JSON.stringify(testScore))}`);
  }
  console.log('');
}

// ── Run ──────────────────────────────────────────────────────────────

async function handleRun(args: string[]) {
  const definitionPath = resolve(flagValue(args, '--definition', '-d') ?? DEFAULT_DEFINITION);
  const configPath = resolve(args[0] && !args[0].startsWith('-') ? args[0] : DEFAULT_CONFIG);

  loadEnv({ path: resolve(dirname(definitionPath), '.env') });
  requireEnvKey();

  const definition = await loadDefinition(definitionPath);
  validateDefinition(definition, false);

  let config: ModelConfig | undefined;
  try {
    const raw = await readFile(configPath, 'utf-8');
    config = JSON.parse(raw);
  } catch {
    // No config — run without training
  }

  const inputFields = Object.entries(definition.input.shape) as unknown as [string, ZodField][];

  const cliInput: Record<string, unknown> = {};
  let allProvided = true;

  for (const [name, field] of inputFields) {
    const value = flagValue(args, `--${name}`);
    if (value != null) {
      const type = field.def.type;
      cliInput[name] = coerce(value, type === 'number' ? 'number' : type === 'boolean' ? 'boolean' : 'string');
    } else {
      allProvided = false;
    }
  }

  if (!allProvided) {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    const ask = (q: string): Promise<string> => new Promise((r) => rl.question(q, r));

    console.log('');
    for (const [name, field] of inputFields) {
      if (name in cliInput) continue;
      const desc = field.description ?? '';
      const type = field.def.type;
      const value = await ask(`  ${cyan(name)} ${dim(desc)}\n  ${dim('>')} `);
      cliInput[name] = coerce(value, type === 'number' ? 'number' : type === 'boolean' ? 'boolean' : 'string');
    }
    rl.close();
  }

  await runModel(definition, cliInput, config);
}

async function runModel(definition: ModelDefinition, input: Record<string, unknown>, config?: ModelConfig) {
  const { generateText } = await import('./generate.js');

  const spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  let frame = 0;
  const interval = setInterval(() => {
    frame = (frame + 1) % spinner.length;
    process.stdout.write(`\r  ${dim(spinner[frame])}`);
  }, 80);

  try {
    const { output, score } = await generateText({ definition, input, config });

    clearInterval(interval);
    process.stdout.write('\r\x1b[K');
    console.log(JSON.stringify(output, null, 2));
    if (score != null) {
      console.log(`\n  ${dim('score:')} ${score}`);
    }
  } catch (err: unknown) {
    clearInterval(interval);
    process.stdout.write('\r\x1b[K');
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`${bold('Error:')} ${msg}`);
    process.exit(1);
  }
}

// ── Validate ─────────────────────────────────────────────────────────

async function handleValidate(args: string[]) {
  const definitionPath = resolve(flagValue(args, '--definition', '-d') ?? DEFAULT_DEFINITION);
  const configPath = resolve(args[0] && !args[0].startsWith('-') ? args[0] : DEFAULT_CONFIG);

  const definition = await loadDefinition(definitionPath);
  validateDefinition(definition, false);

  const raw = await readFile(configPath, 'utf-8');
  const config: ModelConfig = JSON.parse(raw);

  const { validateSchema } = await import('./schema.js');

  let ok = true;

  try {
    validateSchema(definition.input, config.schema.input, 'input');
  } catch {
    console.log(`  ${red('✗')} input schema mismatch`);
    ok = false;
  }

  try {
    validateSchema(definition.output, config.schema.output, 'output');
  } catch {
    console.log(`  ${red('✗')} output schema mismatch`);
    ok = false;
  }

  if (definition.model !== config.model) {
    console.log(`  ${red('✗')} model mismatch: ${dim(definition.model)} ≠ ${dim(config.model)}`);
    ok = false;
  }

  if ((definition.teacher ?? undefined) !== (config.teacher ?? undefined)) {
    console.log(`  ${red('✗')} teacher mismatch: ${dim(definition.teacher ?? 'none')} ≠ ${dim(config.teacher ?? 'none')}`);
    ok = false;
  }

  if (ok) {
    console.log(`  ${green('✓')} config is in sync with definition`);
  } else {
    console.log(`\n  Run ${bold('npx praxis train')} to fix.\n`);
    process.exit(1);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────

function validateDefinition(def: ModelDefinition, requireMetric: boolean) {
  if (!def.input || !def.output)
    throw new Error('Definition must export "input" and "output" Zod objects');
  if (!def.model || typeof def.model !== 'string')
    throw new Error('Definition must export a "model" string');
  if (!Array.isArray(def.examples))
    throw new Error('Definition must export an "examples" array');
  if (requireMetric && !def.metric)
    throw new Error('Definition must export a "metric" function');
}

async function loadDefinition(filePath: string): Promise<ModelDefinition> {
  const { ViteNodeRunner } = await import('vite-node/client');
  const { ViteNodeServer } = await import('vite-node/server');
  const { createServer } = await import('vite');

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

  const mod = await runner.executeFile(filePath);
  await viteServer.close();

  if (mod.default && typeof mod.default === 'object' && 'input' in mod.default) {
    return mod.default as ModelDefinition;
  }
  return mod as ModelDefinition;
}

function coerce(value: string, type: string): unknown {
  if (type === 'number') return parseFloat(value);
  if (type === 'boolean') return value === 'true' || value === '1' || value === 'yes';
  return value;
}

function flagValue(args: string[], long: string, short?: string): string | undefined {
  for (let i = 0; i < args.length; i++) {
    if (args[i] === long || (short && args[i] === short)) return args[i + 1];
    if (args[i].startsWith(`${long}=`)) return args[i].slice(long.length + 1);
  }
  return undefined;
}

function printUsage() {
  console.log(`
  ${bold('praxis')} — Define, train, and use optimized LLM prompts

  ${bold('Commands')}

    ${cyan('train')} [definition] [options]
      Optimize prompts from a definition file.
      Default: ${dim(DEFAULT_DEFINITION)}

      --output, -o <path>    Config output ${dim(`(default: ${DEFAULT_CONFIG})`)}
      --optimizer <type>     ace | gepa | auto ${dim('(default: auto)')}
      --split <ratio>        Train/test split ${dim('(default: 0.7)')}

    ${cyan('run')} [config] [--definition, -d <path>] [--field "value" ...]
      Run the model. Uses trained config if available.
      Default definition: ${dim(DEFAULT_DEFINITION)}
      Default config: ${dim(DEFAULT_CONFIG)}

      All fields via flags → runs once, outputs JSON.
      Missing fields → prompts for them first.

    ${cyan('validate')} [config] [--definition, -d <path>]
      Check that the config matches the definition schema.

  ${dim('Requires OPENROUTER_KEY in env or .env file.')}
`);
}

main().catch((err) => {
  console.error(`\n  ${bold('Error:')} ${err.message ?? err}\n`);
  process.exit(1);
});
