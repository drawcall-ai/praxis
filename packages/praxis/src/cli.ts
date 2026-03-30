import { resolve, dirname } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { Command } from 'commander';
import ora from 'ora';
import { glob } from 'tinyglobby';
import { config as loadEnv } from 'dotenv';
import { createServer } from 'vite';
import { ViteNodeServer } from 'vite-node/server';
import { ViteNodeRunner } from 'vite-node/client';
import { train } from './train.js';
import { generateText } from './generate.js';
import { validateSchema } from './schema.js';
import type { ModelConfig, ModelDefinition, TrainOptions } from './types.js';

const DEFAULT_CONFIG = 'model.config.json';
const DEFINITION_GLOBS = ['**/model.definition.ts', '**/model.definition.js'];

// ── Colors ──────────────────────────────────────────────────────────

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;

// ── Helpers ─────────────────────────────────────────────────────────

async function findDefinition(): Promise<string> {
  const matches = await glob(DEFINITION_GLOBS, {
    ignore: ['**/node_modules/**', '**/dist/**'],
    absolute: true,
  });
  if (matches.length === 0) {
    console.error(`\n  Could not find model.definition.ts or model.definition.js\n`);
    process.exit(1);
  }
  if (matches.length > 1) {
    console.error(`\n  Found multiple definition files:\n`);
    for (const m of matches) console.error(`    ${m}`);
    console.error(`\n  Pass the path explicitly with -d to disambiguate.\n`);
    process.exit(1);
  }
  return matches[0];
}

function loadEnvUp(from: string) {
  let dir = resolve(from);
  for (let i = 0; i < 10; i++) {
    loadEnv({ path: resolve(dir, '.env') });
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
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

async function resolveDefinitionPath(flag?: string): Promise<string> {
  return flag ? resolve(flag) : await findDefinition();
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

// ── Train ───────────────────────────────────────────────────────────

async function handleTrain(opts: { definition?: string; output?: string; optimizer: string; split: string }) {
  const definitionPath = await resolveDefinitionPath(opts.definition);
  const defaultOutput = resolve(dirname(definitionPath), DEFAULT_CONFIG);

  const options: TrainOptions = {
    definitionPath,
    output: opts.output ? resolve(opts.output) : defaultOutput,
    optimizer: opts.optimizer as TrainOptions['optimizer'],
    split: parseFloat(opts.split),
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

  const { config, testScore } = await train(definition, options);

  await writeFile(options.output, JSON.stringify(config, null, 2));

  console.log('');
  console.log(`  ${green('✓')} ${bold(options.output)}`);
  if (testScore !== null) {
    console.log(`  ${green('✓')} test score ${bold(JSON.stringify(testScore))}`);
  }
  console.log('');
}

// ── Run ─────────────────────────────────────────────────────────────

async function handleRun(opts: { definition?: string; config?: string }, extraArgs: string[]) {
  const definitionPath = await resolveDefinitionPath(opts.definition);
  const configPath = opts.config ? resolve(opts.config) : resolve(dirname(definitionPath), DEFAULT_CONFIG);

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

  // Parse dynamic --fieldName flags from extra args
  const cliInput: Record<string, unknown> = {};
  let allProvided = true;

  for (const [name, field] of inputFields) {
    const idx = extraArgs.indexOf(`--${name}`);
    if (idx !== -1 && idx + 1 < extraArgs.length) {
      const type = field.def.type;
      cliInput[name] = coerce(extraArgs[idx + 1], type === 'number' ? 'number' : type === 'boolean' ? 'boolean' : 'string');
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

  const spinner = ora({ text: 'Generating…', indent: 2 }).start();

  try {
    const { output, score } = await generateText({ definition, input: cliInput, config });

    spinner.stop();
    console.log(JSON.stringify(output, null, 2));
    if (score != null) {
      console.log(`\n  ${dim('score:')} ${score}`);
    }
  } catch (err: unknown) {
    spinner.fail('Generation failed');
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`${bold('Error:')} ${msg}`);
    process.exit(1);
  }
}

// ── Validate ────────────────────────────────────────────────────────

async function handleValidate(opts: { definition?: string; config?: string }) {
  const definitionPath = await resolveDefinitionPath(opts.definition);
  const configPath = opts.config ? resolve(opts.config) : resolve(dirname(definitionPath), DEFAULT_CONFIG);

  const definition = await loadDefinition(definitionPath);
  validateDefinition(definition, false);

  const raw = await readFile(configPath, 'utf-8');
  const config: ModelConfig = JSON.parse(raw);

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

// ── CLI ─────────────────────────────────────────────────────────────

loadEnvUp(process.cwd());

const program = new Command()
  .name('praxis')
  .description('Define, train, and use optimized LLM prompts')
  .version('0.0.0');

program
  .command('train')
  .description('Optimize prompts from a definition file (auto-discovers via glob)')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-o, --output <path>', 'config output path (default: model.config.json next to definition)')
  .option('--optimizer <type>', 'ace | gepa | auto', 'auto')
  .option('--split <ratio>', 'train/test split', '0.7')
  .action(handleTrain);

program
  .command('run')
  .description('Run the model (auto-discovers definition and config)')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-c, --config <path>', 'config file (default: model.config.json next to definition)')
  .allowUnknownOption()
  .action((opts, cmd) => handleRun(opts, cmd.args));

program
  .command('validate')
  .description('Check that the config matches the definition schema')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-c, --config <path>', 'config file (default: model.config.json next to definition)')
  .action(handleValidate);

program.parseAsync().catch((err) => {
  console.error(`\n  ${bold('Error:')} ${err.message ?? err}\n`);
  process.exit(1);
});
