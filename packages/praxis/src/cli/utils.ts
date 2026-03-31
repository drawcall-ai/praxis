import { resolve, dirname } from 'node:path';
import { glob } from 'tinyglobby';
import { config as loadEnv } from 'dotenv';
import { createServer } from 'vite';
import { ViteNodeServer } from 'vite-node/server';
import { ViteNodeRunner } from 'vite-node/client';
import type { ModelDefinition } from '../types.js';

export const DEFAULT_CONFIG = 'model.config.json';

const DEFINITION_GLOBS = ['**/model.definition.ts', '**/model.definition.js'];

// ── Colors ──────────────────────────────────────────────────────────

export const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
export const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
export const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
export const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
export const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;

// ── Helpers ─────────────────────────────────────────────────────────

async function findDefinitions(): Promise<string[]> {
  const matches = await glob(DEFINITION_GLOBS, {
    ignore: ['**/node_modules/**', '**/dist/**'],
    absolute: true,
  });
  if (matches.length === 0) {
    console.error(`\n  Could not find model.definition.ts or model.definition.js\n`);
    process.exit(1);
  }
  return matches;
}

async function requireSingleDefinition(flag?: string): Promise<string> {
  if (flag) return resolve(flag);
  const matches = await findDefinitions();
  if (matches.length > 1) {
    console.error(`\n  Found multiple definition files:\n`);
    for (const m of matches) console.error(`    ${m}`);
    console.error(`\n  Pass the path explicitly with -d to disambiguate.\n`);
    process.exit(1);
  }
  return matches[0];
}

export function loadEnvUp(from: string) {
  let dir = resolve(from);
  for (let i = 0; i < 10; i++) {
    loadEnv({ path: resolve(dir, '.env') });
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
}

export function requireEnvKey(): string {
  const key = process.env.OPENROUTER_KEY ?? process.env.OPENROUTER_API_KEY;
  if (!key) {
    console.error(`\n  ${bold('OPENROUTER_KEY')} is not set.\n`);
    console.error(`  Add it to your ${dim('.env')} file:`);
    console.error(`  OPENROUTER_KEY=sk-or-...\n`);
    process.exit(1);
  }
  return key;
}

export async function resolveDefinitionPath(flag?: string): Promise<string> {
  return requireSingleDefinition(flag);
}

export async function resolveDefinitionPaths(flag?: string): Promise<string[]> {
  if (flag) return [resolve(flag)];
  return findDefinitions();
}

export interface ZodField {
  def: { type: string; entries?: Record<string, string> };
  description?: string;
}

export function formatZodSchema(definition: ModelDefinition): string {
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

export function validateDefinition(def: ModelDefinition, requireMetric: boolean) {
  if (!def.input || !def.output)
    throw new Error('Definition must export "input" and "output" Zod objects');
  if (!def.student || typeof def.student !== 'string')
    throw new Error('Definition must export a "student" string');
  const ex = def.examples;
  if (!Array.isArray(ex) && typeof ex !== 'function')
    throw new Error('Definition must export "examples" as an array or async function');
  if (requireMetric && !def.metric)
    throw new Error('Definition must export a "metric" function');
}

export async function loadDefinition(filePath: string): Promise<ModelDefinition> {
  const viteServer = await createServer({
    root: dirname(filePath),
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

export function coerce(value: string, type: string): unknown {
  if (type === 'number') return parseFloat(value);
  if (type === 'boolean') return value === 'true' || value === '1' || value === 'yes';
  return value;
}
