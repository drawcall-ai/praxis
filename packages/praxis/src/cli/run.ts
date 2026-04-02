import { resolve, dirname } from 'node:path';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { config as loadEnv } from 'dotenv';
import ora from 'ora';
import { generateText } from '../generate.js';
import type { ModelConfig } from '../types.js';
import {
  DEFAULT_CONFIG,
  bold, dim, cyan,
  resolveDefinitionPath, requireEnvKey, loadDefinition,
  validateDefinition, coerce,
  type ZodField,
} from './utils.js';

export async function handleRun(opts: { definition?: string; config?: string }, extraArgs: string[]) {
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
      const scoreStr = typeof score === 'object'
        ? Object.entries(score).map(([k, v]) => `${k}: ${v}`).join(', ')
        : String(score);
      console.log(`\n  ${dim('score:')} ${scoreStr}`);
    }
  } catch (err: unknown) {
    spinner.fail('Generation failed');
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`${bold('Error:')} ${msg}`);
    process.exit(1);
  }
}
