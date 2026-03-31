import { resolve, dirname, relative } from 'node:path';
import { readFile } from 'node:fs/promises';
import { detectMismatches } from '../validate.js';
import type { ModelConfig } from '../types.js';
import {
  DEFAULT_CONFIG,
  bold, dim, green, red,
  resolveDefinitionPaths, loadDefinition, validateDefinition,
} from './utils.js';

export async function handleValidate(opts: { definition?: string; config?: string }) {
  if (opts.config && !opts.definition) {
    const paths = await resolveDefinitionPaths();
    if (paths.length > 1) {
      console.error(`\n  ${bold('-c')} cannot be used with multiple definitions. Pass ${bold('-d')} to select one.\n`);
      process.exit(1);
    }
  }

  const definitionPaths = await resolveDefinitionPaths(opts.definition);
  let hasFailure = false;

  for (const definitionPath of definitionPaths) {
    const configPath = opts.config ? resolve(opts.config) : resolve(dirname(definitionPath), DEFAULT_CONFIG);
    const label = relative(process.cwd(), definitionPath);

    const definition = await loadDefinition(definitionPath);
    validateDefinition(definition, false);

    let raw: string;
    try {
      raw = await readFile(configPath, 'utf-8');
    } catch {
      console.log(`  ${red('✗')} ${bold(label)} — config not found`);
      hasFailure = true;
      continue;
    }
    const config: ModelConfig = JSON.parse(raw);

    const mismatches = detectMismatches(definition, config);

    if (mismatches.length === 0) {
      console.log(`  ${green('✓')} ${dim(label)} config is in sync`);
    } else {
      hasFailure = true;
      console.log(`  ${red('✗')} ${bold(label)}`);
      for (const m of mismatches) {
        console.log(`    ${red('✗')} ${m.field}: ${dim(m.expected)} ≠ ${dim(m.actual)}`);
      }
    }
  }

  if (hasFailure) {
    console.log(`\n  Run ${bold('npx praxis train')} to fix.\n`);
    process.exit(1);
  }
}
