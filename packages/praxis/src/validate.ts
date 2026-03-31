import { z } from 'zod';
import type { ModelDefinition, ModelConfig, JsonSchema } from './types.js';

export interface Mismatch {
  field: string;
  expected: string;
  actual: string;
}

function schemaStr(schema: z.ZodObject<z.ZodRawShape>): string {
  return JSON.stringify(z.toJSONSchema(schema));
}

/**
 * Compare a definition against a trained config and return all mismatches.
 * Checks: input schema, output schema, model, teacher, and version (when set).
 */
export function detectMismatches(
  definition: ModelDefinition,
  config: ModelConfig,
): Mismatch[] {
  const mismatches: Mismatch[] = [];

  const currentInput = schemaStr(definition.input);
  const configInput = JSON.stringify(config.schema.input);
  if (currentInput !== configInput) {
    mismatches.push({ field: 'input schema', expected: 'definition', actual: 'config (run train)' });
  }

  const currentOutput = schemaStr(definition.output);
  const configOutput = JSON.stringify(config.schema.output);
  if (currentOutput !== configOutput) {
    mismatches.push({ field: 'output schema', expected: 'definition', actual: 'config (run train)' });
  }

  if (definition.model !== config.model) {
    mismatches.push({ field: 'model', expected: definition.model, actual: config.model });
  }

  const defTeacher = definition.teacher ?? undefined;
  const cfgTeacher = config.teacher ?? undefined;
  if (defTeacher !== cfgTeacher) {
    mismatches.push({
      field: 'teacher',
      expected: defTeacher ?? 'none',
      actual: cfgTeacher ?? 'none',
    });
  }

  if (definition.version != null && definition.version !== config.version) {
    mismatches.push({
      field: 'version',
      expected: definition.version,
      actual: config.version,
    });
  }

  return mismatches;
}
