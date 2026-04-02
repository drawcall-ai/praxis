import { z } from 'zod';
import type { PraxisConfigSchema, JsonSchema } from './types.js';

interface SchemaSource {
  input: z.ZodObject<z.ZodRawShape>;
  output: z.ZodObject<z.ZodRawShape>;
  description?: string;
}

/**
 * Convert a definition's Zod schemas to JSON Schema for storage in model.config.json.
 */
export function serializeSchema(source: SchemaSource): PraxisConfigSchema {
  return {
    input: z.toJSONSchema(source.input) as JsonSchema,
    output: z.toJSONSchema(source.output) as JsonSchema,
  };
}

/**
 * Validate that a Zod schema matches the JSON Schema stored in a config.
 */
export function validateSchema(
  zodSchema: z.ZodObject<z.ZodRawShape>,
  configJsonSchema: JsonSchema,
  label: string,
): void {
  const current = z.toJSONSchema(zodSchema) as JsonSchema;
  const currentStr = JSON.stringify(current, null, 2);
  const configStr = JSON.stringify(configJsonSchema, null, 2);

  if (currentStr !== configStr) {
    throw new Error(
      `Schema mismatch (${label}): the definition schema does not match the trained config.\n` +
      `Run \`npx praxis train\` to retrain.`,
    );
  }
}

/**
 * Format the input/output schema for inclusion in a system prompt.
 * Produces a human-readable description of each field.
 */
export function formatSchemaForPrompt(source: SchemaSource): string {
  const lines: string[] = [];

  lines.push('Input:');
  for (const [key, zodType] of Object.entries(source.input.shape)) {
    lines.push(`  ${key}: ${describeZodType(zodType as z.ZodTypeAny)}`);
  }

  lines.push('');
  lines.push('Output (respond as JSON):');
  for (const [key, zodType] of Object.entries(source.output.shape)) {
    lines.push(`  ${key}: ${describeZodType(zodType as z.ZodTypeAny)}`);
  }

  return lines.join('\n');
}

interface ZodDef {
  type: string;
  innerType?: z.ZodTypeAny;
  entries?: Record<string, string>;
}

function describeZodType(zodType: z.ZodTypeAny): string {
  const description = zodType.description;
  const def = (zodType as unknown as { def: ZodDef }).def;
  const defType = def.type;

  if (defType === 'optional' || defType === 'nullable' || defType === 'default') {
    return describeZodType(def.innerType as z.ZodTypeAny);
  }

  let typeStr: string;
  if (defType === 'enum') {
    const values = Object.keys(def.entries as Record<string, string>);
    typeStr = values.map((v) => `"${v}"`).join(' | ');
  } else if (defType === 'number') {
    typeStr = 'number';
  } else if (defType === 'boolean') {
    typeStr = 'boolean';
  } else if (defType === 'array') {
    typeStr = 'array';
  } else {
    typeStr = 'string';
  }

  return description ? `${typeStr} — ${description}` : typeStr;
}
