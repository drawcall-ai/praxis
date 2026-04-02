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
 * Uses JSON Schema (via Zod v4) for precise, LLM-native type descriptions.
 */
export function formatSchemaForPrompt(source: SchemaSource): string {
  const inputSchema = z.toJSONSchema(source.input);
  const outputSchema = z.toJSONSchema(source.output);

  return [
    'Input schema:',
    '```json',
    JSON.stringify(inputSchema, null, 2),
    '```',
    '',
    'Output schema (respond as JSON matching this schema):',
    '```json',
    JSON.stringify(outputSchema, null, 2),
    '```',
  ].join('\n');
}
