import { z } from 'zod';
import { zodToTs, printNode, createAuxiliaryTypeStore } from 'zod-to-ts';
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
 * Uses zod-to-ts for correct, complete TypeScript type rendering.
 */
export function formatSchemaForPrompt(source: SchemaSource): string {
  const inputType = printNode(zodToTs(source.input, { auxiliaryTypeStore: createAuxiliaryTypeStore() }).node);
  const outputType = printNode(zodToTs(source.output, { auxiliaryTypeStore: createAuxiliaryTypeStore() }).node);

  return [
    `Input: ${inputType}`,
    '',
    `Respond as JSON matching: ${outputType}`,
  ].join('\n');
}
