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
 * Convert a JSON Schema property to a compact TypeScript-like type string.
 */
function jsonSchemaTypeToTS(field: Record<string, unknown>): string {
  if (field.enum && Array.isArray(field.enum)) {
    return (field.enum as unknown[]).map((v) => JSON.stringify(v)).join(' | ');
  }
  switch (field.type) {
    case 'integer':
    case 'number':
      return 'number';
    case 'boolean':
      return 'boolean';
    case 'array': {
      const items = field.items as Record<string, unknown> | undefined;
      if (items) {
        const inner = jsonSchemaTypeToTS(items);
        return inner.includes(' | ') || inner.includes('{') ? `(${inner})[]` : `${inner}[]`;
      }
      return 'unknown[]';
    }
    case 'object': {
      const nested = field.properties as Record<string, Record<string, unknown>> | undefined;
      if (nested) {
        const req = new Set((field.required ?? []) as string[]);
        return formatObjectFields(nested, req);
      }
      return 'Record<string, unknown>';
    }
    default:
      return 'string';
  }
}

/**
 * Format an object's fields as TypeScript-like notation with descriptions as comments.
 */
function formatObjectFields(
  properties: Record<string, Record<string, unknown>>,
  required: Set<string>,
  indent = '  ',
): string {
  const lines: string[] = ['{'];
  for (const [key, field] of Object.entries(properties)) {
    const type = jsonSchemaTypeToTS(field);
    const optional = required.has(key) ? '' : '?';
    const desc = field.description ? `  // ${field.description}` : '';
    lines.push(`${indent}${key}${optional}: ${type}${desc}`);
  }
  lines.push(indent.slice(2) + '}');
  return lines.join('\n');
}

/**
 * Format the input/output schema for inclusion in a system prompt.
 * Uses compact TypeScript-like type definitions for token efficiency.
 */
export function formatSchemaForPrompt(source: SchemaSource): string {
  const inputJsonSchema = z.toJSONSchema(source.input) as Record<string, unknown>;
  const outputJsonSchema = z.toJSONSchema(source.output) as Record<string, unknown>;

  const inputReq = new Set((inputJsonSchema.required ?? []) as string[]);
  const outputReq = new Set((outputJsonSchema.required ?? []) as string[]);

  const inputFields = formatObjectFields(
    inputJsonSchema.properties as Record<string, Record<string, unknown>>,
    inputReq,
  );
  const outputFields = formatObjectFields(
    outputJsonSchema.properties as Record<string, Record<string, unknown>>,
    outputReq,
  );

  return [
    `Input: ${inputFields}`,
    '',
    `Respond as JSON matching: ${outputFields}`,
  ].join('\n');
}
