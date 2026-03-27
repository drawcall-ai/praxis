import { z } from 'zod';
import type { PraxisConfig, PraxisConfigSchema, JsonSchema, PromptResult } from './types.js';
import { validateSchema } from './schema.js';

/**
 * Build system and user prompts from a trained config and input values.
 *
 * Pass your definition's output Zod schema as the third argument
 * to get a fully typed `schema.output` in the return value.
 */
export function buildPrompt<T extends z.ZodRawShape>(
  config: PraxisConfig,
  input: Record<string, unknown>,
  outputSchema: z.ZodObject<T>,
): PromptResult<T>;
export function buildPrompt(
  config: PraxisConfig,
  input: Record<string, unknown>,
): PromptResult;
export function buildPrompt(
  config: PraxisConfig,
  input: Record<string, unknown>,
  outputSchema?: z.ZodObject<z.ZodRawShape>,
): PromptResult {
  const { schema, optimization } = config;

  if (outputSchema) {
    validateSchema(outputSchema, schema.output, 'output');
  }

  const system = optimization.instruction;
  const parts: string[] = [];

  if (optimization.demos.length > 0) {
    for (const demo of optimization.demos) {
      parts.push(formatDemo(schema, demo.input, demo.output));
    }
    parts.push('---');
    parts.push('');
  }

  parts.push(formatInput(schema, input));

  const outputFields = getFieldNames(schema.output);
  parts.push('');
  parts.push(`Provide: ${outputFields.join(', ')}`);

  return {
    system,
    user: parts.join('\n'),
    schema: {
      input: jsonSchemaToZod(schema.input),
      output: outputSchema ?? jsonSchemaToZod(schema.output),
    },
  };
}

/**
 * Reconstruct a Zod object schema from a JSON Schema.
 */
export function jsonSchemaToZod(jsonSchema: JsonSchema): z.ZodObject<z.ZodRawShape> {
  const props = (jsonSchema.properties ?? {}) as Record<string, Record<string, unknown>>;
  const required = new Set((jsonSchema.required ?? []) as string[]);
  const shape: z.ZodRawShape = {};

  for (const [key, field] of Object.entries(props)) {
    let zodType: z.ZodTypeAny = fieldToZod(field);

    if (!required.has(key)) {
      zodType = zodType.optional();
    }

    if (field.description) {
      zodType = zodType.describe(field.description as string);
    }

    shape[key] = zodType;
  }

  return z.object(shape);
}

function fieldToZod(field: Record<string, unknown>): z.ZodTypeAny {
  // Enum
  if (field.enum && Array.isArray(field.enum)) {
    return z.enum(field.enum as [string, ...string[]]);
  }

  switch (field.type) {
    case 'number':
    case 'integer':
      return z.number();
    case 'boolean':
      return z.boolean();
    case 'array': {
      const items = field.items as Record<string, unknown> | undefined;
      if (items) return z.array(fieldToZod(items));
      return z.array(z.unknown());
    }
    case 'object': {
      const nested = field.properties as Record<string, Record<string, unknown>> | undefined;
      if (nested) {
        const shape: z.ZodRawShape = {};
        for (const [k, v] of Object.entries(nested)) {
          shape[k] = fieldToZod(v);
          if (v.description) shape[k] = shape[k].describe(v.description as string);
        }
        return z.object(shape);
      }
      return z.record(z.unknown());
    }
    default:
      return z.string();
  }
}

// ── Formatting helpers ───────────────────────────────────────────────

function getFieldNames(jsonSchema: JsonSchema): string[] {
  const props = jsonSchema.properties as Record<string, unknown> | undefined;
  return props ? Object.keys(props) : [];
}

function formatDemo(
  schema: PraxisConfigSchema,
  input: Record<string, unknown>,
  output: Record<string, unknown>,
): string {
  const lines: string[] = [];
  for (const key of getFieldNames(schema.input)) lines.push(`[${formatLabel(key)}] ${input[key] ?? ''}`);
  for (const key of getFieldNames(schema.output)) lines.push(`[${formatLabel(key)}] ${output[key] ?? ''}`);
  return lines.join('\n') + '\n';
}

function formatInput(schema: PraxisConfigSchema, input: Record<string, unknown>): string {
  return getFieldNames(schema.input).map((key) => `[${formatLabel(key)}] ${input[key] ?? ''}`).join('\n');
}

function formatLabel(key: string): string {
  return key.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
