import { z } from 'zod';
import type { ModelDefinition, ModelConfig, ModelRequest, PraxisConfigSchema, JsonSchema } from './types.js';
import { serializeSchema } from './schema.js';

/**
 * Build a request from a definition and input.
 * Optionally pass a trained config to use optimized instruction and demos.
 */
export function buildRequest<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  definition: ModelDefinition<I, O>,
  input: z.infer<z.ZodObject<I>>,
  config?: ModelConfig,
): ModelRequest<O> {
  const jsonSchema = config?.schema ?? serializeSchema({ input: definition.input, output: definition.output });

  let system: string;
  const parts: string[] = [];

  if (config) {
    system = config.optimization.instruction;

    if (config.optimization.demos.length > 0) {
      for (const demo of config.optimization.demos) {
        parts.push(formatDemo(jsonSchema, demo.input, demo.output));
      }
      parts.push('---');
      parts.push('');
    }
  } else {
    system = generateDefaultInstruction({ input: definition.input, output: definition.output });
  }

  parts.push(formatInput(jsonSchema, input as Record<string, unknown>));

  const outputFields = getFieldNames(jsonSchema.output);
  parts.push('');
  parts.push(`Provide: ${outputFields.join(', ')}`);

  return {
    system,
    user: parts.join('\n'),
    schema: {
      input: definition.input as z.ZodObject<z.ZodRawShape>,
      output: definition.output,
    },
    model: config?.model ?? definition.model,
    metric: definition.metric as ModelRequest['metric'],
  };
}

/**
 * Reconstruct a Zod object schema from a JSON Schema.
 */
export function jsonSchemaToZod(jsonSchema: JsonSchema): z.ZodObject<z.ZodRawShape> {
  const props = (jsonSchema.properties ?? {}) as Record<string, Record<string, unknown>>;
  const required = new Set((jsonSchema.required ?? []) as string[]);
  const shape: Record<string, z.ZodTypeAny> = {};

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
        const shape: Record<string, z.ZodTypeAny> = {};
        for (const [k, v] of Object.entries(nested)) {
          shape[k] = fieldToZod(v);
          if (v.description) shape[k] = shape[k].describe(v.description as string);
        }
        return z.object(shape);
      }
      return z.record(z.string(), z.unknown());
    }
    default:
      return z.string();
  }
}

// ── Helpers ─────────────────────────────────────────────────────────

function generateDefaultInstruction(def: { input: z.ZodObject<z.ZodRawShape>; output: z.ZodObject<z.ZodRawShape> }): string {
  const inputFields = Object.entries(def.input.shape)
    .map(([k, v]) => {
      const desc = (v as z.ZodTypeAny).description;
      return desc ? `${k} (${desc})` : k;
    })
    .join(', ');

  const outputLines = Object.entries(def.output.shape)
    .map(([k, v]) => {
      const zt = v as z.ZodTypeAny;
      const d = zt._def as unknown as Record<string, unknown>;
      const defType = d.type as string;
      const desc = zt.description ? ` — ${zt.description}` : '';
      if (defType === 'enum') {
        const values = Object.keys(d.entries as Record<string, string>).map(v => `"${v}"`).join(' | ');
        return `  ${k}: ${values}${desc}`;
      }
      return `  ${k}: ${defType}${desc}`;
    });

  return `Given the input (${inputFields}), produce the output as a JSON object with these fields:\n${outputLines.join('\n')}\n\nBe precise and follow the output schema exactly.`;
}

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
