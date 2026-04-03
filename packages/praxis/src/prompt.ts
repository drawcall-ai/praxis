import { z } from 'zod';
import type { ModelMessage } from 'ai';
import type { ModelDefinition, ModelConfig, ModelRequest, JsonSchema } from './types.js';
import { formatSchemaForPrompt } from './schema.js';

/**
 * Build the default system prompt from a model definition.
 * This is the starting point for the agentic optimizer.
 */
export function buildDefaultSystemPrompt<I extends z.ZodRawShape, O extends z.ZodRawShape>(definition: ModelDefinition<I, O>): string {
  const parts: string[] = [];

  if (definition.description) {
    parts.push(definition.description);
  }

  parts.push(formatSchemaForPrompt(definition));

  return parts.join('\n\n');
}

/**
 * Build a request from a definition and input.
 * If a trained config is provided, its instruction replaces the default system prompt.
 */
export function buildRequest<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  definition: ModelDefinition<I, O>,
  input: z.infer<z.ZodObject<I>>,
  config?: ModelConfig,
): ModelRequest<O> {
  const system = config?.optimization.instruction || buildDefaultSystemPrompt(definition);

  const userContent = JSON.stringify(input);

  const messages: ModelMessage[] = [
    { role: 'system', content: system },
    { role: 'user', content: userContent },
  ];

  return {
    messages,
    schema: {
      input: definition.input as z.ZodObject<z.ZodRawShape>,
      output: definition.output,
    },
    student: config?.student ?? definition.student,
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
