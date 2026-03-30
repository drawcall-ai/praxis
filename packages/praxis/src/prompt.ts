import { z } from 'zod';
import { AxSignature, AxPromptTemplate } from '@ax-llm/ax';
import type { AxFieldValue } from '@ax-llm/ax';
import type { ModelMessage } from 'ai';
import type { ModelDefinition, ModelConfig, ModelRequest, JsonSchema } from './types.js';
import { toAxSignature } from './schema.js';

/**
 * Build a request from a definition and input.
 * Uses ax's AxPromptTemplate.render() — the same renderer ax uses internally
 * during training — to produce the exact prompt with 100% parity.
 */
export function buildRequest<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  definition: ModelDefinition<I, O>,
  input: z.infer<z.ZodObject<I>>,
  config?: ModelConfig,
): ModelRequest<O> {
  const sig = AxSignature.create(toAxSignature(definition));
  if (definition.description) sig.setDescription(definition.description);

  const template = new AxPromptTemplate(sig);
  const demos = config?.optimization.demos.map(
    (d) => ({ ...d.input, ...d.output }) as Record<string, AxFieldValue>,
  );

  const rendered = template.render(input as Record<string, AxFieldValue>, { demos });

  const messages: ModelMessage[] = [];
  for (const msg of rendered) {
    if (msg.role === 'system') {
      messages.push({ role: 'system', content: msg.content as string });
    } else if (msg.role === 'assistant') {
      messages.push({ role: 'assistant', content: msg.content as string });
    } else if (msg.role === 'user') {
      const content = msg.content;
      messages.push({
        role: 'user',
        content: typeof content === 'string' ? content : JSON.stringify(content),
      });
    }
  }

  return {
    messages,
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
