import { generateObject } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import type { z } from 'zod';
import type { ModelDefinition, ModelConfig, ModelRequest } from './types.js';
import { buildRequest } from './prompt.js';

type GenerateObjectOptions = Parameters<typeof generateObject>[0];

/** All AI SDK options minus the ones Praxis owns (model, schema, system, prompt). */
export type GenerateOptions = Omit<
  GenerateObjectOptions,
  'model' | 'schema' | 'system' | 'prompt' | 'output' | 'schemaName' | 'schemaDescription'
>;

export interface GenerateResult<O extends z.ZodRawShape = z.ZodRawShape> {
  object: z.infer<z.ZodObject<O>>;
  prompt: ModelRequest<O>;
  score?: number | Record<string, number> | null;
}

/**
 * Generate structured output using a model definition and optional trained config.
 * If the definition has a metric, it runs against the result and returns the score.
 */
export async function generateText<I extends z.ZodRawShape, O extends z.ZodRawShape>({
  definition,
  input,
  config,
  ...options
}: {
  definition: ModelDefinition<I, O>;
  input: z.infer<z.ZodObject<I>>;
  config?: ModelConfig;
} & GenerateOptions): Promise<GenerateResult<O>> {
  const prompt = buildRequest(definition, input, config);

  const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_KEY ?? process.env.OPENROUTER_API_KEY,
  });

  const result = await generateObject({
    ...options,
    model: openrouter.chat(prompt.model),
    system: prompt.system,
    prompt: prompt.user,
    schema: prompt.schema.output,
  });

  const object = result.object as z.infer<z.ZodObject<O>>;

  const score = definition.metric
    ? definition.metric({ input, modelOutput: object }) ?? null
    : undefined;

  return { object, prompt, score };
}
