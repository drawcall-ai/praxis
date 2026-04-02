import { generateText as aiGenerateText, Output, wrapLanguageModel, extractJsonMiddleware } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import type { z } from 'zod';
import type { ModelDefinition, ModelConfig, ModelRequest } from './types.js';
import { buildRequest } from './prompt.js';

type GenerateTextOptions = Parameters<typeof aiGenerateText>[0];

/** All AI SDK options minus the ones Praxis owns (model, schema, system, prompt, output). */
export type GenerateOptions = Omit<
  GenerateTextOptions,
  'model' | 'system' | 'prompt' | 'messages' | 'output' | 'tools' | 'toolChoice'
>;

export interface GenerateResult<O extends z.ZodRawShape = z.ZodRawShape> {
  output: z.infer<z.ZodObject<O>>;
  request: ModelRequest<O>;
  score?: Record<string, number> | null;
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

  const model = wrapLanguageModel({
    model: openrouter.chat(prompt.student),
    middleware: extractJsonMiddleware(),
  });

  const result = await aiGenerateText({
    ...options,
    model,
    messages: prompt.messages,
    output: Output.object({ schema: prompt.schema.output }),
    ...(config?.optimization.temperature != null ? { temperature: config.optimization.temperature } : {}),
    providerOptions: {
      openrouter: {
        reasoning: {
          effort: config?.optimization.reasoningEffort ?? 'minimal',
          exclude: false,
        },
      },
    },
  });

  const object = result.output as z.infer<z.ZodObject<O>>;

  const score = definition.metric
    ? definition.metric({ input, modelOutput: object }) ?? null
    : undefined;

  return { output: object, request: prompt, score };
}
