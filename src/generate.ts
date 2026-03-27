import type { z } from 'zod';
import type { PraxisConfig, PromptResult } from './types.js';
import { buildPrompt } from './prompt.js';

export interface GenerateOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

/**
 * Generate structured output using the trained config, AI SDK, and OpenRouter.
 *
 * Pass your definition's output Zod schema as the third argument
 * to get a fully typed `object` in the return value.
 */
export async function generateText<T extends z.ZodRawShape>(
  config: PraxisConfig,
  input: Record<string, unknown>,
  outputSchema: z.ZodObject<T>,
  options?: GenerateOptions,
): Promise<{ object: z.infer<z.ZodObject<T>>; prompt: PromptResult<T> }>;
export async function generateText(
  config: PraxisConfig,
  input: Record<string, unknown>,
  options?: GenerateOptions,
): Promise<{ object: Record<string, unknown>; prompt: PromptResult }>;
export async function generateText(
  config: PraxisConfig,
  input: Record<string, unknown>,
  schemaOrOptions?: z.ZodObject<z.ZodRawShape> | GenerateOptions,
  maybeOptions?: GenerateOptions,
): Promise<{ object: Record<string, unknown>; prompt: PromptResult }> {
  const { generateObject } = await import('ai');
  const { createOpenRouter } = await import('@openrouter/ai-sdk-provider');

  // Disambiguate overloads
  const isZod = schemaOrOptions && 'shape' in schemaOrOptions;
  const outputSchema = isZod ? schemaOrOptions as z.ZodObject<z.ZodRawShape> : undefined;
  const options = isZod ? maybeOptions : schemaOrOptions as GenerateOptions | undefined;

  const prompt = outputSchema
    ? buildPrompt(config, input, outputSchema)
    : buildPrompt(config, input);

  const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_KEY ?? process.env.OPENROUTER_API_KEY,
  });

  const result = await generateObject({
    model: openrouter.chat(options?.model ?? config.model),
    system: prompt.system,
    prompt: prompt.user,
    schema: prompt.schema.output,
    temperature: options?.temperature,
    maxTokens: options?.maxTokens,
  });

  return {
    object: result.object as Record<string, unknown>,
    prompt,
  };
}
