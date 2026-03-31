import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { AxGen, AxMockAIService } from '@ax-llm/ax';
import type { AxFieldValue, AxProgramDemos, AxChatRequest } from '@ax-llm/ax';
import { defineModel } from '../src/define.js';
import { buildRequest } from '../src/prompt.js';
import { toAxSignature } from '../src/schema.js';
import type { ModelConfig } from '../src/types.js';

const definition = defineModel({
  student: 'google/gemini-3-flash-preview',
  description: 'Analyze product reviews to determine sentiment and confidence.',
  input: z.object({
    reviewText: z.string().describe('The text of the product review to analyze'),
  }),
  output: z.object({
    sentiment: z
      .enum(['positive', 'negative', 'neutral'])
      .describe('The overall sentiment of the review'),
    confidence: z.number().describe('Confidence score between 0 and 1'),
  }),
  examples: [],
});

const trainedConfig: ModelConfig = {
  version: '1.0',
  student: 'google/gemini-3-flash-preview',
  schema: { input: {}, output: {} },
  optimization: {
    optimizer: 'ace',
    instruction:
      'You are an expert sentiment analyzer. Classify reviews precisely.',
    demos: [
      {
        input: { reviewText: 'Love it!' },
        output: { sentiment: 'positive', confidence: 0.95 },
      },
      {
        input: { reviewText: 'Terrible.' },
        output: { sentiment: 'negative', confidence: 0.9 },
      },
    ],
    bestScore: 0.95,
  },
};

/**
 * Capture the exact chatPrompt that AxGen.forward() sends to the AI service.
 * This is the ground truth — the same code path used during training.
 */
async function captureAxRequest(
  input: Record<string, AxFieldValue>,
  config?: ModelConfig,
) {
  const program = new AxGen(toAxSignature(definition));
  program.setDescription(definition.description!);

  if (config?.optimization.demos.length) {
    const traces = config.optimization.demos.map(
      (d) => ({ ...d.input, ...d.output }) as Record<string, AxFieldValue>,
    );
    program.setDemos([
      { traces, programId: 'root' },
    ] as AxProgramDemos<any, any>[]);
  }

  let captured!: AxChatRequest['chatPrompt'];

  const interceptor = new AxMockAIService<string>({
    chatResponse: (req: Readonly<AxChatRequest<unknown>>) => {
      captured = req.chatPrompt as AxChatRequest['chatPrompt'];
      return Promise.resolve({
        results: [
          { index: 0, content: 'Sentiment: positive\nConfidence: 0.95' },
        ],
      });
    },
  });

  await program.forward(interceptor, input);
  return captured;
}

function normalize(msgs: any[]) {
  return msgs.map((m: any) => ({
    role: m.role as string,
    content:
      typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
  }));
}

describe('buildRequest', () => {
  it('uses definition model when no config is provided', () => {
    const req = buildRequest(definition, { reviewText: 'test' });
    expect(req.student).toBe('google/gemini-3-flash-preview');
  });

  it('uses config model when provided', () => {
    const overrideConfig: ModelConfig = {
      ...trainedConfig,
      student: 'openai/gpt-4o',
    };
    const req = buildRequest(
      definition,
      { reviewText: 'test' },
      overrideConfig,
    );
    expect(req.student).toBe('openai/gpt-4o');
  });
});

describe('buildRequest parity with AxGen.forward()', () => {
  it('produces identical messages for default (no config)', async () => {
    const input = { reviewText: 'Great product!' };

    const praxis = buildRequest(definition, input).messages;
    const axPrompt = await captureAxRequest(input);

    expect(normalize(praxis)).toEqual(normalize(axPrompt));
  });

  it('produces identical messages for trained config with demos', async () => {
    const input = { reviewText: 'Decent for the price.' };

    const praxis = buildRequest(definition, input, trainedConfig).messages;
    const axPrompt = await captureAxRequest(input, trainedConfig);

    expect(normalize(praxis)).toEqual(normalize(axPrompt));
  });
});
