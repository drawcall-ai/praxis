import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { defineModel } from '../src/define.js';
import { buildRequest, buildDefaultSystemPrompt } from '../src/prompt.js';
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
    instruction:
      'You are an expert sentiment analyzer. Classify reviews precisely.',
    bestScore: { accuracy: 0.95 },
  },
};

describe('buildDefaultSystemPrompt', () => {
  it('includes description', () => {
    const prompt = buildDefaultSystemPrompt(definition);
    expect(prompt).toContain('Analyze product reviews');
  });

  it('includes output schema fields', () => {
    const prompt = buildDefaultSystemPrompt(definition);
    expect(prompt).toContain('sentiment');
    expect(prompt).toContain('confidence');
  });

  it('includes enum values', () => {
    const prompt = buildDefaultSystemPrompt(definition);
    expect(prompt).toContain('"positive"');
    expect(prompt).toContain('"negative"');
    expect(prompt).toContain('"neutral"');
  });
});

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
    const req = buildRequest(definition, { reviewText: 'test' }, overrideConfig);
    expect(req.student).toBe('openai/gpt-4o');
  });

  it('uses default system prompt when no config', () => {
    const req = buildRequest(definition, { reviewText: 'test' });
    expect(req.messages[0].role).toBe('system');
    expect(req.messages[0].content).toContain('Analyze product reviews');
  });

  it('uses trained instruction when config is provided', () => {
    const req = buildRequest(definition, { reviewText: 'test' }, trainedConfig);
    expect(req.messages[0].role).toBe('system');
    expect(req.messages[0].content).toBe(trainedConfig.optimization.instruction);
  });

  it('formats user input as key-value pairs', () => {
    const req = buildRequest(definition, { reviewText: 'Great product!' });
    const userMsg = req.messages.find((m) => m.role === 'user');
    expect(userMsg?.content).toContain('reviewText: Great product!');
  });

  it('produces system + user message structure', () => {
    const req = buildRequest(definition, { reviewText: 'test' });
    expect(req.messages).toHaveLength(2);
    expect(req.messages[0].role).toBe('system');
    expect(req.messages[1].role).toBe('user');
  });
});
