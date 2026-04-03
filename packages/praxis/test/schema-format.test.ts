import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { formatSchemaForPrompt } from '../src/schema.js';

describe('formatSchemaForPrompt', () => {
  it('formats string input + enum/number output with descriptions', () => {
    const source = {
      input: z.object({
        reviewText: z.string().describe('The text of the product review to analyze'),
      }),
      output: z.object({
        sentiment: z
          .enum(['positive', 'negative', 'neutral'])
          .describe('The overall sentiment of the review'),
        confidence: z.number().describe('Confidence score between 0 and 1'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** The text of the product review to analyze */',
        '    reviewText: string;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** The overall sentiment of the review */',
        '    sentiment: "positive" | "negative" | "neutral";',
        '    /** Confidence score between 0 and 1 */',
        '    confidence: number;',
        '}',
      ].join('\n'),
    );
  });

  it('formats multiple enum fields and booleans', () => {
    const source = {
      input: z.object({
        reviewText: z.string().describe('The product review to evaluate'),
      }),
      output: z.object({
        quality: z
          .enum(['helpful', 'unhelpful'])
          .describe('Whether the review is helpful to other buyers'),
        sentiment: z
          .enum(['positive', 'negative', 'neutral'])
          .describe('The overall sentiment of the review'),
        hasSpecificDetails: z
          .boolean()
          .describe('Whether the review mentions specific product details'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** The product review to evaluate */',
        '    reviewText: string;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** Whether the review is helpful to other buyers */',
        '    quality: "helpful" | "unhelpful";',
        '    /** The overall sentiment of the review */',
        '    sentiment: "positive" | "negative" | "neutral";',
        '    /** Whether the review mentions specific product details */',
        '    hasSpecificDetails: boolean;',
        '}',
      ].join('\n'),
    );
  });

  it('formats nested objects and arrays (ARC-style)', () => {
    const Grid = z
      .array(z.array(z.number().int().min(0).max(9)))
      .describe('A 2D grid of integers 0-9');
    const Demonstration = z.object({
      input: Grid.describe('Input grid for this demonstration'),
      output: Grid.describe('Output grid showing the transformation result'),
    });

    const source = {
      input: z.object({
        demonstrations: z
          .array(Demonstration)
          .describe('A list of input/output pairs'),
        testInput: Grid.describe('The test input grid'),
      }),
      output: z.object({
        reasoning: z.string().describe('Step-by-step explanation'),
        grid: Grid.describe('The predicted output grid'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** A list of input/output pairs */',
        '    demonstrations: {',
        '        /** Input grid for this demonstration */',
        '        input: number[][];',
        '        /** Output grid showing the transformation result */',
        '        output: number[][];',
        '    }[];',
        '    /** The test input grid */',
        '    testInput: number[][];',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** Step-by-step explanation */',
        '    reasoning: string;',
        '    /** The predicted output grid */',
        '    grid: number[][];',
        '}',
      ].join('\n'),
    );
  });

  it('marks optional fields with ?', () => {
    const source = {
      input: z.object({
        name: z.string().describe('User name'),
        email: z.string().optional().describe('Email address'),
      }),
      output: z.object({
        greeting: z.string().describe('Generated greeting'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** User name */',
        '    name: string;',
        '    /** Email address */',
        '    email?: string | undefined;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** Generated greeting */',
        '    greeting: string;',
        '}',
      ].join('\n'),
    );
  });

  it('handles nullable fields', () => {
    const source = {
      input: z.object({
        title: z.string().describe('Title'),
      }),
      output: z.object({
        category: z.string().nullable().describe('Category or null'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** Title */',
        '    title: string;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** Category or null */',
        '    category: string | null;',
        '}',
      ].join('\n'),
    );
  });

  it('formats fields without descriptions', () => {
    const source = {
      input: z.object({ x: z.number() }),
      output: z.object({ y: z.string(), flag: z.boolean() }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    x: number;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    y: string;',
        '    flag: boolean;',
        '}',
      ].join('\n'),
    );
  });

  it('formats simple arrays', () => {
    const source = {
      input: z.object({
        tags: z.array(z.string()).describe('List of tags'),
      }),
      output: z.object({
        scores: z.array(z.number()).describe('Score per tag'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** List of tags */',
        '    tags: string[];',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** Score per tag */',
        '    scores: number[];',
        '}',
      ].join('\n'),
    );
  });

  it('formats nested object fields', () => {
    const source = {
      input: z.object({
        user: z.object({
          name: z.string().describe('Full name'),
          age: z.number().describe('Age in years'),
        }),
      }),
      output: z.object({
        result: z.string(),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    user: {',
        '        /** Full name */',
        '        name: string;',
        '        /** Age in years */',
        '        age: number;',
        '    };',
        '}',
        '',
        'Respond as JSON matching: {',
        '    result: string;',
        '}',
      ].join('\n'),
    );
  });

  it('formats array of enums with parentheses', () => {
    const source = {
      input: z.object({ id: z.string() }),
      output: z.object({
        labels: z
          .array(z.enum(['bug', 'feature', 'docs']))
          .describe('Applicable labels'),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    id: string;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    /** Applicable labels */',
        '    labels: ("bug" | "feature" | "docs")[];',
        '}',
      ].join('\n'),
    );
  });

  it('formats union types', () => {
    const source = {
      input: z.object({
        value: z.union([z.string(), z.number()]).describe('A string or number'),
      }),
      output: z.object({
        result: z.string(),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** A string or number */',
        '    value: string | number;',
        '}',
        '',
        'Respond as JSON matching: {',
        '    result: string;',
        '}',
      ].join('\n'),
    );
  });

  it('formats literal types', () => {
    const source = {
      input: z.object({
        status: z.literal('active').describe('Must be active'),
      }),
      output: z.object({
        ok: z.boolean(),
      }),
    };

    expect(formatSchemaForPrompt(source)).toBe(
      [
        'Input: {',
        '    /** Must be active */',
        '    status: "active";',
        '}',
        '',
        'Respond as JSON matching: {',
        '    ok: boolean;',
        '}',
      ].join('\n'),
    );
  });
});
