import type { z } from 'zod';

// ── Definition file exports ──────────────────────────────────────────

export interface PraxisSchema {
  input: z.ZodObject<z.ZodRawShape>;
  output: z.ZodObject<z.ZodRawShape>;
}

export interface PraxisExample {
  input: Record<string, unknown>;
  [key: string]: unknown;
}

export type PraxisMetricFn = (ctx: {
  input: Record<string, unknown>;
  prediction: Record<string, unknown>;
  example?: PraxisExample;
}) => number | null;

export interface PraxisDefinition {
  model: string;
  schema: PraxisSchema;
  examples: PraxisExample[];
  metric?: PraxisMetricFn;
  metrics?: Record<string, PraxisMetricFn>;
}

// ── Config file (model.config.json) ──────────────────────────────────

/** Standard JSON Schema stored in the config. */
export type JsonSchema = Record<string, unknown>;

export interface PraxisConfigSchema {
  input: JsonSchema;
  output: JsonSchema;
}

export interface PraxisDemo {
  input: Record<string, unknown>;
  output: Record<string, unknown>;
}

export interface PraxisConfig {
  version: string;
  model: string;
  schema: PraxisConfigSchema;
  optimization: {
    optimizer: string;
    instruction: string;
    demos: PraxisDemo[];
    bestScore: number | Record<string, number>;
    stats?: Record<string, unknown>;
  };
}

// ── CLI options ──────────────────────────────────────────────────────

export interface TrainOptions {
  definitionPath: string;
  output: string;
  optimizer: 'auto' | 'ace' | 'gepa';
  split: number;
}

// ── Prompt output ────────────────────────────────────────────────────

export interface PromptResult<T extends z.ZodRawShape = z.ZodRawShape> {
  system: string;
  user: string;
  schema: {
    input: z.ZodObject<z.ZodRawShape>;
    output: z.ZodObject<T>;
  };
}
