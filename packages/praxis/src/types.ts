import type { z } from 'zod';

// ── Definition (defineModel) ────────────────────────────────────────

export interface ModelExample<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> {
  input: z.infer<z.ZodObject<I>>;
  output?: z.infer<z.ZodObject<O>>;
}

export type ModelMetricFn<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> = (ctx: {
  input: z.infer<z.ZodObject<I>>;
  modelOutput: z.infer<z.ZodObject<O>>;
  exampleOutput?: z.infer<z.ZodObject<O>>;
}) => number | Record<string, number> | null | undefined;

export interface ModelDefinition<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> {
  model: string;
  input: z.ZodObject<I>;
  output: z.ZodObject<O>;
  examples: ModelExample<I, O>[];
  metric?: ModelMetricFn<I, O>;
}

// ── Config (model.config.json) ──────────────────────────────────────

export type JsonSchema = Record<string, unknown>;

export interface PraxisConfigSchema {
  input: JsonSchema;
  output: JsonSchema;
}

export interface PraxisDemo {
  input: Record<string, unknown>;
  output: Record<string, unknown>;
}

export interface ModelConfig {
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

// ── Request (buildRequest) ──────────────────────────────────────────

export interface ModelRequest<O extends z.ZodRawShape = z.ZodRawShape> {
  system: string;
  user: string;
  schema: {
    input: z.ZodObject<z.ZodRawShape>;
    output: z.ZodObject<O>;
  };
  model: string;
  metric?: ModelMetricFn;
}

// ── CLI options ─────────────────────────────────────────────────────

export interface TrainOptions {
  definitionPath: string;
  output: string;
  optimizer: 'auto' | 'ace' | 'gepa';
  split: number;
}
