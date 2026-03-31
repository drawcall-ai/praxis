import type { z } from 'zod';
import type { ModelMessage } from 'ai';

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

export type ModelExamples<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> =
  | ModelExample<I, O>[]
  | (() => Promise<ModelExample<I, O>[]>);

export interface ModelDefinition<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> {
  model: string;
  version?: string;
  teacher?: string;
  description?: string;
  input: z.ZodObject<I>;
  output: z.ZodObject<O>;
  examples: ModelExamples<I, O>;
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

export interface PraxisEvalRun {
  input: Record<string, unknown>;
  expectedOutput: Record<string, unknown>;
  modelOutput: Record<string, unknown>;
  score: number | Record<string, number>;
}

export interface ModelConfig {
  version: string;
  model: string;
  teacher?: string;
  schema: PraxisConfigSchema;
  optimization: {
    optimizer: string;
    instruction: string;
    demos: PraxisDemo[];
    bestScore: number | Record<string, number>;
    evalRuns?: PraxisEvalRun[];
    stats?: Record<string, unknown>;
  };
}

// ── Request (buildRequest) ──────────────────────────────────────────

export interface ModelRequest<O extends z.ZodRawShape = z.ZodRawShape> {
  messages: ModelMessage[];
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
