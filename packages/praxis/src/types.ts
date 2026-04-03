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
  M extends string = string,
> = (ctx: {
  input: z.infer<z.ZodObject<I>>;
  modelOutput: z.infer<z.ZodObject<O>>;
  exampleOutput?: z.infer<z.ZodObject<O>>;
}) => Record<M, number> | null | undefined;

export interface LazyExampleProvider<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> {
  getLength: () => Promise<number>;
  getExample: (index: number) => Promise<ModelExample<I, O>>;
}

export type ModelExamples<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> =
  | ModelExample<I, O>[]
  | (() => Promise<ModelExample<I, O>[]>)
  | LazyExampleProvider<I, O>;

export interface ModelDefinition<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
  M extends string = string,
> {
  name?: string;
  student: string;
  version?: string;
  teacher?: string;
  description?: string;
  input: z.ZodObject<I>;
  output: z.ZodObject<O>;
  examples: ModelExamples<I, O>;
  metric: ModelMetricFn<I, O, M>;
  metricWeights: Record<M, number>;
  split?: [number, number, number];
  targetScore?: number;
  maxIterations?: number;
  maxTestsPerIteration?: number;
  maxOutputTokens?: number;
}

// ── Config (model.config.json) ──────────────────────────────────────

export type JsonSchema = Record<string, unknown>;

export interface PraxisConfigSchema {
  input: JsonSchema;
  output: JsonSchema;
}

export interface PraxisEvalRun {
  input: Record<string, unknown>;
  expectedOutput: Record<string, unknown>;
  modelOutput: Record<string, unknown>;
  score: Record<string, number>;
}

export interface ModelConfig {
  version: string;
  student: string;
  teacher?: string;
  schema: PraxisConfigSchema;
  optimization: {
    instruction: string;
    reasoningEffort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';
    bestScore: Record<string, number>;
    metricWeights?: Record<string, number>;
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
  student: string;
  metric?: ModelMetricFn;
}

// ── CLI options ─────────────────────────────────────────────────────

export interface TrainOptions {
  definitionPath: string;
  output: string;
}

// ── Scoring ─────────────────────────────────────────────────────────

export function computeCombinedScore(
  scores: Record<string, number>,
  weights?: Record<string, number>,
): number {
  const keys = Object.keys(scores);
  if (keys.length === 0) return 0;
  if (!weights) {
    return keys.reduce((sum, k) => sum + scores[k], 0) / keys.length;
  }
  let totalWeight = 0;
  let weightedSum = 0;
  for (const k of keys) {
    const w = weights[k] ?? 1;
    weightedSum += scores[k] * w;
    totalWeight += w;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}
