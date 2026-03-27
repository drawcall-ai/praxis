import { AxAIOpenRouter, AxGen, AxACE, AxGEPA } from '@ax-llm/ax';
import type { AxACEPlaybook, AxProgramDemos, AxMetricFn, AxMultiMetricFn, AxOptimizationProgress } from '@ax-llm/ax';
import type {
  PraxisDefinition,
  PraxisExample,
  PraxisMetricFn,
  PraxisConfig,
  PraxisDemo,
  TrainOptions,
} from './types.js';
import { toAxSignature, serializeSchema } from './schema.js';

interface TrainResult {
  config: PraxisConfig;
  testScore: number | Record<string, number> | null;
}

// ── Progress display ─────────────────────────────────────────────────

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;

const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const BAR_WIDTH = 20;

class Progress {
  private startTime = Date.now();
  private frame = 0;
  private interval: ReturnType<typeof setInterval> | null = null;
  private line = '';

  spin(label: string) {
    this.stop();
    this.line = label;
    this.interval = setInterval(() => {
      this.frame = (this.frame + 1) % SPINNER.length;
      this.write(`  ${SPINNER[this.frame]} ${this.line}  ${dim(this.elapsed())}`);
    }, 80);
  }

  set(label: string) {
    this.line = label;
  }

  done(msg: string) {
    this.stop();
    console.log(`  ${green('✓')} ${msg} ${dim(this.elapsed())}`);
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
      process.stdout.write('\r\x1b[K');
    }
  }

  private elapsed(): string {
    return `${((Date.now() - this.startTime) / 1000).toFixed(0)}s`;
  }

  private write(s: string) {
    process.stdout.write(`\r\x1b[K${s}`);
  }
}

function bar(ratio: number): string {
  const filled = Math.round(ratio * BAR_WIDTH);
  const empty = BAR_WIDTH - filled;
  return `${'█'.repeat(filled)}${dim('░'.repeat(empty))}`;
}

function formatProgress(p: AxOptimizationProgress): string {
  const pct = p.totalRounds > 0 ? p.round / p.totalRounds : 0;
  const score = typeof p.bestScore === 'number' ? p.bestScore.toFixed(2) : '?';
  const converging = p.convergenceInfo?.isConverging ? dim(' converging') : '';
  return `${bar(pct)} ${dim(`${p.round}/${p.totalRounds}`)} best: ${score}${converging}`;
}

/**
 * Run the full training pipeline.
 */
export async function train(
  definition: PraxisDefinition,
  options: Pick<TrainOptions, 'optimizer' | 'split'>,
): Promise<TrainResult> {
  const { schema, examples, model } = definition;
  const progress = new Progress();

  if (examples.length < 10) {
    throw new Error(`At least 10 examples required, got ${examples.length}`);
  }

  const hasMultipleMetrics = !!definition.metrics && Object.keys(definition.metrics).length > 1;
  const hasSingleMetric = !!definition.metric || (definition.metrics && Object.keys(definition.metrics).length === 1);

  if (!hasSingleMetric && !hasMultipleMetrics) {
    throw new Error('Definition must export "metric" or "metrics"');
  }

  let optimizerType = options.optimizer;
  if (optimizerType === 'auto') {
    optimizerType = hasMultipleMetrics ? 'gepa' : 'ace';
  }

  const signature = toAxSignature(schema);

  // ── Train/test split ─────────────────────────────────────────────
  const shuffled = [...examples].sort(() => Math.random() - 0.5);
  const splitIdx = Math.floor(shuffled.length * options.split);
  const trainExamples = shuffled.slice(0, splitIdx);
  const testExamples = shuffled.slice(splitIdx);

  // ── Build AX components ──────────────────────────────────────────
  const apiKey = process.env.OPENROUTER_KEY ?? process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_KEY not found in environment.');
  }

  const ai = new AxAIOpenRouter({ apiKey, config: { model } });
  const program = new AxGen(signature);

  const toAxExample = (ex: PraxisExample) => {
    const { input, ...outputFields } = ex;
    return { ...input, ...outputFields };
  };

  const axTrainExamples = trainExamples.map(toAxExample);
  const axTestExamples = testExamples.map(toAxExample);

  const onProgress = (p: Readonly<AxOptimizationProgress>) => {
    progress.set(formatProgress(p as AxOptimizationProgress));
  };

  // ── Run optimization ─────────────────────────────────────────────
  let instruction = '';
  let demos: PraxisDemo[] = [];
  let bestScore: number | Record<string, number> = 0;
  let stats: Record<string, unknown> = {};

  if (optimizerType === 'ace') {
    const metricFn = resolveMetric(definition);
    const axMetric = adaptMetric(metricFn, schema);

    progress.spin('Optimizing');

    const optimizer = new AxACE({ studentAI: ai, onProgress });
    const result = await optimizer.compile(program, axTrainExamples, axMetric);

    instruction = renderPlaybook(result.playbook);
    demos = extractDemos(result.demos, schema);
    bestScore = result.bestScore ?? 0;
    stats = result.stats ?? {};

    progress.done(`Optimized — score ${bestScore}`);
  } else {
    const metricFns = resolveMetrics(definition);
    const axMultiMetric = adaptMultiMetric(metricFns, schema);

    progress.spin('Optimizing');

    const optimizer = new AxGEPA({ studentAI: ai, onProgress });
    const result = await optimizer.compilePareto(program, axTrainExamples, axMultiMetric);

    const bestSolution = result.paretoFront?.[0];
    instruction = result.optimizedProgram?.instruction ?? '';
    demos = bestSolution
      ? extractDemos(bestSolution.demos, schema)
      : extractDemos(result.demos, schema);
    bestScore = bestSolution?.scores ?? {};
    stats = result.stats ?? {};

    const scoreStr = Object.entries(bestScore as Record<string, number>)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    progress.done(`Optimized — ${scoreStr}`);
  }

  // ── Evaluate on test set ─────────────────────────────────────────
  let testScore: number | Record<string, number> | null = null;

  if (testExamples.length > 0) {
    let completed = 0;
    const total = testExamples.length;

    progress.spin('Evaluating');

    testScore = await evaluate(ai, program, axTestExamples, definition, schema, () => {
      completed++;
      progress.set(`Evaluating ${dim(`${completed}/${total}`)}`);
    });

    progress.done(`Evaluated — test score ${JSON.stringify(testScore)}`);
  }

  // ── Build config ─────────────────────────────────────────────────
  return {
    config: {
      version: '1.0',
      model,
      schema: serializeSchema(schema),
      optimization: {
        optimizer: optimizerType as 'ace' | 'gepa',
        instruction,
        demos,
        bestScore,
        stats,
      },
    },
    testScore,
  };
}

// ── Helpers ────────────────────────────────────────────────────────────

function resolveMetric(def: PraxisDefinition): PraxisMetricFn {
  if (def.metric) return def.metric;
  if (def.metrics) {
    const fns = Object.values(def.metrics);
    if (fns.length === 1) return fns[0];
  }
  throw new Error('No metric found');
}

function resolveMetrics(def: PraxisDefinition): Record<string, PraxisMetricFn> {
  if (def.metrics && Object.keys(def.metrics).length > 1) return def.metrics;
  if (def.metric) return { default: def.metric };
  throw new Error('No metrics found');
}

function adaptMetric(fn: PraxisMetricFn, schema: PraxisDefinition['schema']): AxMetricFn {
  const inputKeys = Object.keys(schema.input.shape);
  const outputKeys = Object.keys(schema.output.shape);

  return ({ prediction, example }) => {
    const input: Record<string, unknown> = {};
    const pred: Record<string, unknown> = {};
    for (const k of inputKeys) input[k] = (prediction as Record<string, unknown>)[k] ?? example?.[k];
    for (const k of outputKeys) pred[k] = (prediction as Record<string, unknown>)[k];

    let praxisExample: PraxisExample | undefined;
    if (example) {
      const exInput: Record<string, unknown> = {};
      for (const k of inputKeys) exInput[k] = example[k];
      praxisExample = { input: exInput };
      for (const k of outputKeys) (praxisExample as Record<string, unknown>)[k] = example[k];
    }

    return fn({ input, prediction: pred, example: praxisExample }) ?? 0;
  };
}

function adaptMultiMetric(fns: Record<string, PraxisMetricFn>, schema: PraxisDefinition['schema']): AxMultiMetricFn {
  const adapted = Object.entries(fns).map(([name, fn]) => [name, adaptMetric(fn, schema)] as const);

  return ({ prediction, example }) => {
    const scores: Record<string, number> = {};
    for (const [name, fn] of adapted) scores[name] = fn({ prediction, example });
    return scores;
  };
}

function renderPlaybook(playbook: AxACEPlaybook): string {
  const lines: string[] = [];
  for (const [section, bullets] of Object.entries(playbook.sections)) {
    lines.push(`## ${section}`);
    for (const bullet of bullets) lines.push(`- ${bullet.content}`);
    lines.push('');
  }
  return lines.join('\n').trim();
}

function extractDemos(axDemos: AxProgramDemos<unknown, unknown>[] | undefined, schema: PraxisDefinition['schema']): PraxisDemo[] {
  if (!axDemos?.length) return [];
  const inputKeys = Object.keys(schema.input.shape);
  const outputKeys = Object.keys(schema.output.shape);
  const result: PraxisDemo[] = [];

  for (const demoGroup of axDemos) {
    for (const trace of demoGroup.traces) {
      const flat = trace as Record<string, unknown>;
      const input: Record<string, unknown> = {};
      const output: Record<string, unknown> = {};
      for (const k of inputKeys) input[k] = flat[k];
      for (const k of outputKeys) output[k] = flat[k];
      result.push({ input, output });
    }
  }
  return result;
}

async function evaluate(
  ai: InstanceType<typeof AxAIOpenRouter>,
  program: InstanceType<typeof AxGen>,
  testExamples: Record<string, unknown>[],
  definition: PraxisDefinition,
  schema: PraxisDefinition['schema'],
  onProgress?: () => void,
): Promise<number | Record<string, number>> {
  const hasMultiple = !!definition.metrics && Object.keys(definition.metrics).length > 1;

  if (hasMultiple) {
    const metrics = resolveMetrics(definition);
    const scores: Record<string, number[]> = {};
    for (const name of Object.keys(metrics)) scores[name] = [];

    for (const example of testExamples) {
      try {
        const result = await program.forward(ai, example);
        for (const [name, fn] of Object.entries(metrics)) {
          scores[name].push(adaptMetric(fn, schema)({ prediction: result, example }));
        }
      } catch { /* skip */ }
      onProgress?.();
    }

    return Object.fromEntries(
      Object.entries(scores).map(([name, vals]) => [
        name, vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0,
      ]),
    );
  }

  const adapted = adaptMetric(resolveMetric(definition), schema);
  const scores: number[] = [];

  for (const example of testExamples) {
    try {
      scores.push(adapted({ prediction: await program.forward(ai, example), example }));
    } catch { /* skip */ }
    onProgress?.();
  }

  return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}
