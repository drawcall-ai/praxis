import { resolve, dirname } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { config as loadEnv } from 'dotenv';
import { AxAIOpenRouter, AxGen, AxACE, AxGEPA } from '@ax-llm/ax';
import type { AxACEPlaybook, AxProgramDemos, AxMetricFn, AxMultiMetricFn, AxOptimizationProgress, AxFieldValue } from '@ax-llm/ax';
import { toAxSignature, serializeSchema } from '../schema.js';
import { detectMismatches } from '../validate.js';
import type {
  ModelDefinition,
  ModelExample,
  ModelConfig,
  PraxisDemo,
  PraxisEvalRun,
  TrainOptions,
} from '../types.js';
import {
  DEFAULT_CONFIG,
  bold, dim, green,
  resolveDefinitionPath, requireEnvKey, loadDefinition,
  validateDefinition, formatZodSchema,
} from './utils.js';

type AxExample = Record<string, AxFieldValue>;

// ── CLI handler ────────────────────────────────────────────────────────

export async function handleTrain(opts: { definition?: string; output?: string; optimizer: string; split: string; force: boolean }) {
  const definitionPath = await resolveDefinitionPath(opts.definition);
  const defDir = dirname(definitionPath);
  const defaultOutput = resolve(defDir, DEFAULT_CONFIG);

  const options: TrainOptions = {
    definitionPath,
    output: opts.output ? resolve(opts.output) : defaultOutput,
    optimizer: opts.optimizer as TrainOptions['optimizer'],
    split: parseFloat(opts.split),
  };

  loadEnv({ path: resolve(dirname(options.definitionPath), '.env') });
  requireEnvKey();

  const definition = await loadDefinition(options.definitionPath);
  validateDefinition(definition, true);

  // ── Train guard: decide train / skip / error ─────────────────────
  if (!opts.force) {
    let existingConfig: ModelConfig | undefined;
    try {
      existingConfig = JSON.parse(await readFile(options.output, 'utf-8'));
    } catch { /* no config yet */ }

    if (existingConfig) {
      const mismatches = detectMismatches(definition, existingConfig);
      const versionMismatch = mismatches.some((m) => m.field === 'version');
      const otherMismatches = mismatches.filter((m) => m.field !== 'version');

      if (mismatches.length === 0) {
        console.log(`\n  ${green('✓')} config is up to date — nothing to train\n`);
        return;
      }

      if (definition.version != null && otherMismatches.length > 0 && !versionMismatch) {
        const fields = otherMismatches.map((m) => m.field).join(', ');
        throw new Error(
          `${fields} changed but version was not bumped (still ${definition.version}).\n` +
          `  Update "version" in your definition to trigger retraining.`,
        );
      }
    }
  }

  // ── Resolve examples ─────────────────────────────────────────────
  const resolvedExamples = Array.isArray(definition.examples)
    ? definition.examples
    : await definition.examples();
  const resolvedDef = { ...definition, examples: resolvedExamples };

  console.log('');
  const teacherLabel = definition.teacher ? ` · teacher: ${definition.teacher}` : '';
  console.log(`  ${bold(definition.student)} ${dim(`${options.optimizer.toUpperCase()} · ${resolvedExamples.length} examples · ${options.split}/${(1 - options.split).toFixed(1)} split${teacherLabel}`)}`);
  console.log('');
  console.log(formatZodSchema(definition));
  console.log('');

  const { config, testScore } = await train(resolvedDef, options);

  await writeFile(options.output, JSON.stringify(config, null, 2));

  console.log('');
  console.log(`  ${green('✓')} ${bold(options.output)}`);
  if (testScore !== null) {
    console.log(`  ${green('✓')} test score ${bold(JSON.stringify(testScore))}`);
  }
  console.log('');
}

// ── Training pipeline ──────────────────────────────────────────────────

interface TrainResult {
  config: ModelConfig;
  testScore: number | Record<string, number> | null;
  evalRuns: PraxisEvalRun[];
}

// ── Progress display ─────────────────────────────────────────────────

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
 * Probe the metric function with a sample example to determine if it returns
 * a single number or a Record<string, number> (multi-metric).
 */
function probeMetricType(def: ModelDefinition, examples: ModelExample[]): 'single' | 'multi' {
  const example = examples[0];
  if (!example || !def.metric) return 'single';

  const ctx = {
    input: example.input as Record<string, unknown>,
    modelOutput: (example.output ?? {}) as Record<string, unknown>,
    exampleOutput: example.output as Record<string, unknown> | undefined,
  };

  const result = def.metric(ctx);
  if (result != null && typeof result === 'object') return 'multi';
  return 'single';
}

/**
 * Run the full training pipeline.
 */
async function train(
  definition: ModelDefinition,
  options: Pick<TrainOptions, 'optimizer' | 'split'>,
): Promise<TrainResult> {
  const { student } = definition;
  const progress = new Progress();

  if (!Array.isArray(definition.examples)) {
    throw new Error('train() expects resolved examples (plain array). Use resolveExamples() first.');
  }
  const examples = definition.examples;

  if (examples.length < 10) {
    throw new Error(`At least 10 examples required, got ${examples.length}`);
  }

  if (!definition.metric) {
    throw new Error('Definition must export a "metric" function');
  }

  // Validate metric against a sample example to catch null returns early
  validateMetric(definition, examples);

  const metricType = probeMetricType(definition, examples);
  const isMulti = metricType === 'multi';

  let optimizerType = options.optimizer;
  if (optimizerType === 'auto') {
    optimizerType = isMulti ? 'gepa' : 'ace';
  }

  const signature = toAxSignature(definition);

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

  const ai = new AxAIOpenRouter<string>({ apiKey, config: { model: student } });
  const teacherAI = definition.teacher
    ? new AxAIOpenRouter<string>({ apiKey, config: { model: definition.teacher } })
    : undefined;
  const program = new AxGen(signature);
  if (definition.description) {
    program.setDescription(definition.description);
  }

  const toAxExample = (ex: ModelExample): AxExample => {
    return { ...ex.input, ...(ex.output ?? {}) } as AxExample;
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
    const axMetric = adaptSingleMetric(definition);

    progress.spin('Optimizing');

    const optimizer = new AxACE({ studentAI: ai, teacherAI, onProgress });
    const result = await optimizer.compile(program, axTrainExamples, axMetric);

    instruction = renderPlaybook(result.playbook);
    demos = extractDemos(result.demos, definition);
    bestScore = result.bestScore ?? 0;
    stats = { ...result.stats };

    progress.done(`Optimized — score ${bestScore}`);
  } else {
    const axMultiMetric = adaptMultiMetric(definition);

    progress.spin('Optimizing');

    const optimizer = new AxGEPA({ studentAI: ai, teacherAI, onProgress });
    const result = await optimizer.compilePareto(program, axTrainExamples, axMultiMetric);

    const bestSolution = result.paretoFront?.[0];
    instruction = result.optimizedProgram?.instruction ?? '';
    demos = bestSolution
      ? extractDemos(bestSolution.demos, definition)
      : extractDemos(result.demos, definition);
    bestScore = bestSolution?.scores ?? {};
    stats = { ...result.stats };

    const scoreStr = Object.entries(bestScore as Record<string, number>)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    progress.done(`Optimized — ${scoreStr}`);
  }

  // ── Evaluate on test set ─────────────────────────────────────────
  let testScore: number | Record<string, number> | null = null;
  let evalRuns: PraxisEvalRun[] = [];

  if (testExamples.length > 0) {
    let completed = 0;
    const total = testExamples.length;

    progress.spin('Evaluating');

    const evalResult = await evaluate(ai, program, axTestExamples, definition, isMulti, () => {
      completed++;
      progress.set(`Evaluating ${dim(`${completed}/${total}`)}`);
    });

    testScore = evalResult.score;
    evalRuns = evalResult.runs;

    progress.done(`Evaluated — test score ${JSON.stringify(testScore)}`);
  }

  // ── Build config ─────────────────────────────────────────────────
  return {
    config: {
      version: definition.version ?? '1.0',
      student,
      ...(definition.teacher ? { teacher: definition.teacher } : {}),
      schema: serializeSchema(definition),
      optimization: {
        optimizer: optimizerType as 'ace' | 'gepa',
        instruction,
        demos,
        bestScore,
        evalRuns,
        stats,
      },
    },
    testScore,
    evalRuns,
  };
}

// ── Helpers ────────────────────────────────────────────────────────────

function validateMetric(def: ModelDefinition, examples: ModelExample[]): void {
  const example = examples[0];
  if (!example || !def.metric) return;

  const ctx = {
    input: example.input as Record<string, unknown>,
    modelOutput: (example.output ?? {}) as Record<string, unknown>,
    exampleOutput: example.output as Record<string, unknown> | undefined,
  };

  const result = def.metric(ctx);
  if (result == null) {
    throw new Error(
      'Metric returned null/undefined for a training example. ' +
      'If your metric requires the expected output, every example must include an "output" field. ' +
      'For metrics that work without expected output, return a number instead of null.',
    );
  }
}

/**
 * Wrap the user's metric as an AxMetricFn (single number).
 * If the metric returns a Record, averages the values.
 */
function adaptSingleMetric(def: ModelDefinition): AxMetricFn {
  const inputKeys = Object.keys(def.input.shape);
  const outputKeys = Object.keys(def.output.shape);
  const metricFn = def.metric!;

  return ({ prediction, example }) => {
    const input: Record<string, unknown> = {};
    const modelOutput: Record<string, unknown> = {};
    for (const k of inputKeys) input[k] = (prediction as Record<string, unknown>)[k] ?? example?.[k];
    for (const k of outputKeys) modelOutput[k] = (prediction as Record<string, unknown>)[k];

    let exampleOutput: Record<string, unknown> | undefined;
    if (example) {
      exampleOutput = {};
      for (const k of outputKeys) exampleOutput[k] = example[k];
    }

    const result = metricFn({ input, modelOutput, exampleOutput });
    if (result == null) return 0;
    if (typeof result === 'number') return result;
    // Record — average all values
    const vals = Object.values(result);
    return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };
}

/**
 * Wrap the user's metric as an AxMultiMetricFn.
 * The metric must return a Record<string, number>.
 */
function adaptMultiMetric(def: ModelDefinition): AxMultiMetricFn {
  const inputKeys = Object.keys(def.input.shape);
  const outputKeys = Object.keys(def.output.shape);
  const metricFn = def.metric!;

  return ({ prediction, example }) => {
    const input: Record<string, unknown> = {};
    const modelOutput: Record<string, unknown> = {};
    for (const k of inputKeys) input[k] = (prediction as Record<string, unknown>)[k] ?? example?.[k];
    for (const k of outputKeys) modelOutput[k] = (prediction as Record<string, unknown>)[k];

    let exampleOutput: Record<string, unknown> | undefined;
    if (example) {
      exampleOutput = {};
      for (const k of outputKeys) exampleOutput[k] = example[k];
    }

    const result = metricFn({ input, modelOutput, exampleOutput });
    if (result == null) return {};
    if (typeof result === 'number') return { default: result };
    return result;
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

function extractDemos(axDemos: readonly AxProgramDemos<unknown, unknown>[] | undefined, def: ModelDefinition): PraxisDemo[] {
  if (!axDemos?.length) return [];
  const inputKeys = Object.keys(def.input.shape);
  const outputKeys = Object.keys(def.output.shape);
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

interface EvalResult {
  score: number | Record<string, number>;
  runs: PraxisEvalRun[];
}

async function evaluate(
  ai: AxAIOpenRouter<string>,
  program: InstanceType<typeof AxGen>,
  testExamples: AxExample[],
  definition: ModelDefinition,
  isMulti: boolean,
  onProgress?: () => void,
): Promise<EvalResult> {
  const inputKeys = Object.keys(definition.input.shape);
  const outputKeys = Object.keys(definition.output.shape);
  const runs: PraxisEvalRun[] = [];

  const splitExample = (example: AxExample, prediction: Record<string, unknown>) => {
    const input: Record<string, unknown> = {};
    const expectedOutput: Record<string, unknown> = {};
    const modelOutput: Record<string, unknown> = {};
    for (const k of inputKeys) input[k] = example[k];
    for (const k of outputKeys) {
      expectedOutput[k] = example[k];
      modelOutput[k] = prediction[k];
    }
    return { input, expectedOutput, modelOutput };
  };

  if (isMulti) {
    const axMultiMetric = adaptMultiMetric(definition);
    const allScores: Record<string, number[]> = {};

    for (const example of testExamples) {
      try {
        const result = await program.forward(ai, example);
        const prediction = result as Record<string, unknown>;
        const scores = await axMultiMetric({ prediction, example });
        const { input, expectedOutput, modelOutput } = splitExample(example, prediction);
        runs.push({ input, expectedOutput, modelOutput, score: scores });
        for (const [name, val] of Object.entries(scores)) {
          (allScores[name] ??= []).push(val);
        }
      } catch { /* skip */ }
      onProgress?.();
    }

    const score = Object.fromEntries(
      Object.entries(allScores).map(([name, vals]) => [
        name, vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0,
      ]),
    );
    return { score, runs };
  }

  const axMetric = adaptSingleMetric(definition);
  const scores: number[] = [];

  for (const example of testExamples) {
    try {
      const result = await program.forward(ai, example);
      const prediction = result as Record<string, unknown>;
      const s = await axMetric({ prediction, example });
      const { input, expectedOutput, modelOutput } = splitExample(example, prediction);
      runs.push({ input, expectedOutput, modelOutput, score: s });
      scores.push(s);
    } catch { /* skip */ }
    onProgress?.();
  }

  const score = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  return { score, runs };
}
