import { resolve, dirname } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { config as loadEnv } from 'dotenv';
import pMap from 'p-map';
import { AxAIOpenRouter, AxGen, AxACE, AxGEPA } from '@ax-llm/ax';
import type { AxACEPlaybook, AxProgramDemos, AxMetricFn, AxMultiMetricFn, AxFieldValue, AxOptimizerLoggerData } from '@ax-llm/ax';
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

const EVAL_CONCURRENCY = 10;
const BAR_WIDTH = 20;

// ── Progress helpers ──────────────────────────────────────────────────

function progressBar(ratio: number): string {
  const filled = Math.round(ratio * BAR_WIDTH);
  return `${'█'.repeat(filled)}${dim('░'.repeat(BAR_WIDTH - filled))}`;
}

function writeProgress(msg: string) {
  process.stdout.write(`\r\x1b[K  ${msg}`);
}

function writeDone(msg: string) {
  process.stdout.write(`\r\x1b[K  ${green('✓')} ${msg}\n`);
}

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

async function train(
  definition: ModelDefinition,
  options: Pick<TrainOptions, 'optimizer' | 'split'>,
): Promise<TrainResult> {
  const { student } = definition;
  const startTime = Date.now();
  const elapsed = () => dim(`${((Date.now() - startTime) / 1000).toFixed(0)}s`);

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

  // ── Progress tracking ───────────────────────────────────────────
  // ACE's compile() only emits progress via optimizerLogger (not onProgress).
  // We store the latest round info and render it from a single interval.
  let lastRound: { round: number; totalRounds: number; bestScore: number } | undefined;

  const optimizerLogger = (data: AxOptimizerLoggerData) => {
    if (data.name === 'RoundProgress') {
      lastRound = data.value;
    }
  };

  const progressTimer = setInterval(() => {
    if (lastRound) {
      const { round, totalRounds, bestScore: best } = lastRound;
      const pct = totalRounds > 0 ? round / totalRounds : 0;
      const ms = Date.now() - startTime;
      const eta = round > 0 ? Math.ceil((ms / round) * (totalRounds - round) / 1000) : 0;
      writeProgress(`Optimizing ${progressBar(pct)} ${dim(`${round}/${totalRounds}`)} best: ${best.toFixed(2)} ${dim(`~${eta}s left`)}`);
    } else {
      writeProgress(`Optimizing… ${elapsed()}`);
    }
  }, 1000);
  progressTimer.unref();

  // ── Run optimization ─────────────────────────────────────────────
  let instruction = '';
  let demos: PraxisDemo[] = [];
  let bestScore: number | Record<string, number> = 0;
  let stats: Record<string, unknown> = {};

  const useMinibatch = trainExamples.length > 50;
  const optimizerArgs: ConstructorParameters<typeof AxACE>[0] = {
    studentAI: ai,
    teacherAI,
    optimizerLogger,
    debugOptimizer: true,
    ...(useMinibatch ? { minibatch: true, minibatchSize: Math.min(25, Math.ceil(trainExamples.length / 4)) } : {}),
  };

  writeProgress('Optimizing…');

  try {
    if (optimizerType === 'ace') {
      const axMetric = adaptSingleMetric(definition);
      const optimizer = new AxACE(optimizerArgs);
      const result = await optimizer.compile(program, axTrainExamples, axMetric);

      instruction = renderPlaybook(result.playbook);
      demos = extractDemos(result.demos, definition);
      bestScore = result.bestScore ?? 0;
      stats = { ...result.stats };
    } else {
      const axMultiMetric = adaptMultiMetric(definition);
      const optimizer = new AxGEPA(optimizerArgs);
      const result = await optimizer.compilePareto(program, axTrainExamples, axMultiMetric);

      const bestSolution = result.paretoFront?.[0];
      instruction = result.optimizedProgram?.instruction ?? '';
      demos = bestSolution
        ? extractDemos(bestSolution.demos, definition)
        : extractDemos(result.demos, definition);
      bestScore = bestSolution?.scores ?? {};
      stats = { ...result.stats };
    }
  } finally {
    clearInterval(progressTimer);
  }

  const scoreLabel = typeof bestScore === 'number'
    ? `score ${bestScore}`
    : Object.entries(bestScore).map(([k, v]) => `${k}: ${v}`).join(', ');
  writeDone(`Optimized — ${scoreLabel} ${elapsed()}`);

  // ── Evaluate on test set ─────────────────────────────────────────
  let testScore: number | Record<string, number> | null = null;
  let evalRuns: PraxisEvalRun[] = [];

  if (testExamples.length > 0) {
    let completed = 0;
    const total = testExamples.length;

    const evalResult = await evaluate(ai, program, axTestExamples, definition, isMulti, () => {
      completed++;
      writeProgress(`Evaluating ${dim(`${completed}/${total}`)}`);
    });

    testScore = evalResult.score;
    evalRuns = evalResult.runs;
    writeDone(`Evaluated — test score ${JSON.stringify(testScore)} ${elapsed()}`);
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

function probeMetricType(def: ModelDefinition, examples: ModelExample[]): 'single' | 'multi' {
  const example = examples[0];
  if (!example || !def.metric) return 'single';
  const result = def.metric({
    input: example.input as Record<string, unknown>,
    modelOutput: (example.output ?? {}) as Record<string, unknown>,
    exampleOutput: example.output as Record<string, unknown> | undefined,
  });
  return result != null && typeof result === 'object' ? 'multi' : 'single';
}

function validateMetric(def: ModelDefinition, examples: ModelExample[]): void {
  const example = examples[0];
  if (!example || !def.metric) return;
  const result = def.metric({
    input: example.input as Record<string, unknown>,
    modelOutput: (example.output ?? {}) as Record<string, unknown>,
    exampleOutput: example.output as Record<string, unknown> | undefined,
  });
  if (result == null) {
    throw new Error(
      'Metric returned null/undefined for a training example. ' +
      'If your metric requires the expected output, every example must include an "output" field. ' +
      'For metrics that work without expected output, return a number instead of null.',
    );
  }
}

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
    const vals = Object.values(result);
    return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };
}

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

    await pMap(testExamples, async (example) => {
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
    }, { concurrency: EVAL_CONCURRENCY });

    const score = Object.fromEntries(
      Object.entries(allScores).map(([name, vals]) => [
        name, vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0,
      ]),
    );
    return { score, runs };
  }

  const axMetric = adaptSingleMetric(definition);
  const scores: number[] = [];

  await pMap(testExamples, async (example) => {
    try {
      const result = await program.forward(ai, example);
      const prediction = result as Record<string, unknown>;
      const s = await axMetric({ prediction, example });
      const { input, expectedOutput, modelOutput } = splitExample(example, prediction);
      runs.push({ input, expectedOutput, modelOutput, score: s });
      scores.push(s);
    } catch { /* skip */ }
    onProgress?.();
  }, { concurrency: EVAL_CONCURRENCY });

  const score = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  return { score, runs };
}
