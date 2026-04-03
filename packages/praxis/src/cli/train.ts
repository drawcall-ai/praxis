import { resolve, dirname } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { generateText as aiGenerateText, NoObjectGeneratedError, Output, tool, wrapLanguageModel, extractJsonMiddleware } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { z } from 'zod';
import pMap from 'p-map';
import pRetry from 'p-retry';
import { serializeSchema, formatSchemaForPrompt } from '../schema.js';
import { buildDefaultSystemPrompt } from '../prompt.js';
import { detectMismatches } from '../validate.js';
import { computeCombinedScore } from '../types.js';
import type {
  ModelDefinition,
  ModelConfig,
  PraxisEvalRun,
  TrainOptions,
} from '../types.js';
import { resolveExamples } from '../examples.js';
import type { ExampleProvider } from '../examples.js';
import {
  DEFAULT_CONFIG,
  bold, dim, green,
  resolveDefinitionPath, requireEnvKey, loadDefinition,
  validateDefinition, formatZodSchema,
} from './utils.js';
import { formatTable } from '../format.js';

const EVAL_CONCURRENCY = 20;
const MAX_TEST_PER_CALL = 20;
const DEFAULT_MAX_TESTS_PER_ITERATION = 100;
const MAX_VAL_EXAMPLES = 30;
const MAX_TEST_EXAMPLES = 30;
const DEFAULT_MAX_OUTPUT_TOKENS = 16384;
const DEFAULT_REASONING_EFFORT = 'medium' as const;
const DEFAULT_SPLIT = [0.7, 0.15, 0.15] as const;
const DEFAULT_MAX_ITERATIONS = 5;
const DEFAULT_TOKEN_EFFICIENCY_WEIGHT = 0.1;

// ── CLI handler ────────────────────────────────────────────────────────

export async function handleTrain(opts: { definition?: string; output?: string; force: boolean }) {
  const definitionPath = await resolveDefinitionPath(opts.definition);
  const defDir = dirname(definitionPath);
  const defaultOutput = resolve(defDir, DEFAULT_CONFIG);

  const options: TrainOptions = {
    definitionPath,
    output: opts.output ? resolve(opts.output) : defaultOutput,
  };

  requireEnvKey();

  const definition = await loadDefinition(options.definitionPath);
  validateDefinition(definition, true);

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

  const provider = await resolveExamples(definition.examples);

  console.log('');
  const teacherLabel = definition.teacher ? ` · teacher: ${definition.teacher}` : '';
  const splitRatio = definition.split ?? DEFAULT_SPLIT;
  console.log(`  ${bold(definition.student)} ${dim(`${provider.length} examples · ${splitRatio.join('/')} split${teacherLabel}`)}`);
  console.log('');
  console.log(formatZodSchema(definition));
  console.log('');

  const { config, testScore } = await train(definition, provider);

  await writeFile(options.output, JSON.stringify(config, null, 2));

  console.log('');
  console.log(`  ${green('✓')} ${bold(options.output)}`);
  if (testScore !== null) {
    const scoreStr = Object.entries(testScore).map(([k, v]) => `${k}: ${v.toFixed(2)}`).join(', ');
    console.log(`  ${green('✓')} test score: ${bold(scoreStr)}`);
  }
  console.log('');
}

// ── Training pipeline ──────────────────────────────────────────────────

interface TrainResult {
  config: ModelConfig;
  testScore: Record<string, number> | null;
  evalRuns: PraxisEvalRun[];
}

// ── Evaluation types ───────────────────────────────────────────────────

interface EvalResult {
  id: number;
  input: Record<string, unknown>;
  expectedOutput: Record<string, unknown>;
  modelOutput: Record<string, unknown>;
  score: Record<string, number>;
  reasoning: string;
  actualCost: number;
  optimalCost: number;
}

async function train(
  definition: ModelDefinition,
  provider: ExampleProvider,
): Promise<TrainResult> {
  const { student } = definition;
  const split = definition.split ?? [...DEFAULT_SPLIT];
  let targetScore = definition.targetScore;
  const maxIterations = definition.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const maxTestsPerIteration = definition.maxTestsPerIteration ?? DEFAULT_MAX_TESTS_PER_ITERATION;
  const maxAgentSteps = 2 * maxTestsPerIteration;
  let budgetRemaining = maxTestsPerIteration;
  let stepWhenBudgetHitZero: number | undefined;
  const startTime = Date.now();
  const elapsed = () => dim(`${((Date.now() - startTime) / 1000).toFixed(0)}s`);

  if (provider.length < 10) {
    throw new Error(`At least 10 examples required, got ${provider.length}`);
  }

  if (!definition.metric) {
    throw new Error('Definition must export a "metric" function');
  }

  const splitSum = split[0] + split[1] + split[2];
  if (Math.abs(splitSum - 1) > 0.01) {
    throw new Error(`split must sum to 1.0, got [${split.join(', ')}] = ${splitSum}`);
  }

  const userMetricKeys = await probeMetricKeys(definition, provider);
  const metricKeys = [...userMetricKeys, 'tokenEfficiency'];
  const weights = definition.metricWeights;
  const effectiveWeights: Record<string, number> = {
    ...weights,
    tokenEfficiency: definition.metricWeights?.tokenEfficiency ?? DEFAULT_TOKEN_EFFICIENCY_WEIGHT,
  };

  // ── Train/val/test split (index-based, no full materialization) ──
  const indices = [...Array(provider.length).keys()];
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const trainCount = Math.round(provider.length * split[0]);
  const valCount = Math.min(Math.round(provider.length * split[1]), MAX_VAL_EXAMPLES);
  const testCount = Math.min(provider.length - trainCount - valCount, MAX_TEST_EXAMPLES);
  const trainIndices = indices.slice(0, trainCount);
  const valIndices = indices.slice(trainCount, trainCount + valCount);
  const testIndices = indices.slice(trainCount + valCount, trainCount + valCount + testCount);

  console.log(`  ${dim(`${trainIndices.length} train / ${valIndices.length} val / ${testIndices.length} test`)}`);
  console.log(`  ${dim(`metric keys: ${metricKeys.join(', ')}`)}`);
  console.log(`  ${dim(`weights: ${JSON.stringify(effectiveWeights)}`)}`);
  if (targetScore != null) console.log(`  ${dim(`target: ${(targetScore * 100).toFixed(1)}% · max iterations: ${maxIterations}`)}`);

  // ── OpenRouter setup ─────────────────────────────────────────────
  const apiKey = process.env.OPENROUTER_KEY ?? process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_KEY not found in environment.');

  const openrouter = createOpenRouter({ apiKey });
  const teacherModel = definition.teacher ?? student;

  // ── Evaluate helper ──────────────────────────────────────────────
  const runs = new Map<string, EvalResult[]>();
  let runCounter = 0;
  function nextRunId() { return `run-${++runCounter}`; }

  let currentPrompt = buildDefaultSystemPrompt(definition);
  let currentReasoningEffort: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' = DEFAULT_REASONING_EFFORT;
  let baselineMediumCost: number | undefined;

  async function evaluate(
    systemPrompt: string,
    exampleIndices: number[],
  ): Promise<{ runId: string; results: EvalResult[] }> {
    const runId = nextRunId();
    const validIds = exampleIndices.filter((id) => id >= 0 && id < provider.length);
    const resultMap = new Map<number, EvalResult>();

    const model = wrapLanguageModel({
      model: openrouter.chat(student),
      middleware: extractJsonMiddleware(),
    });

    await pMap(validIds, async (id) => {
      const ex = await provider.get(id);
      const input = ex.input as Record<string, unknown>;
      const userContent = Object.entries(input)
        .map(([key, value]) => `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`)
        .join('\n');
      const generateArgs = {
        model,
        system: systemPrompt,
        prompt: userContent,
        output: Output.object({ schema: definition.output }),
        maxOutputTokens: definition.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
        providerOptions: {
          openrouter: { reasoning: { effort: currentReasoningEffort, exclude: false } },
        },
      } as const;
      const result = await pRetry(() => aiGenerateText(generateArgs), {
        retries: 2,
        shouldRetry: (err) => err instanceof NoObjectGeneratedError,
        onFailedAttempt(err: any) {
          console.log(`  ${dim(`[eval] example #${id} failed (${err.message?.slice(0, 80)}), retry ${err.attemptNumber}/3...`)}`);
        },
      });
      const modelOutput = result.output as Record<string, unknown>;
      let reasoning = '';
      if (typeof result.reasoning === 'string') {
        reasoning = result.reasoning;
      } else if (Array.isArray(result.reasoning)) {
        reasoning = (result.reasoning as any[])
          .filter((r) => r.type === 'reasoning' && typeof r.text === 'string' && r.text !== '[REDACTED]')
          .map((r) => r.text)
          .join('\n');
      }
      const userScore = definition.metric!({
        input: ex.input as Record<string, unknown>,
        modelOutput: modelOutput as any,
        exampleOutput: ex.output as any,
      }) ?? Object.fromEntries(metricKeys.map((k) => [k, 0]));

      // Token efficiency: linear scale where 100% = optimal, 50% = baseline
      const promptTokens = result.usage?.inputTokens ?? 0;
      const completionTokens = result.usage?.outputTokens ?? 0;
      const estimatedUserInputTokens = Math.ceil(userContent.length / 4);
      const estimatedOutputTokens = Math.ceil(JSON.stringify(modelOutput).length / 4);
      const optimalCost = estimatedUserInputTokens + estimatedOutputTokens * 4;
      const actualCost = promptTokens + completionTokens * 4;
      let tokenEfficiency: number;
      if (baselineMediumCost != null && baselineMediumCost > optimalCost) {
        tokenEfficiency = Math.max(0, Math.min(1,
          1 - 0.5 * (actualCost - optimalCost) / (baselineMediumCost - optimalCost)));
      } else {
        tokenEfficiency = actualCost > 0 ? Math.min(optimalCost / actualCost, 1) : 1;
      }

      const score = { ...userScore, tokenEfficiency };

      resultMap.set(id, {
        id, input, modelOutput,
        expectedOutput: (ex.output ?? {}) as Record<string, unknown>,
        score, reasoning, actualCost, optimalCost,
      });
    }, { concurrency: EVAL_CONCURRENCY });

    const results = validIds.map((id) => resultMap.get(id)!);
    runs.set(runId, results);
    return { runId, results };
  }

  // ── Formatting helpers ───────────────────────────────────────────

  function formatCompactTable(results: EvalResult[]): string {
    if (results.length === 0) return '(no results)';

    const columns = [
      'Examples',
      ...metricKeys.map((k) => {
        const w = effectiveWeights?.[k];
        return w != null ? `${k} (${w})` : k;
      }),
      'combinedScore',
    ];

    const rows: string[][] = [];
    let rangeStart = results[0].id;
    let rangeEnd = rangeStart;
    let currentScore = results[0].score;

    function emitRow(start: number, end: number, score: Record<string, number>) {
      const label = start === end ? `#${start}` : `#${start}-#${end}`;
      const vals = metricKeys.map((k) => String(score[k] ?? 0));
      vals.push(computeCombinedScore(score, effectiveWeights).toFixed(2));
      rows.push([label, ...vals]);
    }

    function scoresEqual(a: Record<string, number>, b: Record<string, number>) {
      return metricKeys.every((k) => (a[k] ?? 0) === (b[k] ?? 0));
    }

    for (let i = 1; i <= results.length; i++) {
      const r = results[i];
      if (r && scoresEqual(r.score, currentScore)) {
        rangeEnd = r.id;
      } else {
        emitRow(rangeStart, rangeEnd, currentScore);
        if (r) {
          rangeStart = r.id;
          rangeEnd = r.id;
          currentScore = r.score;
        }
      }
    }

    return formatTable(columns, rows);
  }

  function summarize(results: EvalResult[]): string {
    const avg: Record<string, number> = {};
    for (const k of metricKeys) {
      avg[k] = results.length > 0
        ? results.reduce((sum, r) => sum + (r.score[k] ?? 0), 0) / results.length
        : 0;
    }
    const combined = computeCombinedScore(avg, effectiveWeights);
    const parts = metricKeys.map((k) => `${k}: ${(avg[k] * 100).toFixed(1)}%`);
    parts.push(`combined: ${(combined * 100).toFixed(1)}%`);
    return parts.join(' | ');
  }

  function avgScores(results: EvalResult[]): Record<string, number> {
    const avg: Record<string, number> = {};
    for (const k of metricKeys) {
      avg[k] = results.length > 0
        ? results.reduce((sum, r) => sum + (r.score[k] ?? 0), 0) / results.length
        : 0;
    }
    return avg;
  }

  function formatScores(set: 'training' | 'validation' | 'test', scores: Record<string, number>): string {
    const parts = metricKeys.map((k) => `${k}: ${((scores[k] ?? 0) * 100).toFixed(1)}%`);
    parts.push(`combined: ${(computeCombinedScore(scores, effectiveWeights) * 100).toFixed(1)}%`);
    return `Scores on ${set}-set: ${parts.join(' | ')}`;
  }

  function formatGoal(): string {
    return `Target: combined score > ${(targetScore! * 100).toFixed(1)}% on the validation-set. Build a configuration that generalizes to the problem.`;
  }

  const baselinePrompt = currentPrompt;

  // ── Agent system prompt ──────────────────────────────────────────

  const schemaDesc = formatSchemaForPrompt(definition);
  const metricDesc = `Multi-metric: ${metricKeys.join(', ')}. Weights: ${JSON.stringify(effectiveWeights)}.`;

  const agentSystemPrompt = `You are a prompt optimization agent. Your goal is to find the best configuration — system prompt and reasoning effort — for a target LLM.

## Task
${definition.name ? `**${definition.name}** — ` : ''}${definition.description ?? ''}
${schemaDesc}

The target model has internal reasoning enabled. It reasons before answering, then outputs structured JSON.

**Metric:** ${metricDesc} The table shows per-metric scores and a \`combinedScore\` (weighted average).

## Data

- **Training-set** — examples you can inspect and test against via tools.
- **Validation-set** — separate held-out examples you never see. After each iteration your configuration is scored here. You must beat the target score. Good training-set scores do not guarantee good validation-set scores.

## Token Efficiency

The built-in \`tokenEfficiency\` metric (weight: ${effectiveWeights.tokenEfficiency}) uses a linear scale: 100% = optimal (irreducible cost: just user input + model output), 50% = baseline efficiency (the average token cost of the initial unoptimized run). Shorter prompts and less reasoning overhead → higher score. The score is clamped to [0, 1].

## Prompt Design

A system prompt can include any combination of: role/persona, step-by-step algorithms, few-shot examples, lookup tables, rules and constraints, edge case handling, common mistake warnings. You have full creative freedom.

## Workflow

1. **Test** a subset of training examples to get a baseline score and a runId
2. **Diagnose** — call view_input on failing IDs, then view_output on the same runId+IDs to see the model's reasoning and where it went wrong
3. **Improve** — write a new configuration (prompt, reasoning effort) that addresses the observed errors
4. **Verify** — test the failing examples again to confirm the fix
5. **Repeat** from step 2 if failures remain

NEVER write a prompt without first inspecting examples and model reasoning via view_input and view_output.`;

  // ── Pre-optimization val baseline ─────────────────────────────────
  console.log(`  Evaluating baseline on val set... ${elapsed()}`);
  const { results: valBaseline } = await evaluate(currentPrompt, valIndices);

  // Compute mediumCost anchor for tokenEfficiency: max(2 * avgOptimal, avgBaseline)
  const valWithCosts = valBaseline.filter((r) => r.actualCost > 0);
  if (valWithCosts.length > 0) {
    const avgBaselineCost = valWithCosts.reduce((s, r) => s + r.actualCost, 0) / valWithCosts.length;
    const avgOptimalCost = valWithCosts.reduce((s, r) => s + r.optimalCost, 0) / valWithCosts.length;
    baselineMediumCost = Math.max(2 * avgOptimalCost, avgBaselineCost);
  }

  // Re-score baseline with the new mediumCost anchor
  if (baselineMediumCost != null) {
    for (const r of valBaseline) {
      if (r.optimalCost > 0 && baselineMediumCost > r.optimalCost) {
        r.score.tokenEfficiency = Math.max(0, Math.min(1,
          1 - 0.5 * (r.actualCost - r.optimalCost) / (baselineMediumCost - r.optimalCost)));
      }
    }
  }

  const valBaselineScores = avgScores(valBaseline);
  const valBaselineCombined = computeCombinedScore(valBaselineScores, effectiveWeights);
  console.log(`  ${green('✓')} Baseline (val): ${summarize(valBaseline)} ${elapsed()}`);

  // Effective target: must beat both the baseline and the user-defined target
  targetScore = Math.max(targetScore ?? 0, valBaselineCombined);
  console.log(`  ${dim(`effective target: ${(targetScore * 100).toFixed(1)}%`)}`);

  interface Checkpoint {
    prompt: string;
    reasoningEffort: typeof currentReasoningEffort;
    valCombined: number;
  }
  const checkpoints: Checkpoint[] = [{
    prompt: currentPrompt,
    reasoningEffort: currentReasoningEffort,
    valCombined: valBaselineCombined,
  }];

  // ── Build initial user message ──────────────────────────────────
  const userMessage = `You have ${trainIndices.length} training examples available (indices 0-${trainIndices.length - 1}).
You can run up to ${maxTestsPerIteration} test runs this iteration. Each example ID passed to test_examples counts as one test run — re-testing the same example costs another run.

Current reasoning effort: ${currentReasoningEffort}

## Baseline Validation Scores

${formatScores('validation', valBaselineScores)}

## Goals

${formatGoal()}

## Current System Prompt
\`\`\`
${currentPrompt}
\`\`\`

Start by testing a batch of examples, then use view_input and view_output to understand the failures BEFORE writing any new prompt.`;

  // ── Agent tools ─────────────────────────────────────────────────
  const agentTools = {
    write_prompt: tool({
      description: 'Write or update the system prompt for the target model.',
      inputSchema: z.object({
        promptName: z.string().describe('Use "system"'),
        promptContent: z.string().describe('The full system prompt content'),
      }),
      execute: async ({ promptName, promptContent }) => {
        if (promptName === 'system') {
          currentPrompt = promptContent;
          console.log(`  ${dim(`[write_prompt] ${promptContent.length} chars`)}`);
          return `Prompt updated (${promptContent.length} chars).`;
        }
        return `Unknown prompt "${promptName}". Use "system".`;
      },
    }),

    test_examples: tool({
      description: `Run examples against the current system prompt. Returns a runId and compact results table. Max ${MAX_TEST_PER_CALL} examples per call. Each example ID counts as one test run against your per-iteration budget.`,
      inputSchema: z.object({
        ids: z.array(z.number()).describe(`Example indices to test. Max ${MAX_TEST_PER_CALL} per call.`),
      }),
      execute: async ({ ids }) => {
        if (ids.length > MAX_TEST_PER_CALL) {
          return `Error: requested ${ids.length} examples but max is ${MAX_TEST_PER_CALL}. Pick a smaller subset.`;
        }
        if (ids.length > budgetRemaining) {
          return `Error: requested ${ids.length} test runs but only ${budgetRemaining} remaining this iteration. Request fewer examples.`;
        }
        const globalIds = ids.filter((id) => id >= 0 && id < trainIndices.length).map((id) => trainIndices[id]);
        console.log(`  ${dim(`[test_examples] ${globalIds.length} example(s)...`)}`);
        const { runId, results } = await evaluate(currentPrompt, globalIds);
        const localResults = results.map((r, i) => ({ ...r, id: ids[i] }));
        const localRunId = runId;
        runs.set(localRunId, localResults);
        budgetRemaining -= ids.length;
        if (budgetRemaining <= 0 && stepWhenBudgetHitZero == null) {
          stepWhenBudgetHitZero = currentIterStepCount;
        }
        console.log(`  ${dim(`[test_examples] ${runId}: ${summarize(localResults)} (budget: ${budgetRemaining}/${maxTestsPerIteration})`)}`);
        const table = formatCompactTable(localResults);
        return `${localRunId} — ${summarize(localResults)}\n${table}\nTest runs remaining: ${budgetRemaining}/${maxTestsPerIteration}`;
      },
    }),

    view_input: tool({
      description: 'View input fields for specific training examples as a compact table. Inputs are static across runs.',
      inputSchema: z.object({
        exampleIds: z.array(z.number()).describe('Example indices to view'),
      }),
      execute: async ({ exampleIds }) => {
        const valid = exampleIds.filter((id) => id >= 0 && id < trainIndices.length);
        if (valid.length === 0) return 'No matching examples found.';

        const firstEx = await provider.get(trainIndices[valid[0]]);
        const inputKeys = Object.keys(firstEx.input as Record<string, unknown>);
        const rows = await Promise.all(valid.map(async (id) => {
          const ex = await provider.get(trainIndices[id]);
          const input = ex.input as Record<string, unknown>;
          const vals = inputKeys.map((k) => {
            const v = input[k];
            return typeof v === 'string' ? v : JSON.stringify(v);
          });
          return [String(id), ...vals];
        }));
        return formatTable(['#', ...inputKeys], rows);
      },
    }),

    view_output: tool({
      description: "View predictions, scores, and the target model's internal reasoning for specific examples from a previous run. Returns a compact table + reasoning.",
      inputSchema: z.object({
        runId: z.string().describe('The runId from a previous test_examples call'),
        exampleIds: z.array(z.number()).describe('Example indices to view'),
      }),
      execute: async ({ runId, exampleIds }) => {
        const runResults = runs.get(runId);
        if (!runResults) return `Run "${runId}" not found. Available: ${[...runs.keys()].join(', ')}`;

        const requested = exampleIds
          .map((id) => runResults.find((r) => r.id === id))
          .filter(Boolean) as EvalResult[];

        if (requested.length === 0) return 'No matching examples found.';

        const outputKeys = Object.keys(requested[0].modelOutput);
        const columns = ['#', ...outputKeys, ...metricKeys];
        const rows = requested.map((r) => {
          const outVals = outputKeys.map((k) => JSON.stringify(r.modelOutput[k] ?? ''));
          const scoreVals = metricKeys.map((k) => String(r.score[k] ?? 0));
          return [String(r.id), ...outVals, ...scoreVals];
        });
        const table = formatTable(columns, rows);

        const reasonings = requested
          .filter((r) => r.reasoning)
          .map((r) => `#${r.id}: ${r.reasoning}`);

        return reasonings.length > 0
          ? `${table}\n\n${reasonings.join('\n\n')}`
          : table;
      },
    }),

    set_reasoning_effort: tool({
      description: 'Set the reasoning effort level for the target model (default xhigh).',
      inputSchema: z.object({
        level: z.enum(['minimal', 'low', 'medium', 'high', 'xhigh']).describe('Reasoning effort level'),
      }),
      execute: async ({ level }) => {
        currentReasoningEffort = level;
        console.log(`  ${dim(`[set_reasoning_effort] ${level}`)}`);
        return `Reasoning effort set to ${level}.`;
      },
    }),
  };

  // ── Iteration loop ──────────────────────────────────────────────
  console.log(`  Starting optimizer agent (${teacherModel})... ${elapsed()}`);

  function extractReasoning(r: unknown): string {
    if (typeof r === 'string') return r;
    if (Array.isArray(r)) {
      return r
        .filter((p: any) => p.type === 'reasoning' && typeof p.text === 'string' && p.text !== '[REDACTED]')
        .map((p: any) => p.text)
        .join('\n');
    }
    return '';
  }

  let totalSteps = 0;
  let currentIterStepCount = 0;
  let conversationMessages: Array<{ role: string; content: string } | Record<string, unknown>> = [
    { role: 'user' as const, content: userMessage },
  ];

  for (let iter = 0; iter < maxIterations; iter++) {
    budgetRemaining = maxTestsPerIteration;
    stepWhenBudgetHitZero = undefined;
    currentIterStepCount = 0;
    const iterLabel = maxIterations > 1 ? `iter ${iter + 1}/${maxIterations}, ` : '';

    const { text: agentText, steps, response } = await aiGenerateText({
      model: openrouter.chat(teacherModel),
      system: agentSystemPrompt,
      messages: conversationMessages as any,
      providerOptions: {
        openrouter: { reasoning: { effort: 'xhigh', exclude: false } },
      },
      onStepFinish({ stepNumber, toolCalls, reasoning, finishReason }) {
        currentIterStepCount = stepNumber;
        const tools = toolCalls?.length ? ` → ${toolCalls.map((tc) => tc.toolName).join(', ')}` : '';
        console.log(`  [${iterLabel}step ${stepNumber}${tools}] (${finishReason}) ${elapsed()}`);
        const thoughts = extractReasoning(reasoning);
        if (thoughts.trim()) {
          const lines = thoughts.trim().split('\n');
          const preview = lines.slice(0, 4).join('\n');
          for (const line of preview.split('\n')) {
            console.log(`    ${dim(line)}`);
          }
          if (lines.length > 4) console.log(`    ${dim(`... (${lines.length - 4} more lines)`)}`);
        }
      },
      tools: agentTools,
      stopWhen: ({ steps }) => {
        const count = steps.length;
        if (count >= maxAgentSteps) return true;
        if (budgetRemaining <= 0 && stepWhenBudgetHitZero != null && (currentIterStepCount - stepWhenBudgetHitZero) >= 20) return true;
        return false;
      },
    });

    totalSteps += steps.length;
    conversationMessages = [...conversationMessages, ...response.messages];

    if (agentText) {
      console.log(`  ${dim('Agent:')} ${agentText.slice(0, 200)}${agentText.length > 200 ? '...' : ''}`);
    }
    console.log(`  ${green('✓')} Agent finished (${steps.length} steps) ${elapsed()}`);

    // ── Evaluate on val set ─────────────────────────────────────
    console.log(`  Evaluating on val set... ${elapsed()}`);
    const { results: valResults } = await evaluate(currentPrompt, valIndices);
    const valScores = avgScores(valResults);
    const valCombined = computeCombinedScore(valScores, effectiveWeights);

    checkpoints.push({
      prompt: currentPrompt,
      reasoningEffort: currentReasoningEffort,
      valCombined,
    });

    console.log(`  [iter ${iter + 1}/${maxIterations}] val: ${summarize(valResults)} ${elapsed()}`);

    if (valCombined > targetScore!) {
      console.log(`  ${green('✓')} Target score met on validation set`);
      break;
    }

    if (iter < maxIterations - 1) {
      conversationMessages.push({
        role: 'user',
        content: [
          `${formatScores('validation', valScores)}`,
          formatGoal(),
          `Go back to the training examples, diagnose remaining failure patterns, and improve the configuration.`,
          `You have ${maxIterations - iter - 1} iteration(s) remaining.`,
          `You can run up to ${maxTestsPerIteration} test runs this iteration. Each example ID counts as one test run.`,
        ].join('\n'),
      });
    } else {
      break;
    }
  }

  // ── Select best checkpoint by val score ─────────────────────────
  const bestCheckpoint = checkpoints.reduce((best, cp) =>
    cp.valCombined > best.valCombined ? cp : best,
  );
  const acceptedPrompt = bestCheckpoint.prompt;
  currentReasoningEffort = bestCheckpoint.reasoningEffort;

  const bestIdx = checkpoints.indexOf(bestCheckpoint);
  const bestLabel = bestIdx === 0 ? 'baseline' : `iter ${bestIdx}`;
  console.log(`  ${green('✓')} Best checkpoint: ${bestLabel} · val ${(bestCheckpoint.valCombined * 100).toFixed(1)}% ${elapsed()}`);

  // ── Final test set evaluation ───────────────────────────────────
  console.log(`  Evaluating on test set... ${elapsed()}`);
  const [baselineTest, finalTest] = await Promise.all([
    evaluate(baselinePrompt, testIndices),
    evaluate(acceptedPrompt, testIndices),
  ]);
  const baselineTestScores = avgScores(baselineTest.results);
  const baselineCombined = computeCombinedScore(baselineTestScores, effectiveWeights);
  console.log(`  ${green('✓')} Baseline (test): ${summarize(baselineTest.results)} ${elapsed()}`);
  const finalTestScores = avgScores(finalTest.results);
  const finalCombined = computeCombinedScore(finalTestScores, effectiveWeights);
  console.log(`  ${green('✓')} Final (test): ${summarize(finalTest.results)} ${elapsed()}`);

  const delta = (finalCombined - baselineCombined) * 100;
  if (delta >= 0) {
    console.log(`  ${green('✓')} combinedScore improved by ${delta.toFixed(1)} pp`);
  } else {
    console.log(`  ${dim(`combinedScore regressed by ${Math.abs(delta).toFixed(1)} pp`)}`);
  }

  const acceptedScores = finalTestScores;

  // ── Build config ─────────────────────────────────────────────────
  const evalRuns: PraxisEvalRun[] = finalTest.results.map((r) => ({
    input: r.input,
    expectedOutput: r.expectedOutput,
    modelOutput: r.modelOutput,
    score: r.score,
  }));

  return {
    config: {
      version: definition.version ?? '1.0',
      student,
      ...(definition.teacher ? { teacher: definition.teacher } : {}),
      schema: serializeSchema(definition),
      optimization: {
        instruction: acceptedPrompt,
        reasoningEffort: currentReasoningEffort,
        bestScore: acceptedScores,
        metricWeights: effectiveWeights,
        evalRuns,
        stats: {
          agentSteps: totalSteps,
          iterations: checkpoints.length,
          baselineCombined,
          finalCombined,
          elapsedMs: Date.now() - startTime,
        },
      },
    },
    testScore: acceptedScores,
    evalRuns,
  };
}

// ── Helpers ────────────────────────────────────────────────────────────



async function probeMetricKeys(def: ModelDefinition, provider: ExampleProvider): Promise<string[]> {
  if (provider.length === 0 || !def.metric) return ['default'];
  const example = await provider.get(0);
  const result = def.metric({
    input: example.input as Record<string, unknown>,
    modelOutput: (example.output ?? {}) as Record<string, unknown>,
    exampleOutput: example.output as Record<string, unknown> | undefined,
  });
  if (result == null) return ['default'];
  return Object.keys(result);
}
