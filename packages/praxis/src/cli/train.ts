import { resolve, dirname } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { generateText as aiGenerateText, Output, tool, stepCountIs, wrapLanguageModel, extractJsonMiddleware } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { z } from 'zod';
import pMap from 'p-map';
import { serializeSchema, formatSchemaForPrompt } from '../schema.js';
import { buildDefaultSystemPrompt } from '../prompt.js';
import { detectMismatches } from '../validate.js';
import { computeCombinedScore } from '../types.js';
import type {
  ModelDefinition,
  ModelExample,
  ModelConfig,
  PraxisEvalRun,
  TrainOptions,
} from '../types.js';
import {
  DEFAULT_CONFIG,
  bold, dim, green,
  resolveDefinitionPath, requireEnvKey, loadDefinition,
  validateDefinition, formatZodSchema,
} from './utils.js';
import { formatTable } from '../format.js';

const EVAL_CONCURRENCY = 20;
const MAX_TEST_PER_CALL = 20;
const MAX_AGENT_STEPS = 40;

// ── CLI handler ────────────────────────────────────────────────────────

export async function handleTrain(opts: { definition?: string; output?: string; split: string; force: boolean }) {
  const definitionPath = await resolveDefinitionPath(opts.definition);
  const defDir = dirname(definitionPath);
  const defaultOutput = resolve(defDir, DEFAULT_CONFIG);

  const options: TrainOptions = {
    definitionPath,
    output: opts.output ? resolve(opts.output) : defaultOutput,
    split: parseFloat(opts.split),
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

  const resolvedExamples = Array.isArray(definition.examples)
    ? definition.examples
    : await definition.examples();
  const resolvedDef = { ...definition, examples: resolvedExamples };

  console.log('');
  const teacherLabel = definition.teacher ? ` · teacher: ${definition.teacher}` : '';
  console.log(`  ${bold(definition.student)} ${dim(`${resolvedExamples.length} examples · ${options.split}/${(1 - options.split).toFixed(1)} split${teacherLabel}`)}`);
  console.log('');
  console.log(formatZodSchema(definition));
  console.log('');

  const { config, testScore } = await train(resolvedDef, options);

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
}

async function train(
  definition: ModelDefinition,
  options: Pick<TrainOptions, 'split'>,
): Promise<TrainResult> {
  const { student } = definition;
  const startTime = Date.now();
  const elapsed = () => dim(`${((Date.now() - startTime) / 1000).toFixed(0)}s`);

  if (!Array.isArray(definition.examples)) {
    throw new Error('train() expects resolved examples. Use resolveExamples() first.');
  }
  const examples = definition.examples;

  if (examples.length < 10) {
    throw new Error(`At least 10 examples required, got ${examples.length}`);
  }

  if (!definition.metric) {
    throw new Error('Definition must export a "metric" function');
  }

  const metricKeys = probeMetricKeys(definition, examples);
  const weights = definition.metricWeights;

  // ── Train/test split ─────────────────────────────────────────────
  const shuffled = [...examples].sort(() => Math.random() - 0.5);
  const testCount = Math.min(Math.round(shuffled.length * (1 - options.split)), 50);
  const splitIdx = shuffled.length - testCount;
  const trainExamples = shuffled.slice(0, splitIdx);
  const testExamples = shuffled.slice(splitIdx);

  console.log(`  ${dim(`${trainExamples.length} train / ${testExamples.length} test`)}`);
  console.log(`  ${dim(`metric keys: ${metricKeys.join(', ')}`)}`);
  if (weights) console.log(`  ${dim(`weights: ${JSON.stringify(weights)}`)}`);

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
  let currentTemperature = 0;

  async function evaluate(
    systemPrompt: string,
    exampleSet: ModelExample[],
    ids: number[],
  ): Promise<{ runId: string; results: EvalResult[] }> {
    const runId = nextRunId();
    const validIds = ids.filter((id) => exampleSet[id] != null);
    const resultMap = new Map<number, EvalResult>();

    const model = wrapLanguageModel({
      model: openrouter.chat(student),
      middleware: extractJsonMiddleware(),
    });

    await pMap(validIds, async (id) => {
      const ex = exampleSet[id];
      const input = ex.input as Record<string, unknown>;
      const userContent = Object.entries(input)
        .map(([key, value]) => `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`)
        .join('\n');
      try {
        const result = await aiGenerateText({
          model,
          system: systemPrompt,
          prompt: userContent,
          output: Output.object({ schema: definition.output }),
          temperature: currentTemperature,
          providerOptions: {
            openrouter: { reasoning: { effort: 'low', exclude: false } },
          },
        });
        const modelOutput = (result.output ?? {}) as Record<string, unknown>;
        let reasoning = '';
        if (typeof result.reasoning === 'string') {
          reasoning = result.reasoning;
        } else if (Array.isArray(result.reasoning)) {
          reasoning = (result.reasoning as any[])
            .filter((r) => r.type === 'reasoning' && typeof r.text === 'string' && r.text !== '[REDACTED]')
            .map((r) => r.text)
            .join('\n');
        }
        const score = definition.metric!({
          input: ex.input as Record<string, unknown>,
          modelOutput: modelOutput as any,
          exampleOutput: ex.output as any,
        }) ?? Object.fromEntries(metricKeys.map((k) => [k, 0]));

        resultMap.set(id, {
          id, input, modelOutput,
          expectedOutput: (ex.output ?? {}) as Record<string, unknown>,
          score, reasoning,
        });
      } catch {
        resultMap.set(id, {
          id, input,
          modelOutput: {},
          expectedOutput: (ex.output ?? {}) as Record<string, unknown>,
          score: Object.fromEntries(metricKeys.map((k) => [k, 0])),
          reasoning: '',
        });
      }
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
        const w = weights?.[k];
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
      vals.push(computeCombinedScore(score, weights).toFixed(2));
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
    const combined = computeCombinedScore(avg, weights);
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

  // ── Baseline eval on test set ────────────────────────────────────
  console.log(`  Evaluating baseline on test set... ${elapsed()}`);
  const allTestIds = testExamples.map((_, i) => i);
  const baselineTest = await evaluate(currentPrompt, testExamples, allTestIds);
  const baselineTestScores = avgScores(baselineTest.results);
  const baselineCombined = computeCombinedScore(baselineTestScores, weights);
  console.log(`  ${green('✓')} Baseline (test): ${summarize(baselineTest.results)} ${elapsed()}`);

  // ── Agent system prompt ──────────────────────────────────────────
  const allTrainIds = trainExamples.map((_, i) => i);

  const schemaDesc = formatSchemaForPrompt(definition);
  const metricDesc = metricKeys.length === 1 && metricKeys[0] === 'default'
    ? 'Single score (0 or 1).'
    : `Multi-metric: ${metricKeys.join(', ')}.${weights ? ` Weights: ${JSON.stringify(weights)}.` : ' Equal weights.'}`;

  const agentSystemPrompt = `You are a prompt optimization agent. Your goal is to write the best possible system prompt for a target LLM.

## Task Definition
${definition.name ? `Task: ${definition.name}` : ''}
${definition.description ? `Description: ${definition.description}` : ''}

${schemaDesc}

**Important:** The target model has internal reasoning/thinking enabled. It thinks before answering, then outputs structured JSON.

**Metric:** ${metricDesc} The table shows per-metric scores and a \`combinedScore\` (weighted average).

## What a system prompt can contain

A system prompt is not limited to a simple instruction. You can include any combination of:

- **Role and persona** — who the model should act as
- **Step-by-step algorithms** — explicit reasoning procedures to follow
- **Few-shot examples** — worked input/output pairs showing correct behavior
- **Lookup tables or mappings** — reference data the model should consult
- **Rules and constraints** — explicit "always do X" / "never do Y" directives
- **Edge case handling** — specific instructions for known tricky cases
- **Common mistake warnings** — "the model often confuses X with Y, be careful"

The current prompt is just a starting point. You have full creative freedom.

## Tools

- **write_prompt(promptName, promptContent)**: update the system prompt
- **test_examples(ids)**: run up to ${MAX_TEST_PER_CALL} examples. Returns a runId + compact score table.
- **view_input(exampleIds)**: view raw input text for specific examples. Inputs are static across runs.
- **view_output(runId, exampleIds)**: view predictions, scores, AND the target model's internal reasoning/thoughts. Use this to see HOW the model reasoned and where its logic breaks down.
- **set_temperature(temperature)**: set the target model's temperature (default 0).

## MANDATORY workflow — you MUST follow these steps in order:

1. **Test** a subset of examples to get a baseline score and a runId
2. **Diagnose** — call view_input on failing example IDs to see what the questions are, then call view_output on the same runId+IDs to see how the model REASONED and where it went wrong. DO NOT skip this step.
3. **Analyze** the reasoning errors — look for patterns in how the model thinks
4. **Write** an improved prompt that specifically addresses the reasoning errors you observed
5. **Test** on the failing examples to see if your fix works
6. **Repeat** from step 2 if failures remain

NEVER write a prompt without first looking at actual examples and model reasoning via view_input and view_output.`;

  const userMessage = `You have ${trainExamples.length} training examples available (indices 0-${trainExamples.length - 1}).

## Current System Prompt
\`\`\`
${currentPrompt}
\`\`\`

Start by testing a batch of examples, then use view_input and view_output to understand the failures BEFORE writing any new prompt.`;

  // ── Agent loop ───────────────────────────────────────────────────
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

  const { text: agentText, steps } = await aiGenerateText({
    model: openrouter.chat(teacherModel),
    system: agentSystemPrompt,
    prompt: userMessage,
    providerOptions: {
      openrouter: { reasoning: { effort: 'high', exclude: false } },
    },
    onStepFinish({ stepNumber, text, toolCalls, reasoning, finishReason }) {
      const tools = toolCalls?.length ? ` → ${toolCalls.map((tc) => tc.toolName).join(', ')}` : '';
      console.log(`  [step ${stepNumber}${tools}] (${finishReason}) ${elapsed()}`);
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
    tools: {
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
        description: `Run examples against the current system prompt. Returns a runId and compact results table. Max ${MAX_TEST_PER_CALL} examples per call.`,
        inputSchema: z.object({
          ids: z.array(z.number()).describe(`Example indices to test. Max ${MAX_TEST_PER_CALL} per call.`),
        }),
        execute: async ({ ids }) => {
          if (ids.length > MAX_TEST_PER_CALL) {
            return `Error: requested ${ids.length} examples but max is ${MAX_TEST_PER_CALL}. Pick a smaller subset.`;
          }
          console.log(`  ${dim(`[test_examples] ${ids.length} example(s)...`)}`);
          const { runId, results } = await evaluate(currentPrompt, trainExamples, ids);
          console.log(`  ${dim(`[test_examples] ${runId}: ${summarize(results)}`)}`);
          const table = formatCompactTable(results);
          const failed = results.filter((r) => computeCombinedScore(r.score, weights) < 1).map((r) => r.id);
          return `${runId} — ${summarize(results)}\n${table}\n\nFailed IDs: ${failed.length > 0 ? failed.join(', ') : 'none'}`;
        },
      }),

      view_input: tool({
        description: 'View input fields for specific training examples as a compact table. Inputs are static across runs.',
        inputSchema: z.object({
          exampleIds: z.array(z.number()).describe('Example indices to view'),
        }),
        execute: async ({ exampleIds }) => {
          const valid = exampleIds.filter((id) => trainExamples[id] != null);
          if (valid.length === 0) return 'No matching examples found.';

          const inputKeys = Object.keys(trainExamples[valid[0]].input as Record<string, unknown>);
          const rows = valid.map((id) => {
            const input = trainExamples[id].input as Record<string, unknown>;
            const vals = inputKeys.map((k) => {
              const v = input[k];
              return typeof v === 'string' ? v : JSON.stringify(v);
            });
            return [String(id), ...vals];
          });
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
          const columns = ['#', 'status', ...outputKeys, ...metricKeys];
          const rows = requested.map((r) => {
            const status = computeCombinedScore(r.score, weights) >= 1 ? 'OK' : 'FAIL';
            const outVals = outputKeys.map((k) => JSON.stringify(r.modelOutput[k] ?? ''));
            const scoreVals = metricKeys.map((k) => String(r.score[k] ?? 0));
            return [String(r.id), status, ...outVals, ...scoreVals];
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

      set_temperature: tool({
        description: 'Set the temperature for the target model (default 0).',
        inputSchema: z.object({
          temperature: z.number().min(0).max(2).describe('Temperature value'),
        }),
        execute: async ({ temperature }) => {
          currentTemperature = temperature;
          console.log(`  ${dim(`[set_temperature] ${temperature}`)}`);
          return `Temperature set to ${temperature}.`;
        },
      }),
    },
    stopWhen: stepCountIs(MAX_AGENT_STEPS),
  });

  if (agentText) {
    console.log(`  ${dim('Agent:')} ${agentText.slice(0, 200)}${agentText.length > 200 ? '...' : ''}`);
  }

  console.log(`  ${green('✓')} Agent finished (${steps.length} steps) ${elapsed()}`);

  // ── Final eval on test set ───────────────────────────────────────
  console.log(`  Evaluating final prompt on test set... ${elapsed()}`);
  const finalTest = await evaluate(currentPrompt, testExamples, allTestIds);
  const finalTestScores = avgScores(finalTest.results);
  const finalCombined = computeCombinedScore(finalTestScores, weights);
  console.log(`  Final (test): ${summarize(finalTest.results)} ${elapsed()}`);

  // ── Acceptance check ─────────────────────────────────────────────
  let acceptedPrompt = currentPrompt;
  let acceptedScores = finalTestScores;

  if (finalCombined < baselineCombined) {
    console.log(`  ${dim('Final combinedScore regressed — keeping baseline prompt.')}`);
    acceptedPrompt = buildDefaultSystemPrompt(definition);
    acceptedScores = baselineTestScores;
  } else {
    const improvement = ((finalCombined - baselineCombined) * 100).toFixed(1);
    console.log(`  ${green('✓')} Accepted — combinedScore improved by ${improvement} pp`);
  }

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
        ...(currentTemperature !== 0 ? { temperature: currentTemperature } : {}),
        bestScore: acceptedScores,
        evalRuns,
        stats: {
          agentSteps: steps.length,
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

function probeMetricKeys(def: ModelDefinition, examples: ModelExample[]): string[] {
  const example = examples[0];
  if (!example || !def.metric) return ['default'];
  const result = def.metric({
    input: example.input as Record<string, unknown>,
    modelOutput: (example.output ?? {}) as Record<string, unknown>,
    exampleOutput: example.output as Record<string, unknown> | undefined,
  });
  if (result == null) return ['default'];
  return Object.keys(result);
}
