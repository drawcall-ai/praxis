# praxis

> knowledge through action

[![npm version](https://img.shields.io/npm/v/@drawcall/praxis)](https://www.npmjs.com/package/@drawcall/praxis)

**Praxis** turns a schema + examples into an optimized LLM prompt. You define the task, Praxis finds the best prompt automatically using an agentic optimizer that diagnoses failures and iterates, and you get a portable `model.config.json` that works with any LLM.

## Install

```bash
npm install @drawcall/praxis
```

Or add it as a skill for your coding agent ([Claude Code, Cursor, Copilot, Codex, Gemini CLI, and more](https://agentskills.io)):

```bash
npx skills add drawcall-ai/praxis
```

## Workflow: Define → Train (optional) → Build → Run

### 1. Define

Create `model.definition.ts` with a schema, examples, and a metric:

```ts
import { z } from 'zod';
import { defineModel } from '@drawcall/praxis';

export default defineModel({
  name: 'Sentiment Analyzer', // optional: human-readable label shown in the web UI
  student: 'google/gemini-3-flash-preview',
  version: '1.0', // optional: bump to trigger retraining when examples or metrics change
  teacher: 'google/gemini-3.1-pro-preview', // optional: stronger model used as the optimizer agent
  description: 'Classify product review sentiment with confidence.', // optional: task description

  input: z.object({
    reviewText: z.string().describe('The product review to analyze'),
  }),

  output: z.object({
    sentiment: z.enum(['positive', 'negative', 'neutral']),
    confidence: z.number().describe('Confidence score 0-1'),
  }),

  examples: [
    { input: { reviewText: 'Love this product!' }, output: { sentiment: 'positive', confidence: 0.95 } },
    { input: { reviewText: 'Terrible quality.' }, output: { sentiment: 'negative', confidence: 0.92 } },
    { input: { reviewText: 'It works fine.' }, output: { sentiment: 'neutral', confidence: 0.75 } },
    // ... at least 10 examples
  ],

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return { accuracy: modelOutput.sentiment === exampleOutput.sentiment ? 1 : 0 };
  },
});
```

### 2. Train (optional)

```bash
npx praxis train
```

Auto-discovers `model.definition.ts` (or `.js`) anywhere in the project via glob, optimizes the prompt, and writes `model.config.json` next to the definition file. You can also pass an explicit path: `npx praxis train -d path/to/model.definition.ts`.

Training requires a metric and at least 10 examples — without it, Praxis generates a default instruction from the schema.

CLI options: `--output, -o <path>` · `--force`

Training parameters are configured in the definition:

```ts
export default defineModel({
  // ...
  split: [0.7, 0.15, 0.15], // train/val/test split (default: [0.7, 0.15, 0.15])
  targetScore: 0.9,    // target combined score on validation set (enables iterative optimization)
  maxIterations: 5,    // max optimization iterations when target not met (default: 5)
  maxTestsPerIteration: 100, // max example test runs the optimizer can use per iteration (default: 100)
});
```

Training is skipped if the config is already up to date. If `version` is set in the definition, schema/student/teacher changes without a version bump produce an error — bump the version to retrain.

#### How training works

The optimizer splits examples into **train / val / test** sets and runs an agentic loop powered by the teacher model (or student if no teacher is set). It:

1. Evaluates the default prompt on the validation set to establish a baseline
2. Tests subsets of training examples, inspects failing inputs and the target model's reasoning
3. Writes improved prompts that address the specific failure patterns it observes
4. After each iteration, evaluates on the validation set — if `targetScore` is set and not met, continues the conversation with the agent for up to `maxIterations` rounds
5. Selects the best prompt by validation score, then confirms on the held-out test set

A built-in `tokenEfficiency` metric (default weight 0.1) penalizes inference cost overhead — it compares the irreducible cost (user input + output) against actual cost (system prompt + thinking + output). The optimizer starts with maximum reasoning effort (`xhigh`) and finds the best quality/cost tradeoff.

The agent has full creative freedom over the system prompt — it can include step-by-step algorithms, few-shot examples, lookup tables, edge case warnings, and more.

### 3. Build

```ts
import { buildRequest } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';
import modelConfig from './model.config.json'; // optional

const request = buildRequest(modelDefinition, { reviewText: '...' }, modelConfig);
// → { messages, schema, student, metric? }
```

### 4. Run

```ts
import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

const { output, score } = await generateText({
  definition: modelDefinition,
  input: { reviewText: 'Amazing product!' },
});
```

Or from the CLI:

```bash
npx praxis run --reviewText "Great product!"
```

## Multiple metrics

Return a `Record<string, number>` from `metric` to evaluate on multiple dimensions. Use optional `metricWeights` to control importance — if omitted, all metrics are weighted equally:

```ts
export default defineModel({
  student: 'google/gemini-3-flash-preview',

  input: z.object({
    reviewText: z.string().describe('The product review to evaluate'),
  }),

  output: z.object({
    quality: z.enum(['helpful', 'unhelpful']),
    sentiment: z.enum(['positive', 'negative', 'neutral']),
    hasSpecificDetails: z.boolean(),
  }),

  examples: [
    { input: { reviewText: 'Battery lasts 8 hours, charges in 30 min.' }, output: { quality: 'helpful', sentiment: 'positive', hasSpecificDetails: true } },
    { input: { reviewText: 'Bad.' }, output: { quality: 'unhelpful', sentiment: 'negative', hasSpecificDetails: false } },
    // ... at least 10 examples
  ],

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return {
      quality: modelOutput.quality === exampleOutput.quality ? 1 : 0,
      sentiment: modelOutput.sentiment === exampleOutput.sentiment ? 1 : 0,
      details: modelOutput.hasSpecificDetails === exampleOutput.hasSpecificDetails ? 1 : 0,
    };
  },

  metricWeights: { quality: 0.5, sentiment: 0.3, details: 0.2 }, // optional
});
```

`generateText` returns per-metric scores:

```ts
const { output, score } = await generateText({
  definition: modelDefinition,
  input: { reviewText: '...' },
});
// score → { quality: 1, sentiment: 1, details: 0 }
```

## Async examples

Examples can be an async function instead of a plain array — useful for loading from a database or generating synthetic data:

```ts
export default defineModel({
  student: 'google/gemini-3-flash-preview',

  input: z.object({ text: z.string() }),
  output: z.object({ label: z.enum(['a', 'b', 'c']) }),

  examples: async () => {
    const rows = await fetchTrainingData();
    return rows.map(r => ({ input: { text: r.text }, output: { label: r.label } }));
  },

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return { accuracy: modelOutput.label === exampleOutput.label ? 1 : 0 };
  },
});
```

You can also use top-level `await` for simpler cases — no special syntax needed.

## Versioning

The optional `version` field signals when to retrain. Schema, student, and teacher changes are detected automatically, but changes to metric logic or example content are not. Bump the version to trigger retraining:

```ts
export default defineModel({
  student: 'google/gemini-3-flash-preview',
  version: '1.1', // was '1.0' — bumped because metric changed

  // ...
});
```

If `version` is set and schema/student/teacher change without a version bump, `praxis train` errors — this prevents accidental drift. Use `--force` to bypass.

## Examples without expected output

The `output` field on examples is optional. This is useful for metrics that evaluate the prediction on its own (e.g. length, format, safety) without comparing to a ground truth:

```ts
export default defineModel({
  student: 'google/gemini-3-flash-preview',

  input: z.object({ text: z.string() }),
  output: z.object({ summary: z.string() }),

  examples: [
    { input: { text: 'A long article about climate change...' } },
    { input: { text: 'Breaking news: new discovery in physics...' } },
  ],

  metric: ({ modelOutput }) => {
    return { quality: modelOutput.summary.length > 10 && modelOutput.summary.length < 200 ? 1 : 0 };
  },
});
```

If a metric returns `null` during training, Praxis throws an error — this usually means the metric expects `exampleOutput` but the examples don't provide it.

## Runnable examples

See the [examples/](examples/) directory for runnable projects:

- **[sentiment](examples/sentiment/)** — Single metric: classify review sentiment
- **[review-quality](examples/review-quality/)** — Multiple metrics: evaluate quality, sentiment, and detail

Run them:

```bash
pnpm install
pnpm --filter @drawcall/example-sentiment dev
pnpm --filter @drawcall/example-review-quality dev
```

## CLI

| Command | Description |
|---------|-------------|
| `praxis train [-d <path>] [-f]` | Optimize prompts (auto-discovers definition via glob) |
| `praxis run [-d <path>] [-c <config>]` | Run inference (auto-discovers definition and config) |
| `praxis view [-d <path>] [-p <port>]` | Launch web UI to inspect eval runs and test manually |
| `praxis check [-d <path>] [-c <config>]` | Check config matches definition (schema, student, version, weights) |

## Features

- **Schema-driven** — Define inputs and outputs with Zod, get type-safe results
- **Agentic optimization** — An AI agent diagnoses failures, inspects model reasoning, and writes targeted prompt improvements
- **Iterative optimization** — Set `targetScore` in the definition and the optimizer iterates up to N times, using a validation set for checkpoint selection
- **Token efficiency** — Built-in metric penalizes reasoning cost overhead, pushing the optimizer to find the best quality/cost tradeoff
- **Portable output** — `model.config.json` works with any LLM provider via OpenRouter
- **Works without training** — Definitions alone produce results; training makes them better
- **Multi-metric with weights** — Return a `Record<string, number>` with optional `metricWeights` for weighted optimization
- **Observable** — Every optimizer step is logged: agent thinking, prompt iterations, model reasoning
- **CLI + code** — Run from the terminal or import into your app
- **Agent-friendly** — Install as a skill and let your coding agent write definitions

## Requirements

- Node.js >= 18
- `OPENROUTER_KEY` in `.env` ([openrouter.ai](https://openrouter.ai))

## License

MIT
