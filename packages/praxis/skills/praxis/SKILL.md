---
name: praxis
description: Define, train, and run optimized LLM prompt models using Praxis. Use when the user wants to create a model definition, train prompts, or run inference with structured input/output schemas.
---

# Praxis — Prompt Model Framework

You are an expert at building optimized LLM prompt models using Praxis. Praxis uses an agentic optimizer to turn a schema + examples into a high-quality, tested `model.config.json`.

## Workflow: Define → Train (optional) → Build → Run

### 1. Define a model

Create a `model.definition.ts` file with a default export using `defineModel`:

```ts
import { z } from 'zod';
import { defineModel } from '@drawcall/praxis';

export default defineModel({
  name: 'Text Classifier',
  student: 'google/gemini-3-flash-preview',
  teacher: 'google/gemini-3.1-pro-preview',
  description: 'Classify input text into categories.',
  version: '1.0',

  input: z.object({
    text: z.string().describe('The input text'),
  }),

  output: z.object({
    label: z.enum(['a', 'b', 'c']).describe('Classification label'),
    confidence: z.number().describe('Confidence score 0-1'),
  }),

  examples: [
    { input: { text: '...' }, output: { label: 'a', confidence: 0.95 } },
    // ... at least 10 examples with { input, output }
  ],

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return { accuracy: modelOutput.label === exampleOutput.label ? 1 : 0 };
  },

  // Training options (all optional)
  metricWeights: { accuracy: 0.7, confidence: 0.3 },
  split: [0.7, 0.15, 0.15],
  targetScore: 0.9,
  maxIterations: 5,
  maxTestsPerIteration: 100,
});
```

#### Definition fields

**Required:**
- `student` — OpenRouter model ID (e.g. `'google/gemini-3-flash-preview'`)
- `input` — Zod schema for the input
- `output` — Zod schema for the output
- `examples` — Array of `{ input, output? }` (at least 10 for training). Can also be an async function or a lazy provider.

**Optional — model:**
- `name` — Human-readable label shown in the web UI
- `teacher` — Stronger model used as the optimizer agent (e.g. `'google/gemini-3.1-pro-preview'`)
- `description` — Short task description included in the prompt
- `version` — Bump to trigger retraining when examples or metrics change

**Optional — metrics:**
- `metric` — Scoring function. Return `Record<string, number>` (e.g. `{ accuracy: 1 }`) or `null`. Required for training.
- `metricWeights` — Control importance of different metrics (equal weights if omitted). A built-in `tokenEfficiency` metric (default weight 0.1) penalizes reasoning cost overhead — set `metricWeights: { tokenEfficiency: 0 }` to disable it.

**Optional — training:**
- `split` — Train/val/test split as `[train, val, test]` (default: `[0.7, 0.15, 0.15]`). Must sum to 1.0.
- `targetScore` — Target combined score on validation set. When set, the optimizer iterates up to `maxIterations` times, continuing the agent conversation when the val score doesn't meet the target.
- `maxIterations` — Max optimization iterations (default: `5`). Only takes effect when `targetScore` is set.
- `maxTestsPerIteration` — Max example test runs the optimizer agent can use per iteration (default: `100`). Each example ID passed to `test_examples` costs one run. Limits agent steps to `2 × maxTestsPerIteration`.

#### Async examples

Examples can be an async function instead of a plain array. Useful for loading from a database or generating synthetic data:

```ts
export default defineModel({
  // ...
  examples: async () => {
    const rows = await fetchFromAPI('/training-data');
    return rows.map(r => ({ input: { text: r.text }, output: { label: r.label } }));
  },
});
```

You can also use top-level `await` for simpler cases:

```ts
const data = await loadFromDB();
export default defineModel({
  // ...
  examples: data.map(r => ({ input: { text: r.text }, output: { label: r.label } })),
});
```

### 2. Train (optional)

```bash
npx praxis train
```

Auto-discovers `model.definition.ts` (or `.js`) anywhere in the project via glob. Runs an agentic optimizer that diagnoses failures, inspects model reasoning, and writes targeted prompt improvements. Outputs `model.config.json` next to the definition file.

CLI options:
- `-d, --definition <path>` — definition file (default: auto-discover)
- `-o, --output <path>` — output file (default: `model.config.json` next to the definition)
- `-f, --force` — skip version/schema guard and force retraining

All training parameters (split, targetScore, maxIterations, metricWeights) are configured in the definition, not the CLI.

Training requires a metric and at least 10 examples. Without training, the model works using a default instruction generated from the schema.

### 3. Build

```ts
import { buildRequest } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';
import modelConfig from './model.config.json'; // optional

const request = buildRequest(modelDefinition, { text: 'Hello' }, modelConfig);
// → { messages, schema, student, metric? }
```

### 4. Run

```bash
npx praxis run --text "Hello world"
```

The definition and config are auto-discovered via glob. Use `-d <path>` to specify a definition explicitly.

### 5. Use in code

```ts
import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

// Without training
const { output, score } = await generateText({
  definition: modelDefinition,
  input: { text: 'Hello' },
});

// With training
import modelConfig from './model.config.json';
const { output, score } = await generateText({
  definition: modelDefinition,
  input: { text: 'Hello' },
  config: modelConfig,
});

// With AI SDK options (temperature, topP, maxRetries, mode, seed, etc.)
const { output } = await generateText({
  definition: modelDefinition,
  input: { text: 'Hello' },
  temperature: 0.5,
  maxRetries: 3,
});
```

`score` is `Record<string, number> | null` — always a per-metric record.

### 6. Check

```bash
npx praxis check
```

Verifies the config matches the definition (schema, student, teacher, version, metricWeights). Suggests running `npx praxis train` to fix mismatches.

## Key types

- **`ModelDefinition<I, O>`** — Returned by `defineModel()`. Fields: `name?`, `student`, `version?`, `teacher?`, `description?`, `input`, `output`, `examples`, `metric?`, `metricWeights?`, `split?`, `targetScore?`, `maxIterations?`, `maxTestsPerIteration?`.
- **`ModelConfig`** — The trained `model.config.json`. Optional — everything works without it.
- **`ModelRequest<O>`** — Returned by `buildRequest()`. Contains `messages` (AI SDK `ModelMessage[]`), `schema`, `student`, and `metric?`.
- **`ModelExample<I, O>`** — `{ input: z.infer<Input>, output?: z.infer<Output> }`. The `output` field is optional — omit it when the metric evaluates `modelOutput` without comparing to expected values.
- **`ModelExamples<I, O>`** — `ModelExample[] | (() => Promise<ModelExample[]>)`. Examples can be a plain array or an async function.

## When to help with Praxis

- User asks to "create a model", "define a prompt model", "train a prompt", or "optimize a prompt"
- User has a classification, extraction, analysis, or generation task with structured I/O
- User wants to improve LLM output quality with training data

## How to help

1. Ask the user what task they want to solve (classification, extraction, summarization, etc.)
2. Create a `model.definition.ts` with `export default defineModel({...})`:
   - Set `student` to an OpenRouter model ID
   - Define `input` and `output` as Zod schemas
   - Write 10+ examples with `{ input, output }` structure
   - Add a `metric` function returning `Record<string, number>` or `null`
   - Optionally set `name`, `teacher`, `description`, `version`
   - Optionally set `metricWeights`, `split`, `targetScore`, `maxIterations`, `maxTestsPerIteration`
   - If the metric returns `null` during training, Praxis throws an error — ensure examples have `output` if the metric compares against it
3. Tell them to set `OPENROUTER_KEY` in `.env` then run `npx praxis train`
4. Show them how to use `buildRequest` / `generateText` in code, or `npx praxis run` from CLI
5. Use `npx praxis check` to verify the config matches the definition

## Requirements

- `OPENROUTER_KEY` must be set in `.env`
- Node.js >= 18
- At least 10 examples in the definition (for training)
