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
  name: 'Text Classifier', // optional: human-readable label shown in the web UI
  student: 'google/gemini-3-flash-preview',
  teacher: 'google/gemini-3.1-pro-preview', // optional: stronger model used as the optimizer agent
  description: 'Classify input text into categories.', // optional: task description included in prompt

  input: z.object({
    text: z.string().describe('The input text'),
  }),

  output: z.object({
    label: z.enum(['a', 'b', 'c']).describe('Classification label'),
    confidence: z.number().describe('Confidence score 0-1'),
  }),

  version: '1.0', // optional: bump to trigger retraining when examples or metrics change

  examples: [
    { input: { text: '...' }, output: { label: 'a', confidence: 0.95 } },
    // ... at least 10 examples with { input, output }
  ],

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return { accuracy: modelOutput.label === exampleOutput.label ? 1 : 0 };
  },
});
```

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

#### Multiple metrics

Return a `Record<string, number>` from `metric` to evaluate on multiple dimensions. Use optional `metricWeights` to control importance:

```ts
export default defineModel({
  // ... input, output, examples ...

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return {
      accuracy: modelOutput.label === exampleOutput.label ? 1 : 0,
      confidence: Math.abs(modelOutput.confidence - exampleOutput.confidence) < 0.1 ? 1 : 0,
    };
  },

  metricWeights: { accuracy: 0.7, confidence: 0.3 }, // optional: equal weights if omitted
});
```

### 2. Train (optional)

```bash
npx praxis train
```

Auto-discovers `model.definition.ts` (or `.js`) anywhere in the project via glob. Runs an agentic optimizer that diagnoses failures, inspects model reasoning, and writes targeted prompt improvements. Outputs `model.config.json` next to the definition file.

Options:
- `--output, -o <path>` — output file (default: `model.config.json` next to the definition)
- `--split <ratio>` — train/test split (default: 0.7)
- `--force` — skip version/schema guard and force retraining

You can also pass an explicit definition path: `npx praxis train -d path/to/model.definition.ts`

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

## Key types

- **`ModelDefinition<I, O>`** — Returned by `defineModel()`. Fields: `name?`, `student`, `version?`, `teacher?`, `description?`, `input`, `output`, `examples`, `metric?`, `metricWeights?`.
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
   - Optionally set `name` — a human-readable label displayed in the web UI (e.g. `'Sentiment Analyzer'`)
   - Set `student` to an OpenRouter model ID (e.g. `'google/gemini-3-flash-preview'`)
   - Optionally set `version` — bump it when changing examples or metrics to trigger retraining
   - Optionally set `teacher` to a stronger model used as the optimizer agent (e.g. `'google/gemini-3.1-pro-preview'`)
   - Optionally set `description` — a short task description included in the prompt
   - Define `input` and `output` as Zod schemas
   - Write 10+ examples with `{ input: {...}, output: {...} }` structure (`output` is optional when the metric doesn't need expected values)
   - Add a `metric` function — always return `Record<string, number>` (e.g. `{ accuracy: 1 }`) or `null`
   - Optionally add `metricWeights` to control importance of different metrics
   - If the metric returns `null` during training, Praxis throws an error — ensure examples have `output` if the metric compares against it
3. Tell them to set `OPENROUTER_KEY` in `.env` then run `npx praxis train` (auto-discovers the definition; optional)
4. Show them how to use `buildRequest` / `generateText` in code, or `npx praxis run` from CLI

## Requirements

- `OPENROUTER_KEY` must be set in `.env`
- Node.js >= 18
- At least 10 examples in the definition (for training)
