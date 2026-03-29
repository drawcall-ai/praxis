---
name: praxis
description: Define, train, and run optimized LLM prompt models using Praxis. Use when the user wants to create a model definition, train prompts, or run inference with structured input/output schemas.
---

# Praxis — Prompt Model Framework

You are an expert at building optimized LLM prompt models using Praxis. Praxis uses automatic prompt optimization (DSPy-style) to turn a schema + examples into a high-quality, tested `model.config.json`.

## Workflow: Define → Train (optional) → Build → Run

### 1. Define a model

Create a `model.definition.ts` file with a default export using `defineModel`:

```ts
import { z } from 'zod';
import { defineModel } from '@drawcall/praxis';

export default defineModel({
  model: 'anthropic/claude-sonnet-4',

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

  // Single metric
  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return modelOutput.label === exampleOutput.label ? 1 : 0;
  },
});
```

#### Multiple metrics

Return a `Record<string, number>` from `metric` for multi-objective optimization. Praxis auto-selects the GEPA optimizer:

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
});
```

### 2. Train (optional)

```bash
npx praxis train
```

Reads `model.definition.ts`, runs automatic prompt optimization, and writes `model.config.json`.

Options:
- `--output, -o <path>` — output file (default: `model.config.json`)
- `--optimizer <ace|gepa|auto>` — optimizer type (default: auto)
- `--split <ratio>` — train/test split (default: 0.7)

Training requires a metric and at least 10 examples. Without training, the model works using a default instruction generated from the schema.

### 3. Build

```ts
import { buildRequest } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';
import modelConfig from './model.config.json'; // optional

const request = buildRequest(modelDefinition, { text: 'Hello' }, modelConfig);
// → { system, user, schema, model, metric?, metrics? }
```

### 4. Run

```bash
npx praxis run --text "Hello world"
```

### 5. Use in code

```ts
import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

// Without training
const { object, score } = await generateText({
  definition: modelDefinition,
  input: { text: 'Hello' },
});

// With training
import modelConfig from './model.config.json';
const { object, score } = await generateText({
  definition: modelDefinition,
  input: { text: 'Hello' },
  config: modelConfig,
});

// With AI SDK options (temperature, topP, maxRetries, mode, seed, etc.)
const { object } = await generateText({
  definition: modelDefinition,
  input: { text: 'Hello' },
  temperature: 0.5,
  maxRetries: 3,
});
```

`score` is `number | null` for single metric, `Record<string, number>` for multiple metrics.

## Key types

- **`ModelDefinition<I, O>`** — Returned by `defineModel()`. Generic: infers input/output types from Zod schemas.
- **`ModelConfig`** — The trained `model.config.json`. Optional — everything works without it.
- **`ModelRequest<O>`** — Returned by `buildRequest()`. Contains system/user prompts, schemas, model name, and metrics.
- **`ModelExample<I, O>`** — `{ input: z.infer<Input>, output?: z.infer<Output> }`. The `output` field is optional — omit it when the metric evaluates `modelOutput` without comparing to expected values.

## When to help with Praxis

- User asks to "create a model", "define a prompt model", "train a prompt", or "optimize a prompt"
- User has a classification, extraction, analysis, or generation task with structured I/O
- User wants to improve LLM output quality with training data

## How to help

1. Ask the user what task they want to solve (classification, extraction, summarization, etc.)
2. Create a `model.definition.ts` with `export default defineModel({...})`:
   - Define `input` and `output` as Zod schemas
   - Write 10+ examples with `{ input: {...}, output: {...} }` structure (`output` is optional when the metric doesn't need expected values)
   - Add a `metric` function (return a `number` for single, `Record<string, number>` for multi)
   - If the metric returns `null` during training, Praxis throws an error — ensure examples have `output` if the metric compares against it
3. Tell them to set `OPENROUTER_KEY` in `.env` then run `npx praxis train` (optional)
4. Show them how to use `buildRequest` / `generateText` in code, or `npx praxis run` from CLI

## Requirements

- `OPENROUTER_KEY` must be set in `.env`
- Node.js >= 18
- At least 10 examples in the definition (for training)
