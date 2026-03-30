# praxis

> knowledge through action

[![npm version](https://img.shields.io/npm/v/@drawcall/praxis)](https://www.npmjs.com/package/@drawcall/praxis)

**Praxis** turns a schema + examples into an optimized LLM prompt. You define the task, Praxis finds the best prompt automatically using [AX](https://axllm.dev) (TypeScript DSPy), and you get a portable `model.config.json` that works with any LLM.

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
  model: 'google/gemini-3-flash-preview',
  teacher: 'google/gemini-3.1-pro-preview', // optional: stronger model for optimization
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
    return modelOutput.sentiment === exampleOutput.sentiment ? 1 : 0;
  },
});
```

### 2. Train (optional)

```bash
npx praxis train
```

Optimizes the prompt and writes `model.config.json`. Training requires a metric and at least 10 examples — without it, Praxis generates a default instruction from the schema.

Options: `--output, -o <path>` · `--optimizer <ace|gepa|auto>` · `--split <ratio>`

### 3. Build

```ts
import { buildRequest } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';
import modelConfig from './model.config.json'; // optional

const request = buildRequest(modelDefinition, { reviewText: '...' }, modelConfig);
// → { messages, schema, model, metric? }
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

Return a `Record<string, number>` from `metric` to evaluate on multiple dimensions. Praxis auto-selects the GEPA optimizer for multi-objective optimization:

```ts
export default defineModel({
  model: 'google/gemini-3-flash-preview',

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

## Examples without expected output

The `output` field on examples is optional. This is useful for metrics that evaluate the prediction on its own (e.g. length, format, safety) without comparing to a ground truth:

```ts
export default defineModel({
  model: 'google/gemini-3-flash-preview',

  input: z.object({ text: z.string() }),
  output: z.object({ summary: z.string() }),

  examples: [
    { input: { text: 'A long article about climate change...' } },
    { input: { text: 'Breaking news: new discovery in physics...' } },
    // no output needed — metric evaluates the prediction directly
  ],

  metric: ({ modelOutput }) => {
    // Score based on output quality, not comparison to expected
    return modelOutput.summary.length > 10 && modelOutput.summary.length < 200 ? 1 : 0;
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
| `praxis train [definition]` | Optimize prompts from definition file |
| `praxis run [--definition, -d <path>]` | Run inference (config optional) |
| `praxis validate [config]` | Check config matches definition schema |

## Features

- **Schema-driven** — Define inputs and outputs with Zod, get type-safe results
- **Automatic optimization** — Finds the best prompt through systematic search
- **Portable output** — `model.config.json` works with any LLM provider via OpenRouter
- **Works without training** — Definitions alone produce results; training makes them better
- **Single or multi-metric** — Return a `number` for one score, or a `Record<string, number>` for multi-objective optimization
- **CLI + code** — Run from the terminal or import into your app
- **Agent-friendly** — Install as a skill and let your coding agent write definitions

## Requirements

- Node.js >= 18
- `OPENROUTER_KEY` in `.env` ([openrouter.ai](https://openrouter.ai))

## License

MIT
