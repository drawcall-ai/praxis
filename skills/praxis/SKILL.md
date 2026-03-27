---
name: praxis
description: Define, train, and run optimized LLM prompt models using Praxis. Use when the user wants to create a model definition, train prompts, or run inference with structured input/output schemas.
---

# Praxis — Prompt Model Framework

You are an expert at building optimized LLM prompt models using Praxis. Praxis uses automatic prompt optimization (DSPy-style) to turn a schema + examples into a high-quality, tested `model.config.json`.

## Workflow: Define → Train → Run

### 1. Define a model

Create a `model.definition.ts` file. It must export:

- **`model`** — the LLM model string (e.g. `'anthropic/claude-sonnet-4'`)
- **`schema`** — `{ input: z.object({...}), output: z.object({...}) }` using Zod
- **`examples`** — at least 10 input/output pairs for training
- **`metric`** — a function `({ prediction, example }) => 0 | 1` scoring correctness

```ts
import { z } from 'zod';

export const model = 'anthropic/claude-sonnet-4';

export const schema = {
  input: z.object({
    text: z.string().describe('The input text'),
  }),
  output: z.object({
    label: z.enum(['a', 'b', 'c']).describe('Classification label'),
    confidence: z.number().describe('Confidence score 0-1'),
  }),
};

export const examples = [
  { input: { text: '...' }, label: 'a', confidence: 0.95 },
  // ... at least 10 examples
];

export const metric = ({ prediction, example }) => {
  if (!example) return null;
  return prediction.label === example.label ? 1 : 0;
};
```

### 2. Train

```bash
npx praxis train
```

This reads `model.definition.ts`, runs automatic prompt optimization, and writes `model.config.json`.

Options:
- `--output, -o <path>` — output file (default: `model.config.json`)
- `--optimizer <ace|gepa|auto>` — optimizer type (default: auto)
- `--split <ratio>` — train/test split (default: 0.7)

### 3. Run

```bash
# Interactive
npx praxis run

# With flags
npx praxis run --text "Hello world"
```

### 4. Use in code

```ts
import { generateText } from '@drawcall/praxis';
import config from './model.config.json';

const { object } = await generateText(config, { text: 'Hello world' });
```

## When to help with Praxis

- User asks to "create a model", "define a prompt model", "train a prompt", or "optimize a prompt"
- User has a classification, extraction, analysis, or generation task with structured I/O
- User wants to improve LLM output quality with training data

## How to help

1. Ask the user what task they want to solve (classification, extraction, summarization, etc.)
2. Create a `model.definition.ts` with appropriate schema, 10+ examples, and a metric
3. Tell them to set `OPENROUTER_KEY` in `.env` then run `npx praxis train`
4. After training, show them how to use `npx praxis run` or the library API

## Requirements

- `OPENROUTER_KEY` must be set in `.env`
- Node.js >= 18
- At least 10 examples in the definition
