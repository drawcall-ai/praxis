# praxis

> knowledge through action

Praxis uses [AX](https://axllm.dev) (TypeScript DSPy) to automatically optimize your prompts via structured training. You define a schema, provide examples, and praxis generates an optimized `model.config.json` that can be used standalone to build prompts or call any LLM.

## Install

```bash
npm install @drawcall/praxis
```

## Quick start

### 1. Define your model

Create `model.definition.ts`:

```ts
import { z } from 'zod';

export const model = 'anthropic/claude-sonnet-4';

export const schema = {
  input: z.object({
    reviewText: z.string().describe('The product review to analyze'),
  }),
  output: z.object({
    sentiment: z.enum(['positive', 'negative', 'neutral']).describe('Sentiment classification'),
    confidence: z.number().describe('Confidence score 0-1'),
  }),
};

export const examples = [
  { input: { reviewText: 'Love this product!' }, sentiment: 'positive', confidence: 0.95 },
  { input: { reviewText: 'Terrible quality.' }, sentiment: 'negative', confidence: 0.92 },
  { input: { reviewText: 'It works fine.' }, sentiment: 'neutral', confidence: 0.75 },
  // ... at least 10 examples required
];

export const metric = ({ prediction, example }) => {
  if (!example) return null;
  return prediction.sentiment === example.sentiment ? 1 : 0;
};
```

### 2. Train

```bash
npx praxis train
```

Reads `model.definition.ts`, optimizes prompts, writes `model.config.json`.

### 3. Run

```bash
# Interactive — prompts for missing fields
npx praxis run

# Non-interactive — all fields via flags
npx praxis run --reviewText "Great product!"
# → { "sentiment": "positive", "confidence": 0.95 }
```

### 4. Use in code

```ts
import { generateText } from '@drawcall/praxis';
import config from './model.config.json';

const { object } = await generateText(config, {
  reviewText: 'Amazing product!',
});
```