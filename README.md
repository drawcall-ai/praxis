# praxis

> knowledge through action

Praxis turns your problem into an optimized LLM prompt model. You describe what you want (schema + examples), Praxis automatically optimizes the prompt using [AX](https://axllm.dev) (TypeScript DSPy), and you get a portable `model.config.json` that works with any LLM.

**Define** your task with a schema, examples, and a metric. **Train** to optimize prompts automatically. **Run** anywhere — CLI, code, or API.

## Setup

Install the skill for your coding agent (works with [Claude Code, Cursor, Copilot, Codex, Gemini CLI, and more](https://agentskills.io)):

```bash
npx skills add drawcall/praxis
```

Then tell your agent what you need:

> "Create a praxis model that classifies support tickets into billing, technical, and general"

The agent knows how to write definitions, run training, and wire up the output.

## How it works

### 1. Define

Create `model.definition.ts` — a schema, training examples, and a scoring metric:

```ts
import { z } from 'zod';

export const model = 'anthropic/claude-sonnet-4';

export const schema = {
  input: z.object({
    reviewText: z.string().describe('The product review to analyze'),
  }),
  output: z.object({
    sentiment: z.enum(['positive', 'negative', 'neutral']),
    confidence: z.number().describe('Confidence score 0-1'),
  }),
};

export const examples = [
  { input: { reviewText: 'Love this product!' }, sentiment: 'positive', confidence: 0.95 },
  { input: { reviewText: 'Terrible quality.' }, sentiment: 'negative', confidence: 0.92 },
  { input: { reviewText: 'It works fine.' }, sentiment: 'neutral', confidence: 0.75 },
  // ... at least 10 examples
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

Reads your definition, runs automatic prompt optimization, and writes `model.config.json` with the best instruction, demos, and test scores.

Options:
- `--output, -o <path>` — output file (default: `model.config.json`)
- `--optimizer <ace|gepa|auto>` — optimizer type (default: auto-detect)
- `--split <ratio>` — train/test split (default: 0.7)

### 3. Run

```bash
# Interactive — prompts for missing fields
npx praxis run

# All fields via flags
npx praxis run --reviewText "Great product!"
# → { "sentiment": "positive", "confidence": 0.95 }
```

### Use in code

```ts
import { generateText } from '@drawcall/praxis';
import config from './model.config.json';

const { object } = await generateText(config, {
  reviewText: 'Amazing product!',
});
```

## CLI reference

| Command | Description |
|---------|-------------|
| `praxis train [definition]` | Optimize prompts from definition file |
| `praxis run [config]` | Run inference with trained config |
| `praxis validate [config]` | Check config matches definition schema |

## Requirements

- Node.js >= 18
- `OPENROUTER_KEY` in `.env` (get one at [openrouter.ai](https://openrouter.ai))
- At least 10 examples in your definition

## License

MIT
