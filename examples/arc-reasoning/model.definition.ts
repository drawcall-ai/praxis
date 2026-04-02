import { z } from 'zod';
import { defineModel } from '@drawcall/praxis';
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// A single grid cell is 0-9 (color index)
const Grid = z.array(z.array(z.number().int().min(0).max(9))).describe(
  'A 2D grid of integers 0-9, where each integer represents a color'
);

const Demonstration = z.object({
  input: Grid.describe('Input grid for this demonstration'),
  output: Grid.describe('Output grid showing the transformation result'),
});

interface ArcTask {
  taskId: string;
  demonstrations: { input: number[][]; output: number[][] }[];
  testInput: number[][];
  expectedOutput: number[][];
}

// Load tasks lazily from JSON
const loadTasks = async (): Promise<ArcTask[]> => {
  const raw = await readFile(join(__dirname, 'arc-tasks.json'), 'utf-8');
  return JSON.parse(raw);
};

export default defineModel({
  name: 'ARC Reasoning',
  student: 'google/gemini-3-flash-preview',
  teacher: 'google/gemini-3.1-pro-preview',
  description:
    'Solve ARC (Abstraction and Reasoning Corpus) tasks: given a few input→output demonstration pairs that share a hidden transformation rule, predict the output for a new test input. This requires abstract pattern recognition, spatial reasoning, and rule induction.',
  version: '1.0',

  input: z.object({
    demonstrations: z
      .array(Demonstration)
      .describe(
        'A list of input→output pairs demonstrating the transformation rule. Study these carefully to infer the pattern.'
      ),
    testInput: Grid.describe(
      'The test input grid. Apply the inferred transformation rule to produce the output.'
    ),
  }),

  output: z.object({
    reasoning: z
      .string()
      .describe(
        'Step-by-step explanation of the discovered transformation rule and how it applies to the test input'
      ),
    grid: Grid.describe('The predicted output grid after applying the transformation'),
  }),

  examples: async () => {
    const tasks = await loadTasks();
    return tasks.map((task) => ({
      input: {
        demonstrations: task.demonstrations,
        testInput: task.testInput,
      },
      output: {
        reasoning: `Task ${task.taskId}: Apply the transformation pattern observed in the demonstrations.`,
        grid: task.expectedOutput,
      },
    }));
  },

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;

    const expected = exampleOutput.grid;
    const predicted = modelOutput.grid;

    // Dimension check
    if (
      predicted.length !== expected.length ||
      predicted[0]?.length !== expected[0]?.length
    ) {
      return { gridMatch: 0, dimensionMatch: 0, cellAccuracy: 0 };
    }

    // Cell-by-cell accuracy
    let correct = 0;
    let total = 0;
    for (let r = 0; r < expected.length; r++) {
      for (let c = 0; c < expected[r].length; c++) {
        total++;
        if (predicted[r]?.[c] === expected[r][c]) correct++;
      }
    }

    const cellAccuracy = total > 0 ? correct / total : 0;
    const exactMatch = cellAccuracy === 1 ? 1 : 0;

    return {
      gridMatch: exactMatch,
      cellAccuracy,
      dimensionMatch: 1,
    };
  },

  metricWeights: {
    gridMatch: 0.6,
    cellAccuracy: 0.3,
    dimensionMatch: 0.1,
  },

  split: [0.7, 0.15, 0.15],
  targetScore: 0.7,
  maxIterations: 5,
});
