import { z } from 'zod';
import { defineModel } from '@drawcall/praxis';

/**
 * Multi-metric example: evaluate review quality on multiple dimensions.
 * The metric returns a Record<string, number> — Praxis auto-selects the
 * GEPA optimizer for multi-objective optimization.
 */
export default defineModel({
  name: 'Review Quality',
  student: 'google/gemini-3-flash-preview',

  input: z.object({
    reviewText: z.string().describe('The product review to evaluate'),
  }),

  output: z.object({
    quality: z
      .enum(['helpful', 'unhelpful'])
      .describe('Whether the review is helpful to other buyers'),
    sentiment: z
      .enum(['positive', 'negative', 'neutral'])
      .describe('The overall sentiment of the review'),
    hasSpecificDetails: z
      .boolean()
      .describe('Whether the review mentions specific product details'),
  }),

  examples: [
    { input: { reviewText: 'The battery lasts 8 hours and charges in 30 minutes. Great for travel.' }, output: { quality: 'helpful', sentiment: 'positive', hasSpecificDetails: true } },
    { input: { reviewText: 'Bad.' }, output: { quality: 'unhelpful', sentiment: 'negative', hasSpecificDetails: false } },
    { input: { reviewText: 'Screen resolution is 1080p, colors are accurate. Touch response is slightly laggy.' }, output: { quality: 'helpful', sentiment: 'neutral', hasSpecificDetails: true } },
    { input: { reviewText: 'Love it!!!!!!' }, output: { quality: 'unhelpful', sentiment: 'positive', hasSpecificDetails: false } },
    { input: { reviewText: 'Returned after 3 days. The zipper broke on first use and the stitching came apart at the seams.' }, output: { quality: 'helpful', sentiment: 'negative', hasSpecificDetails: true } },
    { input: { reviewText: 'It is what it is.' }, output: { quality: 'unhelpful', sentiment: 'neutral', hasSpecificDetails: false } },
    { input: { reviewText: 'Fits true to size. The waterproof coating held up in heavy rain for 2 hours.' }, output: { quality: 'helpful', sentiment: 'positive', hasSpecificDetails: true } },
    { input: { reviewText: 'DO NOT BUY THIS PRODUCT' }, output: { quality: 'unhelpful', sentiment: 'negative', hasSpecificDetails: false } },
    { input: { reviewText: 'The 12W motor handles frozen fruit easily. Cleanup is simple — dishwasher safe blades.' }, output: { quality: 'helpful', sentiment: 'positive', hasSpecificDetails: true } },
    { input: { reviewText: 'Meh, not impressed but not terrible either. Just average I guess.' }, output: { quality: 'unhelpful', sentiment: 'neutral', hasSpecificDetails: false } },
    { input: { reviewText: 'After 6 months of daily use the non-stick coating started peeling. Heats unevenly on the edges.' }, output: { quality: 'helpful', sentiment: 'negative', hasSpecificDetails: true } },
    { input: { reviewText: 'Perfect gift for my mom. The ceramic finish looks premium and the 2-year warranty is a plus.' }, output: { quality: 'helpful', sentiment: 'positive', hasSpecificDetails: true } },
  ],

  metricWeights: { quality: 1, sentiment: 1, details: 1 },

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return {
      quality: modelOutput.quality === exampleOutput.quality ? 1 : 0,
      sentiment: modelOutput.sentiment === exampleOutput.sentiment ? 1 : 0,
      details: modelOutput.hasSpecificDetails === exampleOutput.hasSpecificDetails ? 1 : 0,
    };
  },
});
