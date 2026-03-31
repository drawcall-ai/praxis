import { z } from 'zod';
import { defineModel } from '@drawcall/praxis';

export default defineModel({
  name: 'Sentiment Analyzer',
  student: 'google/gemini-3-flash-preview',
  teacher: 'google/gemini-3.1-pro-preview',
  description: 'Analyze product reviews to determine sentiment and confidence.',

  input: z.object({
    reviewText: z.string().describe('The text of the product review to analyze'),
  }),

  output: z.object({
    sentiment: z
      .enum(['positive', 'negative', 'neutral'])
      .describe('The overall sentiment of the review'),
    confidence: z
      .number()
      .describe('Confidence score between 0 and 1'),
  }),

  examples: [
    { input: { reviewText: 'Absolutely love this product! Best purchase ever.' }, output: { sentiment: 'positive', confidence: 0.95 } },
    { input: { reviewText: 'Terrible quality, broke after one day.' }, output: { sentiment: 'negative', confidence: 0.92 } },
    { input: { reviewText: 'It works fine, nothing special.' }, output: { sentiment: 'neutral', confidence: 0.80 } },
    { input: { reviewText: 'Amazing customer service and fast delivery!' }, output: { sentiment: 'positive', confidence: 0.90 } },
    { input: { reviewText: 'Would not recommend to anyone.' }, output: { sentiment: 'negative', confidence: 0.88 } },
    { input: { reviewText: 'The product is okay for the price.' }, output: { sentiment: 'neutral', confidence: 0.75 } },
    { input: { reviewText: 'Five stars! Exceeded my expectations.' }, output: { sentiment: 'positive', confidence: 0.97 } },
    { input: { reviewText: 'Complete waste of money.' }, output: { sentiment: 'negative', confidence: 0.94 } },
    { input: { reviewText: 'Does what it says, no more no less.' }, output: { sentiment: 'neutral', confidence: 0.70 } },
    { input: { reviewText: 'I keep coming back to buy more, truly excellent.' }, output: { sentiment: 'positive', confidence: 0.93 } },
    { input: { reviewText: 'Arrived damaged and support was unhelpful.' }, output: { sentiment: 'negative', confidence: 0.91 } },
    { input: { reviewText: 'It is a product. It exists.' }, output: { sentiment: 'neutral', confidence: 0.65 } },
  ],

  metric: ({ modelOutput, exampleOutput }) => {
    if (!exampleOutput) return null;
    return modelOutput.sentiment === exampleOutput.sentiment ? 1 : 0;
  },
});
