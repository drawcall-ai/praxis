import { z } from 'zod';

export const model = 'anthropic/claude-sonnet-3';

export const schema = {
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
};

export const examples = [
  { input: { reviewText: 'Absolutely love this product! Best purchase ever.' }, sentiment: 'positive', confidence: 0.95 },
  { input: { reviewText: 'Terrible quality, broke after one day.' }, sentiment: 'negative', confidence: 0.92 },
  { input: { reviewText: 'It works fine, nothing special.' }, sentiment: 'neutral', confidence: 0.80 },
  { input: { reviewText: 'Amazing customer service and fast delivery!' }, sentiment: 'positive', confidence: 0.90 },
  { input: { reviewText: 'Would not recommend to anyone.' }, sentiment: 'negative', confidence: 0.88 },
  { input: { reviewText: 'The product is okay for the price.' }, sentiment: 'neutral', confidence: 0.75 },
  { input: { reviewText: 'Five stars! Exceeded my expectations.' }, sentiment: 'positive', confidence: 0.97 },
  { input: { reviewText: 'Complete waste of money.' }, sentiment: 'negative', confidence: 0.94 },
  { input: { reviewText: 'Does what it says, no more no less.' }, sentiment: 'neutral', confidence: 0.70 },
  { input: { reviewText: 'I keep coming back to buy more, truly excellent.' }, sentiment: 'positive', confidence: 0.93 },
  { input: { reviewText: 'Arrived damaged and support was unhelpful.' }, sentiment: 'negative', confidence: 0.91 },
  { input: { reviewText: 'It is a product. It exists.' }, sentiment: 'neutral', confidence: 0.65 },
];

export const metric = ({
  prediction,
  example,
}: {
  input: Record<string, unknown>;
  prediction: Record<string, unknown>;
  example?: typeof examples[number];
}) => {
  if (!example) return null;
  return prediction.sentiment === example.sentiment ? 1 : 0;
};
