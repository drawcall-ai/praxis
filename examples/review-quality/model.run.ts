import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

// Generate with the model — score returns per-metric results
const { object, score } = await generateText({
  definition: modelDefinition,
  input: { reviewText: 'Bad product. Would not recommend.' },
});

console.log('Result:', object);
console.log('Scores:', score);
// → { quality: 1, sentiment: 1, details: 0 }
