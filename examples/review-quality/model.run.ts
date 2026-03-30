import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

const { output, score } = await generateText({
  definition: modelDefinition,
  input: { reviewText: 'Bad product. Would not recommend.' },
});

console.log('Result:', output);
console.log('Scores:', score);
