import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

const { output, score } = await generateText({
  definition: modelDefinition,
  input: { reviewText: 'Terrible quality, broke after one day.' },
});

console.log('Result:', output);
console.log('Score:', score);
