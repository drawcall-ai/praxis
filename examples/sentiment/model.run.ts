import { generateText } from '@drawcall/praxis';
import modelDefinition from './model.definition.js';

// Generate with the model
const { object, score } = await generateText({
  definition: modelDefinition,
  input: { reviewText: 'Terrible quality, broke after one day.' },
});

console.log('Result:', object);
console.log('Score:', score);
