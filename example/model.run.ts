import { buildPrompt, generateText } from '@drawcall/praxis';
import { schema } from './model.definition.js';
import config from './model.config.json';

const { system, user } = buildPrompt(config, {
  reviewText: 'This blender is amazing, best kitchen gadget ever!',
}, schema.output);

console.log('System:', system.slice(0, 80) + '...');

const { object } = await generateText(config, {
  reviewText: 'Terrible quality, broke after one day.',
}, schema.output);

console.log('Result:', object);
