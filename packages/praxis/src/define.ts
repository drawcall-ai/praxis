import type { z } from 'zod';
import type { ModelDefinition } from './types.js';

export function defineModel<I extends z.ZodRawShape, O extends z.ZodRawShape, M extends string>(
  definition: ModelDefinition<I, O, M>,
): ModelDefinition<I, O, M> {
  return definition;
}
