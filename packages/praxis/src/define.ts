import type { z } from 'zod';
import type { ModelDefinition } from './types.js';

export function defineModel<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  definition: ModelDefinition<I, O>,
): ModelDefinition<I, O> {
  return definition;
}
