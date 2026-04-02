import type { ModelExample, ModelExamples } from './types.js';
import type { z } from 'zod';

export interface ExampleProvider<
  I extends z.ZodRawShape = z.ZodRawShape,
  O extends z.ZodRawShape = z.ZodRawShape,
> {
  length: number;
  get(index: number): Promise<ModelExample<I, O>>;
}

function arrayProvider<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  arr: ModelExample<I, O>[],
): ExampleProvider<I, O> {
  return {
    length: arr.length,
    get: async (i) => arr[i],
  };
}

function cachedProvider<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  length: number,
  fetch: (index: number) => Promise<ModelExample<I, O>>,
): ExampleProvider<I, O> {
  const cache = new Map<number, ModelExample<I, O>>();
  return {
    length,
    async get(index: number) {
      const cached = cache.get(index);
      if (cached) return cached;
      const example = await fetch(index);
      cache.set(index, example);
      return example;
    },
  };
}

export async function resolveExamples<I extends z.ZodRawShape, O extends z.ZodRawShape>(
  raw: ModelExamples<I, O>,
): Promise<ExampleProvider<I, O>> {
  if (Array.isArray(raw)) return arrayProvider(raw);
  if (typeof raw === 'function') return arrayProvider(await raw());
  const length = await raw.getLength();
  return cachedProvider(length, (i) => raw.getExample(i));
}
