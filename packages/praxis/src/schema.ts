import { z } from 'zod';
import type { PraxisConfigSchema, JsonSchema } from './types.js';

interface SchemaSource {
  input: z.ZodObject<z.ZodRawShape>;
  output: z.ZodObject<z.ZodRawShape>;
}

/**
 * Convert a definition's Zod schemas to JSON Schema for storage in model.config.json.
 */
export function serializeSchema(source: SchemaSource): PraxisConfigSchema {
  return {
    input: z.toJSONSchema(source.input) as JsonSchema,
    output: z.toJSONSchema(source.output) as JsonSchema,
  };
}

/**
 * Validate that a Zod schema matches the JSON Schema stored in a config.
 * Throws a descriptive error if they differ.
 */
export function validateSchema(
  zodSchema: z.ZodObject<z.ZodRawShape>,
  configJsonSchema: JsonSchema,
  label: string,
): void {
  const current = z.toJSONSchema(zodSchema) as JsonSchema;
  const currentStr = JSON.stringify(current, null, 2);
  const configStr = JSON.stringify(configJsonSchema, null, 2);

  if (currentStr !== configStr) {
    throw new Error(
      `Schema mismatch (${label}): the definition schema does not match the trained config.\n` +
      `Run \`npx praxis train\` to retrain.`,
    );
  }
}

/**
 * Convert a definition's schemas to an AX signature string.
 * e.g. 'reviewText:string "desc" -> sentiment:class "positive,negative,neutral" "desc"'
 */
export function toAxSignature(source: SchemaSource): string {
  const inputParts = zodObjectToAxFields(source.input);
  const outputParts = zodObjectToAxFields(source.output);
  return `${inputParts.join(', ')} -> ${outputParts.join(', ')}`;
}

function zodObjectToAxFields(obj: z.ZodObject<z.ZodRawShape>): string[] {
  const fields: string[] = [];
  for (const [key, zodType] of Object.entries(obj.shape)) {
    fields.push(zodTypeToAxField(key, zodType as z.ZodTypeAny));
  }
  return fields;
}

function zodTypeToAxField(name: string, zodType: z.ZodTypeAny): string {
  const description = zodType.description;
  const def = zodType._def as unknown as Record<string, unknown>;
  const defType = def.type as string;

  if (defType === 'optional' || defType === 'nullable' || defType === 'default') {
    return zodTypeToAxField(name, (def.innerType ?? def.type) as z.ZodTypeAny);
  }

  let axType: string;

  if (defType === 'enum') {
    axType = `class "${Object.keys(def.entries as Record<string, string>).join(',')}"`;
  } else if (defType === 'number') {
    axType = 'number';
  } else if (defType === 'boolean') {
    axType = 'boolean';
  } else if (defType === 'array') {
    axType = 'string[]';
  } else {
    axType = 'string';
  }

  const desc = description ? ` "${description}"` : '';
  return `${name}:${axType}${desc}`;
}
