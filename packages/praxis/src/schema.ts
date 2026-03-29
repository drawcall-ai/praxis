import type { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
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
    input: zodToJsonSchema(source.input, { target: 'jsonSchema7' }),
    output: zodToJsonSchema(source.output, { target: 'jsonSchema7' }),
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
  const current = zodToJsonSchema(zodSchema, { target: 'jsonSchema7' });
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
  const def = zodType._def;

  if (def.typeName === 'ZodOptional' || def.typeName === 'ZodNullable' || def.typeName === 'ZodDefault') {
    return zodTypeToAxField(name, def.innerType ?? def.type);
  }

  let axType: string;

  if (def.typeName === 'ZodEnum') {
    axType = `class "${(def.values as string[]).join(',')}"`;
  } else if (def.typeName === 'ZodNumber') {
    axType = 'number';
  } else if (def.typeName === 'ZodBoolean') {
    axType = 'boolean';
  } else if (def.typeName === 'ZodArray') {
    axType = 'string[]';
  } else {
    axType = 'string';
  }

  const desc = description ? ` "${description}"` : '';
  return `${name}:${axType}${desc}`;
}
