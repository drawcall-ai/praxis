import type { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import type { PraxisSchema, PraxisConfigSchema, JsonSchema } from './types.js';

/**
 * Convert Zod schemas to JSON Schema for storage in model.config.json.
 */
export function serializeSchema(schema: PraxisSchema): PraxisConfigSchema {
  return {
    input: zodToJsonSchema(schema.input, { target: 'jsonSchema7' }),
    output: zodToJsonSchema(schema.output, { target: 'jsonSchema7' }),
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
 * Convert a Zod schema to an AX signature string.
 * e.g. 'reviewText:string "desc" -> sentiment:class "positive,negative,neutral" "desc"'
 */
export function toAxSignature(schema: PraxisSchema): string {
  const inputParts = zodObjectToAxFields(schema.input);
  const outputParts = zodObjectToAxFields(schema.output);
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
