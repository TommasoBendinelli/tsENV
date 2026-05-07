import fs from 'fs';
import path from 'path';
import Ajv, { ErrorObject, ValidateFunction } from 'ajv';

const validators = new Map<string, ValidateFunction>();

const getSchemaPath = (schemaFilename: string) =>
  path.join(process.cwd(), '..', 'shared', 'schemas', schemaFilename);

const getValidator = (schemaFilename: string) => {
  const cached = validators.get(schemaFilename);
  if (cached) return cached;

  const schemaPath = getSchemaPath(schemaFilename);
  const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));

  // Repo currently uses Ajv v6; draft-07 schema.
  const ajv = new Ajv({ allErrors: true, jsonPointers: true, schemaId: 'auto' } as any);
  const validate = ajv.compile(schema);
  validators.set(schemaFilename, validate);
  return validate;
};

export const formatAjvErrors = (errors: ErrorObject[] | null | undefined) => {
  if (!errors || errors.length === 0) return '';
  return errors
    .slice(0, 25)
    .map((e) => `${String((e as any).dataPath || '')} ${e.message || ''}`.trim())
    .filter(Boolean)
    .join('; ');
};

export const assertValidAgainstSharedSchema = (schemaFilename: string, payload: unknown) => {
  const validate = getValidator(schemaFilename);
  const ok = validate(payload);
  if (!ok) {
    const preview = formatAjvErrors(validate.errors);
    const suffix =
      validate.errors && validate.errors.length > 25
        ? `; ...and ${validate.errors.length - 25} more`
        : '';
    throw new Error(`Invalid schema (${schemaFilename}): ${preview}${suffix}`);
  }
  return payload as any;
};

