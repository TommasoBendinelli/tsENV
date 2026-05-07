/* eslint-disable no-console */
const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const SCHEMAS_DIR = path.join(REPO_ROOT, "shared", "schemas");
const OUT_DIR = path.join(REPO_ROOT, "web_model_explorer", "generated", "schemas");

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function mkdirp(p) {
  fs.mkdirSync(p, { recursive: true });
}

function asTypeName(name) {
  return String(name || "")
    .replace(/[^a-zA-Z0-9_]/g, "_")
    .replace(/^(\d)/, "_$1");
}

function quoteLit(s) {
  return JSON.stringify(s);
}

function isRef(schema) {
  return schema && typeof schema === "object" && typeof schema.$ref === "string";
}

function refName(ref) {
  const m = String(ref).match(/^#\/definitions\/(.+)$/);
  return m ? asTypeName(m[1]) : "any";
}

function union(types) {
  const flat = [];
  for (const t of types) {
    if (!t) continue;
    if (t.includes(" | ")) {
      flat.push(...t.split(" | ").map((x) => x.trim()));
    } else {
      flat.push(t);
    }
  }
  const uniq = Array.from(new Set(flat));
  return uniq.length ? uniq.join(" | ") : "any";
}

function keyTypeForObject(schema) {
  const pn = schema && schema.propertyNames;
  if (pn && Array.isArray(pn.enum) && pn.enum.length > 0) {
    return union(pn.enum.map((x) => quoteLit(x)));
  }
  return "string";
}

function schemaToTs(schema) {
  if (!schema || typeof schema !== "object") return "any";
  if (isRef(schema)) return refName(schema.$ref);

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    return union(schema.anyOf.map(schemaToTs));
  }
  if (schema.const !== undefined) return quoteLit(schema.const);
  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    return union(schema.enum.map((x) => quoteLit(x)));
  }

  const t = schema.type;
  if (t === "string") return "string";
  if (t === "number") return "number";
  if (t === "integer") return "number";
  if (t === "boolean") return "boolean";
  if (t === "null") return "null";
  if (t === "array") {
    return `(${schemaToTs(schema.items)})[]`;
  }
  if (t === "object") {
    const props = schema.properties && typeof schema.properties === "object" ? schema.properties : null;
    const addl = schema.additionalProperties;
    const keyT = keyTypeForObject(schema);
    if (!props || Object.keys(props).length === 0) {
      if (addl === true) return `Record<${keyT}, any>`;
      if (addl && typeof addl === "object") return `Record<${keyT}, ${schemaToTs(addl)}>`;
      return `Record<${keyT}, any>`;
    }
    if (addl === true) return `{ [k: string]: any }`;
    if (addl && typeof addl === "object") return `{ [k: string]: ${schemaToTs(addl)} }`;
    return `{ [k: string]: never }`;
  }
  return "any";
}

function emitInterface(name, schema) {
  const props = schema.properties || {};
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const lines = [];
  lines.push(`export interface ${name} {`);
  for (const [k, v] of Object.entries(props)) {
    const opt = required.has(k) ? "" : "?";
    const key = /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(k) ? k : quoteLit(k);
    lines.push(`  ${key}${opt}: ${schemaToTs(v)};`);
  }
  lines.push("}");
  return lines.join("\n");
}

function generateOne(schemaPath) {
  const schema = readJson(schemaPath);
  const basename = path.basename(schemaPath).replace(/\.json$/, "");
  const outPath = path.join(OUT_DIR, `${basename}.ts`);

  const defs = schema.definitions && typeof schema.definitions === "object" ? schema.definitions : {};
  const blocks = [];
  blocks.push(
    [
      "/* This file is generated. Do not edit by hand. */",
      `/* Source: ${path.relative(REPO_ROOT, schemaPath)} */`,
      "",
    ].join("\n"),
  );

  for (const [defKey, defSchema] of Object.entries(defs)) {
    const typeName = asTypeName(defKey);
    if (defSchema && defSchema.type === "object" && defSchema.properties) {
      blocks.push(emitInterface(typeName, defSchema));
      blocks.push("");
    } else {
      blocks.push(`export type ${typeName} = ${schemaToTs(defSchema)};`);
      blocks.push("");
    }
  }

  const rootName = asTypeName(schema.title || "SchemaRoot");
  if (schema.type === "object" && schema.properties) {
    blocks.push(emitInterface(rootName, schema));
  } else {
    blocks.push(`export type ${rootName} = ${schemaToTs(schema)};`);
  }
  blocks.push("");

  const text = blocks.join("\n");
  fs.writeFileSync(outPath, text, "utf8");
  return outPath;
}

function main() {
  mkdirp(OUT_DIR);
  const files = fs
    .readdirSync(SCHEMAS_DIR)
    .filter((f) => f.endsWith(".schema.json"))
    .sort();
  const written = [];
  for (const f of files) {
    const p = path.join(SCHEMAS_DIR, f);
    written.push(path.relative(REPO_ROOT, generateOne(p)));
  }
  console.log(JSON.stringify({ written }, null, 2));
}

if (require.main === module) {
  main();
}
