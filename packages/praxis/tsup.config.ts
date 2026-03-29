import { defineConfig, Format } from "tsup";

const shared = {
  format: ["esm"] satisfies Format[],
  clean: true,
  sourcemap: true,
  shims: true,
  external: [
    "vite",
    "vite-node",
    "vite-node/client",
    "vite-node/server",
    "@ax-llm/ax",
    "ai",
    "@openrouter/ai-sdk-provider",
    "dotenv",
    "zod",
    "zod-to-json-schema",
  ],
};

export default defineConfig([
  {
    ...shared,
    entry: ["src/index.ts"],
    dts: true,
    clean: true,
  },
  {
    ...shared,
    entry: ["src/cli.ts"],
    banner: { js: "#!/usr/bin/env node" },
    clean: false,
  },
]);
