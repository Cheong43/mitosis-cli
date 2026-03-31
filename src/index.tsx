import React from 'react';
import { render, Text } from 'ink';
import * as path from 'path';
import * as fs from 'fs';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';

import { App } from './components/App.js';
import { importDocCommand } from './import-doc.js';
import { resolveProjectRoot } from './config/projectPaths.js';
import { logError } from './utils/errorLogger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../.env') });

const projectRoot = resolveProjectRoot(__dirname);
console.log(`[CodeCLI] Using projectRoot: ${projectRoot}`);

// Prevent unhandled promise rejections from silently killing the process.
process.on('unhandledRejection', (reason) => {
  console.error('[CodeCLI] Unhandled rejection:', reason);
  logError(projectRoot, reason, 'unhandledRejection');
});
process.on('uncaughtException', (err) => {
  console.error('[CodeCLI] Uncaught exception:', err);
  logError(projectRoot, err, 'uncaughtException');
});

// ── import-doc sub-command ────────────────────────────────────────────────────
// Usage:
//   node src/index.tsx import-doc --file <path> [--node-id <id>] [--title <t>]
//   node src/index.tsx import-doc --dir <directory>
const args = process.argv.slice(2);
if (args[0] === 'import-doc') {
  importDocCommand(args.slice(1), projectRoot).then((code) => process.exit(code ?? 0));
} else {
  const apiKey = process.env.OPENAI_API_KEY || '';
  const baseURL = process.env.OPENAI_BASE_URL;
  const model = process.env.OPENAI_MODEL;
  const memoryApiKey = process.env.MEMORY_API_KEY;
  const memoryBaseURL = process.env.MEMORY_BASE_URL;
  const memoryModel = process.env.MEMORY_MODEL;

  const anthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN;
  const anthropicBaseURL = process.env.ANTHROPIC_BASE_URL;
  const anthropicModel = process.env.ANTHROPIC_MODEL;

  if (!apiKey && !anthropicAuthToken) {
    render(
      <Text>
        Missing API key. Please set `OPENAI_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in `mitosis-cli/.env`.
      </Text>
    );
    process.exit(1);
  }

  render(
    <App
      apiKey={apiKey}
      projectRoot={projectRoot}
      baseURL={baseURL}
      model={model}
      memoryApiKey={memoryApiKey}
      memoryBaseURL={memoryBaseURL}
      memoryModel={memoryModel}
      anthropicAuthToken={anthropicAuthToken}
      anthropicBaseURL={anthropicBaseURL}
      anthropicModel={anthropicModel}
    />
  );
}
