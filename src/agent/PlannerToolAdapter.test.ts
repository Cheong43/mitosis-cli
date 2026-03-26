import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { PlannerToolAdapter } from './PlannerToolAdapter.js';

function createTempProjectRoot(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function createAdapter(projectRoot: string) {
  const calls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const runtimeHandle = {
    executeTool: async (toolName: string, args: Record<string, unknown>) => {
      calls.push({ toolName, args });
      if (toolName === 'read_file') {
        return {
          success: true,
          result: `read:${String(args.path || '')}`,
          durationMs: 0,
        };
      }
      if (toolName === 'write_file') {
        return {
          success: true,
          result: `write:${String(args.path || '')}`,
          durationMs: 0,
        };
      }
      if (toolName === 'run_shell') {
        return {
          success: true,
          result: JSON.stringify({ kind: 'search_results', results: [{ node_id: 'cap_rate', score: 1.5 }] }),
          durationMs: 0,
        };
      }
      return {
        success: false,
        error: `unexpected tool ${toolName}`,
        durationMs: 0,
      };
    },
    toolRuntime: { resetSession() {} },
    governance: {},
    registry: {},
  } as any;

  return {
    adapter: new PlannerToolAdapter({
      projectRoot,
      codeCliRoot: projectRoot,
      runtimeHandle,
    }),
    calls,
  };
}

test('PlannerToolAdapter routes workspace reads through read_file', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-workspace-');
  const { adapter, calls } = createAdapter(projectRoot);

  const result = await adapter.execute('read', {
    target: 'workspace',
    path: 'src/index.ts',
  });

  assert.equal(result, 'read:src/index.ts');
  assert.deepEqual(calls[0], {
    toolName: 'read_file',
    args: { path: 'src/index.ts' },
  });
});

test('PlannerToolAdapter routes preference edits through semantic preferences path', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-preferences-');
  const { adapter, calls } = createAdapter(projectRoot);

  const result = await adapter.execute('edit', {
    target: 'preferences',
    content: '# Preferences\n- concise answers',
  });

  assert.equal(result, 'write:.mempedia/memory/preferences.md');
  assert.deepEqual(calls[0], {
    toolName: 'write_file',
    args: {
      path: '.mempedia/memory/preferences.md',
      content: '# Preferences\n- concise answers',
    },
  });
});

test('PlannerToolAdapter reads local skills through semantic target', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-skills-');
  const skillDir = path.join(projectRoot, 'skills', 'release-check');
  fs.mkdirSync(skillDir, { recursive: true });
  fs.writeFileSync(
    path.join(skillDir, 'SKILL.md'),
    `---\nname: release-check\ndescription: "Release checklist"\n---\n\nRun smoke tests before tagging.\n`,
    'utf-8',
  );

  const { adapter } = createAdapter(projectRoot);
  const result = await adapter.execute('read', {
    target: 'skills',
    skill_id: 'release-check',
  });

  assert.match(result, /Skill: release-check/);
  assert.match(result, /Run smoke tests before tagging\./);
});

test('PlannerToolAdapter routes memory search through a Mempedia shell action', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-memory-');
  const { adapter, calls } = createAdapter(projectRoot);

  const result = await adapter.execute('search', {
    target: 'memory',
    mode: 'hybrid',
    query: 'cap rate',
    limit: 3,
  });

  assert.match(result, /"kind":"search_results"/);
  assert.equal(calls[0]?.toolName, 'run_shell');
  assert.match(String(calls[0]?.args.command || ''), /\\"action\\":\\"search_hybrid\\"/);
  assert.match(String(calls[0]?.args.command || ''), /\\"query\\":\\"cap rate\\"/);
});
