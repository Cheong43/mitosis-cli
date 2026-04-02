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
  const mempediaCalls: Array<Record<string, unknown>> = [];
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
      mempediaClient: {
        send: async (action: Record<string, unknown>) => {
          mempediaCalls.push(action);
          return { kind: 'search_results', results: [{ node_id: 'cap_rate', score: 1.5 }] };
        },
      },
    }),
    calls,
    mempediaCalls,
  };
}

function createHtmlResponse(html: string, status = 200, statusText = 'OK') {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    text: async () => html,
  } as any;
}

async function withMockFetch(
  mock: (input: string | URL | Request) => Promise<any>,
  fn: () => Promise<void>,
) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request) => mock(input)) as typeof fetch;
  try {
    await fn();
  } finally {
    globalThis.fetch = originalFetch;
  }
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

test('PlannerToolAdapter routes memory search through the Mempedia transport client', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-memory-');
  const { adapter, calls, mempediaCalls } = createAdapter(projectRoot);

  const result = await adapter.execute('search', {
    target: 'memory',
    mode: 'hybrid',
    query: 'cap rate',
    limit: 3,
  });

  assert.match(result, /"kind":"search_results"/);
  assert.equal(calls.length, 0);
  assert.equal(String(mempediaCalls[0]?.action || ''), 'search_hybrid');
  assert.equal(String(mempediaCalls[0]?.query || ''), 'cap rate');
});

test('PlannerToolAdapter rejects removed web search mode with an explicit error', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-search-removed-');
  const { adapter } = createAdapter(projectRoot);

  const raw = await adapter.execute('web', {
    mode: 'search',
    query: 'example docs',
  });
  const result = JSON.parse(raw);
  assert.equal(result.kind, 'error');
  assert.match(String(result.message || ''), /web search has been removed/i);
});

test('PlannerToolAdapter blocks web fetches outside allowed domains', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-fetch-permissions-');
  const { adapter } = createAdapter(projectRoot);

  const raw = await adapter.execute('web', {
    mode: 'fetch',
    url: 'https://news.example.com/story',
    allowed_domains: ['docs.example.com'],
  });
  const result = JSON.parse(raw);
  assert.equal(result.kind, 'error');
  assert.match(String(result.message || ''), /outside allowed_domains/);
});

test('PlannerToolAdapter web fetch returns citation summary instead of full raw page text', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-fetch-summary-');
  const { adapter } = createAdapter(projectRoot);

  await withMockFetch(async () => createHtmlResponse(`
    <html>
      <title>Example Research</title>
      <body>
        <p>Alibaba Group reported quarterly revenue growth and highlighted stronger cloud momentum across international markets.</p>
        <p>The filing also described logistics investments, AI infrastructure spending, and management commentary about margin discipline.</p>
        <p>Executives said the next quarter would focus on monetization quality and operational efficiency across the group.</p>
      </body>
    </html>
  `), async () => {
    const raw = await adapter.execute('web', {
      mode: 'fetch',
      url: 'https://docs.example.com/report',
      allowed_domains: ['docs.example.com'],
    });
    const result = JSON.parse(raw);
    assert.equal(result.kind, 'web_fetch');
    assert.equal(result.domain, 'docs.example.com');
    assert.equal(result.citation, '[fetch]');
    assert.match(String(result.summary || ''), /Fetched page text length/);
    assert.ok(Array.isArray(result.highlights));
    assert.ok(result.highlights.length > 0);
    assert.ok(typeof result.content_preview === 'string' && result.content_preview.length > 0);
    assert.equal('content' in result, false);
    assert.match(String(result.recommended_next_action || ''), /content_preview/);
  });
});
