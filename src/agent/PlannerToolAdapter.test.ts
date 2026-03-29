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

async function withEnv(
  updates: Record<string, string | undefined>,
  fn: () => Promise<void>,
) {
  const originals = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(updates)) {
    originals.set(key, process.env[key]);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  try {
    await fn();
  } finally {
    for (const [key, value] of originals.entries()) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
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

test('PlannerToolAdapter web search sends the full query to the engine, not only the first term', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-query-');
  const { adapter } = createAdapter(projectRoot);
  const requestedUrls: string[] = [];
  const query = 'Qin Shi Huang burning books historical debate';

  await withMockFetch(async (input) => {
    requestedUrls.push(String(input));
    return createHtmlResponse(`
      <html>
        <a class="result__a" href="https://example.com/article">Article</a>
        <a class="result__snippet">Useful snippet</a>
      </html>
    `);
  }, async () => {
    const raw = await adapter.execute('web', { mode: 'search', query });
    const result = JSON.parse(raw);
    assert.equal(result.kind, 'web_search');
    assert.equal(result.query, query);
  });

  assert.equal(requestedUrls.length, 1);
  assert.match(requestedUrls[0], new RegExp(`q=${encodeURIComponent(query)}`));
  assert.doesNotMatch(requestedUrls[0], /q=Qin(?:&|$)/);
});

test('PlannerToolAdapter web search returns citation-oriented metadata with budget and permissions', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-citations-');
  const { adapter } = createAdapter(projectRoot);

  await withMockFetch(async () => createHtmlResponse(`
    <html>
      <a class="result__a" href="https://docs.example.com/article">Docs result</a>
      <a class="result__snippet">Useful snippet</a>
    </html>
  `), async () => {
    const raw = await adapter.execute('web', {
      mode: 'search',
      query: 'example docs',
      allowed_domains: ['docs.example.com'],
    });
    const result = JSON.parse(raw);
    assert.equal(result.kind, 'web_search');
    assert.equal(result.engine, 'duckduckgo');
    assert.equal(result.results[0].citation, '[1]');
    assert.equal(result.results[0].domain, 'docs.example.com');
    assert.deepEqual(result.permissions.allowed_domains, ['docs.example.com']);
    assert.equal(result.budget.used, 1);
    assert.equal(result.budget.remaining, 7);
    assert.match(String(result.summary || ''), /Best citations to inspect next/);
    assert.deepEqual(result.recommended_fetch_urls, ['https://docs.example.com/article']);
  });
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

test('PlannerToolAdapter enforces a per-run web search budget', async () => {
  await withEnv({ MITOSIS_WEB_SEARCH_BUDGET: '2' }, async () => {
    const projectRoot = createTempProjectRoot('planner-tool-adapter-web-budget-');
    const { adapter } = createAdapter(projectRoot);

    await withMockFetch(async () => createHtmlResponse(`
      <html>
        <a class="result__a" href="https://example.com/one">One</a>
        <a class="result__snippet">Snippet</a>
      </html>
    `), async () => {
      const first = JSON.parse(await adapter.execute('web', { mode: 'search', query: 'first' }));
      const second = JSON.parse(await adapter.execute('web', { mode: 'search', query: 'second' }));
      const third = JSON.parse(await adapter.execute('web', { mode: 'search', query: 'third' }));

      assert.equal(first.kind, 'web_search');
      assert.equal(second.kind, 'web_search');
      assert.equal(third.kind, 'error');
      assert.match(String(third.message || ''), /budget exhausted/);
      assert.equal(third.budget.used, 2);
      assert.equal(third.budget.limit, 2);
    });
  });
});

test('PlannerToolAdapter can return identical result sets for different queries when the upstream engine responds with identical HTML', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-same-upstream-');
  const { adapter } = createAdapter(projectRoot);
  const html = `
    <html>
      <a class="result__a" href="https://example.com/qin-overview">Qin Overview</a>
      <a class="result__snippet">A generic Qin snippet</a>
      <a class="result__a" href="https://example.com/qin-timeline">Qin Timeline</a>
      <a class="result__snippet">A generic timeline snippet</a>
    </html>
  `;

  await withMockFetch(async () => createHtmlResponse(html), async () => {
    const first = JSON.parse(await adapter.execute('web', {
      mode: 'search',
      query: 'Qin Shi Huang burning books',
    }));
    const second = JSON.parse(await adapter.execute('web', {
      mode: 'search',
      query: 'Qin Shi Huang terracotta army',
    }));

    assert.notEqual(first.query, second.query);
    assert.deepEqual(first.results, second.results);
  });
});

test('PlannerToolAdapter fallback chain can amplify same-result behavior when DDG and Bing fail and Baidu returns the same head results', async () => {
  const projectRoot = createTempProjectRoot('planner-tool-adapter-web-baidu-fallback-');
  const { adapter } = createAdapter(projectRoot);
  const requestedUrls: string[] = [];
  const baiduHtml = `
    <html>
      <div class="result c-container">
        <h3><a href="https://example.com/qin-intro">秦始皇 - 词条</a></h3>
        <div class="c-abstract">首屏固定结果</div>
      </div>
      <div class="result c-container">
        <h3><a href="https://example.com/qin-history">秦朝历史</a></h3>
        <div class="c-abstract">另一个固定结果</div>
      </div>
    </html>
  `;

  await withMockFetch(async (input) => {
    const url = String(input);
    requestedUrls.push(url);
    if (url.includes('duckduckgo.com')) {
      return createHtmlResponse('<html><body>no parseable results</body></html>');
    }
    if (url.includes('bing.com')) {
      return createHtmlResponse('<html><body>no parseable results</body></html>');
    }
    if (url.includes('baidu.com')) {
      return createHtmlResponse(baiduHtml);
    }
    throw new Error(`Unexpected URL: ${url}`);
  }, async () => {
    const first = JSON.parse(await adapter.execute('web', {
      mode: 'search',
      query: '秦始皇 焚书坑儒',
    }));
    const second = JSON.parse(await adapter.execute('web', {
      mode: 'search',
      query: '秦始皇 兵马俑 阿房宫',
    }));

    assert.notEqual(first.query, second.query);
    assert.deepEqual(first.results, second.results);
  });

  assert.equal(requestedUrls.filter((url) => url.includes('duckduckgo.com')).length, 2);
  assert.equal(requestedUrls.filter((url) => url.includes('bing.com')).length, 2);
  assert.equal(requestedUrls.filter((url) => url.includes('baidu.com')).length, 2);
  assert.ok(requestedUrls.some((url) => url.includes(encodeURIComponent('秦始皇 焚书坑儒'))));
  assert.ok(requestedUrls.some((url) => url.includes(encodeURIComponent('秦始皇 兵马俑 阿房宫'))));
});
