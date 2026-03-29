import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  loadWorkspaceSkills,
  parseSkillMarkdown,
  renderSkillGuidance,
} from './router.js';

function makeTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

async function withHome(homeDir: string, fn: () => Promise<void>) {
  const originalHome = process.env.HOME;
  process.env.HOME = homeDir;
  try {
    await fn();
  } finally {
    if (originalHome === undefined) {
      delete process.env.HOME;
    } else {
      process.env.HOME = originalHome;
    }
  }
}

test('parseSkillMarkdown keeps existing SKILL.md metadata parsing intact', () => {
  const parsed = parseSkillMarkdown(
    `---
name: project-discovery
description: "Summarize the repository"
metadata:
  category: mempedia
  priority: high
  tags: [mempedia, discovery]
---

# Project Discovery
Read the repository carefully.
`,
    'fallback',
  );

  assert.equal(parsed.name, 'project-discovery');
  assert.equal(parsed.description, 'Summarize the repository');
  assert.equal(parsed.category, 'mempedia');
  assert.equal(parsed.priority, 100);
  assert.deepEqual(parsed.tags, ['mempedia', 'discovery']);
});

test('parseSkillMarkdown understands Claude agent tools and top-level always_include', () => {
  const parsed = parseSkillMarkdown(
    `---
name: brave-search
description: "Use Brave Search when a Brave channel is available."
tools: [bash, web]
always_include: true
---

Use Brave as the preferred external search provider.
`,
    'fallback',
  );

  assert.equal(parsed.name, 'brave-search');
  assert.deepEqual(parsed.tools, ['bash', 'web']);
  assert.equal(parsed.alwaysInclude, true);
});

test('loadWorkspaceSkills discovers Claude-compatible agent files and gives project precedence over user home', async () => {
  const projectRoot = makeTempDir('skills-router-project-');
  const codeCliRoot = makeTempDir('skills-router-codecli-');
  const homeDir = makeTempDir('skills-router-home-');

  fs.mkdirSync(path.join(projectRoot, '.claude', 'agents'), { recursive: true });
  fs.mkdirSync(path.join(homeDir, '.claude', 'agents'), { recursive: true });
  fs.mkdirSync(path.join(projectRoot, 'skills', 'workspace-skill'), { recursive: true });

  fs.writeFileSync(
    path.join(projectRoot, '.claude', 'agents', 'brave-search.md'),
    `---
name: brave-search
description: "Project-level Brave skill"
tools:
  - bash
  - web
always_include: true
---

Project Brave guidance.
`,
    'utf-8',
  );
  fs.writeFileSync(
    path.join(homeDir, '.claude', 'agents', 'brave-search.md'),
    `---
name: brave-search
description: "User-level Brave skill"
tools: [bash]
always_include: true
---

User Brave guidance.
`,
    'utf-8',
  );
  fs.writeFileSync(
    path.join(projectRoot, 'skills', 'workspace-skill', 'SKILL.md'),
    `---
name: workspace-skill
description: "Workspace skill"
metadata:
  always_include: true
---

Workspace skill guidance.
`,
    'utf-8',
  );

  await withHome(homeDir, async () => {
    const skills = loadWorkspaceSkills(projectRoot, codeCliRoot);
    const brave = skills.find((skill) => skill.name === 'brave-search');
    const workspace = skills.find((skill) => skill.name === 'workspace-skill');

    assert.ok(brave, 'expected brave-search to load');
    assert.equal(brave?.description, 'Project-level Brave skill');
    assert.deepEqual(brave?.tools, ['bash', 'web']);
    assert.equal(brave?.source, 'claude-agent');
    assert.match(String(brave?.location || ''), /\.claude\/agents\/brave-search\.md$/);

    assert.ok(workspace, 'expected workspace SKILL.md to load');
    assert.equal(workspace?.alwaysInclude, true);
  });
});

test('renderSkillGuidance includes allowed tools for Claude-compatible agents', () => {
  const rendered = renderSkillGuidance([{
    name: 'brave-search',
    description: 'Use Brave Search',
    content: 'Prefer Brave over generic scraping.',
    tools: ['bash', 'web'],
  }]);

  assert.match(rendered, /Skill: brave-search/);
  assert.match(rendered, /Allowed tools: bash, web/);
  assert.match(rendered, /Prefer Brave over generic scraping\./);
});
