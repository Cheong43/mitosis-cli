import test from 'node:test';
import assert from 'node:assert/strict';
import { SubagentRegistry } from './subagents/registry.js';
import { runSubagent } from './subagents/runner.js';
import { researchSubagentHandler } from './subagents/research.js';
import type { SubagentHandler, SubagentRunContext } from './subagents/types.js';

const baseContext: SubagentRunContext = {
  projectRoot: '/tmp/test',
  branch: {
    id: 'B0',
    parentId: null,
    depth: 0,
    label: 'root',
    goal: 'goal',
    steps: 0,
  },
  canonicalPlanState: {
    version: 0,
    canonicalPlanText: '',
    tokenEstimate: 0,
    updatedByBranchId: 'B0',
    updatedAtStep: 0,
    deltaSummary: '',
  },
  originalUserRequest: 'request',
  input: 'request',
  plannerTemperature: 0,
  planSubagentPrompt: 'prompt',
  planSubagentMaxTokens: 1000,
  clipText: (text) => text,
  estimateTokens: (text) => text.length,
  normalizePlannerBranches: (branches) => branches.map((branch) => ({
    label: branch.label,
    goal: branch.goal,
    why: branch.why,
    priority: branch.priority,
    executionGroup: branch.execution_group,
    dependsOn: branch.depends_on,
  })),
  trimTextToTokenBudget: (text) => text,
  renderPlanSubagentFocus: () => 'focus',
  generateToolCalls: async () => ({ calls: [], text: '' }),
};

test('subagent registry rejects duplicate registrations', () => {
  const handler: SubagentHandler<any> = {
    kind: 'plan',
    enabled: true,
    async run() {
      return { traceSummary: 'ok' };
    },
  };
  const registry = new SubagentRegistry([handler]);
  assert.throws(() => registry.register(handler), /already registered/i);
});

test('subagent runner dispatches to the matching handler', async () => {
  const registry = new SubagentRegistry([{
    kind: 'plan',
    enabled: true,
    async run(_ctx, invocation) {
      return {
        traceSummary: `handled ${invocation.subagent}`,
      };
    },
  }]);

  const result = await runSubagent(registry, baseContext, {
    subagent: 'plan',
    task: 'plan',
    context: 'focus',
  });

  assert.equal(result.traceSummary, 'handled plan');
});

test('disabled research subagent returns a stable structured result', async () => {
  const registry = new SubagentRegistry([researchSubagentHandler]);

  const result = await runSubagent(registry, baseContext, {
    subagent: 'research',
    task: 'query',
    context: 'focus',
  });

  assert.equal(result.traceSummary, 'research subagent is registered but not enabled');
  assert.equal(result.nextStep, undefined);
  assert.equal(result.canonicalPlanUpdate, undefined);
  assert.deepEqual(result.metadata, {
    subagent: 'research',
    enabled: false,
    focus: 'focus',
  });
});
