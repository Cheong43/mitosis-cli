import test from 'node:test';
import assert from 'node:assert/strict';
import { planSubagentHandler } from './subagents/plan.js';
import type {
  SubagentRunContext,
  SubagentToolCallOptions,
  SubagentToolCallResult,
} from './subagents/types.js';

function buildCtx(overrides?: Partial<SubagentRunContext>): SubagentRunContext {
  return {
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
      canonicalPlanText: 'state plan text that should be ignored when invocation.plan is set',
      tokenEstimate: 0,
      updatedByBranchId: 'B0',
      updatedAtStep: 0,
      deltaSummary: '',
    },
    originalUserRequest: 'request',
    input: 'request',
    plannerTemperature: 0,
    planSubagentPrompt: 'prompt',
    planSubagentMaxTokens: 1200,
    clipText: (text, maxChars) => (maxChars !== undefined ? text.slice(0, maxChars) : text),
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
    generateToolCalls: async <TParsed>(
      _options: SubagentToolCallOptions<TParsed>,
    ): Promise<SubagentToolCallResult<TParsed>> => ({
      calls: [{
        name: 'plan_subagent_result',
        input: {
          kind: 'plan_subagent_result',
          decision: {
            planVersion: 1,
            canonicalPlan: '1. Gather evidence.\n2. Merge the result.',
            planDeltaSummary: 'Created the first canonical plan.',
            branches: [{
              label: 'Evidence path',
              goal: 'Collect the source evidence.',
              priority: 1,
            }],
            branchAlignments: [{
              label: 'Evidence path',
              planExcerpt: 'Collect the source evidence.',
              alignmentChecks: ['Cite the exact source.'],
            }],
          },
        } as TParsed,
      }],
      text: '',
    }),
    ...overrides,
  };
}

test('plan subagent normalizes the returned plan into nextStep and canonicalPlanUpdate', async () => {
  const result = await planSubagentHandler.run(buildCtx(), {
    subagent: 'plan',
    task: 'Split into one evidence branch.',
    context: 'Need a canonical plan. Design the initial branch graph.',
  });

  assert.equal(result.traceSummary, '[plan-subagent] version=1 branches=1 tokens=40');
  assert.equal(result.nextStep?.kind, 'branch');
  assert.equal(result.canonicalPlanUpdate?.planVersion, 1);
  assert.equal(result.canonicalPlanUpdate?.planDeltaSummary, 'Created the first canonical plan.');
  assert.equal(result.canonicalPlanUpdate?.branches[0]?.planExcerpt, 'Collect the source evidence.');
  assert.deepEqual(result.canonicalPlanUpdate?.branches[0]?.alignmentChecks, ['Cite the exact source.']);
});

test('plan subagent uses invocation.plan when provided instead of canonical plan state', async () => {
  const capturedMessages: Array<{ role: string; content: string }> = [];

  const ctx = buildCtx({
    generateToolCalls: async <TParsed>(
      options: SubagentToolCallOptions<TParsed>,
    ): Promise<SubagentToolCallResult<TParsed>> => {
      // Capture the user message to verify which plan text was used
      const userMsg = options.messages.find((m) => m.role === 'user');
      if (userMsg && typeof userMsg.content === 'string') {
        capturedMessages.push({ role: 'user', content: userMsg.content });
      }
      return {
        calls: [{
          name: 'plan_subagent_result',
          input: {
            kind: 'plan_subagent_result',
            decision: {
              planVersion: 1,
              canonicalPlan: '1. Step A.\n2. Step B.',
              planDeltaSummary: 'Initial plan.',
              branches: [{ label: 'A', goal: 'Do step A.', priority: 5 }],
              branchAlignments: [{ label: 'A', planExcerpt: 'Do step A.' }],
            },
          } as TParsed,
        }],
        text: '',
      };
    },
  });

  const invocationPlan = 'Compact plan passed via tool call (< 2000 chars).';

  await planSubagentHandler.run(ctx, {
    subagent: 'plan',
    task: 'Branch the work.',
    plan: invocationPlan,
  });

  // The task state forwarded to the plan subagent LLM should contain the
  // invocation plan text, NOT the canonical plan state text.
  assert.ok(capturedMessages.length > 0, 'generateToolCalls should have been called');
  const taskStateMsg = capturedMessages[0].content;
  assert.ok(
    taskStateMsg.includes(invocationPlan),
    `Task state should contain the invocation plan text. Got:\n${taskStateMsg}`,
  );
  assert.ok(
    !taskStateMsg.includes('state plan text that should be ignored'),
    'Task state should NOT contain the canonical plan state text when invocation.plan is provided',
  );
});

test('plan subagent falls back to canonical plan state when invocation.plan is absent', async () => {
  const capturedMessages: Array<{ role: string; content: string }> = [];
  const canonicalPlanText = 'canonical plan stored in state';

  const ctx = buildCtx({
    canonicalPlanState: {
      version: 2,
      canonicalPlanText,
      tokenEstimate: 5,
      updatedByBranchId: 'B0',
      updatedAtStep: 1,
      deltaSummary: '',
    },
    generateToolCalls: async <TParsed>(
      options: SubagentToolCallOptions<TParsed>,
    ): Promise<SubagentToolCallResult<TParsed>> => {
      const userMsg = options.messages.find((m) => m.role === 'user');
      if (userMsg && typeof userMsg.content === 'string') {
        capturedMessages.push({ role: 'user', content: userMsg.content });
      }
      return {
        calls: [{
          name: 'plan_subagent_result',
          input: {
            kind: 'plan_subagent_result',
            decision: {
              planVersion: 3,
              canonicalPlan: 'updated plan',
              planDeltaSummary: 'Refreshed.',
              branches: [{ label: 'B', goal: 'Do B.', priority: 3 }],
              branchAlignments: [{ label: 'B', planExcerpt: 'Do B.' }],
            },
          } as TParsed,
        }],
        text: '',
      };
    },
  });

  await planSubagentHandler.run(ctx, {
    subagent: 'plan',
    task: 'Refresh the plan.',
    // no plan field — should fall back to canonicalPlanState
  });

  assert.ok(capturedMessages.length > 0, 'generateToolCalls should have been called');
  const taskStateMsg = capturedMessages[0].content;
  assert.ok(
    taskStateMsg.includes(canonicalPlanText),
    `Task state should fall back to canonical plan state text. Got:\n${taskStateMsg}`,
  );
});
