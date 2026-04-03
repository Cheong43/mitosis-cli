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

test('plan subagent rejects duplicate external workstreams and retries with explicit feedback', async () => {
  const capturedMessages: string[] = [];
  let attempt = 0;

  const ctx = buildCtx({
    branch: {
      id: 'B0.2',
      parentId: 'B0',
      depth: 1,
      label: 'design-system',
      goal: 'Define modern design system with tokens, typography, and responsive patterns.',
      steps: 0,
    },
    kanbanSnapshot: {
      updatedAt: Date.now(),
      summary: {
        total: 3,
        queued: 0,
        active: 2,
        finalizing: 0,
        completed: 1,
        error: 0,
      },
      cards: [
        {
          branchId: 'B0',
          parentBranchId: null,
          label: 'root',
          goal: 'Build the final website.',
          depth: 0,
          step: 1,
          status: 'completed',
          updatedAt: Date.now(),
        },
        {
          branchId: 'B0.1',
          parentBranchId: 'B0',
          label: 'tech-stack',
          goal: 'Research and recommend optimal tech stack for loveconnect.us personal site.',
          depth: 1,
          step: 2,
          status: 'active',
          summary: 'Already researching the stack options.',
          updatedAt: Date.now(),
        },
        {
          branchId: 'B0.2',
          parentBranchId: 'B0',
          label: 'design-system',
          goal: 'Define modern design system with tokens, typography, and responsive patterns.',
          depth: 1,
          step: 1,
          status: 'active',
          updatedAt: Date.now(),
        },
      ],
    },
    generateToolCalls: async <TParsed>(
      options: SubagentToolCallOptions<TParsed>,
    ): Promise<SubagentToolCallResult<TParsed>> => {
      capturedMessages.push(options.messages.map((message) => String(message.content || '')).join('\n\n'));
      attempt += 1;

      if (attempt === 1) {
        return {
          calls: [{
            name: 'plan_subagent_result',
            input: {
              kind: 'plan_subagent_result',
              decision: {
                planVersion: 1,
                canonicalPlan: '1. Revisit the tech stack.\n2. Continue from there.',
                planDeltaSummary: 'Retrying plan.',
                branches: [{
                  label: 'tech-stack',
                  goal: 'Research and recommend optimal tech stack for loveconnect.us personal site.',
                  priority: 5,
                }],
                branchAlignments: [{
                  label: 'tech-stack',
                  planExcerpt: 'Research and recommend optimal tech stack for loveconnect.us personal site.',
                }],
              },
            } as TParsed,
          }],
          text: '',
        };
      }

      return {
        calls: [{
          name: 'plan_subagent_result',
          input: {
            kind: 'plan_subagent_result',
            decision: {
              planVersion: 1,
              canonicalPlan: '1. Define tokens.\n2. Specify typography.\n3. Lock responsive rules.',
              planDeltaSummary: 'Scoped the design-system work.',
              branches: [{
                label: 'design-tokens',
                goal: 'Define the color, spacing, and radius tokens for the site.',
                priority: 6,
              }],
              branchAlignments: [{
                label: 'design-tokens',
                planExcerpt: 'Define the color, spacing, and radius tokens for the site.',
              }],
            },
          } as TParsed,
        }],
        text: '',
      };
    },
  });

  const result = await planSubagentHandler.run(ctx, {
    subagent: 'plan',
    task: 'Break the design-system scope into child workstreams only.',
    context: 'Do not reopen already-owned workstreams.',
  });

  assert.equal(attempt, 2);
  assert.equal(result.canonicalPlanUpdate?.branches[0]?.label, 'design-tokens');
  assert.match(capturedMessages[0] || '', /"existing_workstreams"/);
  assert.match(capturedMessages[0] || '', /"relation": "sibling"/);
  assert.match(capturedMessages[1] || '', /duplicates existing sibling workstream/i);
  assert.match(capturedMessages[1] || '', /do not recreate existing workstreams/i);
});
