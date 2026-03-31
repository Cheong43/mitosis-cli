import test from 'node:test';
import assert from 'node:assert/strict';
import { planSubagentHandler } from './subagents/plan.js';
import type {
  SubagentRunContext,
  SubagentToolCallOptions,
  SubagentToolCallResult,
} from './subagents/types.js';

test('plan subagent normalizes the returned plan into nextStep and canonicalPlanUpdate', async () => {
  const ctx: SubagentRunContext = {
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
    planSubagentMaxTokens: 1200,
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
  };

  const result = await planSubagentHandler.run(ctx, {
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
