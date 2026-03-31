import type {
  ResearchSubagentInvocation,
  SubagentHandler,
  SubagentRunContext,
  SubagentRunResult,
} from './types.js';

export const researchSubagentHandler: SubagentHandler<ResearchSubagentInvocation> = {
  kind: 'research',
  enabled: true,
  buildPlannerHint: () => 'Use research subagent for information gathering, exploration, and analysis tasks.',
  async run(
    ctx: SubagentRunContext,
    invocation: ResearchSubagentInvocation,
  ): Promise<SubagentRunResult> {
    return {
      traceSummary: `Research: ${invocation.task}`,
      metadata: {
        subagent: invocation.subagent,
        enabled: true,
        task: invocation.task,
      },
    };
  },
};
