import { z } from 'zod';
import type { ChatMessage, ParseableFunctionTool } from '../llm.js';
import { fixUnterminatedStrings, fixArrayFormatting } from '../llm.js';
import type {
  PlanSubagentInvocation,
  SubagentHandler,
  SubagentRunContext,
  SubagentRunResult,
} from './types.js';
import { logError } from '../../utils/errorLogger.js';

type ParsedPlanSubagentDecision = {
  planVersion: number;
  canonicalPlan: string;
  planDeltaSummary: string;
  branches: ReturnType<SubagentRunContext['normalizePlannerBranches']>;
  branchAlignments: Array<{
    label: string;
    planExcerpt: string;
    alignmentChecks?: string[];
  }>;
};

const PlanBranchSchema = z.object({
  label: z.string().trim().min(1).max(80),
  goal: z.string().trim().min(1).max(240),
  why: z.string().trim().min(1).max(240).optional(),
  priority: z.number().transform((v) => Math.min(10, Math.max(0, v))),
  execution_group: z.number().int().min(1).max(8),
  depends_on: z.array(z.string().trim().min(1).max(80)).min(0).max(7).optional(),
});

const PlanBranchAlignmentSchema = z.object({
  label: z.string().trim().min(1).max(80),
  plan_excerpt: z.string().trim().min(1).max(4000),
  alignment_checks: z.array(z.string().trim().min(1).max(240)).max(12).optional(),
});

const PlanSubagentDecisionSchema = z.object({
  plan_version: z.number().int().min(1).max(9999),
  canonical_plan: z.string().trim().min(1),
  plan_delta_summary: z.string().trim().min(1).max(400),
  branches: z.array(PlanBranchSchema).min(1).max(8),
  branch_alignments: z.array(PlanBranchAlignmentSchema).min(1).max(8),
}).superRefine((decision, ctx) => {
  const branchLabels = new Set(decision.branches.map((branch) => branch.label));
  const seenAlignmentLabels = new Set<string>();

  decision.branch_alignments.forEach((alignment, index) => {
    if (!branchLabels.has(alignment.label)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['branch_alignments', index, 'label'],
        message: 'branch_alignments must reference labels from the same plan decision.',
      });
      return;
    }
    if (seenAlignmentLabels.has(alignment.label)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['branch_alignments', index, 'label'],
        message: 'branch_alignments labels must be unique.',
      });
      return;
    }
    seenAlignmentLabels.add(alignment.label);
  });

  decision.branches.forEach((branch, index) => {
    if (!seenAlignmentLabels.has(branch.label)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['branches', index, 'label'],
        message: 'Each branch must have a matching branch_alignment entry.',
      });
    }
  });
});

export const planSubagentHandler: SubagentHandler<PlanSubagentInvocation> = {
  kind: 'plan',
  enabled: true,
  buildPlannerHint: () => 'Use subagent=plan when you need initial branching, a plan refresh, or remediation rebranch. Plan subagent can iterate with user confirmation.',
  async run(ctx: SubagentRunContext, invocation: PlanSubagentInvocation): Promise<SubagentRunResult> {
    const planResultTool: ParseableFunctionTool<{
      kind: 'plan_subagent_result';
      decision: ParsedPlanSubagentDecision;
    }> = {
      name: 'plan_subagent_result',
      description: 'Return the canonical plan, branch graph, and branch alignment excerpts for the current planning scope.',
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties: {
          plan_version: { type: 'integer', minimum: 1, maximum: 9999 },
          canonical_plan: { type: 'string', minLength: 1 },
          plan_delta_summary: { type: 'string', minLength: 1, maxLength: 400 },
          branches: {
            type: 'array',
            minItems: 1,
            maxItems: 8,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                label: { type: 'string', minLength: 1, maxLength: 80 },
                goal: { type: 'string', minLength: 1, maxLength: 240 },
                why: { type: 'string', minLength: 1, maxLength: 240 },
                priority: { type: 'number', minimum: 0, maximum: 10 },
                execution_group: { type: 'integer', minimum: 1, maximum: 8 },
                depends_on: {
                  type: 'array',
                  items: { type: 'string', minLength: 1, maxLength: 80 },
                  minItems: 0,
                  maxItems: 7,
                  uniqueItems: true,
                },
              },
              required: ['label', 'goal', 'priority', 'execution_group'],
            },
          },
          branch_alignments: {
            type: 'array',
            minItems: 1,
            maxItems: 8,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                label: { type: 'string', minLength: 1, maxLength: 80 },
                plan_excerpt: { type: 'string', minLength: 1, maxLength: 4000 },
                alignment_checks: {
                  type: 'array',
                  items: { type: 'string', minLength: 1, maxLength: 240 },
                  minItems: 1,
                  maxItems: 12,
                },
              },
              required: ['label', 'plan_excerpt'],
            },
          },
        },
        required: ['plan_version', 'canonical_plan', 'plan_delta_summary', 'branches', 'branch_alignments'],
      },
      parse: (input) => {
        const record = (input && typeof input === 'object' && !Array.isArray(input))
          ? input as Record<string, unknown>
          : {};

        // The LLM occasionally returns branch_alignments as a JSON-encoded string
        // instead of a real array.  Coerce it so Zod validation doesn't throw.
        if (typeof record['branch_alignments'] === 'string') {
          try {
            let fixed = fixUnterminatedStrings(record['branch_alignments'] as string);
            fixed = fixArrayFormatting(fixed);
            record['branch_alignments'] = JSON.parse(fixed);
          } catch (e) {
            logError(ctx.projectRoot, e, 'plan_subagent_parse_branch_alignments');
            console.warn('Failed to parse branch_alignments JSON, will auto-generate');
            record['branch_alignments'] = undefined;
          }
        }
        if (typeof record['branches'] === 'string') {
          try {
            let fixed = fixUnterminatedStrings(record['branches'] as string);
            fixed = fixArrayFormatting(fixed);
            record['branches'] = JSON.parse(fixed);
          } catch {
            logError(ctx.projectRoot, new Error('Failed to parse branches JSON'), 'plan_subagent_parse_branches');
            record['branches'] = [];
          }
        }
        // Auto-generate branch_alignments if missing
        if (!record['branch_alignments'] || !Array.isArray(record['branch_alignments']) || record['branch_alignments'].length === 0) {
          const branches = Array.isArray(record['branches']) ? record['branches'] : [];
          if (branches.length === 0) {
            throw new Error('No branches provided by LLM');
          }

          const labels = new Set<string>();
          record['branch_alignments'] = branches.map((b: any) => {
            const label = b.label;
            if (!label) {
              throw new Error('Branch missing label field');
            }
            if (labels.has(label)) {
              throw new Error(`Duplicate branch label: ${label}`);
            }
            labels.add(label);
            return { label, plan_excerpt: b.goal || b.label };
          });
        }
        const parsed = PlanSubagentDecisionSchema.parse(record);
        return {
          kind: 'plan_subagent_result',
          decision: {
            planVersion: parsed.plan_version,
            canonicalPlan: parsed.canonical_plan,
            planDeltaSummary: parsed.plan_delta_summary,
            branches: ctx.normalizePlannerBranches(parsed.branches),
            branchAlignments: parsed.branch_alignments.map((alignment) => ({
              label: alignment.label,
              planExcerpt: alignment.plan_excerpt,
              alignmentChecks: alignment.alignment_checks,
            })),
          },
        };
      },
    };

    // Build task state object - structured, not prose
    const buildTaskState = (): string => {
      // Prefer the plan text passed explicitly through the tool call (bounded to 2000 chars)
      // to avoid large canonical plan state inflating the subagent context.
      const planText = invocation.plan
        ? ctx.clipText(invocation.plan, 2000)
        : ctx.trimTextToTokenBudget(
            ctx.canonicalPlanState.canonicalPlanText || 'none',
            1500,
            1500 * 4,
          );

      return JSON.stringify({
        user_goal: ctx.clipText(ctx.originalUserRequest || ctx.input, 400),
        plan_task: ctx.clipText(invocation.task, 600),
        context: invocation.context ? ctx.clipText(invocation.context, 300) : null,
        current_plan_version: ctx.canonicalPlanState.version,
        current_plan: planText,
        branch_status: {
          total: ctx.kanbanSnapshot?.summary.total ?? 0,
          completed: ctx.kanbanSnapshot?.summary.completed ?? 0,
          error: ctx.kanbanSnapshot?.summary.error ?? 0,
        },
        last_error: ctx.branch.handoff?.priorIssue || null,
      }, null, 2);
    };

    const executionInstruction = `Task state:
${buildTaskState()}

Output exactly one plan_subagent_result tool call.
- No markdown
- No prose
- No reasoning text
- No <think> tags
Success = valid tool call matching schema exactly.`;

    let messages: ChatMessage[] = [
      { role: 'system', content: ctx.planSubagentPrompt },
      { role: 'user' as const, content: executionInstruction },
    ];

    let parsedDecision: ParsedPlanSubagentDecision | null = null;

    for (let attempt = 0; attempt < 3; attempt += 1) {
      // Add delay between retries
      if (attempt > 0) {
        await new Promise(resolve => setTimeout(resolve, 2000 * attempt));
      }

      try {
        const { calls, text } = await ctx.generateToolCalls({
          messages,
          tools: [planResultTool as ParseableFunctionTool<any>],
          temperature: ctx.plannerTemperature,
          maxTokens: ctx.planSubagentMaxTokens,
          timeoutLabel: `plan_subagent_${ctx.branch.id}_${attempt + 1} llm`,
        });
        const call = calls[0] as { input?: { kind?: string; decision?: ParsedPlanSubagentDecision } } | undefined;
        if (!call?.input || call.input.kind !== 'plan_subagent_result' || !call.input.decision) {
          if (attempt === 2) {
            const error = new Error(`Plan subagent emitted no actionable result. Raw text: ${ctx.clipText(text, 400)}`);
            logError(ctx.projectRoot, error, 'plan_subagent_no_result');
            throw error;
          }
          continue;
        }
        parsedDecision = call.input.decision;
        const tokenEstimate = ctx.estimateTokens(parsedDecision.canonicalPlan);
        if (tokenEstimate <= 4000) {
          break;
        }
        messages.push({
          role: 'user' as const,
          content: `Plan too large: ~${tokenEstimate} tokens. Compress to ≤4000 tokens. Return plan_subagent_result again.`,
        });
        parsedDecision = null;
      } catch (error) {
        logError(ctx.projectRoot, error, `plan_subagent_attempt_${attempt + 1}`);
        if (attempt === 2) throw error;
        messages.push({
          role: 'user' as const,
          content: `Previous attempt failed. Return valid plan_subagent_result tool call only.`,
        });
      }
    }

    if (!parsedDecision) {
      const error = new Error('Plan subagent failed to produce a <= 4000 token canonical plan after retry.');
      logError(ctx.projectRoot, error, 'plan_subagent_final_failure');
      throw error;
    }

    const nextPlanVersion = ctx.canonicalPlanState.version + 1;
    const trimmedCanonicalPlan = ctx.trimTextToTokenBudget(parsedDecision.canonicalPlan, 4000, 16000);
    const normalizedPlanDecision = {
      planVersion: nextPlanVersion,
      canonicalPlan: trimmedCanonicalPlan,
      planDeltaSummary: parsedDecision.planDeltaSummary,
      branches: parsedDecision.branches.map((plannedBranch) => {
        const alignment = parsedDecision!.branchAlignments.find((item) => item.label === plannedBranch.label);
        return {
          ...plannedBranch,
          planExcerpt: alignment?.planExcerpt || plannedBranch.goal,
          alignmentChecks: alignment?.alignmentChecks || [],
          planVersion: nextPlanVersion,
        };
      }),
      branchAlignments: parsedDecision.branchAlignments,
    };

    return {
      traceSummary: `[plan-subagent] version=${normalizedPlanDecision.planVersion} branches=${normalizedPlanDecision.branches.length} tokens=${ctx.estimateTokens(normalizedPlanDecision.canonicalPlan)}`,
      canonicalPlanUpdate: normalizedPlanDecision,
      nextStep: {
        kind: 'branch',
        thought: `Plan subagent created branches [version ${normalizedPlanDecision.planVersion}]`,
        branches: normalizedPlanDecision.branches,
      },
      metadata: {
        subagent: invocation.subagent,
        task: invocation.task,
      },
    };
  },
};
