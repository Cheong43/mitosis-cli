import type {
  AgentStep,
  BranchKanbanSnapshot,
  BranchHandoff,
  CanonicalPlanState,
  PlanSubagentDecision,
  PlannedBranch,
  SynthesisSharedHandoff,
} from '../../runtime/agent/types.js';
import type {
  ChatMessage,
  ParseableFunctionTool,
  ParsedToolCall,
} from '../llm.js';

export type PlanSubagentTrigger =
  | 'initial_branching'
  | 'plan_refresh'
  | 'remediation_rebranch';

export interface PlanSubagentInvocation {
  subagent: 'plan';
  task: string;
  context?: string;
  /** Compact plan text passed from the main agent (max 2000 chars). When provided,
   *  the plan subagent uses this instead of reading the full canonical plan state,
   *  keeping the planning context small and stable. */
  plan?: string;
}

export interface ResearchSubagentInvocation {
  subagent: 'research';
  task: string;
  context?: string;
}

export interface CrawlerSubagentInvocation {
  subagent: 'crawler';
  task: string;
  context?: string;
}

export type SubagentInvocation = PlanSubagentInvocation | ResearchSubagentInvocation | CrawlerSubagentInvocation;
export type SubagentKind = SubagentInvocation['subagent'];

export interface SubagentToolCallOptions<TParsed> {
  messages: ChatMessage[];
  tools: Array<ParseableFunctionTool<TParsed>>;
  temperature?: number;
  maxTokens?: number;
  timeoutLabel: string;
  jsonMode?: boolean;
}

export interface SubagentToolCallResult<TParsed> {
  calls: ParsedToolCall<TParsed>[];
  text: string;
}

export interface SubagentBranchState {
  id: string;
  parentId: string | null;
  depth: number;
  label: string;
  goal: string;
  steps: number;
  handoff?: BranchHandoff;
  sharedHandoff?: SynthesisSharedHandoff;
  planVersionSeen?: number;
  planExcerpt?: string;
  alignmentChecks?: string[];
  completionSummary?: string;
  outcomeReason?: string;
}

export interface SubagentRunContext {
  projectRoot: string;
  branch: SubagentBranchState;
  kanbanSnapshot?: BranchKanbanSnapshot;
  canonicalPlanState: CanonicalPlanState;
  originalUserRequest: string;
  input: string;
  plannerTemperature: number;
  planSubagentPrompt: string;
  planSubagentMaxTokens: number;
  clipText(text: string, maxChars?: number): string;
  estimateTokens(text: string): number;
  normalizePlannerBranches(branches: Array<{
    label: string;
    goal: string;
    why?: string;
    priority?: number;
    execution_group?: number;
    depends_on?: string[];
  }>): PlannedBranch[];
  trimTextToTokenBudget(text: string, maxTokens: number, fallbackChars?: number): string;
  renderPlanSubagentFocus(branch: SubagentBranchState, kanbanSnapshot?: BranchKanbanSnapshot): string;
  /** Compact digest of shared workspace facts and sibling outputs for the planning scope. */
  sharedProgressDigest?: string;
  generateToolCalls<TParsed>(
    options: SubagentToolCallOptions<TParsed>,
  ): Promise<SubagentToolCallResult<TParsed>>;
}

export interface SubagentRunResult {
  traceSummary: string;
  nextStep?: AgentStep;
  canonicalPlanUpdate?: PlanSubagentDecision;
  metadata?: Record<string, unknown>;
}

export interface SubagentHandler<TInvocation extends SubagentInvocation = SubagentInvocation> {
  kind: TInvocation['subagent'];
  enabled: boolean;
  buildPlannerHint?(): string;
  run(ctx: SubagentRunContext, invocation: TInvocation): Promise<SubagentRunResult>;
}
