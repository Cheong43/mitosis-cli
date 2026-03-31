import type { AgentStep, BranchKanbanSnapshot } from '../../runtime/agent/types.js';
import type { AgentBranchState } from '../../runtime/agent/AgentRuntime.js';

export interface CanonicalPlanState {
  version: number;
  canonicalPlanText: string;
  deltaSummary: string;
  tokenEstimate: number;
  updatedByBranchId?: string;
  updatedAtStep?: number;
}

export interface PlanningDecisionContext {
  branch: AgentBranchState;
  kanbanSnapshot?: BranchKanbanSnapshot;
  canonicalPlanState: CanonicalPlanState;
  isInitialBranch: boolean;
  hasActiveBranches: boolean;
}

export interface PlanningInvocationContext extends PlanningDecisionContext {
  originalUserRequest: string;
  planSubagentPrompt: string;
  trigger: 'initial_branching' | 'plan_refresh' | 'remediation_rebranch';
}

export interface PlanSubagentDecision {
  planVersion: number;
  canonicalPlan: string;
  planDeltaSummary: string;
}

export interface PlanningResult {
  decision: AgentStep;
  canonicalPlanUpdate?: PlanSubagentDecision;
  traceSummary: string;
}
