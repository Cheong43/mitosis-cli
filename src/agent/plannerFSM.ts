/**
 * FSM-based planner orchestration.
 *
 * Enforces deterministic state transitions for the plan→execute loop,
 * preventing invalid states like branching during tool execution or
 * selecting multiple tools simultaneously.
 */

export type PlannerState =
  | 'THINKING'
  | 'TOOL_SELECTION'
  | 'TOOL_EXECUTION'
  | 'OBSERVATION'
  | 'BRANCH_DECISION'
  | 'FINAL';

export type PlannerEvent =
  | { type: 'THOUGHT_COMPLETE' }
  | { type: 'TOOL_SELECTED'; toolName: string }
  | { type: 'TOOL_EXECUTED'; success: boolean }
  | { type: 'OBSERVATION_PROCESSED' }
  | { type: 'BRANCH_REQUESTED' }
  | { type: 'FINALIZE' };

const VALID_TRANSITIONS: Record<PlannerState, PlannerState[]> = {
  THINKING: ['TOOL_SELECTION', 'BRANCH_DECISION', 'FINAL'],
  TOOL_SELECTION: ['TOOL_EXECUTION'],
  TOOL_EXECUTION: ['OBSERVATION'],
  OBSERVATION: ['THINKING'],
  BRANCH_DECISION: ['FINAL'],
  FINAL: [],
};

export class PlannerFSM {
  private state: PlannerState = 'THINKING';
  private history: Array<{ from: PlannerState; to: PlannerState; event: PlannerEvent; timestamp: number }> = [];

  getState(): PlannerState {
    return this.state;
  }

  getHistory() {
    return this.history;
  }

  transition(event: PlannerEvent): PlannerState {
    const nextState = this.resolveNextState(event);
    const allowed = VALID_TRANSITIONS[this.state];

    if (!allowed.includes(nextState)) {
      throw new InvalidTransitionError(this.state, nextState, event);
    }

    this.history.push({
      from: this.state,
      to: nextState,
      event,
      timestamp: Date.now(),
    });

    this.state = nextState;
    return this.state;
  }

  /**
   * Return which tools are allowed in the current state.
   * During TOOL_SELECTION, all tools are available.
   * In other states, no tool selection should happen.
   */
  getAllowedToolCount(): number {
    if (this.state === 'TOOL_SELECTION') {
      return 1; // Only one tool at a time
    }
    return 0;
  }

  /**
   * Force a specific tool choice constraint based on the current state.
   */
  enforceToolChoice(): 'required' | 'auto' | undefined {
    switch (this.state) {
      case 'TOOL_SELECTION':
        return 'required';
      case 'THINKING':
        return 'auto';
      default:
        return undefined;
    }
  }

  /**
   * Check whether the FSM is in a terminal state.
   */
  isTerminal(): boolean {
    return this.state === 'FINAL';
  }

  /**
   * Reset the FSM to the initial state (for new branch runs).
   */
  reset(): void {
    this.state = 'THINKING';
    this.history = [];
  }

  private resolveNextState(event: PlannerEvent): PlannerState {
    switch (event.type) {
      case 'THOUGHT_COMPLETE':
        return 'TOOL_SELECTION';
      case 'TOOL_SELECTED':
        return 'TOOL_EXECUTION';
      case 'TOOL_EXECUTED':
        return 'OBSERVATION';
      case 'OBSERVATION_PROCESSED':
        return 'THINKING';
      case 'BRANCH_REQUESTED':
        return 'BRANCH_DECISION';
      case 'FINALIZE':
        return 'FINAL';
    }
  }
}

export class InvalidTransitionError extends Error {
  constructor(
    public readonly fromState: PlannerState,
    public readonly toState: PlannerState,
    public readonly event: PlannerEvent,
  ) {
    super(
      `Invalid FSM transition: ${fromState} → ${toState} (event: ${event.type}). `
      + `Allowed transitions from ${fromState}: ${VALID_TRANSITIONS[fromState].join(', ') || 'none'}`,
    );
    this.name = 'InvalidTransitionError';
  }
}
