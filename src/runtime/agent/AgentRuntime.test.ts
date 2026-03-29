import test from 'node:test';
import assert from 'node:assert/strict';
import { AgentRuntime } from './AgentRuntime.js';
import type { AgentStep, AgentTraceEvent, SynthesisResult, TranscriptMessage } from './types.js';

class FakePlanner {
  constructor(private readonly planFn: (transcript: TranscriptMessage[]) => Promise<AgentStep>) {}

  async plan(transcript: TranscriptMessage[]): Promise<AgentStep> {
    return this.planFn(transcript);
  }
}

class FakeToolRuntime {
  constructor(private readonly executeFn?: (toolName: string, args: Record<string, unknown>) => Promise<unknown>) {}

  resetSession(): void {}

  async execute(toolName: string, args: Record<string, unknown>) {
    try {
      const result = this.executeFn ? await this.executeFn(toolName, args) : { ok: true };
      return {
        success: true,
        result,
        durationMs: 0,
      };
    } catch (error: unknown) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        durationMs: 0,
      };
    }
  }
}

test('child branches get a fresh local step budget', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const lastMessage = transcript[transcript.length - 1]?.content || '';
    if (lastMessage.includes('child branch "Fast path"')) {
      return {
        kind: 'final',
        content: 'child-final-answer',
      };
    }

    return {
      kind: 'branch',
      thought: 'Split into two strategies.',
      branches: [
        { label: 'Fast path', goal: 'Finish quickly', priority: 1 },
        { label: 'Slow path', goal: 'Fallback path', priority: 0.5 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 1,
    maxBranchDepth: 1,
    maxCompletedBranches: 1,
    branchConcurrency: 2,
  });

  const answer = await runtime.run('solve this');
  assert.equal(answer, 'child-final-answer');
});

test('branch transcripts record planner decisions as text markers instead of legacy JSON blobs', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');

    if (serialized.includes('TOOL OBSERVATION for read:')) {
      assert.match(serialized, /PLANNER TOOL DECISION:/);
      assert.doesNotMatch(serialized, /\{"kind":"tool"/);
      return {
        kind: 'final',
        content: 'branch transcript clean',
      };
    }

    if (serialized.includes('child branch "Evidence path"')) {
      assert.match(serialized, /PLANNER BRANCH DECISION:/);
      assert.doesNotMatch(serialized, /\{"kind":"branch"/);
      return {
        kind: 'tool',
        toolCalls: [{ name: 'read', arguments: { path: 'note.txt' } }],
      };
    }

    return {
      kind: 'branch',
      thought: 'Split root from child evidence collection.',
      branches: [
        { label: 'Evidence path', goal: 'Read note.txt before answering', priority: 1 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async () => 'ok'),
    maxSteps: 3,
    maxBranchDepth: 1,
    maxCompletedBranches: 1,
  });

  const answer = await runtime.run('inspect branch transcript');
  assert.equal(answer, 'branch transcript clean');
});

test('web observations keep the search query in the branch transcript', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => String(message.content)).join('\n');

    if (serialized.includes('TOOL OBSERVATION for web search query=')) {
      assert.match(serialized, /TOOL OBSERVATION for web search query="Qin Shi Huang tomb army"/);
      return {
        kind: 'final',
        content: 'web transcript keeps query',
      };
    }

    return {
      kind: 'tool',
      toolCalls: [
        { name: 'web', arguments: { mode: 'search', query: 'Qin Shi Huang tomb army' } },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async () => '{"kind":"web_search","query":"Qin Shi Huang tomb army","results":[]}'),
    maxSteps: 2,
  });

  const answer = await runtime.run('search history should be preserved');
  assert.equal(answer, 'web transcript keeps query');
});

test('root branch blocks a third near-duplicate web-search round and forces replanning', async () => {
  const executedQueries: string[] = [];
  let sawGuardMessage = false;
  let planCount = 0;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => String(message.content)).join('\n');
    if (serialized.includes('Blocked near-duplicate web search loop')) {
      sawGuardMessage = true;
      return {
        kind: 'final',
        content: 'guarded root final',
      };
    }

    planCount += 1;
    if (planCount === 1) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'web', arguments: { mode: 'search', query: 'Alibaba latest revenue' } }],
      };
    }
    if (planCount === 2) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'web', arguments: { mode: 'search', query: 'Alibaba latest revenue today' } }],
      };
    }
    return {
      kind: 'tool',
      toolCalls: [{ name: 'web', arguments: { mode: 'search', query: 'Alibaba latest revenue current' } }],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async (_toolName, args) => {
      const query = String(args.query || '');
      executedQueries.push(query);
      return JSON.stringify({
        kind: 'web_search',
        query,
        results: [{ title: 'Alibaba Group', url: 'https://www.alibabagroup.com/en/news' }],
      });
    }),
    maxSteps: 5,
  });

  const answer = await runtime.run('research alibaba');
  assert.equal(answer, 'guarded root final');
  assert.equal(sawGuardMessage, true);
  assert.deepEqual(executedQueries, [
    'Alibaba latest revenue',
    'Alibaba latest revenue today',
  ]);
});

test('child branches inherit the same repeated web-search loop guard', async () => {
  const executedQueries: string[] = [];
  let sawChildGuardMessage = false;
  let childPlanCount = 0;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => String(message.content)).join('\n');
    if (!serialized.includes('child branch "Evidence path"')) {
      return {
        kind: 'branch',
        branches: [
          { label: 'Evidence path', goal: 'Collect web evidence', priority: 1 },
        ],
      };
    }

    if (serialized.includes('Blocked near-duplicate web search loop')) {
      sawChildGuardMessage = true;
      return {
        kind: 'final',
        content: 'guarded child final',
      };
    }

    childPlanCount += 1;
    if (childPlanCount === 1) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'web', arguments: { mode: 'search', query: 'Alibaba cloud revenue' } }],
      };
    }
    if (childPlanCount === 2) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'web', arguments: { mode: 'search', query: 'Alibaba cloud revenue latest' } }],
      };
    }
    return {
      kind: 'tool',
      toolCalls: [{ name: 'web', arguments: { mode: 'search', query: 'Alibaba cloud revenue current' } }],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async (_toolName, args) => {
      const query = String(args.query || '');
      executedQueries.push(query);
      return JSON.stringify({
        kind: 'web_search',
        query,
        results: [{ title: 'Alibaba Cloud', url: 'https://www.alibabacloud.com/company/news' }],
      });
    }),
    maxSteps: 5,
    maxBranchDepth: 1,
    maxBranchWidth: 1,
    maxCompletedBranches: 1,
  });

  const answer = await runtime.run('branch for evidence');
  assert.equal(answer, 'guarded child final');
  assert.equal(sawChildGuardMessage, true);
  assert.deepEqual(executedQueries, [
    'Alibaba cloud revenue',
    'Alibaba cloud revenue latest',
  ]);
});

test('branches execute concurrently up to branchConcurrency', async () => {
  let maxObservedConcurrency = 0;
  let currentConcurrency = 0;

  const toolRuntime = new FakeToolRuntime(async () => {
    currentConcurrency += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, currentConcurrency);
    await new Promise((resolve) => setTimeout(resolve, 80));
    currentConcurrency -= 1;
    return { ok: true };
  });

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');
    if (serialized.includes('TOOL OBSERVATION for wait_tool')) {
      const childMatch = serialized.match(/child branch "([^"]+)"/);
      const childLabel = childMatch ? childMatch[1] : '';
      return {
        kind: 'final',
        content: `${childLabel} done`,
      };
    }

    if (serialized.includes('child branch "')) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'wait_tool', arguments: {} }],
      };
    }

    return {
      kind: 'branch',
      branches: [
        { label: 'A', goal: 'Wait in branch A', priority: 1 },
        { label: 'B', goal: 'Wait in branch B', priority: 1 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime,
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 2,
    maxCompletedBranches: 2,
    branchConcurrency: 2,
    synthesizeFinal: async ({ branches }) => branches.map((branch) => branch.finalAnswer).join(' | '),
  });

  const answer = await runtime.run('run two waits');
  assert.match(answer, /A done/);
  assert.match(answer, /B done/);
  assert.equal(maxObservedConcurrency, 2);
});

test('execution groups let sibling branches run in waves', async () => {
  let maxObservedConcurrency = 0;
  let currentConcurrency = 0;
  const completedLabels: string[] = [];

  const toolRuntime = new FakeToolRuntime(async (_toolName, args) => {
    const label = String(args.label || '');
    if (label === 'C') {
      assert.ok(completedLabels.includes('A'), 'C should wait for A to complete');
      assert.ok(completedLabels.includes('B'), 'C should wait for B to complete');
    }
    currentConcurrency += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, currentConcurrency);
    await new Promise((resolve) => setTimeout(resolve, label === 'C' ? 20 : 80));
    currentConcurrency -= 1;
    return { ok: true };
  });

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');
    const childMatch = serialized.match(/child branch "([^"]+)"/);
    const childLabel = childMatch ? childMatch[1] : '';

    if (serialized.includes('TOOL OBSERVATION for wait_tool')) {
      completedLabels.push(childLabel);
      return {
        kind: 'final',
        content: `${childLabel} done`,
      };
    }

    if (childLabel) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'wait_tool', arguments: { label: childLabel } }],
      };
    }

    return {
      kind: 'branch',
      branches: [
        { label: 'A', goal: 'First wave A', priority: 1, executionGroup: 1 },
        { label: 'B', goal: 'First wave B', priority: 1, executionGroup: 1 },
        { label: 'C', goal: 'Second wave C', priority: 1, executionGroup: 2 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime,
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 3,
    maxCompletedBranches: 3,
    branchConcurrency: 2,
    synthesizeFinal: async ({ branches }) => branches.map((branch) => branch.finalAnswer).join(' | '),
  });

  const answer = await runtime.run('run grouped waves');
  assert.match(answer, /A done/);
  assert.match(answer, /B done/);
  assert.match(answer, /C done/);
  assert.equal(maxObservedConcurrency, 2);
  assert.equal(completedLabels.includes('C'), true);
});

test('depends_on gates same-group siblings until prerequisites complete', async () => {
  let maxObservedConcurrency = 0;
  let currentConcurrency = 0;
  const completedLabels: string[] = [];

  const toolRuntime = new FakeToolRuntime(async (_toolName, args) => {
    const label = String(args.label || '');
    if (label === 'C') {
      assert.ok(completedLabels.includes('A'), 'C should wait for A to complete');
      assert.ok(completedLabels.includes('B'), 'C should wait for B to complete');
    }
    currentConcurrency += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, currentConcurrency);
    await new Promise((resolve) => setTimeout(resolve, label === 'C' ? 20 : 80));
    currentConcurrency -= 1;
    return { ok: true };
  });

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');
    const childMatch = serialized.match(/child branch "([^"]+)"/);
    const childLabel = childMatch ? childMatch[1] : '';

    if (serialized.includes('TOOL OBSERVATION for wait_tool')) {
      completedLabels.push(childLabel);
      return {
        kind: 'final',
        content: `${childLabel} done`,
      };
    }

    if (childLabel) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'wait_tool', arguments: { label: childLabel } }],
      };
    }

    return {
      kind: 'branch',
      branches: [
        { label: 'A', goal: 'First dependency A', priority: 1, executionGroup: 1 },
        { label: 'B', goal: 'First dependency B', priority: 1, executionGroup: 1 },
        { label: 'C', goal: 'Follow-up C', priority: 1, executionGroup: 1, dependsOn: ['A', 'B'] },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime,
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 3,
    maxCompletedBranches: 3,
    branchConcurrency: 3,
    synthesizeFinal: async ({ branches }) => branches.map((branch) => branch.finalAnswer).join(' | '),
  });

  const answer = await runtime.run('run sibling prerequisites');
  assert.match(answer, /A done/);
  assert.match(answer, /B done/);
  assert.match(answer, /C done/);
  assert.equal(maxObservedConcurrency, 2);
  assert.equal(completedLabels.includes('C'), true);
});

test('branches are unlimited by default and rely on external throttling unless a cap is set', async () => {
  let maxObservedConcurrency = 0;
  let currentConcurrency = 0;

  const toolRuntime = new FakeToolRuntime(async () => {
    currentConcurrency += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, currentConcurrency);
    await new Promise((resolve) => setTimeout(resolve, 80));
    currentConcurrency -= 1;
    return { ok: true };
  });

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');
    if (serialized.includes('TOOL OBSERVATION for wait_tool')) {
      const childMatch = serialized.match(/child branch "([^"]+)"/);
      const childLabel = childMatch ? childMatch[1] : '';
      return {
        kind: 'final',
        content: `${childLabel} done`,
      };
    }

    if (serialized.includes('child branch "')) {
      return {
        kind: 'tool',
        toolCalls: [{ name: 'wait_tool', arguments: {} }],
      };
    }

    return {
      kind: 'branch',
      branches: [
        { label: 'A', goal: 'Wait in branch A', priority: 1 },
        { label: 'B', goal: 'Wait in branch B', priority: 1 },
        { label: 'C', goal: 'Wait in branch C', priority: 1 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime,
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 3,
    maxCompletedBranches: 3,
    synthesizeFinal: async ({ branches }) => branches.map((branch) => branch.finalAnswer).join(' | '),
  });

  const answer = await runtime.run('run three waits');
  assert.match(answer, /A done/);
  assert.match(answer, /B done/);
  assert.match(answer, /C done/);
  assert.equal(maxObservedConcurrency, 3);
});

test('later branches receive a shared kanban snapshot in their planning transcript', async () => {
  let sawKanbanSnapshot = false;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');

    if (serialized.includes('child branch "Alpha"')) {
      return {
        kind: 'final' as const,
        content: 'alpha ready',
        outcome: 'success',
      };
    }

    if (serialized.includes('child branch "Beta"')) {
      assert.match(serialized, /Shared branch kanban snapshot:/);
      assert.match(serialized, /\[B0\.1\] Alpha:/);
      assert.match(serialized, /status=completed/);
      assert.match(serialized, /result=alpha ready/);
      sawKanbanSnapshot = true;
      return {
        kind: 'final' as const,
        content: 'beta ready',
        outcome: 'success',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'split into alpha then beta',
      branches: [
        { label: 'Alpha', goal: 'Finish alpha first', priority: 1 },
        { label: 'Beta', goal: 'Reuse sibling results', priority: 0.2 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 2,
    branchConcurrency: 1,
    maxCompletedBranches: 2,
    synthesizeFinal: async ({ branches }) => branches.map((branch) => branch.finalAnswer).join(' + '),
  });

  const answer = await runtime.run('share branch board');
  assert.equal(answer, 'alpha ready + beta ready');
  assert.equal(sawKanbanSnapshot, true);
});

test('kanban sync stays in the leading system message instead of inserting a later system role', async () => {
  let inspectedChildTranscript = false;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => String(message.content)).join('\n');

    if (serialized.includes('child branch "Alpha"')) {
      return {
        kind: 'final' as const,
        content: 'alpha ok',
        outcome: 'success',
      };
    }

    if (serialized.includes('child branch "Beta"')) {
      const systemIndexes = transcript
        .map((message, index) => ({ role: message.role, index }))
        .filter((message) => message.role === 'system')
        .map((message) => message.index);
      assert.deepEqual(systemIndexes, [0]);
      assert.match(String(transcript[0]?.content || ''), /Shared branch kanban snapshot:/);
      inspectedChildTranscript = true;
      return {
        kind: 'final' as const,
        content: 'beta ok',
        outcome: 'success',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'split into alpha then beta',
      branches: [
        { label: 'Alpha', goal: 'Finish alpha first', priority: 1 },
        { label: 'Beta', goal: 'Run after alpha and inspect sync placement', priority: 0.2, executionGroup: 2 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 2,
    branchConcurrency: 1,
    maxCompletedBranches: 2,
    synthesizeFinal: async ({ branches }) => branches.map((branch) => branch.finalAnswer).join(' + '),
  });

  const answer = await runtime.run('inspect kanban sync placement', [
    { role: 'system', content: 'base system prompt' },
  ]);
  assert.equal(answer, 'alpha ok + beta ok');
  assert.equal(inspectedChildTranscript, true);
});

test('trace metadata carries structured kanban cards with artifacts and summaries', async () => {
  const traces: AgentTraceEvent[] = [];

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');
    if (serialized.includes('TOOL OBSERVATION for read:')) {
      return {
        kind: 'final' as const,
        content: 'note checked',
        completionSummary: 'Verified note.txt',
        outcome: 'success',
      };
    }

    return {
      kind: 'tool' as const,
      toolCalls: [{ name: 'read', arguments: { path: 'note.txt' } }],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async () => 'contents ok'),
    maxSteps: 2,
    onTrace: (event) => {
      traces.push(event);
    },
  });

  const answer = await runtime.run('inspect note');
  assert.equal(answer, 'note checked');

  const toolObservation = traces.find((event) => event.type === 'observation' && String(event.content).includes('contents ok'));
  const toolCard = toolObservation?.metadata?.kanbanCard as any;
  assert.ok(toolCard, 'tool observation should include kanban card metadata');
  assert.equal(toolCard?.status, 'active');
  assert.deepEqual(toolCard?.artifacts, ['note.txt']);
  assert.match(String(toolCard?.summary || ''), /read:/i);

  const completionObservation = traces.find((event) => event.type === 'observation' && String(event.content).includes('Branch completed'));
  const completionCard = completionObservation?.metadata?.kanbanCard as any;
  assert.ok(completionCard, 'completion observation should include kanban card metadata');
  assert.equal(completionCard?.status, 'completed');
  assert.match(String(completionCard?.summary || ''), /Verified note\.txt|note checked/);
});

test('planner fallback finals trigger explicit branch finalization before synthesis', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');
    if (serialized.includes('child branch "Needs finalize"')) {
      return {
        kind: 'final',
        content: '抱歉，刚才内部规划结果格式异常，没有生成可展示的自然语言回答。请重试一次。',
        finalizationMode: 'planner_fallback',
      };
    }

    return {
      kind: 'branch',
      branches: [
        { label: 'Needs finalize', goal: 'Recover from invalid planner output', priority: 1 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    finalizeBranch: async ({ branch, reason }) => `finalized:${branch.id}:${reason}`,
  });

  const answer = await runtime.run('recover from fallback');
  assert.equal(answer, 'finalized:B0.1:planner format fallback after schema repair retries');
});

test('safe tool calls inside one branch execute concurrently and preserve observation order', async () => {
  let maxObservedConcurrency = 0;
  let currentConcurrency = 0;

  const toolRuntime = new FakeToolRuntime(async (toolName, args) => {
    currentConcurrency += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, currentConcurrency);
    const delay = Number(args.delayMs ?? 0);
    await new Promise((resolve) => setTimeout(resolve, delay));
    currentConcurrency -= 1;
    return `${toolName}:${String(args.label ?? '')}`;
  });

  const planner = new FakePlanner(async (transcript) => {
    const lastMessage = transcript[transcript.length - 1]?.content || '';
    if (lastMessage.includes('TOOL OBSERVATION for read:')) {
      const readIndex = lastMessage.indexOf('TOOL OBSERVATION for read:\n"read:first"');
      const searchIndex = lastMessage.indexOf('TOOL OBSERVATION for search:\n"search:second"');
      assert.ok(readIndex >= 0, 'missing read observation');
      assert.ok(searchIndex > readIndex, 'observations should remain in planner order');
      return {
        kind: 'final',
        content: 'safe parallel complete',
      };
    }

    return {
      kind: 'tool',
      toolCalls: [
        { name: 'read', arguments: { label: 'first', delayMs: 80 } },
        { name: 'search', arguments: { label: 'second', delayMs: 20 } },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime,
    maxSteps: 2,
  });

  const answer = await runtime.run('run safe tools');
  assert.equal(answer, 'safe parallel complete');
  assert.equal(maxObservedConcurrency, 2);
});

test('unsafe tool batches stay serial within one branch', async () => {
  let maxObservedConcurrency = 0;
  let currentConcurrency = 0;

  const toolRuntime = new FakeToolRuntime(async (_toolName, args) => {
    currentConcurrency += 1;
    maxObservedConcurrency = Math.max(maxObservedConcurrency, currentConcurrency);
    const delay = Number(args.delayMs ?? 0);
    await new Promise((resolve) => setTimeout(resolve, delay));
    currentConcurrency -= 1;
    return { ok: true };
  });

  const planner = new FakePlanner(async (transcript) => {
    const lastMessage = transcript[transcript.length - 1]?.content || '';
    if (lastMessage.includes('TOOL OBSERVATION for edit:')) {
      return {
        kind: 'final',
        content: 'serial complete',
      };
    }

    return {
      kind: 'tool',
      toolCalls: [
        { name: 'read', arguments: { delayMs: 30 } },
        { name: 'edit', arguments: { delayMs: 30 } },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime,
    maxSteps: 2,
  });

  const answer = await runtime.run('run mixed tools');
  assert.equal(answer, 'serial complete');
  assert.equal(maxObservedConcurrency, 1);
});

// ── Post-synthesis re-branch tests ────────────────────────────────────────

test('synthesizeFinal returning done=false triggers remediation re-branch', async () => {
  let branchCallCount = 0;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((m) => m.content).join('\n');

    // Remediation child branch: succeed immediately.
    if (serialized.includes('child branch "fix:')) {
      return {
        kind: 'final' as const,
        content: 'fixed-answer',
        outcome: 'success',
      };
    }

    // Original children: one succeeds, one fails.
    if (serialized.includes('child branch "CodeA"')) {
      return {
        kind: 'final' as const,
        content: 'codeA-ok',
        outcome: 'success',
      };
    }
    if (serialized.includes('child branch "CodeB"')) {
      return {
        kind: 'final' as const,
        content: 'codeB-fail',
        outcome: 'failed',
        outcomeReason: 'test assertion error line 42',
      };
    }

    // Root: branch into two.
    branchCallCount++;
    return {
      kind: 'branch' as const,
      thought: 'Split into A and B.',
      branches: [
        { label: 'CodeA', goal: 'Write snippet A', priority: 1 },
        { label: 'CodeB', goal: 'Write snippet B', priority: 1 },
      ],
    };
  });

  let synthesisCallCount = 0;

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 3,
    maxBranchDepth: 2,
    maxBranchWidth: 3,
    maxCompletedBranches: 8,
    maxSynthesisRetries: 2,
    synthesizeFinal: async ({ branches, synthesisRetry, maxSynthesisRetries }) => {
      synthesisCallCount++;

      // After remediation, look for fix branches that supersede older failed ones.
      const fixBranches = branches.filter((b) => b.label.startsWith('fix:') && b.outcome === 'success');
      const unresolvedFailed = branches.filter(
        (b) => (b.outcome === 'failed' || b.outcome === 'partial')
          && !fixBranches.some((f) => f.label === `fix: ${b.label}`),
      );

      if (unresolvedFailed.length > 0 && synthesisRetry < maxSynthesisRetries) {
        return {
          done: false,
          branches: unresolvedFailed.map((b) => ({
            label: `fix: ${b.label}`,
            goal: `Retry: ${b.outcomeReason}`,
            priority: 0.9,
          })),
          context: branches.filter((b) => b.outcome === 'success').map((b) => `${b.label}: ${b.finalAnswer}`).join('\n'),
        } satisfies SynthesisResult;
      }

      // All resolved (or retries exhausted) → done.
      const successAnswers = branches
        .filter((b) => b.outcome === 'success')
        .map((b) => b.finalAnswer);
      return {
        done: true,
        answer: successAnswers.join(' + '),
      } satisfies SynthesisResult;
    },
  });

  const answer = await runtime.run('test re-branch');
  assert.equal(branchCallCount, 1, 'root should branch exactly once');
  // Synthesis: call 1 sees CodeB failed → re-branch; call 2 sees fix succeeded → done.
  assert.equal(synthesisCallCount, 2, 'synthesis should be called twice (initial + after remediation)');
  assert.ok(answer.includes('fixed-answer'), 'final answer should include the remediation result');
  assert.ok(answer.includes('codeA-ok'), 'final answer should include the successful branch result');
});

test('structured remediation handoff is injected into retry child transcripts', async () => {
  let sawStructuredHandoff = false;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');

    if (serialized.includes('child branch "fix: Needs fix"')) {
      assert.match(serialized, /Structured remediation handoff:/);
      assert.match(serialized, /canonical_task: Needs fix/);
      assert.match(serialized, /must_not_repeat:/);
      assert.match(serialized, /successful_attempts:/);
      sawStructuredHandoff = true;
      return {
        kind: 'final' as const,
        content: 'retry-fixed',
        outcome: 'success',
        disposition: 'resolved',
      };
    }

    if (serialized.includes('child branch "Stable"')) {
      return {
        kind: 'final' as const,
        content: 'stable-ok',
        outcome: 'success',
        disposition: 'resolved',
      };
    }

    if (serialized.includes('child branch "Needs fix"')) {
      return {
        kind: 'final' as const,
        content: 'needs-fix-partial',
        outcome: 'partial',
        outcomeReason: 'missing the exact API field',
        disposition: 'missing_evidence',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'split root',
      branches: [
        { label: 'Stable', goal: 'Return a stable answer', priority: 1 },
        { label: 'Needs fix', goal: 'Create a retryable evidence gap', priority: 0.9 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 3,
    maxBranchDepth: 2,
    maxBranchWidth: 3,
    maxCompletedBranches: 8,
    maxSynthesisRetries: 2,
    synthesizeFinal: async ({ branches, synthesisRetry, maxSynthesisRetries }) => {
      const fixSucceeded = branches.some((branch) => branch.label === 'fix: Needs fix' && branch.outcome === 'success');
      const unresolved = branches.filter((branch) =>
        (branch.outcome === 'partial' || branch.outcome === 'failed')
        && !fixSucceeded,
      );
      if (unresolved.length > 0 && synthesisRetry < maxSynthesisRetries) {
        return {
          done: false,
          branches: [{
            label: 'fix: Needs fix',
            goal: 'Continue from the previous partial result',
            priority: 0.9,
            handoff: {
              canonicalTaskId: 'Needs fix',
              retryOfBranchId: unresolved[0].id,
              disposition: 'missing_evidence',
              missingFields: ['exact API field'],
              mustNotRepeat: ['Do not rerun the same generic search'],
            },
          }],
          context: 'Stable succeeded, Needs fix is missing one exact API field.',
          handoff: {
            retryIndex: synthesisRetry + 1,
            successfulAttempts: [{
              branchId: 'B0.1',
              label: 'Stable',
              canonicalTaskId: 'Stable',
              summary: 'stable-ok',
            }],
            unresolvedAttempts: [{
              canonicalTaskId: 'Needs fix',
              retryOfBranchId: unresolved[0].id,
              disposition: 'missing_evidence',
            }],
          },
        } satisfies SynthesisResult;
      }
      return {
        done: true,
        answer: branches
          .filter((branch) => branch.outcome === 'success')
          .map((branch) => branch.finalAnswer)
          .join(' + '),
      } satisfies SynthesisResult;
    },
  });

  const answer = await runtime.run('test structured remediation handoff');
  assert.equal(answer, 'stable-ok + retry-fixed');
  assert.equal(sawStructuredHandoff, true);
});

test('re-branch respects maxSynthesisRetries and forces done on limit', async () => {
  let synthesisCallCount = 0;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((m) => m.content).join('\n');

    if (serialized.includes('child branch "')) {
      return {
        kind: 'final' as const,
        content: 'always-fails',
        outcome: 'failed',
        outcomeReason: 'persistent error',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'branch',
      branches: [
        { label: 'Task', goal: 'Do something', priority: 1 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 3,
    maxBranchDepth: 2,
    maxBranchWidth: 3,
    maxCompletedBranches: 8,
    maxSynthesisRetries: 2,
    synthesizeFinal: async ({ branches, synthesisRetry, maxSynthesisRetries }) => {
      synthesisCallCount++;
      const failed = branches.filter((b) => b.outcome === 'failed');
      if (failed.length > 0 && synthesisRetry < maxSynthesisRetries) {
        return {
          done: false,
          branches: [{ label: `fix: retry-${synthesisRetry}`, goal: 'Retry', priority: 0.9 }],
          context: 'previous attempts failed',
        };
      }
      // Exhausted retries → force done.
      return { done: true, answer: 'gave-up' };
    },
  });

  const answer = await runtime.run('test retry limit');
  // 1 initial + 2 retries = 3 synthesis calls
  assert.equal(synthesisCallCount, 3, 'synthesis called 3 times (initial + 2 retries)');
  assert.equal(answer, 'gave-up');
});

test('branch execution errors are recorded and still reach synthesis', async () => {
  let sawFailedBranch = false;

  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');

    if (serialized.includes('child branch "Pass"')) {
      return {
        kind: 'final' as const,
        content: 'pass-ok',
        outcome: 'success',
      };
    }

    if (serialized.includes('child branch "Boom"')) {
      throw new Error('simulated planner crash');
    }

    return {
      kind: 'branch' as const,
      thought: 'split into success and failure',
      branches: [
        { label: 'Pass', goal: 'Return a success answer', priority: 1 },
        { label: 'Boom', goal: 'Crash while planning', priority: 0.9 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    maxBranchWidth: 2,
    maxCompletedBranches: 4,
    synthesizeFinal: async ({ branches }) => {
      sawFailedBranch = branches.some((branch) =>
        branch.label === 'Boom'
        && branch.outcome === 'failed'
        && /simulated planner crash/i.test(branch.outcomeReason || ''),
      );
      return 'synthesized-after-error';
    },
  });

  const answer = await runtime.run('handle branch errors');
  assert.equal(answer, 'synthesized-after-error');
  assert.equal(sawFailedBranch, true);
});

test('finalizeBranch failures fall back to the runtime placeholder answer', async () => {
  const planner = new FakePlanner(async () => ({
    kind: 'tool' as const,
    toolCalls: [{ name: 'read', arguments: { path: 'note.txt' } }],
  }));

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async () => 'still-working'),
    maxSteps: 1,
    maxTotalSteps: 2,
    finalizeBranch: async () => {
      throw new Error('finalize network failed');
    },
  });

  const answer = await runtime.run('force finalization failure');
  assert.equal(answer, 'Branch B0 stopped because step budget reached.');
});

test('synthesis hook failures fall back to the default runtime synthesis answer', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');

    if (serialized.includes('child branch "Only"')) {
      return {
        kind: 'final' as const,
        content: 'single-branch-answer',
        outcome: 'success',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'one child',
      branches: [{ label: 'Only', goal: 'finish once', priority: 1 }],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    synthesizeFinal: async () => {
      throw new Error('synthesis provider timeout');
    },
  });

  const answer = await runtime.run('fallback after synthesis error');
  assert.equal(answer, 'single-branch-answer');
});

test('empty remediation requests do not loop forever and fall back to synthesis answer', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((message) => message.content).join('\n');

    if (serialized.includes('child branch "Needs fix"')) {
      return {
        kind: 'final' as const,
        content: 'branch-failed',
        outcome: 'failed',
        outcomeReason: 'still broken',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'create one failing branch',
      branches: [{ label: 'Needs fix', goal: 'fail once', priority: 1 }],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    synthesizeFinal: async () => ({
      done: false as const,
      branches: [],
      context: 'nothing to retry',
    }),
  });

  const answer = await runtime.run('avoid empty remediation loop');
  assert.equal(answer, 'branch-failed');
});

test('synthesizeFinal returning plain string still works (backward compat)', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const serialized = transcript.map((m) => m.content).join('\n');

    if (serialized.includes('child branch "')) {
      return {
        kind: 'final' as const,
        content: 'child-answer',
      };
    }

    return {
      kind: 'branch' as const,
      thought: 'one child',
      branches: [
        { label: 'Only', goal: 'Do it', priority: 1 },
      ],
    };
  });

  const runtime = new AgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    maxSteps: 2,
    maxBranchDepth: 1,
    // Return plain string (old API).
    synthesizeFinal: async ({ branches }) => branches[0].finalAnswer,
  });

  const answer = await runtime.run('test compat');
  assert.equal(answer, 'child-answer');
});
