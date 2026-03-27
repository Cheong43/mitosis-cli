import test from 'node:test';
import assert from 'node:assert/strict';
import { AgentRuntime } from './AgentRuntime.js';
import type { AgentStep, TranscriptMessage } from './types.js';

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
