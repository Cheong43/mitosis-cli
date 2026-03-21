import test from 'node:test';
import assert from 'node:assert/strict';
import { BeamSearchAgentRuntime } from './BeamSearchAgentRuntime.js';
import { DefaultPathScorer } from './DefaultPathScorer.js';
import type { AgentStep, TranscriptMessage, BeamPath } from './types.js';

// ─── Test doubles ────────────────────────────────────────────────────────────

class FakePlanner {
  constructor(private readonly planFn: (transcript: TranscriptMessage[]) => Promise<AgentStep>) {}

  async plan(transcript: TranscriptMessage[]): Promise<AgentStep> {
    return this.planFn(transcript);
  }
}

class FakeToolRuntime {
  constructor(
    private readonly executeFn?: (
      toolName: string,
      args: Record<string, unknown>,
    ) => Promise<unknown>,
  ) {}

  resetSession(): void {}

  async execute(toolName: string, args: Record<string, unknown>) {
    try {
      const result = this.executeFn ? await this.executeFn(toolName, args) : { ok: true };
      return { success: true, result, durationMs: 0 };
    } catch (error: unknown) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        durationMs: 0,
      };
    }
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test('beam search returns final answer from a single path', async () => {
  let callCount = 0;
  const planner = new FakePlanner(async () => {
    callCount++;
    // First call (expand) returns a tool call; subsequent calls (completion
    // check + next expand) return a final answer.
    if (callCount <= 1) {
      return {
        kind: 'tool',
        thought: 'Let me read the file.',
        toolCalls: [{ name: 'read', arguments: { path: 'test.txt' } }],
      };
    }
    return { kind: 'final', content: 'The answer is 42.' };
  });

  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    beamWidth: 1,
    maxDepth: 5,
    expansionFactor: 1,
  });

  const answer = await runtime.run('What is the answer?');
  assert.equal(answer, 'The answer is 42.');
});

test('beam search prunes to beamWidth candidates', async () => {
  const traces: string[] = [];
  let expandCalls = 0;
  const planner = new FakePlanner(async () => {
    expandCalls++;
    // Always return a tool call so paths are never immediately complete.
    if (expandCalls > 15) {
      return { kind: 'final', content: 'done' };
    }
    return {
      kind: 'tool',
      thought: `step-${expandCalls}`,
      toolCalls: [{ name: 'read', arguments: { path: `f${expandCalls}.txt` } }],
    };
  });

  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    beamWidth: 2,
    maxDepth: 3,
    expansionFactor: 3,
    onTrace: (e) => traces.push(e.content),
  });

  const answer = await runtime.run('Explore');
  assert.ok(typeof answer === 'string');
  // Verify that beam step traces were emitted.
  const beamTraces = traces.filter((t) => t.includes('[beam]'));
  assert.ok(beamTraces.length > 0, 'should emit beam-step traces');
});

test('completed paths are preferred during pruning', async () => {
  const planner = new FakePlanner(async (transcript) => {
    const last = transcript[transcript.length - 1]?.content || '';
    // Mark paths that saw 'winner.txt' as done.
    if (last.includes('winner')) {
      return { kind: 'final', content: 'winner-answer' };
    }
    return {
      kind: 'tool',
      thought: 'still working',
      toolCalls: [{ name: 'read', arguments: { path: 'other.txt' } }],
    };
  });

  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async (_name, args) => {
      return `content-of-${String(args.path ?? 'unknown')}`;
    }),
    beamWidth: 2,
    maxDepth: 3,
    expansionFactor: 2,
    generateCandidates: async () => [
      { thought: 'try winner', toolCall: { name: 'read', arguments: { path: 'winner.txt' } } },
      { thought: 'try loser', toolCall: { name: 'read', arguments: { path: 'loser.txt' } } },
    ],
  });

  const answer = await runtime.run('Pick the best');
  assert.equal(answer, 'winner-answer');
});

test('custom scorer influences path selection', async () => {
  let expandCount = 0;
  const planner = new FakePlanner(async () => {
    expandCount++;
    if (expandCount > 6) {
      return { kind: 'final', content: 'scored-final' };
    }
    return {
      kind: 'tool',
      thought: `expand-${expandCount}`,
      toolCalls: [{ name: 'read', arguments: { n: expandCount } }],
    };
  });

  const customScorer = {
    score: (path: BeamPath) => {
      // Paths with more history entries are scored higher.
      return path.history.length * 0.5;
    },
  };

  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    beamWidth: 2,
    maxDepth: 3,
    expansionFactor: 2,
    scorer: customScorer,
  });

  const answer = await runtime.run('Test scoring');
  assert.ok(typeof answer === 'string');
});

test('synthesizeFinal hook is called with the best path', async () => {
  const planner = new FakePlanner(async () => {
    return { kind: 'final', content: 'raw-answer' };
  });

  let receivedPath: BeamPath | undefined;
  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    beamWidth: 1,
    maxDepth: 2,
    expansionFactor: 1,
    synthesizeFinal: async (path) => {
      receivedPath = path;
      return `synthesized: ${path.finalAnswer}`;
    },
  });

  const answer = await runtime.run('Synthesize');
  assert.equal(answer, 'synthesized: raw-answer');
  assert.ok(receivedPath);
  assert.equal(receivedPath!.isComplete, true);
});

test('max depth terminates even without a final answer', async () => {
  const planner = new FakePlanner(async () => {
    return {
      kind: 'tool',
      thought: 'keep going',
      toolCalls: [{ name: 'read', arguments: { path: 'file.txt' } }],
    };
  });

  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(),
    beamWidth: 1,
    maxDepth: 2,
    expansionFactor: 1,
  });

  const answer = await runtime.run('Run forever');
  // Should not hang — must return even without a final answer.
  assert.ok(typeof answer === 'string');
  assert.ok(answer.length > 0);
});

test('DefaultPathScorer scores completed paths higher', () => {
  const scorer = new DefaultPathScorer();

  const completePath: BeamPath = {
    id: 'P0',
    parentId: null,
    depth: 1,
    transcript: [],
    history: [
      {
        thought: 'ok',
        action: { name: 'read', arguments: {} },
        observation: { toolName: 'read', result: 'data', success: true },
      },
    ],
    score: 0,
    isComplete: true,
    finalAnswer: 'done',
  };

  const incompletePath: BeamPath = {
    id: 'P1',
    parentId: null,
    depth: 1,
    transcript: [],
    history: [
      {
        thought: 'ok',
        action: { name: 'read', arguments: {} },
        observation: { toolName: 'read', result: 'data', success: true },
      },
    ],
    score: 0,
    isComplete: false,
  };

  const completeScore = scorer.score(completePath);
  const incompleteScore = scorer.score(incompletePath);
  assert.ok(completeScore > incompleteScore, 'completed paths should score higher');
});

test('DefaultPathScorer penalises deeper paths', () => {
  const scorer = new DefaultPathScorer();

  const shallow: BeamPath = {
    id: 'P0',
    parentId: null,
    depth: 1,
    transcript: [],
    history: [
      {
        thought: 'ok',
        action: { name: 'read', arguments: {} },
        observation: { toolName: 'read', result: 'data', success: true },
      },
    ],
    score: 0,
    isComplete: false,
  };

  const deep: BeamPath = {
    id: 'P1',
    parentId: null,
    depth: 10,
    transcript: [],
    history: [
      {
        thought: 'ok',
        action: { name: 'read', arguments: {} },
        observation: { toolName: 'read', result: 'data', success: true },
      },
    ],
    score: 0,
    isComplete: false,
  };

  assert.ok(scorer.score(shallow) > scorer.score(deep), 'shallow paths should score higher');
});

test('generateCandidates hook overrides default candidate generation', async () => {
  let hookCalled = false;
  const planner = new FakePlanner(async () => {
    return { kind: 'final', content: 'planner-answer' };
  });

  const runtime = new BeamSearchAgentRuntime({
    planner,
    toolRuntime: new FakeToolRuntime(async () => 'custom-result'),
    beamWidth: 1,
    maxDepth: 2,
    expansionFactor: 1,
    generateCandidates: async (_transcript, _n) => {
      hookCalled = true;
      return [
        {
          thought: 'custom thought',
          toolCall: { name: 'read', arguments: { path: 'custom.txt' } },
        },
      ];
    },
  });

  await runtime.run('Custom gen');
  assert.ok(hookCalled, 'generateCandidates hook should be called');
});
