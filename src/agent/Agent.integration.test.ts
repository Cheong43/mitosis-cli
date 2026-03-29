import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { Agent, type TraceEvent } from './index.js';

process.env.AUTO_QUEUE_MEMORY_SAVE = '0';

(Agent.prototype as any).decideExecutionDiscipline = async () => ({
  mode: 'branching',
  reason: 'test-default-branching',
});

function createTempProjectRoot(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  fs.mkdirSync(path.join(dir, '.mempedia', 'memory', 'index'), { recursive: true });
  return dir;
}

function installAgentTestDoubles(agent: Agent, answerForRequest: (request: string) => Promise<string> | string): void {
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    const requestMatch = joined.match(/Original user request:\n([\s\S]*?)(?:\n\nStart with the root loop\.|\n\nActive branch:|$)/);
    const request = requestMatch?.[1]?.trim() || 'unknown';
    const answer = await answerForRequest(request);
    return {
      kind: 'final',
      thought: 'Return final answer.',
      final_answer: answer,
      completion_summary: answer,
    };
  };
}

test('conversationId isolates follow-up grounding state', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-conv-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, (request) => `ok:${request.slice(0, 8)}`);

  const tracesA1: TraceEvent[] = [];
  await agent.run('first topic', (event) => { tracesA1.push(event); }, { conversationId: 'thread-A' });
  assert.ok(tracesA1.some((event) => event.content.includes('Selected 0 recent conversation turns')));

  const tracesB: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesB.push(event); }, { conversationId: 'thread-B' });
  assert.ok(tracesB.some((event) => event.content.includes('Selected 0 recent conversation turns')));

  const tracesA2: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesA2.push(event); }, { conversationId: 'thread-A' });
  assert.ok(tracesA2.some((event) => event.content.includes('Selected 1 relevant recent conversation turn(s)')));

  agent.stop();
});

test('one Agent instance supports concurrent runs across different conversation ids', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-concurrent-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, async (request) => {
    await new Promise((resolve) => setTimeout(resolve, 40));
    return request.includes('alpha') ? 'ok:alpha' : 'ok:beta';
  });

  const [alpha, beta] = await Promise.all([
    agent.run('alpha request', () => {}, { conversationId: 'thread-alpha', sessionId: 'session-alpha' }),
    agent.run('beta request', () => {}, { conversationId: 'thread-beta', sessionId: 'session-beta' }),
  ]);

  assert.equal(alpha, 'ok:alpha');
  assert.equal(beta, 'ok:beta');

  const tracesAlphaFollowUp: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesAlphaFollowUp.push(event); }, { conversationId: 'thread-alpha' });
  assert.ok(tracesAlphaFollowUp.some((event) => event.content.includes('Selected 1 relevant recent conversation turn(s)')));

  const tracesGamma: TraceEvent[] = [];
  await agent.run('继续', (event) => { tracesGamma.push(event); }, { conversationId: 'thread-gamma' });
  assert.ok(tracesGamma.some((event) => event.content.includes('Selected 0 recent conversation turns')));

  agent.stop();
});

test('live turns skip context retrieval before planning', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-skip-context-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  let retrievalCalls = 0;

  anyAgent.retrieveRelevantContext = async () => {
    retrievalCalls += 1;
    throw new Error('retrieveRelevantContext should not be called');
  };
  anyAgent.generatePlannerDecision = async () => ({
    kind: 'final',
    thought: 'Done.',
    final_answer: 'ok',
    completion_summary: 'ok',
  });

  const traces: TraceEvent[] = [];
  const answer = await agent.run('hi', (event) => { traces.push(event); }, { conversationId: 'thread-skip-context' });

  assert.equal(answer, 'ok');
  assert.equal(retrievalCalls, 0);
  assert.ok(traces.some((event) => event.content.includes('Context retrieval is disabled for live turns.')));

  agent.stop();
});

test('planner uses structured decision generation instead of raw JSON text repair', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-structured-');
  const agent = new Agent({ apiKey: 'test-key', model: 'Doubao-Seed-2.0-pro' }, projectRoot);
  const anyAgent = agent as any;
  let plannerCalls = 0;

  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });
  anyAgent.generatePlannerDecision = async () => {
    plannerCalls += 1;
    return {
      kind: 'final',
      thought: 'Done.',
      final_answer: 'ok',
      completion_summary: 'ok',
    };
  };
  anyAgent.generateJsonPromptText = async () => {
    throw new Error('planner should not call generateJsonPromptText');
  };

  const answer = await agent.run('测试结构化 planner', () => {}, { conversationId: 'thread-structured-planner' });
  assert.equal(answer, 'ok');
  assert.ok(plannerCalls >= 1);

  agent.stop();
});

test('unwraps raw planner JSON from final answer before returning to user', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-final-unpack-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, () => '{"kind":"final","final_answer":"这是正常回答"}');

  const answer = await agent.run('测试请求', () => {}, { conversationId: 'thread-json-wrap' });
  assert.equal(answer, '这是正常回答');

  agent.stop();
});

test('planner generation failure returns a user-safe fallback answer', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-fallback-natural-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });
  anyAgent.generatePlannerDecision = async () => {
    throw new Error('```bash\nfind . -name "mempedia"\nls -la ./target/release/\n```');
  };

  const answer = await agent.run('我已经重新build', () => {}, { conversationId: 'thread-fallback-natural' });
  assert.match(answer, /内部规划步骤失败/);
  assert.doesNotMatch(answer, /```bash/);
  assert.doesNotMatch(answer, /find \./);

  agent.stop();
});

test('synthesis rebranch only retries unresolved latest attempts and stops after fix succeeds', async () => {
  const previousRebranchEnabled = process.env.REACT_REBRANCH_ENABLED;
  process.env.REACT_REBRANCH_ENABLED = '1';
  try {
    const projectRoot = createTempProjectRoot('mempedia-agent-rebranch-latest-');
    const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
    const anyAgent = agent as any;

    anyAgent.retrieveRelevantContext = async () => ({
      contextText: '',
      recalledNodeIds: [],
      selectedNodeIds: [],
      rationale: 'test',
    });

    anyAgent.measure = async (_perfEntries: unknown, label: string, fn: () => Promise<unknown>) => {
      if (label === 'branch_synthesis') {
        return { text: 'merged-final-answer' };
      }
      return fn();
    };

    anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
      const joined = Array.isArray(messages)
        ? messages.map((message) => String(message?.content || '')).join('\n\n')
        : '';

      if (joined.includes('child branch "fix: Needs fix"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Repair succeeded.',
            final_answer: 'fixed-result',
            completion_summary: 'fixed-result',
            outcome: 'success',
          },
        };
      }

      if (joined.includes('child branch "Stable path"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Stable path succeeded.',
            final_answer: 'stable-result',
            completion_summary: 'stable-result',
            outcome: 'success',
          },
        };
      }

      if (joined.includes('child branch "Needs fix"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'First attempt failed.',
            final_answer: 'initial-failed-result',
            completion_summary: 'initial-failed-result',
            outcome: 'failed',
            outcome_reason: 'apple refurb page blocked',
          },
        };
      }

      return {
        decision: {
          kind: 'branch',
          thought: 'Split into one stable path and one path that needs remediation.',
          branches: [
            { label: 'Stable path', goal: 'Return a stable answer', priority: 1 },
            { label: 'Needs fix', goal: 'Attempt a path that will fail once', priority: 0.8 },
          ],
        },
      };
    };

    const traces: TraceEvent[] = [];
    const answer = await agent.run('test synthesis rebranch resolution', (event) => { traces.push(event); }, {
      conversationId: 'thread-rebranch-latest',
    });

    assert.equal(answer, 'merged-final-answer');
    assert.equal(
      traces.filter((event) => event.content.includes('Requesting remediation re-branch')).length,
      1,
    );
    assert.ok(
      traces.some((event) => event.content.includes('Synthesizing 2 frontier completed branch attempt(s)')),
    );

    agent.stop();
  } finally {
    if (previousRebranchEnabled === undefined) {
      delete process.env.REACT_REBRANCH_ENABLED;
    } else {
      process.env.REACT_REBRANCH_ENABLED = previousRebranchEnabled;
    }
  }
});

test('synthesis retries only the latest leaf task when a remediation branch fans out into narrower children', async () => {
  const previousRebranchEnabled = process.env.REACT_REBRANCH_ENABLED;
  process.env.REACT_REBRANCH_ENABLED = '1';
  try {
    const projectRoot = createTempProjectRoot('mempedia-agent-rebranch-leaf-');
    const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
    const anyAgent = agent as any;

    anyAgent.retrieveRelevantContext = async () => ({
      contextText: '',
      recalledNodeIds: [],
      selectedNodeIds: [],
      rationale: 'test',
    });

    anyAgent.measure = async (_perfEntries: unknown, label: string, fn: () => Promise<unknown>) => {
      if (label === 'branch_synthesis') {
        return { text: 'leaf-only-final-answer' };
      }
      return fn();
    };

    anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
      const joined = Array.isArray(messages)
        ? messages.map((message) => String(message?.content || '')).join('\n\n')
        : '';

      if (joined.includes('child branch "fix: Narrow child"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Final leaf repair succeeded.',
            final_answer: 'narrow-child-fixed',
            completion_summary: 'narrow-child-fixed',
            outcome: 'success',
            disposition: 'resolved',
          },
        };
      }

      if (joined.includes('child branch "Narrow child"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Still missing one narrow field.',
            final_answer: 'narrow-child-partial',
            completion_summary: 'Still missing exact futures point',
            outcome: 'partial',
            outcome_reason: 'missing exact futures point',
            disposition: 'missing_evidence',
          },
        };
      }

      if (joined.includes('child branch "Covered child"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Covered child succeeded.',
            final_answer: 'covered-child-success',
            completion_summary: 'covered-child-success',
            outcome: 'success',
            disposition: 'resolved',
          },
        };
      }

      if (joined.includes('child branch "fix: Needs broad fix"')) {
        return {
          decision: {
            kind: 'branch',
            thought: 'Split the retry into narrower evidence streams.',
            branches: [
              { label: 'Narrow child', goal: 'Recover the exact futures point', priority: 1 },
              { label: 'Covered child', goal: 'Preserve the already recovered cash index context', priority: 0.8 },
            ],
          },
        };
      }

      if (joined.includes('child branch "Stable path"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Stable path succeeded.',
            final_answer: 'stable-success',
            completion_summary: 'stable-success',
            outcome: 'success',
            disposition: 'resolved',
          },
        };
      }

      if (joined.includes('child branch "Needs broad fix"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'First attempt has a broad gap.',
            final_answer: 'broad-partial',
            completion_summary: 'Have market color, missing exact futures point',
            outcome: 'partial',
            outcome_reason: 'missing exact futures point',
            disposition: 'missing_evidence',
          },
        };
      }

      return {
        decision: {
          kind: 'branch',
          thought: 'Split root into a stable path and a path that needs remediation.',
          branches: [
            { label: 'Stable path', goal: 'Return a stable answer', priority: 1 },
            { label: 'Needs broad fix', goal: 'Attempt a path that needs a focused retry', priority: 0.8 },
          ],
        },
      };
    };

    const traces: TraceEvent[] = [];
    const answer = await agent.run('test leaf-only remediation retry', (event) => { traces.push(event); }, {
      conversationId: 'thread-rebranch-leaf-only',
    });

    assert.equal(answer, 'leaf-only-final-answer');
    assert.equal(
      traces.filter((event) => event.content.includes('Synthesis requested 1 remediation branch(es)')).length,
      2,
    );
    assert.ok(
      traces.some((event) =>
        event.metadata?.branchId === 'R2.1'
        && event.metadata?.branchLabel === 'fix: Narrow child',
      ),
    );
    assert.ok(
      !traces.some((event) =>
        event.metadata?.branchId === 'R2.1'
        && event.metadata?.branchLabel === 'fix: Needs broad fix',
      ),
    );

    agent.stop();
  } finally {
    if (previousRebranchEnabled === undefined) {
      delete process.env.REACT_REBRANCH_ENABLED;
    } else {
      process.env.REACT_REBRANCH_ENABLED = previousRebranchEnabled;
    }
  }
});

test('synthesis keeps replanning unmet work from current progress across retries', async () => {
  const previousRebranchEnabled = process.env.REACT_REBRANCH_ENABLED;
  process.env.REACT_REBRANCH_ENABLED = '1';
  try {
    const projectRoot = createTempProjectRoot('mempedia-agent-rebranch-first-principles-');
    const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
    const anyAgent = agent as any;

    anyAgent.retrieveRelevantContext = async () => ({
      contextText: '',
      recalledNodeIds: [],
      selectedNodeIds: [],
      rationale: 'test',
    });

    anyAgent.measure = async (_perfEntries: unknown, label: string, fn: () => Promise<unknown>) => {
      if (label === 'branch_synthesis') {
        return { text: 'first-principles-final-answer' };
      }
      return fn();
    };

    anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
      const joined = Array.isArray(messages)
        ? messages.map((message) => String(message?.content || '')).join('\n\n')
        : '';

      if (joined.includes('branch_id: R2.1') && joined.includes('child branch "fix: Needs access workaround"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Second remediation closed the gap.',
            final_answer: 'access-gap-fixed',
            completion_summary: 'access-gap-fixed',
            outcome: 'success',
            disposition: 'resolved',
          },
        };
      }

      if (joined.includes('branch_id: R1.1') && joined.includes('child branch "fix: Needs access workaround"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Still blocked but with stronger intermediate progress.',
            final_answer: 'mirror-found-but-blocked',
            completion_summary: 'Recovered mirror candidates, still blocked by login wall',
            outcome: 'partial',
            outcome_reason: 'blocked by login wall',
            disposition: 'blocked_external',
          },
        };
      }

      if (joined.includes('child branch "Stable path"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'Stable path succeeded.',
            final_answer: 'stable-success',
            completion_summary: 'stable-success',
            outcome: 'success',
            disposition: 'resolved',
          },
        };
      }

      if (joined.includes('child branch "Needs access workaround"')) {
        return {
          decision: {
            kind: 'final',
            thought: 'First attempt hit an external blocker.',
            final_answer: 'initial-blocked-result',
            completion_summary: 'Recovered API host, still blocked by login wall',
            outcome: 'partial',
            outcome_reason: 'blocked by login wall',
            disposition: 'blocked_external',
          },
        };
      }

      return {
        decision: {
          kind: 'branch',
          thought: 'Split root into a stable path and a blocked path that needs incremental completion.',
          branches: [
            { label: 'Stable path', goal: 'Return a stable answer', priority: 1 },
            { label: 'Needs access workaround', goal: 'Recover the blocked evidence path', priority: 0.8 },
          ],
        },
      };
    };

    const traces: TraceEvent[] = [];
    const answer = await agent.run('test first-principles remediation retry', (event) => { traces.push(event); }, {
      conversationId: 'thread-rebranch-first-principles',
    });

    assert.equal(answer, 'first-principles-final-answer');
    assert.equal(
      traces.filter((event) => event.content.includes('Requesting remediation re-branch')).length,
      2,
    );
    assert.ok(
      traces.some((event) => event.content.includes('Goal assessment before synthesis')),
    );
    assert.ok(
      traces.some((event) =>
        event.metadata?.branchId === 'R2.1'
        && event.metadata?.branchLabel === 'fix: Needs access workaround',
      ),
    );

    agent.stop();
  } finally {
    if (previousRebranchEnabled === undefined) {
      delete process.env.REACT_REBRANCH_ENABLED;
    } else {
      process.env.REACT_REBRANCH_ENABLED = previousRebranchEnabled;
    }
  }
});

test('synthesis skips remediation rebranch when REACT_REBRANCH_ENABLED is disabled', async () => {
  const previousRebranchEnabled = process.env.REACT_REBRANCH_ENABLED;
  process.env.REACT_REBRANCH_ENABLED = '0';

  const projectRoot = createTempProjectRoot('mempedia-agent-rebranch-disabled-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;

  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  anyAgent.measure = async (_perfEntries: unknown, label: string, fn: () => Promise<unknown>) => {
    if (label === 'branch_synthesis') {
      return { text: 'no-rebranch-final-answer' };
    }
    return fn();
  };

  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';

    if (joined.includes('child branch "Stable path"')) {
      return {
        decision: {
          kind: 'final',
          thought: 'Stable path succeeded.',
          final_answer: 'stable-success',
          completion_summary: 'stable-success',
          outcome: 'success',
          disposition: 'resolved',
        },
      };
    }

    if (joined.includes('child branch "Needs fix"')) {
      return {
        decision: {
          kind: 'final',
          thought: 'First attempt is still unresolved.',
          final_answer: 'still-missing',
          completion_summary: 'Missing exact futures point',
          outcome: 'partial',
          outcome_reason: 'missing exact futures point',
          disposition: 'missing_evidence',
        },
      };
    }

    return {
      decision: {
        kind: 'branch',
        thought: 'Split into one stable path and one path that would normally need remediation.',
        branches: [
          { label: 'Stable path', goal: 'Return a stable answer', priority: 1 },
          { label: 'Needs fix', goal: 'Attempt a path that remains partial', priority: 0.8 },
        ],
      },
    };
  };

  try {
    const traces: TraceEvent[] = [];
    const answer = await agent.run('test rebranch disabled by env', (event) => { traces.push(event); }, {
      conversationId: 'thread-rebranch-disabled',
    });

    assert.equal(answer, 'no-rebranch-final-answer');
    assert.equal(
      traces.filter((event) => event.content.includes('Requesting remediation re-branch')).length,
      0,
    );
    assert.ok(
      traces.some((event) => event.content.includes('Rebranch disabled by REACT_REBRANCH_ENABLED')),
    );
    assert.ok(
      traces.some((event) => event.content.includes('Synthesizing 2 frontier completed branch attempt(s)')),
    );
  } finally {
    agent.stop();
    if (previousRebranchEnabled === undefined) {
      delete process.env.REACT_REBRANCH_ENABLED;
    } else {
      process.env.REACT_REBRANCH_ENABLED = previousRebranchEnabled;
    }
  }
});

test('planner timeout enters planner fallback finalization instead of returning the generic retry message', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-timeout-finalize-');
  fs.writeFileSync(path.join(projectRoot, 'evidence.txt'), 'Observed brand facts already gathered.');

  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  let plannerCalls = 0;

  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  anyAgent.measure = async (_perfEntries: unknown, label: string, fn: () => Promise<unknown>) => {
    if (label === 'finalize_B0') {
      return { text: 'finalized-after-timeout' };
    }
    return fn();
  };

  anyAgent.generatePlannerDecision = async () => {
    plannerCalls += 1;
    if (plannerCalls === 1) {
      return {
        decision: {
          kind: 'tool',
          thought: 'Read local evidence first.',
          tool_calls: [
            {
              name: 'read',
              arguments: { path: path.join(projectRoot, 'evidence.txt'), target: 'workspace' },
              goal: 'Inspect gathered evidence',
            },
          ],
        },
      };
    }
    throw new Error('planBranch_B0 llm timeout after 120000ms');
  };

  const traces: TraceEvent[] = [];
  const answer = await agent.run('use gathered evidence and then timeout', (event) => { traces.push(event); }, {
    conversationId: 'thread-timeout-finalize',
  });

  assert.equal(answer, 'finalized-after-timeout');
  assert.ok(
    traces.some((event) => event.content.includes('Planner timed out while choosing the next step')),
  );
  assert.ok(
    traces.some((event) => event.content.includes('Forcing finalization for B0')),
  );
  assert.ok(
    traces.some((event) => event.content.includes('Branch finalized after planner fallback.')),
  );
  assert.ok(
    !traces.some((event) => event.content.includes('抱歉，内部规划步骤失败了，请重试一次。')),
  );

  agent.stop();
});

test('planner prompt keeps branching guidance principle-based without root coercion', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-branch-guidance-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  let capturedPrompt = '';
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    capturedPrompt = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    return {
      kind: 'final',
      thought: 'Done.',
      final_answer: 'ok',
      completion_summary: 'ok',
    };
  };

  const answer = await agent.run('竭尽所能收集中英今天的盘前信源', () => {}, { conversationId: 'thread-branch-guidance' });
  assert.equal(answer, 'ok');
  assert.match(capturedPrompt, /Optimize for correct end-to-end completion of the user's deliverable/i);
  assert.match(capturedPrompt, /classify candidate workstreams as one of four relationships/i);
  assert.match(capturedPrompt, /If work is inseparable, or if dependency is unclear, prefer fewer branches/i);
  assert.match(capturedPrompt, /Branch only when genuine independent progress is available/i);
  assert.match(capturedPrompt, /For integrated artifacts such as websites, reports, refactors, or single deliverables/i);
  assert.match(capturedPrompt, /Raise the priority of execution-structure planning over immediate tool use/i);
  assert.match(capturedPrompt, /Treat branching as parallel task planning/i);
  assert.match(capturedPrompt, /call read\/search\/edit\/bash\/web directly for work/i);
  assert.match(capturedPrompt, /Prefer materially diverse search strategies/i);
  assert.match(capturedPrompt, /Branches may share the same execution_group only when each branch can make useful progress without waiting for sibling outputs/i);
  assert.match(capturedPrompt, /Re-branch only when the current branch still contains multiple independent workstreams with low coordination cost/i);
  assert.match(capturedPrompt, /each must target a genuinely different evidence stream or hypothesis/i);
  assert.match(capturedPrompt, /After 2-3 consecutive web\/search rounds without clear new information/i);
  assert.match(capturedPrompt, /Keep every `goal` concise and comfortably under the 240-character schema limit/i);
  assert.match(capturedPrompt, /Planning view only/i);
  assert.doesNotMatch(capturedPrompt, /Shared context for this request:/i);
  assert.doesNotMatch(capturedPrompt, /Always provide at least 2 branches/i);
  assert.doesNotMatch(capturedPrompt, /NEVER branch on the first step/i);
  assert.doesNotMatch(capturedPrompt, /default to one `web` tool call before branching/i);
  assert.doesNotMatch(capturedPrompt, /Return exactly one JSON object/i);
  assert.doesNotMatch(capturedPrompt, /planner_tool/i);

  agent.stop();
});

test('sequential execution-discipline decisions use strict plan-and-execute instead of direct branching writes', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-mutation-react-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.decideExecutionDiscipline = async () => ({
    mode: 'sequential',
    reason: 'Shared workspace mutations should stay on one executor.',
  });
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  const capturedPrompts: string[] = [];
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    capturedPrompts.push(joined);
    if (joined.includes('PLAN STAGE ONLY.')) {
      return {
        kind: 'final',
        thought: 'Plan is ready.',
        final_answer: '1. Inspect the current UI shell.\n2. Add the TarotCard3D component.\n3. Wire it into the app sequentially.\n4. Validate the changed files.',
        completion_summary: 'execution plan ready',
      };
    }
    return {
      kind: 'final',
      thought: 'Execution is complete.',
      final_answer: 'ok',
      completion_summary: 'ok',
    };
  };

  const traces: TraceEvent[] = [];
  const answer = await agent.run('让 mitosis cli 生成一个 3d 塔罗牌游戏', (event) => { traces.push(event); }, {
    conversationId: 'thread-mutation-react',
    agentMode: 'branching',
  });

  assert.equal(answer, 'ok');
  assert.ok(traces.some((event) => event.content.includes('Using strict plan-and-execute mode.')));
  assert.ok(traces.some((event) => event.content.includes('Planning stage completed. Switching to the sequential execute stage.')));
  const planningPrompt = capturedPrompts.find((prompt) => prompt.includes('PLAN STAGE ONLY.'));
  const executionPrompt = capturedPrompts.find((prompt) => prompt.includes('APPROVED EXECUTION PLAN:'));
  assert.ok(planningPrompt);
  assert.ok(executionPrompt);
  assert.match(planningPrompt!, /You are a branching ReAct agent/i);
  assert.match(planningPrompt!, /PLAN stage/i);
  assert.match(planningPrompt!, /Raise the priority of execution-structure planning over immediate tool use/i);
  assert.doesNotMatch(planningPrompt!, /Planning is strictly read-only/i);
  assert.match(executionPrompt!, /You are a classic ReAct agent/i);
  assert.match(executionPrompt!, /EXECUTE stage/i);
  assert.match(executionPrompt!, /APPROVED EXECUTION PLAN:/i);

  agent.stop();
});

test('branching execution-discipline decisions keep the root request in normal branching mode', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-design-react-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.decideExecutionDiscipline = async () => ({
    mode: 'branching',
    reason: 'Independent workstreams are safe to branch.',
  });
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  const capturedPrompts: string[] = [];
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    capturedPrompts.push(joined);
    return {
      kind: 'final',
      thought: 'Execution is complete.',
      final_answer: 'ok',
      completion_summary: 'ok',
    };
  };

  const traces: TraceEvent[] = [];
  const answer = await agent.run('请评估这次任务是否适合拆成独立调研分枝', (event) => { traces.push(event); }, {
    conversationId: 'thread-design-temp-react',
    agentMode: 'branching',
  });

  assert.equal(answer, 'ok');
  assert.ok(!traces.some((event) => event.content.includes('Using strict plan-and-execute mode.')));
  assert.ok(!capturedPrompts.some((prompt) => prompt.includes('PLAN STAGE ONLY.')));

  agent.stop();
});

test('execution-discipline gate can switch a follow-up turn into sequential mode', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-followup-react-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  let gateCalls = 0;
  anyAgent.decideExecutionDiscipline = async () => {
    gateCalls += 1;
    return gateCalls === 1
      ? { mode: 'branching', reason: 'Initial turn can branch.' }
      : { mode: 'sequential', reason: 'Follow-up now needs one integrating executor.' };
  };
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    if (joined.includes('PLAN STAGE ONLY.')) {
      return {
        kind: 'final',
        thought: 'Plan is ready.',
        final_answer: '1. 顺序执行计划。\n2. 写入文件。\n3. 验证结果。',
        completion_summary: 'execution plan ready',
      };
    }
    return {
      kind: 'final',
      thought: 'Execution is complete.',
      final_answer: 'ok',
      completion_summary: 'ok',
    };
  };

  await agent.run('先做多路调研，把证据分开收集', () => {}, {
    conversationId: 'thread-followup-sequential',
    agentMode: 'branching',
  });

  const followUpTraces: TraceEvent[] = [];
  const answer = await agent.run('继续，把引用和生平补进去', (event) => { followUpTraces.push(event); }, {
    conversationId: 'thread-followup-sequential',
    agentMode: 'branching',
  });

  assert.equal(answer, 'ok');
  assert.ok(followUpTraces.some((event) => event.content.includes('Using strict plan-and-execute mode.')));

  agent.stop();
});

test('plan stage allows mutating tool calls when the planner chooses them', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-plan-readonly-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  let plannerCalls = 0;
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    plannerCalls += 1;
    if (plannerCalls === 1) {
      return {
        kind: 'tool',
        thought: 'Write the draft immediately.',
        tool_calls: [
          {
            name: 'edit',
            arguments: { target: 'workspace', path: 'draft.txt', content: 'oops' },
            goal: 'Write the draft directly',
          },
        ],
      };
    }

    return {
      kind: 'final',
      thought: 'Return the plan after the edit.',
      final_answer: '已完成计划草稿并继续顺序执行。',
      completion_summary: 'plan ready',
    };
  };

  const traces: TraceEvent[] = [];
  const answer = await agent.run('Original user request:\n生成一个 3d 塔罗牌游戏\n\nPLAN STAGE ONLY.', (event) => { traces.push(event); }, {
    conversationId: 'thread-plan-readonly',
    agentMode: 'branching',
    runPhase: 'plan',
  });

  assert.equal(answer, '已完成计划草稿并继续顺序执行。');
  assert.equal(fs.readFileSync(path.join(projectRoot, 'draft.txt'), 'utf-8'), 'oops');
  assert.ok(!traces.some((event) => event.content.includes('PLAN stage is read-only.')));

  agent.stop();
});

test('child branch prompts stay on direct work tools instead of inheriting legacy planner JSON', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-child-tool-regression-');
  fs.writeFileSync(path.join(projectRoot, 'note.txt'), 'evidence from note\n', 'utf-8');

  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  let sawChildPrompt = false;
  let sawPostToolPrompt = false;
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';

    if (!joined.includes('child branch "Source audit"')) {
      return {
        kind: 'branch',
        thought: 'Split root from evidence collection.',
        branches: [
          {
            label: 'Source audit',
            goal: 'Read note.txt and ground the answer in workspace evidence.',
            why: 'The answer should be evidence-backed.',
            priority: 1,
          },
        ],
      };
    }

    assert.match(joined, /Call read\/search\/edit\/bash\/web directly for work/i);
    assert.match(joined, /Planning view only/i);
    assert.doesNotMatch(joined, /Return exactly one JSON object/i);
    assert.doesNotMatch(joined, /\{"kind":"branch"/);
    assert.doesNotMatch(joined, /planner_tool/i);
    assert.doesNotMatch(joined, /TOOL OBSERVATION for read:/);
    assert.doesNotMatch(joined, /Shared context for this request:/);

    if (sawChildPrompt) {
      sawPostToolPrompt = true;
      assert.doesNotMatch(joined, /\{"kind":"tool"/);
      return {
        kind: 'final',
        thought: 'Done.',
        final_answer: '已基于 note.txt 完成验证。',
        completion_summary: '已基于 note.txt 完成验证。',
      };
    }

    sawChildPrompt = true;
    return {
      kind: 'tool',
      thought: 'Read the note file for concrete evidence.',
      tool_calls: [
        {
          name: 'read',
          arguments: { target: 'workspace', path: 'note.txt' },
          goal: 'Read note.txt',
        },
      ],
    };
  };

  const answer = await agent.run('检查子分支 planner transcript', () => {}, { conversationId: 'thread-child-tool-regression' });
  assert.equal(answer, '已基于 note.txt 完成验证。');
  assert.equal(sawChildPrompt, true);
  assert.equal(sawPostToolPrompt, true);
  assert.equal(sawPostToolPrompt, true);

  agent.stop();
});

test('child branches may execute workspace edits when the planner chooses them', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-child-edit-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });

  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';

    if (!joined.includes('child branch "Writer"')) {
      return {
        kind: 'branch',
        thought: 'Split root from child writer.',
        branches: [
          {
            label: 'Writer',
            goal: 'Write branch-note.txt from the child branch.',
            priority: 1,
          },
        ],
      };
    }

    if (!joined.includes('TOOL OBSERVATION for edit:')) {
      return {
        kind: 'tool',
        thought: 'Write the note from the child branch.',
        tool_calls: [
          {
            name: 'edit',
            arguments: { target: 'workspace', path: 'branch-note.txt', content: 'child write ok' },
            goal: 'Write branch-note.txt',
          },
        ],
      };
    }

    return {
      kind: 'final',
      thought: 'Child write completed.',
      final_answer: 'child branch edit complete',
      completion_summary: 'child branch edit complete',
    };
  };

  const answer = await agent.run('验证子分支可执行写入', () => {}, { conversationId: 'thread-child-edit' });
  assert.equal(answer, 'child branch edit complete');
  assert.equal(fs.readFileSync(path.join(projectRoot, 'branch-note.txt'), 'utf-8'), 'child write ok');

  agent.stop();
});

test('sanitizes leaked bracketed tool-call markup from final answers', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-tool-leak-');
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(
    agent,
    () => '[TOOL_CALL]\n{tool => "web", args => { --query "OpenAI latest" }}\n[/TOOL_CALL]',
  );

  const answer = await agent.run('测试工具泄漏清洗', () => {}, { conversationId: 'thread-tool-leak' });
  assert.doesNotMatch(answer, /\[TOOL_CALL\]/i);
  assert.match(answer, /(不可安全展示|暂时没能生成可展示的回答)/);

  agent.stop();
});

test('planner executes structured tool decisions directly', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-pseudo-tool-call-');
  fs.writeFileSync(path.join(projectRoot, 'note.txt'), 'hello from note\n', 'utf-8');

  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  const anyAgent = agent as any;
  anyAgent.retrieveRelevantContext = async () => ({
    contextText: '',
    recalledNodeIds: [],
    selectedNodeIds: [],
    rationale: 'test',
  });
  anyAgent.generatePlannerDecision = async ({ messages }: { messages: Array<{ content?: string }> }) => {
    const joined = Array.isArray(messages)
      ? messages.map((message) => String(message?.content || '')).join('\n\n')
      : '';
    if (joined.includes('TOOL OBSERVATION for read:')) {
      return {
        kind: 'final',
        thought: 'Done.',
        final_answer: '已读取 note 文件。',
        completion_summary: '已读取 note 文件。',
      };
    }
    return {
      kind: 'tool',
      thought: 'Read the workspace note.',
      tool_calls: [
        {
          name: 'read',
          arguments: { target: 'workspace', path: 'note.txt' },
          goal: 'Read the note file',
        },
      ],
    };
  };

  const traces: TraceEvent[] = [];
  const answer = await agent.run('读取 note 文件', (event) => { traces.push(event); }, { conversationId: 'thread-pseudo-tool-call' });

  assert.equal(answer, '已读取 note 文件。');
  assert.ok(traces.some((event) => event.type === 'action' && event.content.includes('Calling read')));
  assert.ok(traces.some((event) => event.type === 'observation' && event.content.includes('hello from note')));

  agent.stop();
});

test('persists recent conversation context across agent instances for the same conversation id', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-persisted-context-');
  const conversationId = 'thread:persisted-context';

  const agent1 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent1, () => '第一轮回答');
  await agent1.run('请记住这个话题：美光财报保存到 Mempedia', () => {}, { conversationId });
  agent1.stop();

  const statePath = path.join(projectRoot, '.mitosis', 'conversation_state', 'thread_persisted-context.json');
  assert.ok(fs.existsSync(statePath), 'expected persisted conversation state file to be created');

  const agent2 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent2, (request) => `follow-up:${request}`);
  const traces: TraceEvent[] = [];
  await agent2.run('继续', (event) => { traces.push(event); }, { conversationId });

  assert.ok(traces.some((event) => event.content.includes('Recovered persisted thread context with 1 recent turn(s).')));
  assert.ok(traces.some((event) => event.content.includes('Selected 1 relevant recent conversation turn(s)')));
  assert.ok(traces.some((event) => event.content.includes('Injected persisted thread working state into the current run context.')));

  agent2.stop();
});

test('writes a turn summary journal entry for each persisted turn', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-turn-summary-');
  const conversationId = 'thread:journal-check';
  const agent = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent, () => '这是用于摘要日志的回答。');

  await agent.run('记录这轮摘要', () => {}, { conversationId });

  const journalPath = path.join(projectRoot, '.mitosis', 'logs', 'thread_turn_summaries.jsonl');
  assert.ok(fs.existsSync(journalPath), 'expected turn summary journal to exist');
  const rows = fs.readFileSync(journalPath, 'utf-8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].conversation_id, conversationId);
  assert.equal(rows[0].user_intent, '记录这轮摘要');
  assert.match(rows[0].assistant_outcome, /这是用于摘要日志的回答/);
  assert.ok(Array.isArray(rows[0].tool_findings));

  agent.stop();
});

test('reconstructs thread context from turn summary journal when state file is missing', async () => {
  const projectRoot = createTempProjectRoot('mempedia-agent-journal-recovery-');
  const conversationId = 'thread:journal-recovery';

  const agent1 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent1, () => '上一轮已经确认目标是继续保存美光财报。');
  await agent1.run('请继续跟进美光财报保存', () => {}, { conversationId });
  agent1.stop();

  const statePath = path.join(projectRoot, '.mitosis', 'conversation_state', 'thread_journal-recovery.json');
  if (fs.existsSync(statePath)) {
    fs.unlinkSync(statePath);
  }

  const agent2 = new Agent({ apiKey: 'test-key' }, projectRoot);
  installAgentTestDoubles(agent2, () => '已从 journal 恢复上下文。');
  const traces: TraceEvent[] = [];
  await agent2.run('继续', (event) => { traces.push(event); }, { conversationId });

  assert.ok(traces.some((event) => event.content.includes('Recovered persisted thread context with 1 recent turn(s).')));
  assert.ok(traces.some((event) => event.content.includes('Selected 1 relevant turn summary record(s) from journal replay.')));
  assert.ok(traces.some((event) => event.content.includes('Injected journal replay summaries into the current run context.')));

  agent2.stop();
});
