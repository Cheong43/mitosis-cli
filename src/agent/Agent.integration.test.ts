import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';
import { Agent, type TraceEvent } from './index.js';

process.env.AUTO_QUEUE_MEMORY_SAVE = '0';

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
  assert.equal(plannerCalls, 1);

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
  assert.match(capturedPrompt, /independent sub-goals that are worth exploring in parallel/i);
  assert.match(capturedPrompt, /Decompose aggressively/i);
  assert.match(capturedPrompt, /default toward planner_branch on the first meaningful step/i);
  assert.match(capturedPrompt, /task can be decomposed into independent parts/i);
  assert.match(capturedPrompt, /multiple plausible independent avenues to explore/i);
  assert.match(capturedPrompt, /call read\/search\/edit\/bash\/web directly for work/i);
  assert.match(capturedPrompt, /Prefer materially diverse search strategies/i);
  assert.match(capturedPrompt, /If this branch can still be split into 2 or more genuinely independent evidence streams or workstreams/i);
  assert.match(capturedPrompt, /each must target a genuinely different evidence stream or hypothesis/i);
  assert.match(capturedPrompt, /After 2-3 consecutive web\/search rounds without clear new information/i);
  assert.doesNotMatch(capturedPrompt, /Always provide at least 2 branches/i);
  assert.doesNotMatch(capturedPrompt, /NEVER branch on the first step/i);
  assert.doesNotMatch(capturedPrompt, /default to one `web` tool call before branching/i);
  assert.doesNotMatch(capturedPrompt, /Return exactly one JSON object/i);
  assert.doesNotMatch(capturedPrompt, /planner_tool/i);

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
    assert.doesNotMatch(joined, /Return exactly one JSON object/i);
    assert.doesNotMatch(joined, /\{"kind":"branch"/);
    assert.doesNotMatch(joined, /planner_tool/i);

    if (joined.includes('TOOL OBSERVATION for read:')) {
      sawPostToolPrompt = true;
      assert.match(joined, /PLANNER TOOL DECISION:/);
      assert.doesNotMatch(joined, /\{"kind":"tool"/);
      return {
        kind: 'final',
        thought: 'Done.',
        final_answer: '已基于 note.txt 完成验证。',
        completion_summary: '已基于 note.txt 完成验证。',
      };
    }

    sawChildPrompt = true;
    assert.match(joined, /PLANNER BRANCH DECISION:/);
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
