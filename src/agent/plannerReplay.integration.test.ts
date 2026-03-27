import test from 'node:test';
import assert from 'node:assert/strict';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const cliRoot = path.resolve(__dirname, '../..');

dotenv.config({ path: path.join(cliRoot, '.env') });

interface TraceEventLike {
  type: string;
  content: string;
}

interface ReplayOutcome {
  prompt: string;
  answer: string;
  durationMs: number;
  traceCount: number;
  branchCount: number;
  toolCallCount: number;
  plannerErrorCount: number;
  connectionError: boolean;
  exhaustedWithoutAnswer: boolean;
  emittedBudgetTrace: boolean;
  leakedToolCallMarkup: boolean;
}

function parseReplayPrompts(): string[] {
  const raw = (process.env.PLANNER_REPLAY_PROMPTS || '').trim();
  if (!raw) {
    return ['挽救计划好看不'];
  }
  return raw
    .split('||')
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseReplayIterations(): number {
  const parsed = Number.parseInt(process.env.PLANNER_REPLAY_ITERATIONS || '2', 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 2;
  }
  return Math.min(parsed, 10);
}

function slugifyPrompt(prompt: string): string {
  const ascii = prompt
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  if (ascii) {
    return ascii.slice(0, 24);
  }
  return `prompt-${Buffer.from(prompt).toString('hex').slice(0, 12)}`;
}

function detectToolCallLeak(answer: string): boolean {
  const trimmed = answer.trim();
  return /```tool_call/i.test(trimmed)
    || /\[TOOL_CALL\]/i.test(trimmed)
    || /^```(?:bash|json|sh)\b/i.test(trimmed)
    || /^tool_call\s*$/im.test(trimmed)
    || /^PLANNER (?:TOOL|BRANCH|FINAL|SKILLS)\b/im.test(trimmed);
}

function summarizeOutcome(prompt: string, answer: string, traces: TraceEventLike[], durationMs: number): ReplayOutcome {
  const combined = traces.map((trace) => trace.content).join('\n');
  return {
    prompt,
    answer,
    durationMs,
    traceCount: traces.length,
    branchCount: traces.filter((trace) => trace.content.includes('[scheduler] Branch')).length,
    toolCallCount: traces.filter((trace) => trace.content.startsWith('Calling ')).length,
    plannerErrorCount: traces.filter((trace) => trace.content.includes('Planner decision generation failed:')).length,
    connectionError: traces.some((trace) => trace.content.includes('Connection error')),
    exhaustedWithoutAnswer: answer.trim() === 'No branch produced a final answer.',
    emittedBudgetTrace: traces.some((trace) => trace.content.includes('Context budget:')),
    leakedToolCallMarkup: detectToolCallLeak(answer) || /```tool_call/i.test(combined),
  };
}

test('live: planner replay stays on structured planner decisions for configured prompts', { timeout: 300_000 }, async () => {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseURL = process.env.OPENAI_BASE_URL;
  const model = process.env.OPENAI_MODEL;

  if (!apiKey) {
    console.log('  SKIP: OPENAI_API_KEY not set');
    return;
  }

  const prompts = parseReplayPrompts();
  const iterations = parseReplayIterations();

  const { Agent } = await import('./index.js');
  const agent = new Agent(
    {
      apiKey,
      baseURL,
      model: model || 'gpt-4o',
      hmacAccessKey: process.env.HMAC_ACCESS_KEY,
      hmacSecretKey: process.env.HMAC_SECRET_KEY,
      gatewayApiKey: process.env.GATEWAY_API_KEY,
    },
    cliRoot,
  );

  const outcomes: ReplayOutcome[] = [];
  await agent.start();
  try {
    for (const prompt of prompts) {
      for (let i = 0; i < iterations; i += 1) {
        const traces: TraceEventLike[] = [];
        const conversationId = `planner-replay:${slugifyPrompt(prompt)}:${Date.now()}:${i}`;
        const startedAt = Date.now();
        const answer = await agent.run(
          prompt,
          (event) => {
            traces.push({ type: event.type, content: event.content });
          },
          { conversationId },
        );
        const outcome = summarizeOutcome(prompt, answer, traces, Date.now() - startedAt);
        outcomes.push(outcome);
        console.log(
          `  replay prompt="${prompt}" run=${i + 1}/${iterations}`
          + ` duration=${outcome.durationMs}ms traces=${outcome.traceCount}`
          + ` branches=${outcome.branchCount} tools=${outcome.toolCallCount}`
          + ` plannerErrors=${outcome.plannerErrorCount}`
          + ` connectionError=${outcome.connectionError}`
          + ` exhausted=${outcome.exhaustedWithoutAnswer}`
          + ` leakedToolCall=${outcome.leakedToolCallMarkup}`
          + ` answer=${JSON.stringify(outcome.answer.slice(0, 180))}`,
        );
      }
    }
  } finally {
    await agent.shutdown(5000).catch(() => {});
  }

  assert.ok(outcomes.length > 0, 'Expected at least one replay outcome');
  assert.ok(outcomes.every((item) => item.emittedBudgetTrace), 'Each replay should emit a context budget trace');
  assert.ok(outcomes.every((item) => item.answer.trim().length > 0), 'Each replay should produce a non-empty answer');
  assert.ok(outcomes.every((item) => !item.leakedToolCallMarkup), 'Live replay should not leak raw tool_call/code fences');

  const aggregate = {
    runs: outcomes.length,
    prompts: prompts.length,
    iterations,
    connectionErrors: outcomes.filter((item) => item.connectionError).length,
    exhaustedWithoutAnswer: outcomes.filter((item) => item.exhaustedWithoutAnswer).length,
    plannerErrorRuns: outcomes.filter((item) => item.plannerErrorCount > 0).length,
    avgTraceCount: Number((outcomes.reduce((sum, item) => sum + item.traceCount, 0) / outcomes.length).toFixed(1)),
    avgDurationMs: Math.round(outcomes.reduce((sum, item) => sum + item.durationMs, 0) / outcomes.length),
  };
  console.log(`  replay summary: ${JSON.stringify(aggregate)}`);
});
