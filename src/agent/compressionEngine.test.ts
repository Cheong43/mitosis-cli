import test from 'node:test';
import assert from 'node:assert/strict';

import { snipStaleToolResults } from './compressionEngine.js';
import type { TranscriptMessage } from './compressionEngine.js';

test('snipStaleToolResults snips provider replay observations and downgrades paired tool decisions', () => {
  const messages: TranscriptMessage[] = [
    { role: 'user', content: 'Original request' },
    {
      role: 'assistant',
      content: {
        type: 'openai_assistant_message',
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_prev',
              type: 'function',
              function: {
                name: 'read',
                arguments: '{"path":"notes.txt"}',
              },
            },
          ],
        },
      },
    },
    {
      role: 'user',
      content: {
        type: 'openai_tool_result',
        tool_call_id: 'call_prev',
        tool_name: 'read',
        content: 'x'.repeat(500),
      },
    },
    { role: 'assistant', content: 'Midpoint note' },
    { role: 'user', content: 'Another local observation' },
    { role: 'assistant', content: 'Recent assistant message' },
    { role: 'user', content: 'Recent user message' },
  ];

  const result = snipStaleToolResults(messages, { staleThreshold: 3, recentKeep: 2 });

  assert.equal(typeof result.messages[1]?.content, 'string');
  assert.match(String(result.messages[1]?.content || ''), /^PLANNER TOOL DECISION:/);
  assert.match(String(result.messages[2]?.content || ''), /^\[snipped: read result, \d+ chars, step ~2\]$/);
  assert.equal(result.snippedCount, 1);
  assert.ok(result.freedChars > 0);
});

test('snipStaleToolResults still snips legacy tool observations', () => {
  const legacyObservation = `TOOL OBSERVATION for search:\n${'result\n'.repeat(80)}`;
  const messages: TranscriptMessage[] = [
    { role: 'user', content: 'Original request' },
    { role: 'assistant', content: 'PLANNER TOOL DECISION:\n- search | arguments: {"q":"qin"}' },
    { role: 'user', content: legacyObservation },
    { role: 'assistant', content: 'Intermediate note' },
    { role: 'user', content: 'Recent user note' },
    { role: 'assistant', content: 'Recent assistant note' },
  ];

  const result = snipStaleToolResults(messages, { staleThreshold: 3, recentKeep: 2 });

  assert.match(String(result.messages[1]?.content || ''), /^PLANNER TOOL DECISION:/);
  assert.match(String(result.messages[2]?.content || ''), /^\[snipped: search result, \d+ chars, step ~2\]$/);
  assert.equal(result.snippedCount, 1);
});
