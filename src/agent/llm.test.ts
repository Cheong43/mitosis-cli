import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildAnthropicLanguageModel,
  buildLanguageModel,
  generateToolCalls,
  type ParseableFunctionTool,
  type OpenAIAssistantReplayContent,
  type OpenAIToolResultReplayContent,
} from './llm.js';

interface DemoParsedInput {
  value: string;
}

const DEMO_TOOL: ParseableFunctionTool<DemoParsedInput> = {
  name: 'demo_tool',
  description: 'Demo tool for tests.',
  parameters: {
    type: 'object',
    additionalProperties: false,
    properties: {
      value: { type: 'string' },
    },
    required: ['value'],
  },
  parse: (input: unknown) => {
    const record = input && typeof input === 'object' && !Array.isArray(input)
      ? input as Record<string, unknown>
      : {};
    return {
      value: String(record.value || ''),
    };
  },
};

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

async function readJsonBody(init: RequestInit | undefined): Promise<any> {
  const body = init?.body;
  if (typeof body === 'string') {
    return JSON.parse(body);
  }
  if (body instanceof Uint8Array) {
    return JSON.parse(new TextDecoder().decode(body));
  }
  throw new Error(`Unsupported request body in test: ${typeof body}`);
}

test('generateToolCalls parses native OpenAI function tool calls through AI SDK', async () => {
  let requestCount = 0;
  const model = buildLanguageModel({
    model: 'fake-openai',
    apiKey: 'test-key',
    baseURL: 'https://example.test/v1',
    fetch: async (_input, init) => {
      requestCount += 1;
      const payload = await readJsonBody(init);
      assert.equal(payload.tool_choice, 'required');
      return jsonResponse({
        id: 'chatcmpl_test',
        object: 'chat.completion',
        created: 1,
        model: 'fake-openai',
        choices: [
          {
            index: 0,
            finish_reason: 'tool_calls',
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  id: 'call_demo',
                  type: 'function',
                  function: {
                    name: 'demo_tool',
                    arguments: '{"value":"native-openai"}',
                  },
                },
              ],
            },
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
    },
  });

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requestCount, 1);
  assert.deepEqual(result.calls, [{
    name: 'demo_tool',
    input: { value: 'native-openai' },
  }]);
  assert.deepEqual(result.providerMessage?.openaiAssistantMessage, {
    role: 'assistant',
    content: '',
    tool_calls: [
      {
        id: 'call_demo',
        type: 'function',
        function: {
          name: 'demo_tool',
          arguments: '{"value":"native-openai"}',
        },
      },
    ],
  });
});

test('generateToolCalls falls back to JSON repair for OpenAI-compatible models when native tool calls are missing', async () => {
  const requests: any[] = [];
  const model = buildLanguageModel({
    model: 'fake-openai',
    apiKey: 'test-key',
    baseURL: 'https://example.test/v1',
    fetch: async (_input, init) => {
      const payload = await readJsonBody(init);
      requests.push(payload);
      if (requests.length === 1) {
        return jsonResponse({
          id: 'chatcmpl_test_1',
          object: 'chat.completion',
          created: 1,
          model: 'fake-openai',
          choices: [
            {
              index: 0,
              finish_reason: 'stop',
              message: {
                role: 'assistant',
                content: 'I want to call the demo tool, but here is plain text only.',
              },
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        });
      }

      return jsonResponse({
        id: 'chatcmpl_test_2',
        object: 'chat.completion',
        created: 2,
        model: 'fake-openai',
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            message: {
              role: 'assistant',
              content: '{"tool_calls":[{"name":"demo_tool","input":{"value":"fallback-openai"}}]}',
            },
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
    },
  });

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requests.length, 2);
  assert.equal(requests[1].tools, undefined);
  assert.match(String(requests[1].messages?.[0]?.content || ''), /JSON REPAIR MODE/i);
  assert.match(String(requests[1].messages?.[0]?.content || ''), /valid JSON only/i);
  const fallbackPrompt = String(requests[1].messages?.[requests[1].messages.length - 1]?.content || '');
  assert.match(fallbackPrompt, /exactly one JSON object and nothing else/i);
  assert.match(fallbackPrompt, /No markdown fences/i);
  assert.deepEqual(result.calls, [{
    name: 'demo_tool',
    input: { value: 'fallback-openai' },
  }]);
});

test('generateToolCalls falls back to JSON repair when native tool calling is rejected with Not Found', async () => {
  const requests: any[] = [];
  const model = buildLanguageModel({
    model: 'MiniMax-M2.7',
    apiKey: 'test-key',
    baseURL: 'https://example.test/v1',
    fetch: async (_input, init) => {
      const payload = await readJsonBody(init);
      requests.push(payload);
      if (requests.length === 1) {
        return new Response(JSON.stringify({
          error: { message: 'Not Found' },
        }), {
          status: 404,
          headers: { 'content-type': 'application/json' },
        });
      }

      return jsonResponse({
        id: 'chatcmpl_test_2',
        object: 'chat.completion',
        created: 2,
        model: 'MiniMax-M2.7',
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            message: {
              role: 'assistant',
              content: '{"tool_calls":[{"name":"demo_tool","input":{"value":"fallback-after-404"}}]}',
            },
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
    },
  });

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requests.length, 2);
  assert.ok(requests[0].tools);
  assert.equal(requests[1].tools, undefined);
  assert.match(String(requests[1].messages?.[0]?.content || ''), /JSON REPAIR MODE/i);
  assert.deepEqual(result.calls, [{
    name: 'demo_tool',
    input: { value: 'fallback-after-404' },
  }]);
});

test('generateToolCalls enables MiniMax reasoning_split and replays prior OpenAI tool messages', async () => {
  const requests: any[] = [];
  const model = buildLanguageModel({
    model: 'MiniMax-M2.7',
    apiKey: 'test-key',
    baseURL: 'https://example.test/v1',
    fetch: async (_input, init) => {
      const payload = await readJsonBody(init);
      requests.push(payload);
      return jsonResponse({
        id: 'chatcmpl_test',
        object: 'chat.completion',
        created: 1,
        model: 'MiniMax-M2.7',
        choices: [
          {
            index: 0,
            finish_reason: 'tool_calls',
            message: {
              role: 'assistant',
              content: '',
              reasoning_content: 'demo reasoning',
              tool_calls: [
                {
                  id: 'call_demo',
                  type: 'function',
                  function: {
                    name: 'demo_tool',
                    arguments: '{"value":"native-openai"}',
                  },
                },
              ],
            },
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
    },
  });

  const replayAssistant: OpenAIAssistantReplayContent = {
    type: 'openai_assistant_message',
    message: {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'call_prev',
          type: 'function',
          function: {
            name: 'demo_tool',
            arguments: '{"value":"previous"}',
          },
        },
      ],
      reasoning_details: [{ type: 'reasoning.text', text: 'previous reasoning' }],
    },
  };
  const replayTool: OpenAIToolResultReplayContent = {
    type: 'openai_tool_result',
    tool_call_id: 'call_prev',
    content: 'previous tool output',
  };

  await generateToolCalls({
    model,
    messages: [
      { role: 'assistant', content: replayAssistant },
      { role: 'user', content: replayTool },
      { role: 'user', content: 'Use the demo tool again.' },
    ],
    tools: [DEMO_TOOL],
  });

  assert.equal(requests.length, 1);
  assert.equal(requests[0].reasoning_split, true);
  assert.equal(requests[0].messages[0].role, 'assistant');
  assert.deepEqual(requests[0].messages[0].tool_calls, [
    {
      id: 'call_prev',
      type: 'function',
      function: {
        name: 'demo_tool',
        arguments: '{"value":"previous"}',
      },
    },
  ]);
  assert.equal(requests[0].messages[0].reasoning_content, 'previous reasoning');
  assert.deepEqual(requests[0].messages[1], {
    role: 'tool',
    tool_call_id: 'call_prev',
    content: 'previous tool output',
  });
});

test('generateToolCalls parses native Anthropic tool_use blocks through AI SDK', async () => {
  let requestCount = 0;
  const model = buildAnthropicLanguageModel({
    model: 'fake-anthropic',
    authToken: 'test-token',
    baseURL: 'https://anthropic.example.test/v1',
    fetch: async (_input, init) => {
      requestCount += 1;
      const payload = await readJsonBody(init);
      assert.equal(payload.tool_choice.type, 'any');
      return jsonResponse({
        id: 'msg_test',
        type: 'message',
        role: 'assistant',
        model: 'fake-anthropic',
        content: [
          { type: 'text', text: 'Using a native tool.' },
          { type: 'tool_use', id: 'toolu_1', name: 'demo_tool', input: { value: 'native-anthropic' } },
        ],
        stop_reason: 'tool_use',
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
      });
    },
  });

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requestCount, 1);
  assert.deepEqual(result.calls, [{
    name: 'demo_tool',
    input: { value: 'native-anthropic' },
    providerToolUseId: 'toolu_1',
  }]);
});

test('generateToolCalls falls back to JSON repair for Anthropic models when tool_use blocks are missing', async () => {
  const requests: any[] = [];
  const model = buildAnthropicLanguageModel({
    model: 'fake-anthropic',
    authToken: 'test-token',
    baseURL: 'https://anthropic.example.test/v1',
    fetch: async (_input, init) => {
      const payload = await readJsonBody(init);
      requests.push(payload);
      if (requests.length === 1) {
        return jsonResponse({
          id: 'msg_test_1',
          type: 'message',
          role: 'assistant',
          model: 'fake-anthropic',
          content: [
            { type: 'text', text: 'Need to use a tool, but no tool_use block was emitted.' },
          ],
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
        });
      }

      return jsonResponse({
        id: 'msg_test_2',
        type: 'message',
        role: 'assistant',
        model: 'fake-anthropic',
        content: [
          { type: 'text', text: '{"tool_calls":[{"name":"demo_tool","input":{"value":"fallback-anthropic"}}]}' },
        ],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
      });
    },
  });

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requests.length, 2);
  assert.match(JSON.stringify(requests[1].system || ''), /JSON REPAIR MODE/i);
  assert.match(JSON.stringify(requests[1].system || ''), /valid JSON only/i);
  assert.match(
    JSON.stringify(requests[1].messages?.[requests[1].messages.length - 1]?.content || ''),
    /exactly one JSON object and nothing else/i,
  );
  assert.deepEqual(result.calls, [{
    name: 'demo_tool',
    input: { value: 'fallback-anthropic' },
  }]);
});
