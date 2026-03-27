import test from 'node:test';
import assert from 'node:assert/strict';

import {
  generateToolCalls,
  type LanguageModelV1,
  type ParseableFunctionTool,
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

test('generateToolCalls parses native OpenAI function tool calls', async () => {
  let createCalls = 0;
  const model: LanguageModelV1 = {
    provider: 'openai',
    client: {
      chat: {
        completions: {
          create: async () => {
            createCalls += 1;
            return {
              choices: [
                {
                  message: {
                    content: '',
                    tool_calls: [
                      {
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
            };
          },
        },
      },
    } as any,
    model: 'fake-openai',
    supportsStructuredOutputs: false,
  };

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(createCalls, 1);
  assert.deepEqual(result.calls, [{ name: 'demo_tool', input: { value: 'native-openai' } }]);
});

test('generateToolCalls falls back to json_object for OpenAI-compatible models when native tool calls are missing', async () => {
  const requests: any[] = [];
  const model: LanguageModelV1 = {
    provider: 'openai',
    client: {
      chat: {
        completions: {
          create: async (payload: any) => {
            requests.push(payload);
            if (requests.length === 1) {
              return {
                choices: [
                  {
                    message: {
                      content: 'I want to call the demo tool, but here is plain text only.',
                      tool_calls: [],
                    },
                  },
                ],
              };
            }
            return {
              choices: [
                {
                  message: {
                    content: '{"tool_calls":[{"name":"demo_tool","input":{"value":"fallback-openai"}}]}',
                  },
                },
              ],
            };
          },
        },
      },
    } as any,
    model: 'fake-openai',
    supportsStructuredOutputs: false,
  };

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requests.length, 2);
  assert.deepEqual(requests[1].response_format, { type: 'json_object' });
  assert.match(String(requests[1].messages?.[0]?.content || ''), /JSON REPAIR MODE/i);
  const fallbackPrompt = String(requests[1].messages?.[requests[1].messages.length - 1]?.content || '');
  assert.match(fallbackPrompt, /exactly one JSON object and nothing else/i);
  assert.match(fallbackPrompt, /No markdown fences/i);
  assert.deepEqual(result.calls, [{ name: 'demo_tool', input: { value: 'fallback-openai' } }]);
});

test('generateToolCalls parses native Anthropic tool_use blocks', async () => {
  let createCalls = 0;
  const model: LanguageModelV1 = {
    provider: 'anthropic',
    client: {
      messages: {
        create: async () => {
          createCalls += 1;
          return {
            content: [
              { type: 'text', text: 'Using a native tool.' },
              { type: 'tool_use', name: 'demo_tool', input: { value: 'native-anthropic' } },
            ],
          };
        },
      },
    } as any,
    model: 'fake-anthropic',
    supportsStructuredOutputs: false,
  };

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(createCalls, 1);
  assert.deepEqual(result.calls, [{ name: 'demo_tool', input: { value: 'native-anthropic' } }]);
});

test('generateToolCalls falls back to json_object for Anthropic models when tool_use blocks are missing', async () => {
  const requests: any[] = [];
  const model: LanguageModelV1 = {
    provider: 'anthropic',
    client: {
      messages: {
        create: async (payload: any) => {
          requests.push(payload);
          if (requests.length === 1) {
            return {
              content: [
                { type: 'text', text: 'Need to use a tool, but no tool_use block was emitted.' },
              ],
            };
          }
          return {
            content: [
              { type: 'text', text: '{"tool_calls":[{"name":"demo_tool","input":{"value":"fallback-anthropic"}}]}' },
            ],
          };
        },
      },
    } as any,
    model: 'fake-anthropic',
    supportsStructuredOutputs: false,
  };

  const result = await generateToolCalls({
    model,
    messages: [{ role: 'user', content: 'Use the demo tool.' }],
    tools: [DEMO_TOOL],
  });

  assert.equal(requests.length, 2);
  assert.match(String(requests[1].system || ''), /valid JSON only/i);
  assert.match(String(requests[1].system || ''), /JSON REPAIR MODE/i);
  assert.match(
    String(requests[1].messages?.[requests[1].messages.length - 1]?.content || ''),
    /exactly one JSON object and nothing else/i,
  );
  assert.deepEqual(result.calls, [{ name: 'demo_tool', input: { value: 'fallback-anthropic' } }]);
});
