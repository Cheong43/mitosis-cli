/**
 * LLM factory and helpers for OpenAI and Anthropic format APIs.
 *
 * Auth priority (OpenAI): HMAC > Gateway > plain API key.
 */
import crypto from 'crypto';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';

export interface OpenAIChatSettings {
  structuredOutputs?: boolean;
}

export type LLMProvider = 'openai' | 'anthropic';

export interface AnthropicTextContentBlock {
  type: 'text';
  text: string;
}

export interface AnthropicToolUseContentBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: unknown;
}

export interface AnthropicToolResultContentBlock {
  type: 'tool_result';
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}

export interface AnthropicThinkingContentBlock {
  type: 'thinking';
  thinking?: string;
  text?: string;
  signature?: string;
}

export type AnthropicMessageContentBlock =
  | AnthropicTextContentBlock
  | AnthropicToolUseContentBlock
  | AnthropicToolResultContentBlock
  | AnthropicThinkingContentBlock
  | Record<string, unknown>;

export type ChatMessageContent = string | AnthropicMessageContentBlock[];

export interface LanguageModelV1 {
  provider: LLMProvider;
  client: OpenAI | Anthropic;
  model: string;
  supportsStructuredOutputs: boolean;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: ChatMessageContent;
}

interface GenerateTextOptions {
  model: LanguageModelV1;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  providerOptions?: {
    openai?: {
      responseFormat?: {
        type: 'json_object';
      };
    };
  };
}

interface GenerateObjectOptions<TSchema extends z.ZodTypeAny> {
  model: LanguageModelV1;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  mode?: 'json';
  schema: TSchema;
}

export interface ParseableFunctionTool<TParsed> {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
  parse: (input: unknown) => TParsed;
}

export interface ParsedToolCall<TParsed> {
  name: string;
  input: TParsed;
  providerToolUseId?: string;
}

export interface ToolCallProviderMessage {
  anthropicAssistantContent?: AnthropicMessageContentBlock[];
}

interface GenerateToolCallsOptions<TParsed> {
  model: LanguageModelV1;
  messages: ChatMessage[];
  tools: Array<ParseableFunctionTool<TParsed>>;
  temperature?: number;
  maxTokens?: number;
}

interface JsonFallbackParseResult<TParsed> {
  calls: ParsedToolCall<TParsed>[];
  text: string;
}

export class NoObjectGeneratedError extends Error {
  static isInstance(error: unknown): error is NoObjectGeneratedError {
    return error instanceof NoObjectGeneratedError;
  }

  constructor(message = 'No object generated') {
    super(message);
    this.name = 'NoObjectGeneratedError';
  }
}

export class NoToolCallGeneratedError extends Error {
  static isInstance(error: unknown): error is NoToolCallGeneratedError {
    return error instanceof NoToolCallGeneratedError;
  }

  constructor(message = 'No tool call generated') {
    super(message);
    this.name = 'NoToolCallGeneratedError';
  }
}

export interface LLMEndpointConfig {
  model: string;
  apiKey: string;
  baseURL?: string;
  hmacAccessKey?: string;
  hmacSecretKey?: string;
  gatewayApiKey?: string;
}

export interface AnthropicEndpointConfig {
  model: string;
  authToken: string;
  baseURL?: string;
}

export function buildHmacFetch(accessKey: string, secretKey: string): typeof globalThis.fetch {
  return async (input, init) => {
    const bodyStr = init?.body ? String(init.body) : '';
    const digestHash = crypto.createHash('sha256').update(bodyStr).digest('base64');
    const digest = `SHA-256=${digestHash}`;
    const date = new Date().toUTCString();
    const parsedUrl = new URL(String(input));
    const requestPath = `${parsedUrl.pathname}${parsedUrl.search || ''}`;
    const requestLine = `POST ${requestPath} HTTP/1.1`;
    const host = parsedUrl.host;
    const signingData = `Digest: ${digest}\nX-Date: ${date}\nhost: ${host}\n${requestLine}`;
    const signature = crypto.createHmac('sha256', secretKey).update(signingData).digest('base64');
    const authorization = `hmac username="${accessKey}", algorithm="hmac-sha256", headers="Digest X-Date host request-line", signature="${signature}"`;
    const headers = new Headers(init?.headers as HeadersInit | undefined);
    headers.set('X-Date', date);
    headers.set('Digest', digest);
    headers.set('Authorization', authorization);
    return fetch(input, { ...init, headers });
  };
}

function extractTextParts(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') {
          return part;
        }
        if (part && typeof part === 'object') {
          if ('text' in part && typeof (part as { text?: unknown }).text === 'string') {
            return (part as { text: string }).text;
          }
          if ('thinking' in part && typeof (part as { thinking?: unknown }).thinking === 'string') {
            return (part as { thinking: string }).thinking;
          }
          if ('content' in part && typeof (part as { content?: unknown }).content === 'string') {
            return (part as { content: string }).content;
          }
        }
        return '';
      })
      .join('');
  }
  return '';
}

function extractMessageText(message: unknown): string {
  if (!message || typeof message !== 'object') {
    return '';
  }
  const msg = message as Record<string, unknown>;
  if ('content' in msg) {
    const text = extractTextParts(msg.content);
    if (text.trim()) {
      return text;
    }
  }
  // Fallback: some models (e.g. Qwen3 thinking mode) return the answer in
  // reasoning_content or thinking_content when content is null/empty.
  for (const field of ['reasoning_content', 'thinking_content']) {
    if (field in msg) {
      const alt = extractTextParts((msg as Record<string, unknown>)[field]);
      if (alt.trim()) {
        return alt;
      }
    }
  }
  return '';
}

function normalizeMessages(messages: ChatMessage[]): Array<{ role: string; content: string }> {
  return messages.map((message) => ({
    role: message.role,
    content: extractTextParts(message.content),
  }));
}

function parseJsonObjectText(text: string): unknown {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new NoObjectGeneratedError('No object generated: empty response body.');
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      return JSON.parse(trimmed.slice(firstBrace, lastBrace + 1));
    }
    throw new NoObjectGeneratedError(`No object generated: ${trimmed.slice(0, 240)}`);
  }
}

function clipPreview(text: string, maxChars = 240): string {
  return text.replace(/\s+/g, ' ').trim().slice(0, maxChars);
}

function buildJsonFallbackPrompt<TParsed>(
  tools: Array<ParseableFunctionTool<TParsed>>,
  priorReplyPreview: string,
): string {
  const toolSchemas = tools
    .map((tool) => `- ${tool.name}: ${JSON.stringify(tool.parameters)}`)
    .join('\n');

  return [
    'Your previous reply was invalid because it did not produce usable native tool calls.',
    'Reply again with exactly one JSON object and nothing else. No markdown fences. No prose before or after the JSON. Do not output raw shell commands or code unless they are escaped inside a JSON string value.',
    priorReplyPreview ? `Previous reply preview: ${priorReplyPreview}` : '',
    'Use this JSON object schema:',
    '{"tool_calls":[{"name":"<allowed tool name>","input":{}}]}',
    'Rules:',
    '- Use only the allowed tool names below.',
    '- Put every tool input inside the "input" object.',
    '- Return at least one tool call.',
    '- If you need a final answer, return planner_final inside the tool_calls array with its required input fields.',
    '- If you need branching or skills loading, return planner_branch or planner_skills inside the tool_calls array with their required input fields.',
    '- Do not include markdown fences, XML, tags, or prose outside the JSON object.',
    'Allowed tools:',
    toolSchemas,
  ]
    .filter(Boolean)
    .join('\n');
}

function buildJsonFallbackModeInstruction(): string {
  return [
    'JSON REPAIR MODE:',
    'Your previous reply was invalid because it did not produce usable native tool calls.',
    'For this reply only, ignore any earlier instruction that says to emit native tool calls directly.',
    'Instead, return exactly one JSON object and nothing else. No markdown fences. No prose before or after the JSON. Do not output raw shell commands or code unless they are escaped inside a JSON string value.',
  ].join('\n');
}

function buildJsonFallbackMessages<TParsed>(
  messages: ChatMessage[],
  tools: Array<ParseableFunctionTool<TParsed>>,
  priorReplyPreview: string,
): ChatMessage[] {
  const fallbackPrompt = buildJsonFallbackPrompt(tools, priorReplyPreview);
  const fallbackSystem = buildJsonFallbackModeInstruction();
  let injectedSystem = false;
  const nextMessages = messages.map((message) => {
    if (!injectedSystem && message.role === 'system') {
      injectedSystem = true;
      return {
        ...message,
        content: `${message.content}\n\n${fallbackSystem}`,
      };
    }
    return message;
  });
  if (!injectedSystem) {
    nextMessages.unshift({ role: 'system', content: fallbackSystem });
  }
  nextMessages.push({ role: 'user', content: fallbackPrompt });
  return nextMessages;
}

function parseToolCallsFromFallbackPayload<TParsed>(
  payload: unknown,
  toolMap: Map<string, ParseableFunctionTool<TParsed>>,
  rawText: string,
): JsonFallbackParseResult<TParsed> {
  const parsed = payload && typeof payload === 'object' && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {};
  const thoughtText = typeof parsed.thought === 'string' ? parsed.thought.trim() : '';

  let rawCalls: unknown[] = [];
  if (Array.isArray(parsed.tool_calls)) {
    rawCalls = parsed.tool_calls;
  } else if (Array.isArray(parsed.calls)) {
    rawCalls = parsed.calls;
  } else if (typeof parsed.name === 'string') {
    rawCalls = [parsed];
  } else if (parsed.kind === 'tool' && Array.isArray(parsed.tool_calls)) {
    rawCalls = parsed.tool_calls;
  } else if (parsed.kind === 'final' && toolMap.has('planner_final')) {
    rawCalls = [{
      name: 'planner_final',
      input: {
        thought: parsed.thought,
        final_answer: parsed.final_answer,
        completion_summary: parsed.completion_summary,
      },
    }];
  } else if (parsed.kind === 'branch' && toolMap.has('planner_branch')) {
    rawCalls = [{
      name: 'planner_branch',
      input: {
        thought: parsed.thought,
        branches: parsed.branches,
      },
    }];
  } else if (parsed.kind === 'skills' && toolMap.has('planner_skills')) {
    rawCalls = [{
      name: 'planner_skills',
      input: {
        thought: parsed.thought,
        skills_to_load: parsed.skills_to_load,
      },
    }];
  }

  if (rawCalls.length === 0) {
    const preview = clipPreview(rawText);
    throw new NoToolCallGeneratedError(preview ? `No tool call generated: ${preview}` : 'No tool call generated.');
  }

  return {
    calls: rawCalls.map((rawCall) => {
      if (!rawCall || typeof rawCall !== 'object') {
        throw new NoToolCallGeneratedError('Fallback JSON contained a non-object tool call.');
      }
      const record = rawCall as Record<string, unknown>;
      const toolName = typeof record.name === 'string' ? record.name : '';
      if (!toolName) {
        throw new NoToolCallGeneratedError('Fallback JSON tool call was missing a name.');
      }
      const tool = toolMap.get(toolName);
      if (!tool) {
        throw new NoToolCallGeneratedError(`Unknown tool call generated: ${toolName}`);
      }
      const input = record.input ?? record.arguments ?? record.args ?? {};
      return {
        name: tool.name,
        input: tool.parse(input),
      };
    }),
    text: thoughtText || '',
  };
}

/**
 * Try to parse XML-formatted tool calls that some models (MiniMax, Qwen, etc.)
 * emit in the text body instead of native tool-call blocks.
 * Supports patterns:
 *   <minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>
 *   <tool_call><invoke name="...">...</invoke></tool_call>
 *   <invoke name="..."><parameter name="...">...</parameter></invoke>
 * Returns null if no parseable XML tool calls are found.
 */
function tryParseXmlToolCalls<TParsed>(
  text: string,
  toolMap: Map<string, ParseableFunctionTool<TParsed>>,
): JsonFallbackParseResult<TParsed> | null {
  if (!text || !/<invoke\s+name=/i.test(text)) {
    return null;
  }

  // Extract all <invoke name="toolName">...</invoke> blocks
  const invokePattern = /<invoke\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/invoke>/gi;
  const rawCalls: Array<{ name: string; args: Record<string, unknown> }> = [];
  let match: RegExpExecArray | null;
  while ((match = invokePattern.exec(text)) !== null) {
    const toolName = match[1];
    const body = match[2];
    const args: Record<string, unknown> = {};
    // Extract <parameter name="key">value</parameter>
    const paramPattern = /<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/parameter>/gi;
    let paramMatch: RegExpExecArray | null;
    while ((paramMatch = paramPattern.exec(body)) !== null) {
      const key = paramMatch[1];
      let value: unknown = paramMatch[2];
      // Try to parse JSON values (objects, arrays, numbers, booleans)
      const trimmedValue = String(value).trim();
      if (/^[{\[]/.test(trimmedValue)) {
        try { value = JSON.parse(trimmedValue); } catch { /* keep as string */ }
      }
      args[key] = value;
    }
    rawCalls.push({ name: toolName, args });
  }

  if (rawCalls.length === 0) {
    return null;
  }

  // Verify all tool names are known
  const calls: ParsedToolCall<TParsed>[] = [];
  for (const rawCall of rawCalls) {
    const tool = toolMap.get(rawCall.name);
    if (!tool) {
      return null; // Unknown tool — fall through to JSON fallback
    }
    try {
      calls.push({
        name: tool.name,
        input: tool.parse(rawCall.args),
      });
    } catch {
      return null; // Parse failure — fall through to JSON fallback
    }
  }

  // Extract thought text (everything before the first XML tool block)
  const firstXmlIndex = text.search(/<(?:[a-z_][a-z0-9_-]*:)?(?:tool_call|invoke)\b/i);
  const thoughtText = firstXmlIndex > 0 ? text.slice(0, firstXmlIndex).trim() : '';

  return { calls, text: thoughtText };
}

async function generateToolCallsJsonFallback<TParsed>(
  options: GenerateToolCallsOptions<TParsed>,
  toolMap: Map<string, ParseableFunctionTool<TParsed>>,
  priorReplyPreview: string,
): Promise<JsonFallbackParseResult<TParsed>> {
  const { text } = await generateText({
    model: options.model,
    messages: buildJsonFallbackMessages(options.messages, options.tools, priorReplyPreview),
    temperature: options.temperature,
    maxTokens: options.maxTokens,
    providerOptions: {
      openai: {
        responseFormat: { type: 'json_object' },
      },
    },
  });

  return parseToolCallsFromFallbackPayload(parseJsonObjectText(text), toolMap, text);
}

function buildClient(cfg: LLMEndpointConfig): OpenAI {
  const { apiKey, baseURL, hmacAccessKey, hmacSecretKey, gatewayApiKey } = cfg;

  if (hmacAccessKey && hmacSecretKey) {
    if (!baseURL) throw new Error('HMAC auth requires baseURL');
    return new OpenAI({
      apiKey: hmacAccessKey || apiKey || 'placeholder',
      baseURL,
      fetch: buildHmacFetch(hmacAccessKey, hmacSecretKey),
    });
  }

  if (gatewayApiKey) {
    if (!baseURL) throw new Error('Gateway auth requires baseURL');
    return new OpenAI({
      apiKey: apiKey || gatewayApiKey || 'placeholder',
      baseURL,
      defaultHeaders: {
        'x-gatewat-apikey': `Bearer ${gatewayApiKey}`,
        'x-gateway-apikey': `Bearer ${gatewayApiKey}`,
      },
    });
  }

  return new OpenAI({ apiKey, baseURL });
}

export function buildLanguageModel(cfg: LLMEndpointConfig, chatSettings?: OpenAIChatSettings): LanguageModelV1 {
  return {
    provider: 'openai',
    client: buildClient(cfg),
    model: cfg.model,
    supportsStructuredOutputs: Boolean(chatSettings?.structuredOutputs),
  };
}

export function buildAnthropicLanguageModel(cfg: AnthropicEndpointConfig): LanguageModelV1 {
  const client = new Anthropic({
    apiKey: cfg.authToken,
    ...(cfg.baseURL ? { baseURL: cfg.baseURL } : {}),
  });
  return {
    provider: 'anthropic',
    client,
    model: cfg.model,
    supportsStructuredOutputs: false,
  };
}

/* ── Anthropic helpers ────────────────────────────────────────────────────── */

function splitSystemAndMessages(
  messages: ChatMessage[],
): { system: string | undefined; messages: Array<{ role: 'user' | 'assistant'; content: ChatMessageContent }> } {
  let system: string | undefined;
  const out: Array<{ role: 'user' | 'assistant'; content: ChatMessageContent }> = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const systemChunk = extractTextParts(m.content);
      system = system ? `${system}\n\n${systemChunk}` : systemChunk;
    } else {
      out.push({ role: m.role, content: typeof m.content === 'string' ? m.content : m.content });
    }
  }
  // Anthropic requires at least one message and it must start with 'user'.
  if (out.length === 0) {
    out.push({ role: 'user', content: system || '' });
    system = undefined;
  }
  return { system, messages: out };
}

function extractAnthropicText(response: Anthropic.Message): string {
  if (!Array.isArray(response.content)) {
    // Some proxies return content as a plain string or undefined
    if (typeof response.content === 'string') return response.content;
    return '';
  }
  for (const block of response.content) {
    if (block.type === 'text') {
      return block.text;
    }
  }
  return '';
}

async function anthropicGenerateText(
  client: Anthropic,
  model: string,
  messages: ChatMessage[],
  jsonMode?: boolean,
  temperature?: number,
  maxTokens?: number,
): Promise<string> {
  const { system, messages: anthropicMsgs } = splitSystemAndMessages(messages);
  const systemText = jsonMode && system
    ? `${system}\n\nIMPORTANT: You MUST respond with valid JSON only. No other text.`
    : jsonMode
      ? 'You MUST respond with valid JSON only. No other text.'
      : system;

  const maxRetries = 3;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await client.messages.create({
        model,
        max_tokens: maxTokens ?? 8192,
        ...(typeof temperature === 'number' ? { temperature } : {}),
        ...(systemText ? { system: systemText } : {}),
        messages: anthropicMsgs as any,
      });
      return extractAnthropicText(response);
    } catch (error: any) {
      const status = error?.status ?? error?.statusCode;
      const isRetryable = status === 429 || status === 500 || status === 502 || status === 503 || status === 520;
      if (isRetryable && attempt < maxRetries - 1) {
        const baseDelay = status === 429 ? 1000 : 2000;
        const delay = Math.min(baseDelay * Math.pow(2, attempt), 16000);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw error;
    }
  }
  throw new Error('anthropicGenerateText: exhausted retries');
}

/* ── Public API ───────────────────────────────────────────────────────────── */

export async function generateText(options: GenerateTextOptions): Promise<{ text: string }> {
  if (options.model.provider === 'anthropic') {
    const jsonMode = options.providerOptions?.openai?.responseFormat?.type === 'json_object';
    const text = await anthropicGenerateText(
      options.model.client as Anthropic,
      options.model.model,
      options.messages,
      jsonMode,
      options.temperature,
      options.maxTokens,
    );
    return { text };
  }

  // OpenAI path
  const client = options.model.client as OpenAI;
  const responseFormat = options.providerOptions?.openai?.responseFormat;
  const completion = await client.chat.completions.create({
    model: options.model.model,
    messages: normalizeMessages(options.messages),
    ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
    ...(typeof options.maxTokens === 'number' ? { max_completion_tokens: options.maxTokens } : {}),
    ...(responseFormat?.type === 'json_object'
      ? { response_format: { type: 'json_object' as const } }
      : {}),
  } as any);

  return {
    text: extractMessageText(completion.choices?.[0]?.message),
  };
}

export async function generateObject<TSchema extends z.ZodTypeAny>(options: GenerateObjectOptions<TSchema>): Promise<{ object: z.infer<TSchema> }> {
  if (options.mode && options.mode !== 'json') {
    throw new Error(`Unsupported object generation mode: ${options.mode}`);
  }

  if (options.model.provider === 'anthropic') {
    const text = await anthropicGenerateText(
      options.model.client as Anthropic,
      options.model.model,
      options.messages,
      true,
      options.temperature,
      options.maxTokens,
    );
    const parsed = parseJsonObjectText(text);
    return { object: options.schema.parse(parsed) };
  }

  // OpenAI path
  const client = options.model.client as OpenAI;

  if (options.model.supportsStructuredOutputs) {
    const completion = await client.chat.completions.create({
      model: options.model.model,
      messages: normalizeMessages(options.messages),
      ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
      ...(typeof options.maxTokens === 'number' ? { max_completion_tokens: options.maxTokens } : {}),
      response_format: zodResponseFormat(options.schema, 'response'),
    } as any);

    const text = extractMessageText(completion.choices?.[0]?.message);
    if (!text.trim()) {
      throw new NoObjectGeneratedError('No object generated in structured-output mode.');
    }
    const parsed = parseJsonObjectText(text);

    return {
      object: options.schema.parse(parsed),
    };
  }

  const completion = await client.chat.completions.create({
    model: options.model.model,
    messages: normalizeMessages(options.messages),
    ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
    ...(typeof options.maxTokens === 'number' ? { max_completion_tokens: options.maxTokens } : {}),
    response_format: { type: 'json_object' as const },
  } as any);

  const text = extractMessageText(completion.choices?.[0]?.message);
  const parsed = parseJsonObjectText(text);
  return {
    object: options.schema.parse(parsed),
  };
}

export async function generateToolCalls<TParsed>(
  options: GenerateToolCallsOptions<TParsed>,
): Promise<{ calls: ParsedToolCall<TParsed>[]; text: string; providerMessage?: ToolCallProviderMessage }> {
  const toolMap = new Map(options.tools.map((tool) => [tool.name, tool] as const));

  if (options.model.provider === 'anthropic') {
    const client = options.model.client as Anthropic;
    const { system, messages } = splitSystemAndMessages(options.messages);
    const response = await client.messages.create({
      model: options.model.model,
      max_tokens: options.maxTokens ?? 8192,
      ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
      ...(system ? { system } : {}),
      messages: messages as any,
      tools: options.tools.map((tool) => ({
        name: tool.name,
        ...(tool.description ? { description: tool.description } : {}),
        input_schema: tool.parameters,
      })),
      tool_choice: { type: 'any' },
    } as any);

    const text = extractAnthropicText(response);
    try {
      const toolUses = Array.isArray(response.content)
        ? response.content.filter((block: any) => block?.type === 'tool_use' && typeof block?.name === 'string') as
            Array<{ id?: string; name: string; input: unknown }>
        : [];
      if (toolUses.length === 0) {
        const preview = clipPreview(text);
        throw new NoToolCallGeneratedError(preview ? `No tool call generated: ${preview}` : 'No tool call generated.');
      }

      return {
        calls: toolUses.map((toolUse) => {
          const tool = toolMap.get(toolUse.name);
          if (!tool) {
            throw new NoToolCallGeneratedError(`Unknown tool call generated: ${toolUse.name}`);
          }
        return {
          name: tool.name,
          input: tool.parse(toolUse.input),
          ...(typeof toolUse.id === 'string' && toolUse.id.trim()
            ? { providerToolUseId: toolUse.id }
            : {}),
        };
      }),
        text,
        providerMessage: Array.isArray(response.content)
          ? { anthropicAssistantContent: response.content as AnthropicMessageContentBlock[] }
          : undefined,
      };
    } catch (error) {
      if (!NoToolCallGeneratedError.isInstance(error) && !NoObjectGeneratedError.isInstance(error)) {
        throw error;
      }
      // Middle fallback: try parsing XML tool calls from text before re-prompting
      const xmlResult = tryParseXmlToolCalls(text, toolMap);
      if (xmlResult) {
        return xmlResult;
      }
      return generateToolCallsJsonFallback(options, toolMap, clipPreview(text));
    }
  }

  const client = options.model.client as OpenAI;
  const completion = await client.chat.completions.create({
    model: options.model.model,
    messages: normalizeMessages(options.messages),
    ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
    ...(typeof options.maxTokens === 'number' ? { max_completion_tokens: options.maxTokens } : {}),
    tools: options.tools.map((tool) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        ...(tool.description ? { description: tool.description } : {}),
        parameters: tool.parameters,
        strict: true,
      },
    })),
    tool_choice: 'required',
    parallel_tool_calls: true,
  } as any);

  const text = extractMessageText(completion.choices?.[0]?.message);
  try {
    const toolCalls = (completion.choices?.[0]?.message?.tool_calls || []).filter(
      (toolCall: any) => toolCall?.type === 'function',
    );
    if (toolCalls.length === 0) {
      const preview = clipPreview(text);
      throw new NoToolCallGeneratedError(preview ? `No tool call generated: ${preview}` : 'No tool call generated.');
    }

    return {
      calls: toolCalls.map((toolCall: any) => {
        const tool = toolMap.get(toolCall.function.name);
        if (!tool) {
          throw new NoToolCallGeneratedError(`Unknown tool call generated: ${toolCall.function.name}`);
        }
        return {
          name: tool.name,
          input: tool.parse(parseJsonObjectText(toolCall.function.arguments)),
        };
      }),
      text,
    };
  } catch (error) {
    if (!NoToolCallGeneratedError.isInstance(error) && !NoObjectGeneratedError.isInstance(error)) {
      throw error;
    }
    // Middle fallback: try parsing XML tool calls from text before re-prompting
    const xmlResult = tryParseXmlToolCalls(text, toolMap);
    if (xmlResult) {
      return xmlResult;
    }
    return generateToolCallsJsonFallback(options, toolMap, clipPreview(text));
  }
}
