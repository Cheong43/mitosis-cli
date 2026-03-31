/**
 * LLM factory and helpers built on top of the Vercel AI SDK.
 *
 * Auth priority (OpenAI-compatible): HMAC > Gateway > plain API key.
 */
import crypto from 'crypto';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import {
  NoObjectGeneratedError as AiNoObjectGeneratedError,
  generateObject as aiGenerateObject,
  generateText as aiGenerateText,
  jsonSchema,
  tool,
  type AssistantModelMessage,
  type LanguageModel as AiSdkLanguageModel,
  type ModelMessage,
} from 'ai';
import type { ProviderOptions } from '@ai-sdk/provider-utils';
import { z } from 'zod';

const OPENAI_COMPAT_PROVIDER_NAME = 'openaiCompatible';
const JSON_ONLY_INSTRUCTION = 'IMPORTANT: You MUST respond with valid JSON only. No other text.';

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

export interface OpenAIAssistantReplayContent {
  type: 'openai_assistant_message';
  message: Record<string, unknown>;
}

export interface OpenAIToolResultReplayContent {
  type: 'openai_tool_result';
  tool_call_id: string;
  content: string;
}

export type AnthropicMessageContentBlock =
  | AnthropicTextContentBlock
  | AnthropicToolUseContentBlock
  | AnthropicToolResultContentBlock
  | AnthropicThinkingContentBlock
  | Record<string, unknown>;

export type ChatMessageContent =
  | string
  | AnthropicMessageContentBlock[]
  | OpenAIAssistantReplayContent
  | OpenAIToolResultReplayContent;

export interface LanguageModelV1 {
  provider: LLMProvider;
  instance: AiSdkLanguageModel;
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
  openaiAssistantMessage?: Record<string, unknown>;
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
    return error instanceof NoObjectGeneratedError
      || error instanceof AiNoObjectGeneratedError;
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
  headers?: Record<string, string>;
  fetch?: typeof globalThis.fetch;
}

export interface AnthropicEndpointConfig {
  model: string;
  authToken: string;
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: typeof globalThis.fetch;
}

export function buildHmacFetch(
  accessKey: string,
  secretKey: string,
  upstreamFetch: typeof globalThis.fetch = fetch,
): typeof globalThis.fetch {
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
    return upstreamFetch(input, { ...init, headers });
  };
}

function extractTextParts(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }
  if (isOpenAIAssistantReplayContent(content)) {
    return extractOpenAIAssistantMessageText(content.message);
  }
  if (isOpenAIToolResultReplayContent(content)) {
    return content.content;
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

function isOpenAIAssistantReplayContent(content: unknown): content is OpenAIAssistantReplayContent {
  return Boolean(
    content
    && typeof content === 'object'
    && (content as { type?: unknown }).type === 'openai_assistant_message'
    && (content as { message?: unknown }).message
    && typeof (content as { message?: unknown }).message === 'object',
  );
}

function isOpenAIToolResultReplayContent(content: unknown): content is OpenAIToolResultReplayContent {
  return Boolean(
    content
    && typeof content === 'object'
    && (content as { type?: unknown }).type === 'openai_tool_result'
    && typeof (content as { tool_call_id?: unknown }).tool_call_id === 'string'
    && typeof (content as { content?: unknown }).content === 'string',
  );
}

function cloneJsonValue<T>(value: T): T {
  if (value == null) {
    return value;
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function extractOpenAIAssistantMessageText(message: Record<string, unknown>): string {
  const parts: string[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    parts.push(content);
  } else if (Array.isArray(content)) {
    parts.push(extractTextParts(content));
  }
  if (Array.isArray(message.reasoning_details)) {
    parts.push(JSON.stringify(message.reasoning_details));
  }
  if (Array.isArray(message.tool_calls)) {
    parts.push(JSON.stringify(message.tool_calls));
  }
  return parts.filter(Boolean).join('\n');
}

function shouldUseMiniMaxReasoningSplit(model: LanguageModelV1): boolean {
  const normalizedModel = String(model.model || '').trim().toLowerCase();
  return normalizedModel.startsWith('minimax-m2');
}

function parseJsonObjectText(text: string): unknown {
  let trimmed = text.trim();
  if (!trimmed) {
    throw new NoObjectGeneratedError('No object generated: empty response body.');
  }
  // Strip <think> tags if present
  trimmed = trimmed.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  if (!trimmed) {
    throw new NoObjectGeneratedError('No object generated: only <think> tags found.');
  }
  try {
    return JSON.parse(trimmed);
  } catch (error) {
    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      let extracted = trimmed.slice(firstBrace, lastBrace + 1);
      try {
        return JSON.parse(extracted);
      } catch {
        // Try to fix common array formatting issues
        extracted = fixArrayFormatting(extracted);
        try {
          return JSON.parse(extracted);
        } catch {
          // If still fails, return original error
        }
      }
    }
    const errorMsg = error instanceof Error ? error.message : String(error);
    throw new NoObjectGeneratedError(`No object generated: ${errorMsg}. Preview: ${trimmed.slice(0, 240)}`);
  }
}

function fixArrayFormatting(json: string): string {
  // Fix missing commas between array elements
  let fixed = json.replace(/\}\s*\{/g, '},{');
  // Fix trailing commas before closing brackets
  fixed = fixed.replace(/,(\s*[\]}])/g, '$1');
  // Fix missing closing brackets for truncated arrays
  const openBrackets = (fixed.match(/\[/g) || []).length;
  const closeBrackets = (fixed.match(/\]/g) || []).length;
  if (openBrackets > closeBrackets) {
    fixed += ']'.repeat(openBrackets - closeBrackets);
  }
  return fixed;
}

function clipPreview(text: string, maxChars = 240): string {
  return text.replace(/\s+/g, ' ').trim().slice(0, maxChars);
}

function shouldRetryToolCallsWithJsonFallback(error: unknown): boolean {
  const message = String((error as { message?: unknown })?.message || error || '').toLowerCase();
  if (!message) {
    return false;
  }
  return /(^|[^a-z])(404|not found)([^a-z]|$)/i.test(message)
    || /(tool[_ -]?calls?|tool[_ -]?choice|function[_ -]?call|native tool calls?)/i.test(message)
    || /unsupported.*(tools?|tool[_ -]?choice|function[_ -]?call)/i.test(message);
}

function buildJsonFallbackPrompt<TParsed>(
  tools: Array<ParseableFunctionTool<TParsed>>,
  priorReplyPreview: string,
): string {
  const toolSchemas = tools
    .map((entry) => `- ${entry.name}: ${JSON.stringify(entry.parameters)}`)
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
      const entry = toolMap.get(toolName);
      if (!entry) {
        throw new NoToolCallGeneratedError(`Unknown tool call generated: ${toolName}`);
      }
      const input = record.input ?? record.arguments ?? record.args ?? {};
      return {
        name: entry.name,
        input: entry.parse(input),
      };
    }),
    text: thoughtText || '',
  };
}

/**
 * Try to parse XML-formatted tool calls that some models (MiniMax, Qwen, etc.)
 * emit in the text body instead of native tool-call blocks.
 */
function tryParseXmlToolCalls<TParsed>(
  text: string,
  toolMap: Map<string, ParseableFunctionTool<TParsed>>,
): JsonFallbackParseResult<TParsed> | null {
  if (!text || !/<invoke\s+name=/i.test(text)) {
    return null;
  }

  const invokePattern = /<invoke\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/invoke>/gi;
  const rawCalls: Array<{ name: string; args: Record<string, unknown> }> = [];
  let match: RegExpExecArray | null;
  while ((match = invokePattern.exec(text)) !== null) {
    const toolName = match[1];
    const body = match[2];
    const args: Record<string, unknown> = {};
    const paramPattern = /<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/parameter>/gi;
    let paramMatch: RegExpExecArray | null;
    while ((paramMatch = paramPattern.exec(body)) !== null) {
      const key = paramMatch[1];
      let value: unknown = paramMatch[2];
      const trimmedValue = String(value).trim();
      if (/^[{\[]/.test(trimmedValue)) {
        try {
          value = JSON.parse(trimmedValue);
        } catch {
          // Leave non-JSON fragments as strings.
        }
      }
      args[key] = value;
    }
    rawCalls.push({ name: toolName, args });
  }

  if (rawCalls.length === 0) {
    return null;
  }

  const calls: ParsedToolCall<TParsed>[] = [];
  for (const rawCall of rawCalls) {
    const entry = toolMap.get(rawCall.name);
    if (!entry) {
      return null;
    }
    try {
      calls.push({
        name: entry.name,
        input: entry.parse(rawCall.args),
      });
    } catch {
      return null;
    }
  }

  const firstXmlIndex = text.search(/<(?:[a-z_][a-z0-9_-]*:)?(?:tool_call|invoke)\b/i);
  const thoughtText = firstXmlIndex > 0 ? text.slice(0, firstXmlIndex).trim() : '';

  return { calls, text: thoughtText };
}

function addJsonOnlyInstruction(messages: ChatMessage[]): ChatMessage[] {
  let injectedSystem = false;
  const nextMessages = messages.map((message) => {
    if (!injectedSystem && message.role === 'system') {
      injectedSystem = true;
      return {
        ...message,
        content: `${extractTextParts(message.content)}\n\n${JSON_ONLY_INSTRUCTION}`,
      };
    }
    return message;
  });
  if (!injectedSystem) {
    nextMessages.unshift({ role: 'system', content: JSON_ONLY_INSTRUCTION });
  }
  return nextMessages;
}

function buildOpenAICompatibleFetch(cfg: LLMEndpointConfig): typeof globalThis.fetch | undefined {
  const upstreamFetch = cfg.fetch ?? fetch;
  if (cfg.hmacAccessKey && cfg.hmacSecretKey) {
    return buildHmacFetch(cfg.hmacAccessKey, cfg.hmacSecretKey, upstreamFetch);
  }
  return cfg.fetch;
}

function buildOpenAICompatibleHeaders(cfg: LLMEndpointConfig): Record<string, string> | undefined {
  const headers: Record<string, string> = { ...(cfg.headers || {}) };
  if (cfg.gatewayApiKey) {
    headers['x-gatewat-apikey'] = `Bearer ${cfg.gatewayApiKey}`;
    headers['x-gateway-apikey'] = `Bearer ${cfg.gatewayApiKey}`;
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

function extractReasoningTextFromReplayMessage(message: Record<string, unknown>): string {
  if (typeof message.reasoning_content === 'string' && message.reasoning_content.trim()) {
    return message.reasoning_content;
  }
  if (typeof message.thinking_content === 'string' && message.thinking_content.trim()) {
    return message.thinking_content;
  }
  if (Array.isArray(message.reasoning_details)) {
    return message.reasoning_details
      .map((entry) => {
        if (!entry || typeof entry !== 'object') {
          return '';
        }
        if (typeof (entry as { text?: unknown }).text === 'string') {
          return (entry as { text: string }).text;
        }
        return JSON.stringify(entry);
      })
      .filter(Boolean)
      .join('\n');
  }
  return '';
}

function normalizeOpenAIAssistantReplayContent(
  message: Record<string, unknown>,
  openAIToolNameById: Map<string, string>,
): AssistantModelMessage {
  const parts: Exclude<AssistantModelMessage['content'], string> = [];
  const text = extractTextParts(message.content);
  if (text.trim()) {
    parts.push({ type: 'text', text });
  }

  const reasoningText = extractReasoningTextFromReplayMessage(message);
  if (reasoningText.trim()) {
    parts.push({ type: 'reasoning', text: reasoningText });
  }

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const rawCall of toolCalls) {
    if (!rawCall || typeof rawCall !== 'object') {
      continue;
    }
    const record = rawCall as Record<string, unknown>;
    const toolCallId = typeof record.id === 'string' && record.id.trim()
      ? record.id
      : '';
    const functionRecord = record.function && typeof record.function === 'object'
      ? record.function as Record<string, unknown>
      : {};
    const toolName = typeof functionRecord.name === 'string' ? functionRecord.name : '';
    if (!toolCallId || !toolName) {
      continue;
    }
    openAIToolNameById.set(toolCallId, toolName);
    const rawArguments = typeof functionRecord.arguments === 'string'
      ? functionRecord.arguments
      : '{}';
    parts.push({
      type: 'tool-call',
      toolCallId,
      toolName,
      input: parseJsonObjectText(rawArguments),
    });
  }

  return {
    role: 'assistant',
    content: parts.length > 0 ? parts : extractTextParts(message),
  };
}

function normalizeAnthropicArrayContent(
  role: 'assistant' | 'user',
  content: AnthropicMessageContentBlock[],
  anthropicToolNameById: Map<string, string>,
): ModelMessage {
  if (role === 'assistant') {
    const parts: Exclude<AssistantModelMessage['content'], string> = [];
    for (const block of content) {
      if (!block || typeof block !== 'object') {
        continue;
      }
      const type = (block as { type?: unknown }).type;
      if (type === 'text' && typeof (block as { text?: unknown }).text === 'string') {
        parts.push({ type: 'text', text: (block as { text: string }).text });
      } else if (type === 'thinking') {
        const thinkingText = typeof (block as { thinking?: unknown }).thinking === 'string'
          ? (block as { thinking: string }).thinking
          : typeof (block as { text?: unknown }).text === 'string'
            ? (block as { text: string }).text
            : '';
        if (thinkingText.trim()) {
          parts.push({ type: 'reasoning', text: thinkingText });
        }
      } else if (type === 'tool_use') {
        const toolUse = block as AnthropicToolUseContentBlock;
        anthropicToolNameById.set(toolUse.id, toolUse.name);
        parts.push({
          type: 'tool-call',
          toolCallId: toolUse.id,
          toolName: toolUse.name,
          input: cloneJsonValue(toolUse.input),
        });
      }
    }

    return {
      role: 'assistant',
      content: parts.length > 0 ? parts : extractTextParts(content),
    };
  }

  const toolResults = content
    .filter((block): block is AnthropicToolResultContentBlock => (
      Boolean(block)
      && typeof block === 'object'
      && (block as { type?: unknown }).type === 'tool_result'
      && typeof (block as { tool_use_id?: unknown }).tool_use_id === 'string'
    ))
    .map((block) => ({
      type: 'tool-result' as const,
      toolCallId: block.tool_use_id,
      toolName: anthropicToolNameById.get(block.tool_use_id) || 'tool',
      output: {
        type: block.is_error ? 'error-text' as const : 'text' as const,
        value: block.content,
      },
    }));

  if (toolResults.length > 0) {
    return {
      role: 'tool',
      content: toolResults,
    };
  }

  return {
    role: 'user',
    content: extractTextParts(content),
  };
}

function normalizeMessages(messages: ChatMessage[]): ModelMessage[] {
  const normalized: ModelMessage[] = [];
  const openAIToolNameById = new Map<string, string>();
  const anthropicToolNameById = new Map<string, string>();
  // Track whether we have already emitted the single allowed system message.
  // Models like MiniMax only allow one system role; extra system messages are
  // merged into the first one to prevent API errors (error code 2013).
  let systemEmitted = false;

  for (const message of messages) {
    if (message.role === 'system') {
      const content = extractTextParts(message.content);
      if (!systemEmitted) {
        normalized.push({ role: 'system', content });
        systemEmitted = true;
      } else {
        // Merge extra system content: find the existing system entry and append.
        const existing = normalized.find((m) => m.role === 'system') as { role: 'system'; content: string } | undefined;
        if (existing) {
          existing.content = `${existing.content}\n\n${content}`;
        }
      }
      continue;
    }

    if (message.role === 'assistant' && isOpenAIAssistantReplayContent(message.content)) {
      normalized.push(normalizeOpenAIAssistantReplayContent(message.content.message, openAIToolNameById));
      continue;
    }

    if (message.role === 'user' && isOpenAIToolResultReplayContent(message.content)) {
      normalized.push({
        role: 'tool',
        content: [{
          type: 'tool-result',
          toolCallId: message.content.tool_call_id,
          toolName: openAIToolNameById.get(message.content.tool_call_id) || 'tool',
          output: { type: 'text', value: message.content.content },
        }],
      });
      continue;
    }

    if (Array.isArray(message.content)) {
      normalized.push(normalizeAnthropicArrayContent(message.role, message.content, anthropicToolNameById));
      continue;
    }

    normalized.push({
      role: message.role,
      content: extractTextParts(message.content),
    } as ModelMessage);
  }

  return normalized;
}

function buildOpenAIProviderOptions(
  model: LanguageModelV1,
  extraOptions?: Record<string, unknown>,
): ProviderOptions | undefined {
  if (model.provider !== 'openai') {
    return undefined;
  }
  const merged = {
    ...(extraOptions || {}),
    ...(model.supportsStructuredOutputs ? {} : { strictJsonSchema: false }),
  };
  return Object.keys(merged).length > 0
    ? { [OPENAI_COMPAT_PROVIDER_NAME]: merged } as ProviderOptions
    : undefined;
}

function buildProviderMessage(
  model: LanguageModelV1,
  responseMessages: ModelMessage[],
): ToolCallProviderMessage | undefined {
  const assistantMessage = responseMessages.find((message): message is AssistantModelMessage => message.role === 'assistant');
  if (!assistantMessage) {
    return undefined;
  }

  const contentArray = typeof assistantMessage.content === 'string'
    ? [{ type: 'text' as const, text: assistantMessage.content }]
    : assistantMessage.content;
  const text = contentArray
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('');
  const reasoningText = contentArray
    .filter((part): part is { type: 'reasoning'; text: string } => part.type === 'reasoning')
    .map((part) => part.text)
    .join('\n');
  const toolCalls = contentArray
    .filter((part): part is { type: 'tool-call'; toolCallId: string; toolName: string; input: unknown } => part.type === 'tool-call')
    .map((part) => ({
      id: part.toolCallId,
      type: 'function' as const,
      function: {
        name: part.toolName,
        arguments: JSON.stringify(part.input),
      },
    }));

  if (model.provider === 'anthropic') {
    const anthropicAssistantContent: AnthropicMessageContentBlock[] = [];
    if (text.trim()) {
      anthropicAssistantContent.push({ type: 'text', text });
    }
    if (reasoningText.trim()) {
      anthropicAssistantContent.push({ type: 'thinking', thinking: reasoningText });
    }
    anthropicAssistantContent.push(...toolCalls.map((call) => ({
      type: 'tool_use' as const,
      id: call.id,
      name: call.function.name,
      input: parseJsonObjectText(call.function.arguments),
    })));
    return anthropicAssistantContent.length > 0
      ? { anthropicAssistantContent }
      : undefined;
  }

  return {
    openaiAssistantMessage: {
      role: 'assistant',
      content: text,
      ...(reasoningText.trim()
        ? {
          reasoning_details: [{ type: 'reasoning.text', text: reasoningText }],
        }
        : {}),
      ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
    },
  };
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

export function buildLanguageModel(cfg: LLMEndpointConfig, chatSettings?: OpenAIChatSettings): LanguageModelV1 {
  const provider = createOpenAICompatible({
    name: OPENAI_COMPAT_PROVIDER_NAME,
    apiKey: cfg.apiKey,
    baseURL: cfg.baseURL || 'https://api.openai.com/v1',
    headers: buildOpenAICompatibleHeaders(cfg),
    fetch: buildOpenAICompatibleFetch(cfg),
    supportsStructuredOutputs: Boolean(chatSettings?.structuredOutputs),
  });

  return {
    provider: 'openai',
    instance: provider.chatModel(cfg.model),
    model: cfg.model,
    supportsStructuredOutputs: Boolean(chatSettings?.structuredOutputs),
  };
}

export function buildAnthropicLanguageModel(cfg: AnthropicEndpointConfig): LanguageModelV1 {
  const provider = createAnthropic({
    authToken: cfg.authToken,
    ...(cfg.baseURL ? { baseURL: cfg.baseURL } : {}),
    ...(cfg.headers ? { headers: cfg.headers } : {}),
    ...(cfg.fetch ? { fetch: cfg.fetch } : {}),
  });

  return {
    provider: 'anthropic',
    instance: provider.messages(cfg.model),
    model: cfg.model,
    supportsStructuredOutputs: false,
  };
}

export async function generateText(options: GenerateTextOptions): Promise<{ text: string }> {
  const useJsonOnlyInstruction = options.providerOptions?.openai?.responseFormat?.type === 'json_object';
  const messages = normalizeMessages(useJsonOnlyInstruction ? addJsonOnlyInstruction(options.messages) : options.messages);

  const result = await aiGenerateText({
    model: options.model.instance,
    messages,
    ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
    ...(typeof options.maxTokens === 'number' ? { maxOutputTokens: options.maxTokens } : {}),
    ...(options.model.provider === 'openai'
      ? { providerOptions: buildOpenAIProviderOptions(options.model) }
      : {}),
  });

  return {
    text: result.text || result.reasoningText || '',
  };
}

export async function generateObject<TSchema extends z.ZodTypeAny>(
  options: GenerateObjectOptions<TSchema>,
): Promise<{ object: z.infer<TSchema> }> {
  if (options.mode && options.mode !== 'json') {
    throw new Error(`Unsupported object generation mode: ${options.mode}`);
  }

  try {
    const result = await aiGenerateObject({
      model: options.model.instance,
      messages: normalizeMessages(options.messages),
      schema: options.schema,
      output: 'object',
      ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
      ...(typeof options.maxTokens === 'number' ? { maxOutputTokens: options.maxTokens } : {}),
      ...(options.model.provider === 'openai'
        ? { providerOptions: buildOpenAIProviderOptions(options.model) }
        : {}),
    });

    return {
      object: options.schema.parse(result.object),
    };
  } catch (error) {
    if (NoObjectGeneratedError.isInstance(error)) {
      throw new NoObjectGeneratedError(error instanceof Error ? error.message : undefined);
    }
    throw error;
  }
}

export async function generateToolCalls<TParsed>(
  options: GenerateToolCallsOptions<TParsed>,
): Promise<{ calls: ParsedToolCall<TParsed>[]; text: string; providerMessage?: ToolCallProviderMessage }> {
  const toolMap = new Map(options.tools.map((entry) => [entry.name, entry] as const));
  const sdkTools = Object.fromEntries(options.tools.map((entry) => [
    entry.name,
    tool({
      ...(entry.description ? { description: entry.description } : {}),
      inputSchema: jsonSchema(entry.parameters),
    }),
  ]));

  const runNativeToolCallGeneration = () => aiGenerateText({
    model: options.model.instance,
    messages: normalizeMessages(options.messages),
    tools: sdkTools,
    toolChoice: 'required',
    ...(typeof options.temperature === 'number' ? { temperature: options.temperature } : {}),
    ...(typeof options.maxTokens === 'number' ? { maxOutputTokens: options.maxTokens } : {}),
    ...(options.model.provider === 'openai'
      ? {
        providerOptions: buildOpenAIProviderOptions(
          options.model,
          shouldUseMiniMaxReasoningSplit(options.model) ? { reasoning_split: true } : undefined,
        ),
      }
      : {}),
  });

  let result;
  try {
    result = await runNativeToolCallGeneration();
  } catch (error) {
    if (!shouldRetryToolCallsWithJsonFallback(error)) {
      throw error;
    }
    try {
      return await generateToolCallsJsonFallback(
        options,
        toolMap,
        clipPreview(String((error as { message?: unknown })?.message || error || '')),
      );
    } catch {
      throw error;
    }
  }

  const text = result.text || result.reasoningText || '';
  if (result.toolCalls.length === 0) {
    const xmlResult = tryParseXmlToolCalls(text, toolMap);
    if (xmlResult) {
      return xmlResult;
    }
    return generateToolCallsJsonFallback(options, toolMap, clipPreview(text));
  }

  return {
    calls: result.toolCalls.map((call) => {
      const entry = toolMap.get(call.toolName);
      if (!entry) {
        throw new NoToolCallGeneratedError(`Unknown tool call generated: ${call.toolName}`);
      }
      return {
        name: entry.name,
        input: entry.parse(call.input),
        ...(options.model.provider === 'anthropic'
          ? { providerToolUseId: call.toolCallId }
          : {}),
      };
    }),
    text,
    providerMessage: buildProviderMessage(options.model, result.response.messages),
  };
}
