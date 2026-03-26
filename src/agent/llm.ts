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

export interface LanguageModelV1 {
  provider: LLMProvider;
  client: OpenAI | Anthropic;
  model: string;
  supportsStructuredOutputs: boolean;
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
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

export class NoObjectGeneratedError extends Error {
  static isInstance(error: unknown): error is NoObjectGeneratedError {
    return error instanceof NoObjectGeneratedError;
  }

  constructor(message = 'No object generated') {
    super(message);
    this.name = 'NoObjectGeneratedError';
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
        if (part && typeof part === 'object' && 'text' in part && typeof (part as { text?: unknown }).text === 'string') {
          return (part as { text: string }).text;
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
    content: typeof message.content === 'string' ? message.content : String(message.content ?? ''),
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
): { system: string | undefined; messages: Array<{ role: 'user' | 'assistant'; content: string }> } {
  let system: string | undefined;
  const out: Array<{ role: 'user' | 'assistant'; content: string }> = [];
  for (const m of messages) {
    if (m.role === 'system') {
      system = system ? `${system}\n\n${m.content}` : m.content;
    } else {
      out.push({ role: m.role, content: typeof m.content === 'string' ? m.content : String(m.content ?? '') });
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
        messages: anthropicMsgs,
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
