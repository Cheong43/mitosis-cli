/**
 * LLM factory and helpers built directly on the official OpenAI SDK.
 *
 * Auth priority: HMAC > Gateway > plain API key.
 */
import crypto from 'crypto';
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';

export interface OpenAIChatSettings {
  structuredOutputs?: boolean;
}

export interface LanguageModelV1 {
  client: OpenAI;
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
    client: buildClient(cfg),
    model: cfg.model,
    supportsStructuredOutputs: Boolean(chatSettings?.structuredOutputs),
  };
}

export async function generateText(options: GenerateTextOptions): Promise<{ text: string }> {
  const responseFormat = options.providerOptions?.openai?.responseFormat;
  const completion = await options.model.client.chat.completions.create({
    model: options.model.model,
    messages: normalizeMessages(options.messages),
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

  if (options.model.supportsStructuredOutputs) {
    const completion = await options.model.client.chat.completions.create({
      model: options.model.model,
      messages: normalizeMessages(options.messages),
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

  const completion = await options.model.client.chat.completions.create({
    model: options.model.model,
    messages: normalizeMessages(options.messages),
    response_format: { type: 'json_object' as const },
  } as any);

  const text = extractMessageText(completion.choices?.[0]?.message);
  const parsed = parseJsonObjectText(text);
  return {
    object: options.schema.parse(parsed),
  };
}
