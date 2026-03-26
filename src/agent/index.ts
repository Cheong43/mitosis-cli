import {
  NoObjectGeneratedError,
  buildLanguageModel,
  buildAnthropicLanguageModel,
  generateObject,
  generateText,
  type LanguageModelV1,
} from './llm.js';
import { MempediaClient } from '../mempedia/client.js';
import { ToolAction } from '../mempedia/types.js';
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { z } from 'zod';
import { resolveCodeCliRoot } from '../config/projectPaths.js';
import { AgentRuntime, createRuntime, BeamSearchAgentRuntime, CallbackApprovalEngine } from '../runtime/index.js';
import type { ApprovalCallback } from '../runtime/index.js';
import type { AgentBranchState, BranchSynthesisInput } from '../runtime/agent/AgentRuntime.js';
import { TOOLS, TOOL_NAMES } from '../tools/definitions.js';
import { installWorkspaceSkillFromLibraryViaCli } from '../mempedia/cli.js';
import { MemoryClassifierAgent } from './MemoryClassifierAgent.js';
import { fileURLToPath } from 'url';
import { extractJsonishCandidates, extractJsonishText, parseJsonish } from '../utils/jsonish.js';
import {
  loadWorkspaceSkills,
  renderSkillCatalog,
  renderSkillGuidance,
  resolveSkillsByName,
  type SkillRecord,
} from '../skills/router.js';
import {
  computeContextBudget,
  estimateTokens,
  estimateTranscriptTokens,
  compressTranscript,
  getCompressionLevel,
  type ContextBudgetResult,
} from './contextBudget.js';
import { SessionCompressor, isRunExhausted } from './sessionCompressor.js';
import { PlannerToolAdapter } from './PlannerToolAdapter.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const PlannerToolNameSchema = z.enum(TOOL_NAMES);

const PlannerToolCallSchema = z.object({
  name: PlannerToolNameSchema,
  arguments: z.record(z.any()).default({}),
  goal: z.string().trim().min(1).max(240).optional(),
});

// OpenAI structured outputs require all fields to be present.
const PlannerToolCallStructuredSchema = z.object({
  name: PlannerToolNameSchema,
  arguments: z.record(z.any()),
  goal: z.string().trim().min(1).max(240).nullable(),
});

const PlannerBranchSchema = z.object({
  label: z.string().trim().min(1).max(80),
  goal: z.string().trim().min(1).max(240),
  why: z.string().trim().min(1).max(240).optional(),
  priority: z.number().min(0).max(1).optional(),
});

const PlannerBranchStructuredSchema = z.object({
  label: z.string().trim().min(1).max(80),
  goal: z.string().trim().min(1).max(240),
  why: z.string().trim().min(1).max(240).nullable(),
  priority: z.number().min(0).max(1).nullable(),
});

const PlannerDecisionSchema = z.object({
  kind: z.enum(['tool', 'branch', 'final', 'skills']),
  thought: z.string().trim().min(1),
  tool_calls: z.array(PlannerToolCallSchema).optional(),
  branches: z.array(PlannerBranchSchema).optional(),
  skills_to_load: z.array(z.string().trim().min(1)).max(2).optional(),
  final_answer: z.string().optional(),
  completion_summary: z.string().trim().min(1).max(280).optional(),
});

const PlannerDecisionStructuredSchema = z.object({
  kind: z.enum(['tool', 'branch', 'final', 'skills']),
  thought: z.string().trim().min(1),
  tool_calls: z.array(PlannerToolCallStructuredSchema).nullable(),
  branches: z.array(PlannerBranchStructuredSchema).nullable(),
  skills_to_load: z.array(z.string().trim().min(1)).max(2).nullable(),
  final_answer: z.string().nullable(),
  completion_summary: z.string().trim().min(1).max(280).nullable(),
});

type PlannerDecision = z.infer<typeof PlannerDecisionSchema>;

const ContextSelectionSchema = z.object({
  relevant_node_ids: z.array(z.string()).max(4).default([]),
  rationale: z.string().trim().min(1).max(280).optional(),
});

type ContextSelection = z.infer<typeof ContextSelectionSchema>;

interface BranchTranscriptMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ContextCandidate {
  nodeId: string;
  searchScore: number;
  markdown: string;
  preview: string;
}

interface RetrievedContext {
  contextText: string;
  recalledNodeIds: string[];
  selectedNodeIds: string[];
  rationale: string;
}

interface BranchState {
  id: string;
  parentId: string | null;
  depth: number;
  label: string;
  goal: string;
  priority: number;
  steps: number;
  transcript: BranchTranscriptMessage[];
  savedNodeIds: string[];
  completionSummary?: string;
  finalAnswer?: string;
}

export interface TraceEvent {
  type: 'thought' | 'action' | 'observation' | 'error';
  content: string;
  metadata?: {
    branchId?: string;
    parentBranchId?: string | null;
    branchLabel?: string;
    depth?: number;
    step?: number;
    toolName?: string;
    [key: string]: any;
  };
}

export interface AgentConfig {
  apiKey: string;
  baseURL?: string;
  model?: string;
  memoryApiKey?: string;
  memoryBaseURL?: string;
  memoryModel?: string;
  gatewayApiKey?: string;
  memoryGatewayApiKey?: string;
  hmacAccessKey?: string;
  hmacSecretKey?: string;
  memoryHmacAccessKey?: string;
  memoryHmacSecretKey?: string;
  anthropicAuthToken?: string;
  anthropicBaseURL?: string;
  anthropicModel?: string;
}

export interface AgentRunOptions {
  conversationId?: string;
  agentId?: string;
  sessionId?: string;
  /** Interactive approval callback for governance `ask` decisions. */
  onApproval?: ApprovalCallback;
}

interface PerfEntry {
  label: string;
  ms: number;
}

interface ConversationTurn {
  user: string;
  assistant: string;
}

interface MemoryExtraction {
  user_preferences: Array<{ topic: string; preference: string; evidence: string }>;
  agent_skills: Array<{ skill_id: string; title: string; content: string; tags: string[] }>;
  atomic_knowledge: Array<{ keyword: string; summary: string; description: string; evolution: string; relations: string[] }>;
}

interface MemorySaveJob {
  input: string;
  traces: TraceEvent[];
  answer: string;
  reason: string;
  focus?: string;
  savePreferences: boolean;
  saveSkills: boolean;
  saveAtomic: boolean;
  saveEpisodic: boolean;
  branchId?: string;
}

interface StructuredRelationInput {
  target: string;
  label?: string;
  weight?: number;
}

interface StructuredSavePayload {
  requestedNodeId: string;
  title: string;
  summary: string;
  body: string;
  facts: Record<string, string>;
  evidence: string[];
  relations: StructuredRelationInput[];
  source: string;
  comparableText: string;
}

type MempediaActionSender = (action: ToolAction) => Promise<any>;

export class Agent {
  private readonly projectRoot: string;
  private readonly codeCliRoot: string;
  private openai: LanguageModelV1;
  private beamPlannerStructuredOpenai: LanguageModelV1;
  private beamPlannerCompatOpenai: LanguageModelV1;
  private memoryOpenai: LanguageModelV1;
  private beamStructuredResponseFormatSupported = true;
  private jsonObjectResponseFormatSupported = true;
  private mempedia: MempediaClient;
  private model: string;
  private memoryModel: string;
  private interactionCounter: number;
  private readonly maxConversationTurns: number;
  private readonly conversationTurnsByConversation: Map<string, ConversationTurn[]>;
  private onBackgroundTaskCallback: ((task: string, status: 'started' | 'completed') => void) | null = null;
  private saveQueue: MemorySaveJob[] = [];
  private saveInProgress = false;
  private saveCurrentPromise: Promise<void> | null = null;
  private savePendingDrain = false;
  private readonly extractionMaxChars: number;
  private readonly autoLinkEnabled: boolean;
  private readonly autoLinkMaxNodes: number;
  private readonly autoLinkLimit: number;
  private readonly memoryTaskTimeoutMs: number;
  private readonly memoryExtractTimeoutMs: number;
  private readonly memoryActionTimeoutMs: number;
  private readonly memoryLogPath: string;
  private readonly conversationLogDir: string;
  private readonly nodeConversationMapPath: string;
  private readonly relationSearchMinScore: number;
  private readonly relationSearchLimit: number;
  private readonly relationMax: number;
  private readonly branchMaxDepth: number;
  private readonly branchMaxWidth: number;
  private readonly branchMaxSteps: number;
  private readonly branchMaxCompleted: number;
  private readonly branchConcurrency: number;
  private readonly agentLlmTimeoutMs: number;
  /** When true the agent uses the beam-search loop instead of branching. */
  private readonly useBeamSearch: boolean;
  private readonly beamWidth: number;
  private readonly beamMaxDepth: number;
  private readonly beamExpansionFactor: number;
  private readonly memoryClassifier: MemoryClassifierAgent;
  /** Per-session compressor for exhausted-run carry-over. */
  private readonly sessionCompressor: SessionCompressor;

  constructor(config: AgentConfig, projectRoot: string, binaryPath?: string) {
    this.projectRoot = projectRoot;
    this.codeCliRoot = resolveCodeCliRoot(__dirname);
    this.model = config.anthropicModel || config.model || 'gpt-4o';
    this.memoryModel = config.memoryModel || this.model;

    // ── Primary LLM: prefer Anthropic if configured ────────────────────
    const useAnthropic = Boolean(config.anthropicAuthToken);
    if (useAnthropic) {
      const anthropicModel = buildAnthropicLanguageModel({
        model: this.model,
        authToken: config.anthropicAuthToken!,
        baseURL: config.anthropicBaseURL,
      });
      this.openai = anthropicModel;
      this.beamPlannerStructuredOpenai = anthropicModel;
      this.beamPlannerCompatOpenai = anthropicModel;
    } else {
      this.openai = buildLanguageModel({
        model: this.model,
        apiKey: config.apiKey,
        baseURL: config.baseURL,
        hmacAccessKey: config.hmacAccessKey,
        hmacSecretKey: config.hmacSecretKey,
        gatewayApiKey: config.gatewayApiKey,
      });
      this.beamPlannerStructuredOpenai = buildLanguageModel(
        {
          model: this.model,
          apiKey: config.apiKey,
          baseURL: config.baseURL,
          hmacAccessKey: config.hmacAccessKey,
          hmacSecretKey: config.hmacSecretKey,
          gatewayApiKey: config.gatewayApiKey,
        },
        { structuredOutputs: true }
      );
      this.beamPlannerCompatOpenai = buildLanguageModel(
        {
          model: this.model,
          apiKey: config.apiKey,
          baseURL: config.baseURL,
          hmacAccessKey: config.hmacAccessKey,
          hmacSecretKey: config.hmacSecretKey,
          gatewayApiKey: config.gatewayApiKey,
        },
        { structuredOutputs: false }
      );
    }

    // ── Memory LLM: reuse Anthropic if no separate memory config ───────
    const memoryBaseURL = config.memoryBaseURL || config.baseURL;
    const memoryAccessKey = config.memoryHmacAccessKey || config.hmacAccessKey;
    const memorySecretKey = config.memoryHmacSecretKey || config.hmacSecretKey;
    const memoryGatewayKey = config.memoryGatewayApiKey || config.gatewayApiKey;
    if (useAnthropic && !config.memoryApiKey) {
      this.memoryOpenai = buildAnthropicLanguageModel({
        model: this.memoryModel,
        authToken: config.anthropicAuthToken!,
        baseURL: config.anthropicBaseURL,
      });
    } else {
      this.memoryOpenai = buildLanguageModel({
        model: this.memoryModel,
        apiKey: config.memoryApiKey || config.apiKey,
        baseURL: memoryBaseURL,
        hmacAccessKey: memoryAccessKey,
        hmacSecretKey: memorySecretKey,
        gatewayApiKey: memoryGatewayKey,
      });
    }
    this.mempedia = new MempediaClient(projectRoot, binaryPath);
    this.interactionCounter = 0;
    this.maxConversationTurns = 5;
    this.conversationTurnsByConversation = new Map();
    const rawExtractionMaxChars = Number(process.env.MEMORY_EXTRACTION_MAX_CHARS ?? 12000);
    this.extractionMaxChars = Number.isFinite(rawExtractionMaxChars) ? Math.max(2000, Math.floor(rawExtractionMaxChars)) : 12000;
    const rawAutoLinkEnabled = String(process.env.MEMORY_AUTO_LINK_ENABLED ?? '1').toLowerCase();
    this.autoLinkEnabled = rawAutoLinkEnabled !== '0' && rawAutoLinkEnabled !== 'false' && rawAutoLinkEnabled !== 'off';
    const rawAutoLinkMaxNodes = Number(process.env.MEMORY_AUTO_LINK_MAX_NODES ?? 6);
    this.autoLinkMaxNodes = Number.isFinite(rawAutoLinkMaxNodes) ? Math.max(0, Math.min(50, Math.floor(rawAutoLinkMaxNodes))) : 6;
    const rawAutoLinkLimit = Number(process.env.MEMORY_AUTO_LINK_LIMIT ?? 5);
    this.autoLinkLimit = Number.isFinite(rawAutoLinkLimit) ? Math.max(1, Math.min(20, Math.floor(rawAutoLinkLimit))) : 5;
    const rawMemoryTaskTimeoutMs = Number(process.env.MEMORY_TASK_TIMEOUT_MS ?? 180000);
    this.memoryTaskTimeoutMs = Number.isFinite(rawMemoryTaskTimeoutMs) ? Math.max(1000, Math.floor(rawMemoryTaskTimeoutMs)) : 180000;
    const rawMemoryExtractTimeoutMs = Number(process.env.MEMORY_EXTRACT_TIMEOUT_MS ?? 90000);
    this.memoryExtractTimeoutMs = Number.isFinite(rawMemoryExtractTimeoutMs) ? Math.max(1000, Math.floor(rawMemoryExtractTimeoutMs)) : 90000;
    const rawMemoryActionTimeoutMs = Number(process.env.MEMORY_SAVE_ACTION_TIMEOUT_MS ?? 20000);
    this.memoryActionTimeoutMs = Number.isFinite(rawMemoryActionTimeoutMs) ? Math.max(1000, Math.floor(rawMemoryActionTimeoutMs)) : 20000;
    const rawRelationMinScore = Number(process.env.MEMORY_RELATION_MIN_SCORE ?? 1.2);
    this.relationSearchMinScore = Number.isFinite(rawRelationMinScore) ? Math.max(0, rawRelationMinScore) : 1.2;
    const rawRelationSearchLimit = Number(process.env.MEMORY_RELATION_SEARCH_LIMIT ?? 3);
    this.relationSearchLimit = Number.isFinite(rawRelationSearchLimit) ? Math.max(1, Math.min(10, Math.floor(rawRelationSearchLimit))) : 3;
    const rawRelationMax = Number(process.env.MEMORY_RELATION_MAX ?? 6);
    this.relationMax = Number.isFinite(rawRelationMax) ? Math.max(0, Math.min(20, Math.floor(rawRelationMax))) : 6;
    const rawBranchMaxDepth = Number(process.env.REACT_BRANCH_MAX_DEPTH ?? 2);
    this.branchMaxDepth = Number.isFinite(rawBranchMaxDepth) ? Math.max(0, Math.min(4, Math.floor(rawBranchMaxDepth))) : 2;
    const rawBranchMaxWidth = Number(process.env.REACT_BRANCH_MAX_WIDTH ?? 3);
    this.branchMaxWidth = Number.isFinite(rawBranchMaxWidth) ? Math.max(1, Math.min(5, Math.floor(rawBranchMaxWidth))) : 3;
    const rawBranchMaxSteps = Number(process.env.REACT_BRANCH_MAX_STEPS ?? 8);
    this.branchMaxSteps = Number.isFinite(rawBranchMaxSteps) ? Math.max(2, Math.min(24, Math.floor(rawBranchMaxSteps))) : 8;
    const rawBranchMaxCompleted = Number(process.env.REACT_BRANCH_MAX_COMPLETED ?? 4);
    this.branchMaxCompleted = Number.isFinite(rawBranchMaxCompleted) ? Math.max(1, Math.min(8, Math.floor(rawBranchMaxCompleted))) : 4;
    const rawBranchConcurrency = Number(process.env.REACT_BRANCH_CONCURRENCY ?? 3);
    this.branchConcurrency = Number.isFinite(rawBranchConcurrency) ? Math.max(1, Math.min(8, Math.floor(rawBranchConcurrency))) : 3;
    const rawAgentLlmTimeoutMs = Number(process.env.AGENT_LLM_TIMEOUT_MS ?? 120000);
    this.agentLlmTimeoutMs = Number.isFinite(rawAgentLlmTimeoutMs) ? Math.max(5000, Math.floor(rawAgentLlmTimeoutMs)) : 120000;
    const rawUseBeamSearch = String(process.env.REACT_BEAM_SEARCH_ENABLED ?? '0').toLowerCase();
    this.useBeamSearch = rawUseBeamSearch === '1' || rawUseBeamSearch === 'true' || rawUseBeamSearch === 'on';
    const rawBeamWidth = Number(process.env.REACT_BEAM_WIDTH ?? 3);
    this.beamWidth = Number.isFinite(rawBeamWidth) ? Math.max(1, Math.min(10, Math.floor(rawBeamWidth))) : 3;
    const rawBeamMaxDepth = Number(process.env.REACT_BEAM_MAX_DEPTH ?? 5);
    this.beamMaxDepth = Number.isFinite(rawBeamMaxDepth) ? Math.max(1, Math.min(10, Math.floor(rawBeamMaxDepth))) : 5;
    const rawBeamExpansionFactor = Number(process.env.REACT_BEAM_EXPANSION_FACTOR ?? 3);
    this.beamExpansionFactor = Number.isFinite(rawBeamExpansionFactor) ? Math.max(1, Math.min(10, Math.floor(rawBeamExpansionFactor))) : 3;
    this.memoryLogPath = path.join(projectRoot, '.mitosis', 'logs', 'mitosis_save.log');
    this.conversationLogDir = path.join(projectRoot, '.mitosis', 'conversations');
    this.nodeConversationMapPath = path.join(projectRoot, '.mitosis', 'logs', 'node_conversations.jsonl');

    this.memoryClassifier = new MemoryClassifierAgent({
      chatClient: this.memoryOpenai,
      codeCliRoot: this.codeCliRoot,
      extractionMaxChars: this.extractionMaxChars,
      memoryExtractTimeoutMs: this.memoryExtractTimeoutMs,
      memoryActionTimeoutMs: this.memoryActionTimeoutMs,
      autoLinkEnabled: this.autoLinkEnabled,
      autoLinkMaxNodes: this.autoLinkMaxNodes,
      autoLinkLimit: this.autoLinkLimit,
    });
    this.sessionCompressor = new SessionCompressor();
  }

  onBackgroundTask(callback: (task: string, status: 'started' | 'completed') => void) {
      this.onBackgroundTaskCallback = callback;
      return () => { this.onBackgroundTaskCallback = null; };
  }

  private notifyBackgroundTask(task: string, status: 'started' | 'completed') {
      if (this.onBackgroundTaskCallback) {
          this.onBackgroundTaskCallback(task, status);
      }
  }

  async start() {
    this.mempedia.start();
  }

  async sendMempediaAction(action: ToolAction): Promise<any> {
    return this.mempedia.send(action);
  }

  stop() {
    this.mempedia.stop();
  }

  private isJsonResponseFormatUnsupported(error: unknown): boolean {
    const message = String((error as any)?.message || error || '');
    return /(response_format|text\.format|structured outputs?)/i.test(message)
      && /(json_object|json_schema|json|schema)/i.test(message)
      && /(not supported|unsupported)/i.test(message);
  }

  private async generateJsonPromptText(options: {
    model: LanguageModelV1;
    messages: any;
    timeoutMs: number;
    timeoutLabel: string;
    onFallback?: () => void | Promise<void>;
  }): Promise<string> {
    const run = (messages: any, useJsonObject: boolean) => this.withTimeout(
      generateText({
        model: options.model,
        messages,
        ...(useJsonObject
          ? { providerOptions: { openai: { responseFormat: { type: 'json_object' as const } } } }
          : {}),
      }),
      options.timeoutMs,
      options.timeoutLabel,
    );

    const maybeRepairInvalidJsonReply = async (text: string, useJsonObject: boolean): Promise<string> => {
      if (this.hasParseableJsonishPayload(text) || !Array.isArray(options.messages)) {
        return text;
      }
      const repairedMessages = [
        ...options.messages,
        {
          role: 'user',
          content: 'Your previous reply was invalid because it was not exactly one JSON object. Reply again with exactly one JSON object and nothing else. No markdown fences. No prose before or after the JSON. Do not output raw shell commands or code unless they are escaped inside a JSON string value.',
        },
      ];
      const { text: repairedText } = await run(repairedMessages, useJsonObject);
      const normalized = typeof repairedText === 'string' ? repairedText : String(repairedText || '');
      return normalized.trim() ? normalized : text;
    };

    try {
      const { text } = await run(options.messages, this.jsonObjectResponseFormatSupported);
      const resolved = typeof text === 'string' ? text : String(text || '');
      // Empty response with json_object mode — retry without it (e.g., Qwen3 thinking mode
      // may return null content when response_format constrains the output).
      if (!resolved.trim() && this.jsonObjectResponseFormatSupported) {
        this.jsonObjectResponseFormatSupported = false;
        await options.onFallback?.();
        const { text: text2 } = await run(options.messages, false);
        const retried = typeof text2 === 'string' ? text2 : String(text2 || '');
        return maybeRepairInvalidJsonReply(retried, false);
      }
      return maybeRepairInvalidJsonReply(resolved, this.jsonObjectResponseFormatSupported);
    } catch (error) {
      if (!this.jsonObjectResponseFormatSupported || !this.isJsonResponseFormatUnsupported(error)) {
        throw error;
      }
      this.jsonObjectResponseFormatSupported = false;
      await options.onFallback?.();
      const { text } = await run(options.messages, false);
      const resolved = typeof text === 'string' ? text : String(text || '');
      return maybeRepairInvalidJsonReply(resolved, false);
    }
  }

  async shutdown(timeoutMs = 12000): Promise<void> {
    if (this.saveQueue.length > 0 && !this.saveInProgress) {
      this.drainSaveQueue();
    }
    const startedAt = Date.now();
    while ((this.saveInProgress || this.saveCurrentPromise) && Date.now() - startedAt < timeoutMs) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    this.stop();
  }

  private normalizeItems(items: unknown, limit: number): string[] {
    if (!Array.isArray(items)) {
      return [];
    }
    const out: string[] = [];
    for (const item of items) {
      if (typeof item !== 'string') {
        continue;
      }
      const cleaned = item.replace(/\s+/g, ' ').trim();
      if (!cleaned) {
        continue;
      }
      if (out.includes(cleaned)) {
        continue;
      }
      out.push(cleaned);
      if (out.length >= limit) {
        break;
      }
    }
    return out;
  }

  private isNoiseLine(line: string): boolean {
    const lc = line.toLowerCase();
    if (lc.length < 8) {
      return true;
    }
    return lc.includes('command executed successfully')
      || lc.includes('deprecatedwarning')
      || lc.includes('unknown tool')
      || lc.includes('initializing react agent context')
      || lc.includes('initializing branching react context')
      || lc.includes('mitosis process exited with code');
  }

  private buildNodeId(prefix: string): string {
    const now = new Date();
    const stamp = `${now.getUTCFullYear()}${String(now.getUTCMonth() + 1).padStart(2, '0')}${String(now.getUTCDate()).padStart(2, '0')}${String(now.getUTCHours()).padStart(2, '0')}${String(now.getUTCMinutes()).padStart(2, '0')}${String(now.getUTCSeconds()).padStart(2, '0')}`;
    this.interactionCounter += 1;
    return `${prefix}_${stamp}_${this.interactionCounter}`;
  }

  private toSlug(value: string): string {
    const normalized = value
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 72);
    return normalized || 'empty';
  }

  private stableNodeId(type: 'atomic', text: string): string {
    return `kg_${type}_${this.toSlug(text)}`;
  }

  async installWorkspaceSkillFromLibrary(skillId: string, overwrite = false): Promise<{ kind: string; skill_id: string; path?: string; message: string }> {
    return installWorkspaceSkillFromLibraryViaCli(__dirname, this.projectRoot, skillId, overwrite, this.codeCliRoot);
  }

  private loadSoulsMarkdown(): string {
    const candidates = [
      path.join(this.codeCliRoot, 'souls.md'),
    ];
    for (const candidate of candidates) {
      try {
        if (!fs.existsSync(candidate)) {
          continue;
        }
        const content = fs.readFileSync(candidate, 'utf-8').trim();
        if (content) {
          return this.clipText(content, 8000);
        }
      } catch {}
    }
    return '';
  }

  /**
   * Merge extracted user preferences into the project preferences markdown file.
   * Each topic is maintained under a stable heading for idempotent updates.
   */
  private mergeUserPreferencesMarkdown(
    existing: string,
    preferences: Array<{ topic: string; preference: string; evidence: string }>,
    updatedAt: string
  ): string {
    let content = existing || `# User Preferences\n\n_Last updated: ${updatedAt}_\n`;

    for (const pref of preferences) {
      const topic = pref.topic.replace(/\s+/g, ' ').trim();
      if (!topic) {
        continue;
      }
      const heading = `### ${topic}`;
      const preference = pref.preference.replace(/\s+/g, ' ').trim();
      const evidence = pref.evidence.replace(/\s+/g, ' ').trim();
      const blockLines = [
        heading,
        `- **Preference**: ${preference}`,
        evidence ? `- **Evidence**: ${evidence}` : null,
        `- _updated: ${updatedAt}_`,
        '',
      ].filter((line): line is string => line !== null);
      const block = blockLines.join('\n');
      const idx = content.indexOf(heading);
      if (idx >= 0) {
        const nextIdx = content.indexOf('\n### ', idx + 1);
        if (nextIdx >= 0) {
          content = content.slice(0, idx) + block + content.slice(nextIdx);
        } else {
          const nextSection = content.indexOf('\n## ', idx + 1);
          if (nextSection >= 0) {
            content = content.slice(0, idx) + block + content.slice(nextSection);
          } else {
            content = content.slice(0, idx) + block;
          }
        }
      } else {
        const prefSection = '## Preferences';
        if (content.includes(prefSection)) {
          const sIdx = content.indexOf(prefSection);
          const nextSection = content.indexOf('\n## ', sIdx + 1);
          if (nextSection >= 0) {
            content = content.slice(0, nextSection) + '\n' + block + '\n' + content.slice(nextSection);
          } else {
            content = content.trimEnd() + '\n\n' + block;
          }
        } else {
          content = content.trimEnd() + '\n\n## Preferences\n\n' + block;
        }
      }
    }
    return content;
  }

  private isPreferenceLine(line: string): boolean {
    const lc = line.toLowerCase();
    return lc.includes('prefer')
      || lc.includes('preference')
      || lc.includes('习惯')
      || lc.includes('偏好')
      || lc.includes('默认')
      || lc.includes('希望')
      || lc.includes('请用')
      || lc.includes('请保持');
  }

  private isValuableKnowledgeLine(line: string): boolean {
    if (this.isNoiseLine(line)) {
      return false;
    }
    const cleaned = line.trim();
    if (cleaned.length < 12) {
      return false;
    }
    const lc = cleaned.toLowerCase();
    if (lc.includes('hello') || lc.includes('hi') || lc.includes('thanks') || lc.includes('你好')) {
      return false;
    }
    return true;
  }

  private async measure<T>(
    entries: PerfEntry[] | null,
    label: string,
    work: () => Promise<T>
  ): Promise<T> {
    const start = Date.now();
    try {
      return await work();
    } finally {
      if (entries) {
        entries.push({ label, ms: Date.now() - start });
      }
    }
  }

  private async withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
    if (timeoutMs <= 0) {
      return promise;
    }
    let timer: NodeJS.Timeout | null = null;
    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timer = setTimeout(() => reject(new Error(`${label} timeout after ${timeoutMs}ms`)), timeoutMs);
        })
      ]);
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }
  }

  private normalizeSummary(summary: unknown, fallback: string): string {
    const raw = typeof summary === 'string' ? summary : '';
    const compact = raw.replace(/\s+/g, ' ').trim();
    if (compact.length >= 8) {
      return compact.slice(0, 140);
    }
    const fb = fallback.replace(/\s+/g, ' ').trim();
    if (fb.length >= 8) {
      return fb.slice(0, 140);
    }
    return `${(fb || 'memory').slice(0, 120)} summary`;
  }

  private normalizeDetails(details: unknown, fallback: string): string {
    const raw = typeof details === 'string' ? details : '';
    const compact = raw.trim();
    if (compact.length > 0) {
      return compact;
    }
    return fallback;
  }

  private normalizeOptional(details: unknown): string {
    const raw = typeof details === 'string' ? details : '';
    return raw.trim();
  }

  private yamlEscape(value: string): string {
    return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\r?\n/g, ' ').trim();
  }

  private normalizeRelations(relations: string[]): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const rel of relations) {
      const cleaned = rel.replace(/\s+/g, ' ').trim();
      if (!cleaned) {
        continue;
      }
      const slug = this.toSlug(cleaned);
      if (seen.has(slug)) {
        continue;
      }
      seen.add(slug);
      out.push(cleaned);
      if (out.length >= this.relationMax) {
        break;
      }
    }
    return out;
  }

  private async resolveRelationTargets(
    relations: string[]
  ): Promise<Array<{ label: string; target?: string }>> {
    const normalized = this.normalizeRelations(relations);
    const resolved: Array<{ label: string; target?: string }> = [];
    for (const rel of normalized) {
      let target: string | undefined;
      const directId = rel.trim();
      const maybeIds = [directId, `kg_atomic_${this.toSlug(rel)}`];
      for (const candidate of maybeIds) {
        if (!candidate || candidate.length < 2) {
          continue;
        }
        try {
          const open = await this.withTimeout(
            this.sendMempediaAction({ action: 'open_node', node_id: candidate, markdown: false }),
            this.memoryActionTimeoutMs,
            'relation open'
          );
          if (open && (open as any).kind !== 'error') {
            target = candidate;
            break;
          }
        } catch {}
      }
      if (!target) {
        try {
          const search = await this.withTimeout(
            this.sendMempediaAction({
              action: 'search_nodes',
              query: rel,
              limit: this.relationSearchLimit,
              include_highlight: false
            }),
            this.memoryActionTimeoutMs,
            'relation search'
          );
          if (search && (search as any).kind === 'search_results') {
            const results = (search as any).results || [];
            if (results.length > 0) {
              const top = results[0];
              const score = typeof top?.score === 'number' ? top.score : null;
              if (score === null || score >= this.relationSearchMinScore) {
                target = top.node_id;
              }
            }
          }
        } catch {}
      }
      resolved.push({ label: rel, target });
    }
    return resolved;
  }

  private firstSentence(text: string): string {
    const trimmed = text.replace(/\s+/g, ' ').trim();
    if (!trimmed) {
      return '';
    }
    const match = trimmed.match(/^[^。.!?\n]{12,200}[。.!?\n]/u);
    if (match) {
      return match[0].replace(/[\n\r]+/g, ' ').trim();
    }
    return trimmed.slice(0, 200);
  }

  private normalizeUserFacingAnswer(answer: string): string {
    let current = String(answer || '').trim();
    if (!current) {
      return '';
    }

    for (let depth = 0; depth < 3; depth += 1) {
      const extracted = this.extractFinalAnswerFromJsonPayload(current);
      if (!extracted || extracted === current) {
        break;
      }
      current = extracted;
    }

    return current;
  }

  private hasParseableJsonishPayload(raw: string): boolean {
    const trimmed = String(raw || '').trim();
    if (!trimmed) {
      return false;
    }

    const candidates = extractJsonishCandidates(trimmed);
    for (const candidate of candidates) {
      try {
        parseJsonish(candidate);
        return true;
      } catch {
        continue;
      }
    }
    return false;
  }

  private looksLikeNonUserFacingPlannerOutput(raw: string): boolean {
    const trimmed = String(raw || '').trim();
    if (!trimmed) {
      return false;
    }
    if (/^```[\s\S]*```$/u.test(trimmed)) {
      return true;
    }
    if (/^\s*[{[][\s\S]*[}\]]\s*$/u.test(trimmed) && !this.hasParseableJsonishPayload(trimmed)) {
      return true;
    }
    return /(^|\n)\s*(find|ls|cat|grep|rg|echo|printf|cd|pwd|npm|pnpm|yarn|cargo|python|python3|node|git|curl|BIN=|\[\[)\b/m.test(trimmed);
  }

  private buildPlannerFallbackFinalAnswer(raw: string, looksLikePromptEcho: boolean): string {
    if (looksLikePromptEcho || !raw.trim()) {
      return '抱歉，刚才内部规划结果格式异常，没有生成可展示的回答。请重试一次。';
    }
    if (this.looksLikeNonUserFacingPlannerOutput(raw)) {
      return '抱歉，刚才内部规划器输出成了命令或草稿，而不是正式答复。我没有继续把那段代码直接返回给你。请重试一次；如果你愿意，我也可以继续根据上一步结果接着处理。';
    }
    return '抱歉，刚才内部规划结果格式异常，没有生成可展示的自然语言回答。请重试一次。';
  }

  private extractFinalAnswerFromJsonPayload(raw: string): string | null {
    const trimmed = String(raw || '').trim();
    if (!trimmed) {
      return null;
    }

    const candidates = extractJsonishCandidates(trimmed);
    for (const candidate of candidates) {
      let parsed: unknown;
      try {
        parsed = parseJsonish(candidate);
      } catch {
        continue;
      }
      const obj = Array.isArray(parsed) ? (parsed[0] || {}) : parsed;
      if (!obj || typeof obj !== 'object') {
        continue;
      }

      const record = obj as Record<string, unknown>;
      const finalAnswer = [record.final_answer, record.finalAnswer, record.answer, record.content, record.message]
        .find((value) => typeof value === 'string' && value.trim().length > 0);
      if (typeof finalAnswer !== 'string') {
        continue;
      }

      const kind = typeof record.kind === 'string' ? record.kind.trim().toLowerCase() : '';
      if (kind === 'final' || candidate.trim() === trimmed) {
        return finalAnswer.trim();
      }
    }

    return null;
  }

  private collectAtomicCandidates(input: string, answer: string): string[] {
    const candidates: string[] = [];
    const seen = new Set<string>();
    const push = (value: string) => {
      const cleaned = value.replace(/\s+/g, ' ').trim();
      if (cleaned.length < 2 || cleaned.length > 80) {
        return;
      }
      if (this.isPreferenceLine(cleaned)) {
        return;
      }
      const slug = this.toSlug(cleaned);
      if (seen.has(slug)) {
        return;
      }
      seen.add(slug);
      candidates.push(cleaned);
    };

    const backtickRegex = /`([^`]{2,80})`/g;
    const quotedRegex = /"([^"]{2,80})"/g;
    const pathRegex = /\b[\w.-]+\/[\w./-]+\b/g;

    const textPool = `${answer}\n${input}`;
    let match: RegExpExecArray | null = null;
    while ((match = backtickRegex.exec(textPool))) {
      push(match[1]);
    }
    while ((match = quotedRegex.exec(textPool))) {
      push(match[1]);
    }
    while ((match = pathRegex.exec(textPool))) {
      push(match[0]);
    }

    const answerLines = answer.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    for (const line of answerLines) {
      if (line.startsWith('#')) {
        push(line.replace(/^#+\s*/, ''));
        continue;
      }
      const colonIndex = line.indexOf(':') >= 0 ? line.indexOf(':') : line.indexOf('：');
      if (colonIndex > 1 && colonIndex < 60) {
        push(line.slice(0, colonIndex));
        continue;
      }
      if (line.includes(' - ')) {
        const [left] = line.split(' - ');
        push(left);
      }
    }

    const inputLine = input.split(/\r?\n/).map((line) => line.trim()).find((line) => line.length >= 8);
    if (inputLine) {
      const trimmed = inputLine.replace(/[\p{P}\p{S}]+/gu, ' ').trim();
      push(trimmed.slice(0, 60));
    }

    return candidates.slice(0, 8);
  }

  private fallbackExtractAtomic(input: string, answer: string): Array<{ keyword: string; summary: string; description: string; evolution: string; relations: string[] }> {
    const candidates = this.collectAtomicCandidates(input, answer);
    if (candidates.length === 0) {
      return [];
    }
    const answerLines = answer
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && this.isValuableKnowledgeLine(line));
    const summarySeed = this.firstSentence(answer) || this.firstSentence(input) || '';
    const candidateSet = new Set(candidates.map((c) => c.toLowerCase()));

    return candidates.map((candidate) => {
      const candidateLower = candidate.toLowerCase();
      const matchingLines = answerLines.filter((line) => line.toLowerCase().includes(candidateLower));
      const detailsSource = matchingLines.slice(0, 3).join('\n') || answerLines.slice(0, 3).join('\n') || summarySeed || candidate;
      const summarySource = matchingLines[0] || summarySeed || candidate;
      const evolutionSource = detailsSource === summarySource ? '' : detailsSource;
      const relations = candidates
        .filter((other) => other.toLowerCase() !== candidateLower && candidateSet.has(other.toLowerCase()))
        .slice(0, 4);
      return {
        keyword: candidate,
        summary: this.normalizeSummary(summarySource, candidate),
        description: this.normalizeDetails(detailsSource, candidate),
        evolution: this.normalizeOptional(evolutionSource),
        relations
      };
    }).filter((item) => item.keyword && item.summary);
  }

  private clipText(value: string, maxChars: number): string {
    if (maxChars <= 0 || value.length <= maxChars) {
      return value;
    }
    return value.slice(value.length - maxChars);
  }

  private parseFrontmatter(markdown: string): { frontmatter: Record<string, string>; body: string } {
    const match = markdown.match(/^---\s*[\r\n]+([\s\S]*?)\s*[\r\n]+---\s*[\r\n]*/);
    if (!match) {
      return { frontmatter: {}, body: markdown };
    }
    const frontmatter: Record<string, string> = {};
    for (const line of match[1].split(/\r?\n/)) {
      const [rawKey, ...rest] = line.split(':');
      if (!rawKey || rest.length === 0) {
        continue;
      }
      frontmatter[rawKey.trim()] = rest.join(':').trim().replace(/^"|"$/g, '');
    }
    return { frontmatter, body: markdown.slice(match[0].length) };
  }

  private extractMarkdownTitle(markdown: string): string {
    const { frontmatter, body } = this.parseFrontmatter(markdown);
    if (frontmatter.title) {
      return frontmatter.title.trim();
    }
    const heading = body.match(/^#\s+(.+)$/m);
    if (heading) {
      return heading[1].trim();
    }
    return '';
  }

  private parseStructuredRelation(value: unknown): StructuredRelationInput | null {
    if (!value) {
      return null;
    }
    if (typeof value === 'object' && !Array.isArray(value)) {
      const record = value as Record<string, unknown>;
      const target = typeof record.target === 'string' ? record.target.trim() : '';
      if (!target) {
        return null;
      }
      const label = typeof record.label === 'string' && record.label.trim() ? record.label.trim() : undefined;
      const weight = Number(record.weight);
      return {
        target,
        label,
        weight: Number.isFinite(weight) ? weight : undefined,
      };
    }
    const raw = String(value).trim().replace(/^[-*+]\s+/, '');
    if (!raw) {
      return null;
    }
    if (raw.includes('|')) {
      const parts = raw.split('|').map((part) => part.trim());
      const target = parts[0];
      if (!target) {
        return null;
      }
      const label = parts[1] || undefined;
      const weight = Number(parts[2]);
      return {
        target,
        label,
        weight: Number.isFinite(weight) ? weight : undefined,
      };
    }
    const fnStyle = raw.match(/^(.*?)\((.*)\)$/);
    if (fnStyle) {
      const target = fnStyle[1].trim();
      if (!target) {
        return null;
      }
      let label: string | undefined;
      let weight: number | undefined;
      for (const part of fnStyle[2].split(',')) {
        const [key, rawValue] = part.split('=').map((item) => item?.trim());
        if (!key || !rawValue) {
          continue;
        }
        if (key === 'label') {
          label = rawValue;
        }
        if (key === 'weight') {
          const parsed = Number(rawValue);
          if (Number.isFinite(parsed)) {
            weight = parsed;
          }
        }
      }
      return { target, label, weight };
    }
    return { target: raw };
  }

  private normalizeStructuredRelations(value: unknown): StructuredRelationInput[] {
    const input = Array.isArray(value) ? value : typeof value === 'string' ? value.split(/\r?\n/) : [];
    const seen = new Set<string>();
    const out: StructuredRelationInput[] = [];
    for (const item of input) {
      const parsed = this.parseStructuredRelation(item);
      if (!parsed) {
        continue;
      }
      const key = `${this.toSlug(parsed.target)}__${this.toSlug(parsed.label || 'related')}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      out.push(parsed);
    }
    return out;
  }

  private normalizeStructuredFacts(value: unknown): Record<string, string> {
    const out: Record<string, string> = {};
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
        const factKey = String(key || '').trim();
        const factValue = typeof raw === 'string' ? raw.trim() : String(raw ?? '').trim();
        if (factKey && factValue) {
          out[factKey] = factValue;
        }
      }
      return out;
    }
    const lines = Array.isArray(value) ? value : typeof value === 'string' ? value.split(/\r?\n/) : [];
    for (const item of lines) {
      const raw = String(item || '').trim().replace(/^[-*+]\s+/, '');
      if (!raw) {
        continue;
      }
      const match = raw.match(/^([^:=]+)\s*[:=]\s*(.+)$/);
      if (!match) {
        continue;
      }
      const key = match[1].trim();
      const factValue = match[2].trim();
      if (key && factValue) {
        out[key] = factValue;
      }
    }
    return out;
  }

  private normalizeStructuredEvidence(value: unknown): string[] {
    const input = Array.isArray(value) ? value : typeof value === 'string' ? value.split(/\r?\n/) : [];
    const seen = new Set<string>();
    const out: string[] = [];
    for (const item of input) {
      const evidence = String(item || '').trim().replace(/^[-*+]\s+/, '');
      if (!evidence) {
        continue;
      }
      const key = evidence.toLowerCase();
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      out.push(evidence);
    }
    return out;
  }

  private normalizeStructuredSectionName(name: string): 'facts' | 'relations' | 'evidence' | null {
    const lower = name.trim().toLowerCase();
    if (['facts', 'fact', 'claims', 'claim'].includes(lower)) {
      return 'facts';
    }
    if (['relations', 'relation', 'links', 'link', 'related', 'related nodes', 'connections'].includes(lower)) {
      return 'relations';
    }
    if (['evidence', 'sources', 'source'].includes(lower)) {
      return 'evidence';
    }
    return null;
  }

  private extractStructuredSaveSections(body: string): {
    narrative: string;
    facts: Record<string, string>;
    evidence: string[];
    relations: StructuredRelationInput[];
  } {
    const facts: Record<string, string> = {};
    const evidence: string[] = [];
    const relations: StructuredRelationInput[] = [];
    const narrative: string[] = [];
    let current: 'facts' | 'relations' | 'evidence' | null = null;

    for (const rawLine of body.split(/\r?\n/)) {
      const trimmed = rawLine.trim();
      const heading = trimmed.match(/^#{2,3}\s+(.+)$/);
      if (heading) {
        const section = this.normalizeStructuredSectionName(heading[1]);
        if (section) {
          current = section;
          continue;
        }
        current = null;
        narrative.push(rawLine);
        continue;
      }

      if (current === 'facts') {
        const match = trimmed.replace(/^[-*+]\s+/, '').match(/^([^:=]+)\s*[:=]\s*(.+)$/);
        if (match) {
          facts[match[1].trim()] = match[2].trim();
        }
        continue;
      }

      if (current === 'relations') {
        const relation = this.parseStructuredRelation(trimmed);
        if (relation) {
          relations.push(relation);
        }
        continue;
      }

      if (current === 'evidence') {
        const item = trimmed.replace(/^[-*+]\s+/, '').trim();
        if (item) {
          evidence.push(item);
        }
        continue;
      }

      narrative.push(rawLine);
    }

    return {
      narrative: narrative.join('\n').trim(),
      facts,
      evidence: this.normalizeStructuredEvidence(evidence),
      relations: this.normalizeStructuredRelations(relations),
    };
  }

  private normalizeTextForSimilarity(value: string): string[] {
    return String(value || '')
      .toLowerCase()
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/[`#>*_\-:[\]()/\\|.,!?]/g, ' ')
      .replace(/\s+/g, ' ')
      .split(' ')
      .map((item) => item.trim())
      .filter((item) => item.length >= 3);
  }

  private lexicalOverlapScore(left: string, right: string): number {
    const leftTokens = new Set(this.normalizeTextForSimilarity(left));
    const rightTokens = new Set(this.normalizeTextForSimilarity(right));
    if (leftTokens.size === 0 || rightTokens.size === 0) {
      return 0;
    }
    let shared = 0;
    for (const token of leftTokens) {
      if (rightTokens.has(token)) {
        shared += 1;
      }
    }
    return shared / Math.max(1, Math.min(leftTokens.size, rightTokens.size));
  }

  private buildStructuredSavePayload(rawArgs: Record<string, unknown>): StructuredSavePayload {
    const requestedNodeId = String(rawArgs.node_id || '').trim();
    const explicitFacts = this.normalizeStructuredFacts(rawArgs.facts);
    const explicitEvidence = this.normalizeStructuredEvidence(rawArgs.evidence);
    const explicitRelations = this.normalizeStructuredRelations(rawArgs.relations);
    const legacyContent = typeof rawArgs.content === 'string' ? rawArgs.content.trim() : '';

    let title = typeof rawArgs.title === 'string' ? rawArgs.title.trim() : '';
    let summary = typeof rawArgs.summary === 'string' ? rawArgs.summary.trim() : '';
    let body = typeof rawArgs.body === 'string' ? rawArgs.body.trim() : '';
    let facts: Record<string, string> = { ...explicitFacts };
    let evidence = [...explicitEvidence];
    let relations = [...explicitRelations];

    if (legacyContent) {
      const { frontmatter, body: markdownBody } = this.parseFrontmatter(legacyContent);
      const structured = this.extractStructuredSaveSections(markdownBody);
      title = title || frontmatter.title?.trim() || this.extractMarkdownTitle(legacyContent) || '';
      summary = summary || frontmatter.summary?.trim() || this.firstSentence(structured.narrative || markdownBody) || '';
      body = body || structured.narrative || markdownBody.trim();
      facts = { ...structured.facts, ...facts };
      evidence = this.normalizeStructuredEvidence([...structured.evidence, ...evidence]);
      relations = this.normalizeStructuredRelations([...structured.relations, ...relations]);
    }

    title = title || requestedNodeId || this.firstSentence(body) || 'Untitled';
    summary = summary || this.firstSentence(body) || title;
    body = body || summary;

    const comparableText = [
      title,
      summary,
      body,
      ...Object.entries(facts).map(([key, value]) => `${key}: ${value}`),
      ...evidence,
      ...relations.map((relation) => `${relation.target} ${relation.label || 'related'} ${relation.weight ?? ''}`.trim()),
    ].filter(Boolean).join('\n');

    return {
      requestedNodeId,
      title,
      summary,
      body,
      facts,
      evidence,
      relations,
      source: typeof rawArgs.source === 'string' && rawArgs.source.trim() ? rawArgs.source.trim() : 'agent',
      comparableText,
    };
  }

  private deriveSaveNodeId(requestedNodeId: string, payload: StructuredSavePayload): string {
    const requested = requestedNodeId.trim();
    if (requested) {
      return requested;
    }
    const slug = this.toSlug(payload.title || this.firstSentence(payload.body) || payload.summary || 'saved_note');
    if (slug.includes('dir') || slug.includes('directory') || slug.includes('structure') || slug.includes('source')) {
      return `kg_code_${slug}`;
    }
    return `kg_doc_${slug}`;
  }

  private async guardedMempediaSave(rawArgs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.guardedMempediaSaveWithSender(rawArgs, (action) => this.sendMempediaAction(action));
  }

  private async guardedMempediaSaveWithSender(
    rawArgs: Record<string, unknown>,
    sendAction: MempediaActionSender,
  ): Promise<Record<string, unknown>> {
    const payload = this.buildStructuredSavePayload(rawArgs);
    const originalNodeId = this.deriveSaveNodeId(payload.requestedNodeId, payload);
    const title = payload.title || originalNodeId;
    let resolvedNodeId = originalNodeId;
    let redirected = false;
    let redirectReason = '';

    const titleAlignment = this.lexicalOverlapScore(originalNodeId.replace(/_/g, ' '), title);
    if (payload.requestedNodeId.trim() && titleAlignment < 0.18) {
      resolvedNodeId = this.deriveSaveNodeId('', payload);
      redirected = resolvedNodeId !== originalNodeId;
      redirectReason = `save redirected from ${originalNodeId} to ${resolvedNodeId} because requested node id does not align with markdown title`;
    }

    try {
      const existing = await sendAction({
        action: 'open_node',
        node_id: redirected ? resolvedNodeId : originalNodeId,
        markdown: true,
      });
      if ((existing as any)?.kind === 'markdown' && typeof (existing as any)?.markdown === 'string') {
        const existingMarkdown = String((existing as any).markdown || '');
        const overlap = this.lexicalOverlapScore(payload.comparableText, existingMarkdown);
        const titleSlug = this.toSlug(title);
        const idLooksAligned = resolvedNodeId.includes(titleSlug) || titleSlug.includes(this.toSlug(resolvedNodeId));
        if (overlap < 0.16 && !idLooksAligned) {
          resolvedNodeId = this.deriveSaveNodeId('', payload);
          redirected = resolvedNodeId !== originalNodeId;
          redirectReason = `save redirected from ${originalNodeId} to ${resolvedNodeId} due to low content overlap (${overlap.toFixed(2)})`;
        }
      }
    } catch {
      // missing node is acceptable; keep original node id
    }

    const result = await sendAction({
      action: 'ingest',
      node_id: resolvedNodeId,
      title: title,
      text: payload.body,
      summary: payload.summary,
      facts: Object.keys(payload.facts).length > 0 ? payload.facts : undefined,
      relations: payload.relations.length > 0 ? payload.relations : undefined,
      evidence: payload.evidence.length > 0 ? payload.evidence : undefined,
      source: payload.source,
      agent_id: 'mitosis-cli',
      reason: redirected ? `Branching ReAct task completion (${redirectReason})` : 'Branching ReAct task completion',
      importance: 1.0,
    });

    return {
      requested_node_id: payload.requestedNodeId || originalNodeId,
      resolved_node_id: resolvedNodeId,
      stored_mode: 'structured_fields',
      redirected,
      redirect_reason: redirectReason || undefined,
      result,
    };
  }

  private extractSavedNodeId(payload: string): string | null {
    try {
      const parsed = JSON.parse(payload);
      const direct = parsed?.resolved_node_id || parsed?.node_id || parsed?.version?.node_id || parsed?.result?.version?.node_id;
      return typeof direct === 'string' && direct.trim() ? direct.trim() : null;
    } catch {
      return null;
    }
  }

  private extractOriginalUserRequest(input: string): string {
    const match = input.match(/^Original user request:\n([\s\S]*?)(?:\n\nActive branch:|$)/);
    return (match?.[1] || input).trim();
  }

  private isTrivialMemoryCandidate(text: string): boolean {
    const compact = text.replace(/\s+/g, ' ').trim().toLowerCase();
    if (!compact) {
      return true;
    }
    // Single-word ultra-short inputs (e.g. "edit", "hi", "ok") carry no extractable knowledge.
    if (compact.length < 8 && !compact.includes(' ')) {
      return true;
    }
    if (compact.length <= 24 && compact.split(/\s+/).length <= 5) {
      return /^(hi|hello|hey|thanks|thank you|yo|sup|你好|嗨|哈喽|谢谢|在吗|早上好|下午好|晚上好)\b/.test(compact);
    }
    return /^(hi|hello|hey|yo|sup|你好|嗨|哈喽)\b/.test(compact);
  }

  private shouldAutoSaveAtomicKnowledge(input: string, traces: TraceEvent[], answer: string): boolean {
    const request = this.extractOriginalUserRequest(input);
    const normalizedAnswer = answer.replace(/\s+/g, ' ').trim();
    if (normalizedAnswer.length < 140) {
      return false;
    }
    if (this.isTrivialMemoryCandidate(request) || this.isTrivialMemoryCandidate(normalizedAnswer)) {
      return false;
    }
    // Don't save atomic knowledge when the answer is an error/failure message.
    if (/\b(no such file or directory|binary not found|inaccessible or invalid|failed to fetch|connection refused|permission denied|requested wikipedia url)\b/i.test(normalizedAnswer)) {
      return false;
    }
    // Don't save when the answer is a negative-result / "no info found" response.
    if (/\b(no information about|contain no information|no results? (?:found|about|for)|nothing (?:found|about|related)|not found in (?:the|this)|does not contain|had no results?|web search (?:attempt )?failed)\b/i.test(normalizedAnswer)) {
      return false;
    }
    if (normalizedAnswer.length < 300 && /^(error|an error|the system encountered|failed to|could not|unable to|sorry,|unfortunately)\b/i.test(normalizedAnswer)) {
      return false;
    }

    const groundedInProjectFiles = traces.some((trace) => {
      const toolName = String(trace.metadata?.toolName || '');
      if (toolName === 'read') {
        const args = trace.metadata?.args as Record<string, unknown> | undefined;
        const target = typeof args?.target === 'string' ? args.target.toLowerCase() : '';
        const filePath = typeof args?.path === 'string' ? args.path.toLowerCase() : '';
        if (target === 'memory' || target === 'preferences' || target === 'skills') {
          return true;
        }
        // Any non-empty local file path (not a URL) counts as workspace-grounded.
        return filePath.length > 0 && !filePath.startsWith('http');
      }
      if (toolName === 'search') {
        const args = trace.metadata?.args as Record<string, unknown> | undefined;
        const target = typeof args?.target === 'string' ? args.target.toLowerCase() : '';
        return target === 'memory' || target === 'workspace' || target === 'skills' || target === 'preferences';
      }
      return false;
    });
    if (!groundedInProjectFiles) {
      return false;
    }

    const requestLooksLikeProjectDiscovery = /(check|inspect|summari[sz]e|what(?:'s| is)? in|readme|repo|repository|project|codebase|architecture|structure|features|documentation|analy[sz]e|查看|总结|项目|仓库|代码库|结构|功能|文档)/i.test(request);
    const answerLooksReusable = /(implements|architecture|storage structure|key features|knowledge system|api|layer|module|repository|project|rust|typescript|markdown|jsonl|目录|结构|功能|实现|支持|接口|存储|分层)/i.test(normalizedAnswer)
      || /(^|\n)\d+\./.test(answer)
      || answer.includes('- ');

    return requestLooksLikeProjectDiscovery && answerLooksReusable;
  }

  private shouldAutoSaveEpisodic(input: string, answer: string): boolean {
    const request = this.extractOriginalUserRequest(input);
    const normalizedAnswer = answer.replace(/\s+/g, ' ').trim();
    if (normalizedAnswer.length < 20) {
      return false;
    }
    if (this.isTrivialMemoryCandidate(request)) {
      return false;
    }
    // Don't record episodic entries for error/failure answers — they have no useful knowledge.
    if (/\b(no such file or directory|binary not found|inaccessible or invalid|failed to fetch|connection refused|permission denied|requested wikipedia url)\b/i.test(normalizedAnswer)) {
      return false;
    }
    // Don't record episodic entries for negative-result responses.
    if (/\b(no information about|contain no information|no results? (?:found|about|for)|nothing (?:found|about|related)|not found in (?:the|this)|does not contain|had no results?|web search (?:attempt )?failed)\b/i.test(normalizedAnswer)) {
      return false;
    }
    if (normalizedAnswer.length < 400 && /^(error|an error|the system encountered|failed to|could not|unable to|sorry,|unfortunately)\b/i.test(normalizedAnswer)) {
      return false;
    }
    return true;
  }

  private isLikelyFollowUp(input: string): boolean {
    const text = input.trim().toLowerCase();
    if (!text) {
      return false;
    }
    const explicitMarkers = [
      '继续', '接着', '刚才', '上一个', '上个问题', '上述', '前面', '这个', '那个', '它', '他们', '这些',
      'that', 'those', 'it', 'them', 'previous', 'earlier', 'continue', 'follow up', 'same topic', 'also', 'then'
    ];
    if (explicitMarkers.some((marker) => text.includes(marker))) {
      return true;
    }
    const compactTokens = this.normalizeTextForSimilarity(text);
    return compactTokens.length <= 4;
  }

  private getConversationTurns(conversationId = 'default'): ConversationTurn[] {
    return this.conversationTurnsByConversation.get(conversationId) || [];
  }

  private appendConversationTurn(conversationId: string, turn: ConversationTurn): void {
    const current = this.getConversationTurns(conversationId);
    const next = [...current, turn];
    const bounded = next.length > this.maxConversationTurns ? next.slice(-this.maxConversationTurns) : next;
    this.conversationTurnsByConversation.set(conversationId, bounded);
  }

  private selectRelevantConversationTurns(input: string, conversationId = 'default'): ConversationTurn[] {
    const conversationTurns = this.getConversationTurns(conversationId);
    if (conversationTurns.length === 0) {
      return [];
    }
    const followUp = this.isLikelyFollowUp(input);
    const scored = conversationTurns
      .map((turn, index) => {
        const combined = `${turn.user}\n${turn.assistant}`;
        const overlap = this.lexicalOverlapScore(input, combined);
        const recency = (index + 1) / Math.max(1, conversationTurns.length) * 0.18;
        const score = overlap + (followUp ? recency : recency * 0.5);
        return { turn, score, index };
      })
      .sort((a, b) => b.score - a.score || b.index - a.index);

    const threshold = followUp ? 0.06 : 0.12;
    const selected = scored.filter((item) => item.score >= threshold).slice(0, followUp ? 2 : 1);
    if (selected.length > 0) {
      return selected
        .sort((a, b) => a.index - b.index)
        .map((item) => item.turn);
    }
    if (followUp) {
      return conversationTurns.slice(-1);
    }
    return [];
  }

  private buildContextCandidatePreview(markdown: string): string {
    const title = this.extractMarkdownTitle(markdown);
    const compact = this.clipText(markdown.replace(/\s+/g, ' ').trim(), 1200);
    return title ? `${title} :: ${compact}` : compact;
  }

  private heuristicSelectContextCandidates(input: string, candidates: ContextCandidate[], selectedTurns: ConversationTurn[]): ContextCandidate[] {
    const anchor = [
      input,
      ...selectedTurns.flatMap((turn) => [turn.user, turn.assistant]),
    ].join('\n');
    return candidates
      .map((candidate) => {
        const overlap = this.lexicalOverlapScore(anchor, candidate.preview || candidate.markdown);
        const score = candidate.searchScore + overlap * 2.2;
        return { candidate, score };
      })
      .sort((a, b) => b.score - a.score)
      .filter((item, index) => item.score >= 0.18 || index < 2)
      .slice(0, 3)
      .map((item) => item.candidate);
  }

  private async selectRelevantContextCandidates(
    input: string,
    candidates: ContextCandidate[],
    selectedTurns: ConversationTurn[],
    perfEntries: PerfEntry[] | null,
  ): Promise<{ selected: ContextCandidate[]; rationale: string }> {
    if (candidates.length <= 1) {
      return {
        selected: candidates,
        rationale: candidates.length === 1 ? 'Only one recalled context candidate was available.' : 'No recalled context candidates were available.',
      };
    }

    const candidateList = candidates.map((candidate, index) => [
      `${index + 1}. node_id=${candidate.nodeId}`,
      `score=${candidate.searchScore.toFixed(2)}`,
      `preview=${this.clipText(candidate.preview, 600)}`,
    ].join('\n')).join('\n\n');

    const recentTurnsText = selectedTurns.length > 0
      ? selectedTurns.map((turn, index) => `Turn ${index + 1}\nUser: ${turn.user}\nAssistant: ${turn.assistant}`).join('\n\n')
      : '(none)';

    try {
      const _contextRaw = await this.measure(perfEntries, 'context_selection', async () =>
        this.generateJsonPromptText({
          model: this.openai,
          messages: [
            {
              role: 'system',
              content: 'You are a context selector. First assume all recalled context may be noisy. Then choose only the context candidates that are directly relevant to the current user request. Return JSON only: {"relevant_node_ids":[...],"rationale":"..."}. Select at most 3 node ids.',
            },
            {
              role: 'user',
              content: `Current user request:\n${input}\n\nSelected recent conversation turns:\n${recentTurnsText}\n\nRecalled context candidates:\n${candidateList}`,
            },
          ],
          timeoutMs: this.agentLlmTimeoutMs,
          timeoutLabel: 'context_selection llm',
        })
      );
      const parsed = (() => {
        const candidates = extractJsonishCandidates(_contextRaw);
        for (const candidate of candidates) {
          try {
            const parsedCandidate = ContextSelectionSchema.parse(parseJsonish(candidate));
            if (Array.isArray(parsedCandidate.relevant_node_ids)) {
              return parsedCandidate;
            }
          } catch {}
        }
        return ContextSelectionSchema.parse(parseJsonish(_contextRaw));
      })();
      const allowed = new Set(parsed.relevant_node_ids);
      const selected = candidates.filter((candidate) => allowed.has(candidate.nodeId)).slice(0, 3);
      if (selected.length > 0) {
        return {
          selected,
          rationale: parsed.rationale || `Selected ${selected.length} context candidates after relevance filtering.`,
        };
      }
    } catch {
      // fall through to heuristic selection
    }

    const selected = this.heuristicSelectContextCandidates(input, candidates, selectedTurns);
    return {
      selected,
      rationale: `Selected ${selected.length} context candidates with heuristic relevance filtering.`,
    };
  }

  private async retrieveRelevantContext(
    input: string,
    selectedTurns: ConversationTurn[],
    perfEntries: PerfEntry[] | null,
    sendAction: MempediaActionSender = (action) => this.sendMempediaAction(action),
  ): Promise<RetrievedContext> {
    const query = [input, ...selectedTurns.map((turn) => turn.user)].filter(Boolean).join('\n');
    const searchResults = await sendAction({
      action: 'search_hybrid',
      query,
      limit: 10,
    });

    if (searchResults.kind !== 'search_results' || !Array.isArray(searchResults.results) || searchResults.results.length === 0) {
      return {
        contextText: '',
        recalledNodeIds: [],
        selectedNodeIds: [],
        rationale: 'No context candidates were recalled from Mempedia.',
      };
    }

    const recalledNodeIds = searchResults.results.map((item: any) => String(item.node_id));
    const candidates: ContextCandidate[] = [];
    for (const hit of searchResults.results.slice(0, 5)) {
      const opened = await sendAction({
        action: 'open_node',
        node_id: String(hit.node_id),
        markdown: true,
      });
      if (opened.kind !== 'markdown' || !opened.markdown) {
        continue;
      }
      candidates.push({
        nodeId: String(hit.node_id),
        searchScore: typeof hit.score === 'number' ? hit.score : 0,
        markdown: String(opened.markdown),
        preview: this.buildContextCandidatePreview(String(opened.markdown)),
      });
    }

    const { selected, rationale } = await this.selectRelevantContextCandidates(input, candidates, selectedTurns, perfEntries);
    const contextText = selected
      .map((candidate) => `--- Context: ${candidate.nodeId} (score=${candidate.searchScore.toFixed(2)}) ---\n${candidate.markdown}\n--- End Context: ${candidate.nodeId} ---`)
      .join('\n\n');

    return {
      contextText,
      recalledNodeIds,
      selectedNodeIds: selected.map((candidate) => candidate.nodeId),
      rationale,
    };
  }

  private appendMemoryLog(runId: string, phase: string, data: Record<string, unknown> = {}) {
    try {
      fs.mkdirSync(path.dirname(this.memoryLogPath), { recursive: true });
      const row = {
        ts: new Date().toISOString(),
        run_id: runId,
        phase,
        ...data
      };
      fs.appendFileSync(this.memoryLogPath, `${JSON.stringify(row)}\n`, 'utf-8');
    } catch {}
  }

  private appendConversationLog(runId: string, input: string, traces: TraceEvent[], answer: string): string {
    const conversationId = `conv_${runId}`;
    try {
      fs.mkdirSync(this.conversationLogDir, { recursive: true });
      const payload = {
        id: conversationId,
        timestamp: new Date().toISOString(),
        input,
        answer,
        traces
      };
      const filePath = path.join(this.conversationLogDir, `${conversationId}.json`);
      fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf-8');
    } catch {}
    return conversationId;
  }

  private appendNodeConversationMap(nodeId: string, conversationId: string, reason: string) {
    try {
      fs.mkdirSync(path.dirname(this.nodeConversationMapPath), { recursive: true });
      const row = {
        ts: new Date().toISOString(),
        node_id: nodeId,
        conversation_id: conversationId,
        reason
      };
      fs.appendFileSync(this.nodeConversationMapPath, `${JSON.stringify(row)}\n`, 'utf-8');
    } catch {}
  }

  private readNodeConversationRows(limit = 200): Array<{ node_id: string; conversation_id: string; reason?: string; ts?: string }> {
    try {
      if (!fs.existsSync(this.nodeConversationMapPath)) {
        return [];
      }
      const text = fs.readFileSync(this.nodeConversationMapPath, 'utf-8');
      const rows = text
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return null;
          }
        })
        .filter((row) => row && typeof row.node_id === 'string' && typeof row.conversation_id === 'string');
      return rows.slice(-Math.max(1, limit));
    } catch {
      return [];
    }
  }

  private lookupMappedConversations(nodeId: string, limit = 3): Array<{
    node_id: string;
    conversation_id: string;
    ts?: string;
    reason?: string;
    input?: string;
    answer?: string;
  }> {
    const rows = this.readNodeConversationRows(400)
      .filter((row) => row.node_id === nodeId)
      .reverse();
    const seen = new Set<string>();
    const picked: Array<{ node_id: string; conversation_id: string; ts?: string; reason?: string }> = [];
    for (const row of rows) {
      if (seen.has(row.conversation_id)) {
        continue;
      }
      seen.add(row.conversation_id);
      picked.push(row);
      if (picked.length >= Math.max(1, limit)) {
        break;
      }
    }
    return picked.map((row) => {
      const filePath = path.join(this.conversationLogDir, `${row.conversation_id}.json`);
      if (!fs.existsSync(filePath)) {
        return row;
      }
      try {
        const payload = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        return {
          ...row,
          input: this.clipText(String(payload?.input || ''), 500),
          answer: this.clipText(String(payload?.answer || ''), 500),
        };
      } catch {
        return row;
      }
    });
  }

  private async extractMemoryPayload(input: string, traces: TraceEvent[], answer: string): Promise<MemoryExtraction> {
    const traceLines = traces
      .slice(-30)
      .map((t) => `${t.type.toUpperCase()}: ${t.content}`)
      .join('\n');
    const compactInput = this.clipText(input, this.extractionMaxChars);
    const compactTraces = this.clipText(traceLines, Math.max(2000, Math.floor(this.extractionMaxChars / 2)));
    const compactAnswer = this.clipText(answer, Math.max(1000, Math.floor(this.extractionMaxChars / 3)));
    const extractionPrompt = `请按 Mempedia 最新四层模型提取长期记忆，输出必须是 JSON（不要 markdown）：
{
  "user_preferences": [
    { "topic": "偏好主题", "preference": "稳定偏好结论", "evidence": "证据摘要" }
  ],
  "agent_skills": [
    { "skill_id": "稳定技能ID", "title": "技能标题", "content": "可复用步骤", "tags": ["tag1", "tag2"] }
  ],
  "atomic_knowledge": [
    { "keyword": "核心关键词", "summary": "短摘要", "description": "完整描述", "evolution": "演进信息", "relations": ["相关项"] }
  ]
}

规则：
1. Layer 1 Core Knowledge：atomic_knowledge 必须是可独立复用的知识点。
1.1 如果内容明确来自 README、源码、配置、项目目录或其他仓库文件，并且回答总结了项目架构、模块职责、存储结构、接口能力、构建方式等稳定事实，应优先提取到 atomic_knowledge。
2. Layer 3 User Preferences：仅提取稳定偏好，不要临时状态。
3. Layer 4 Agent Skills：只提取可复用流程/策略，避免一次性日志。
4. 不要逐字复制对话；保留抽象、可复用长期知识。
5. 严禁输出寒暄、临时上下文、错误堆栈。
6. 忽略类似“Original user request”“Active branch”“Branch goal”这类调度包装文本，不要把它们当作知识点。`;

    const userPayload = `用户输入:\n${compactInput}\n\n执行轨迹:\n${compactTraces}\n\n最终回答:\n${compactAnswer}`;
    try {
      const _extractionText = await this.generateJsonPromptText({
        model: this.memoryOpenai,
        messages: [
          { role: 'system', content: extractionPrompt },
          { role: 'user', content: userPayload },
        ],
        timeoutMs: this.agentLlmTimeoutMs,
        timeoutLabel: 'memory extraction llm',
      });
      const content = _extractionText || '{}';
      const parsed = JSON.parse(content);
      const preferences = Array.isArray(parsed.user_preferences)
        ? parsed.user_preferences.map((item: any) => {
            if (typeof item === 'string') {
              const topic = item.replace(/\s+/g, ' ').trim().slice(0, 64) || 'general';
              return {
                topic,
                preference: this.normalizeSummary(item, topic),
                evidence: this.normalizeDetails(item, topic),
              };
            }
            const topic = typeof item?.topic === 'string'
              ? item.topic.replace(/\s+/g, ' ').trim().slice(0, 64)
              : '';
            const fallback = topic || item?.preference || 'general';
            return {
              topic: topic || this.toSlug(String(fallback)).slice(0, 64),
              preference: this.normalizeSummary(item?.preference, String(fallback)),
              evidence: this.normalizeDetails(item?.evidence, String(fallback)),
            };
          }).filter((x: any) => x.topic && x.preference)
        : [];
      const skills = Array.isArray(parsed.agent_skills)
        ? parsed.agent_skills.map((item: any) => {
            if (typeof item === 'string') {
              const skillId = `skill_${this.toSlug(item).slice(0, 56) || 'general'}`;
              return {
                skill_id: skillId,
                title: this.normalizeSummary(item, skillId),
                content: this.normalizeDetails(item, skillId),
                tags: ['auto'],
              };
            }
            const rawTitle = typeof item?.title === 'string' ? item.title.trim() : '';
            const rawSkillId = typeof item?.skill_id === 'string' ? item.skill_id.trim() : '';
            const fallback = rawTitle || rawSkillId || 'general_skill';
            const tags = Array.isArray(item?.tags)
              ? item.tags.map((tag: any) => String(tag || '').trim()).filter((tag: string) => tag.length > 0).slice(0, 8)
              : [];
            return {
              skill_id: rawSkillId || `skill_${this.toSlug(fallback).slice(0, 56) || 'general'}`,
              title: rawTitle || this.normalizeSummary(fallback, fallback),
              content: this.normalizeDetails(item?.content || item?.details || item?.summary, fallback),
              tags,
            };
          }).filter((x: any) => x.skill_id && x.title && x.content)
        : [];
      const atomic = Array.isArray(parsed.atomic_knowledge)
        ? parsed.atomic_knowledge.map((item: any) => {
            const keyword = typeof item?.keyword === 'string' ? item.keyword.replace(/\s+/g, ' ').trim() : '';
            if (!keyword) {
              return null;
            }
            const rawRelations = Array.isArray(item?.relations)
              ? item.relations
              : Array.isArray(item?.related_keywords)
                ? item.related_keywords
                : [];
            const relations = rawRelations
              .map((rel: any) => typeof rel === 'string' ? rel.replace(/\s+/g, ' ').trim() : '')
              .filter((rel: string) => rel.length > 0 && rel.toLowerCase() !== keyword.toLowerCase())
              .slice(0, 8);
            return {
              keyword,
              summary: this.normalizeSummary(item?.summary || item?.description, keyword),
              description: this.normalizeDetails(item?.description || item?.summary, keyword),
              evolution: this.normalizeOptional(item?.evolution || item?.details),
              relations
            };
          }).filter((x: any) => Boolean(x))
        : [];
      
      return {
        user_preferences: preferences.slice(0, 12),
        agent_skills: skills.slice(0, 12),
        atomic_knowledge: atomic.slice(0, 20) as Array<{ keyword: string; summary: string; description: string; evolution: string; relations: string[] }>
      };
  } catch (_) {
    return this.fallbackExtractMemory(input, answer);
  }
  }

  private fallbackExtractMemory(input: string, answer: string): MemoryExtraction {
    const text = `${input}\n${answer}`;
    const preferences: Array<{ topic: string; preference: string; evidence: string }> = [];
    const skills: Array<{ skill_id: string; title: string; content: string; tags: string[] }> = [];

    const habitRegex = /(偏好|喜欢|习惯|不喜欢|讨厌|避免)[^。\\n]{0,120}/;
    const habitMatch = text.match(habitRegex);
    if (habitMatch) {
      const phrase = habitMatch[0].trim();
      const topic = phrase.slice(0, 32);
      preferences.push({
        topic,
        preference: this.normalizeSummary(phrase, topic),
        evidence: this.normalizeDetails(phrase, topic)
      });
    }

    const hasSteps = /步骤|流程|最佳实践|注意事项|操作方法|建议/.test(text) || /\\n\\s*\\d+\\./.test(text);
    if (hasSteps) {
      const key = this.toSlug(answer.slice(0, 64) || 'general_skill').slice(0, 56) || 'general_skill';
      skills.push({
        skill_id: `skill_${key}`,
        title: this.normalizeSummary(answer.slice(0, 120), key),
        content: this.normalizeDetails(answer.slice(0, 600), key),
        tags: ['auto'],
      });
    }

    return {
      user_preferences: preferences.slice(0, 8),
      agent_skills: skills.slice(0, 8),
      atomic_knowledge: this.fallbackExtractAtomic(input, answer)
    };
  }

  private async persistInteractionMemory(
    job: MemorySaveJob,
    perfEntries: PerfEntry[] | null
  ): Promise<void> {
    const input = job.input;
    const traces = job.traces;
    const answer = job.answer;
    this.notifyBackgroundTask('Saving memory...', 'started');
    const runId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const startedAt = Date.now();
    const conversationId = this.appendConversationLog(runId, input, traces, answer);
    this.appendMemoryLog(runId, 'memory_save_started', {
      reason: job.reason,
      focus: job.focus || '',
      branch_id: job.branchId || null,
      save_preferences: job.savePreferences,
      save_skills: job.saveSkills,
      save_atomic: job.saveAtomic,
      save_episodic: job.saveEpisodic,
      input_chars: input.length,
      traces_count: traces.length,
      answer_chars: answer.length
    });
    try {
      await this.withTimeout(
        this.measure(perfEntries, 'memory_classifier_persist', async () =>
          this.memoryClassifier.persist(job, {
            runId,
            conversationId,
            sendAction: (action) => this.sendMempediaAction(action),
            appendMemoryLog: (phase, data = {}) => this.appendMemoryLog(runId, phase, data),
            appendNodeConversationMap: (nodeId, mappedConversationId, reason) =>
              this.appendNodeConversationMap(nodeId, mappedConversationId, reason),
            resolveRelationTargets: (relations) => this.resolveRelationTargets(relations),
            mergeUserPreferencesMarkdown: (existing, preferences, updatedAt) =>
              this.mergeUserPreferencesMarkdown(existing, preferences, updatedAt),
          })
        ),
        this.memoryTaskTimeoutMs,
        'memory background task'
      );
      this.appendMemoryLog(runId, 'memory_save_done', {
        elapsed_ms: Date.now() - startedAt
      });
      this.notifyBackgroundTask('Saving memory...', 'completed');
    } catch (e: any) {
      this.appendMemoryLog(runId, 'memory_save_failed', {
        elapsed_ms: Date.now() - startedAt,
        error: String(e?.message || e || 'unknown error')
      });
      console.error('Background memory save failed:', e);
      this.notifyBackgroundTask('Saving memory...', 'completed');
    }
  }

  private drainSaveQueue() {
    if (this.saveInProgress) {
      this.savePendingDrain = true;
      return;
    }
    if (this.saveQueue.length === 0) {
      return;
    }
    const job = this.saveQueue.shift()!;
    this.saveInProgress = true;
    this.savePendingDrain = false;

    this.saveCurrentPromise = this.persistInteractionMemory(job, null);
    this.saveCurrentPromise
      .catch(() => {
        // errors are already logged inside persistInteractionMemory
      })
      .finally(() => {
        this.saveCurrentPromise = null;
        this.saveInProgress = false;
        if (this.savePendingDrain) {
          this.savePendingDrain = false;
        }
        if (this.saveQueue.length > 0) {
          this.drainSaveQueue();
        }
      });
  }

  /**
   * Returns true if the agent explicitly called the mempedia CLI during the turn
   * to write any memory layer (atomic knowledge, episodic, preferences, skills).
   * When this is the case, the automatic post-turn async memory job should be
   * skipped to avoid double-writing the same turn.
   */
  private hadExplicitMemoryWrite(traces: TraceEvent[]): boolean {
    const writeActionPattern = /"action"\s*:\s*"(agent_upsert_markdown|upsert_skill|update_user_preferences|record_episodic|write_node|save_node)"/;
    return traces.some((trace) => {
      if (trace.metadata?.toolName !== 'bash') {
        return false;
      }
      const command = String((trace.metadata?.args as Record<string, unknown>)?.command || '');
      return writeActionPattern.test(command);
    });
  }

  private scheduleMemorySave(job: MemorySaveJob) {
    this.saveQueue.push({
      ...job,
      traces: job.traces.slice(),
    });
    this.appendMemoryLog(`${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, 'memory_save_enqueued', {
      reason: job.reason,
      focus: job.focus || '',
      branch_id: job.branchId || null,
      save_preferences: job.savePreferences,
      save_skills: job.saveSkills,
      save_atomic: job.saveAtomic,
      save_episodic: job.saveEpisodic,
      queue_depth: this.saveQueue.length,
    });
    if (!this.saveInProgress) {
      this.drainSaveQueue();
    }
  }

  async run(input: string, onTrace: (event: TraceEvent) => void, options: AgentRunOptions = {}): Promise<string> {
    const perfEnabled = process.env.AGENT_PERF !== '0';
    const perfEntries: PerfEntry[] | null = perfEnabled ? [] : null;
    const traceBuffer: TraceEvent[] = [];
    const conversationId = options.conversationId || 'default';
    const runAgentId = options.agentId || 'agent-main';
    const runSessionId = options.sessionId || `agent-run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const runRuntimeHandle = createRuntime({
      projectRoot: this.projectRoot,
      agentId: runAgentId,
      sessionId: runSessionId,
      ...(options.onApproval ? { approvalEngine: new CallbackApprovalEngine(options.onApproval) } : {}),
    });
    const plannerToolAdapter = new PlannerToolAdapter({
      projectRoot: this.projectRoot,
      codeCliRoot: this.codeCliRoot,
      runtimeHandle: runRuntimeHandle,
    });
    const emitTrace = (event: TraceEvent) => {
      traceBuffer.push(event);
      onTrace(event);
    };
    emitTrace({ type: 'thought', content: 'Initializing branching ReAct context...' });

    const selectedConversationTurns = this.selectRelevantConversationTurns(input, conversationId);
    emitTrace({
      type: 'observation',
      content: selectedConversationTurns.length > 0
        ? `Selected ${selectedConversationTurns.length} relevant recent conversation turn(s) for follow-up grounding.`
        : 'Selected 0 recent conversation turns; treating this request as context-isolated.',
    });

    let context = '';
    let recalledNodeIds: string[] = [];
    let selectedNodeIds: string[] = [];
    const soulsGuidance = this.loadSoulsMarkdown();
    const availableSkills = loadWorkspaceSkills(this.projectRoot, this.codeCliRoot);
    const alwaysIncludeSkills = availableSkills.filter((s) => s.alwaysInclude);
    const localSkillsIndex = renderSkillCatalog(availableSkills);
    const rawAutoQueueMemorySave = String(process.env.AUTO_QUEUE_MEMORY_SAVE ?? '1').toLowerCase();
    const autoQueueMemorySave = rawAutoQueueMemorySave !== '0' && rawAutoQueueMemorySave !== 'false' && rawAutoQueueMemorySave !== 'off';
    let memoryQueuedThisRun = false;
    try {
      const retrieved = await this.measure(perfEntries, 'context_retrieval', async () =>
        this.retrieveRelevantContext(input, selectedConversationTurns, perfEntries)
      );
      context = retrieved.contextText;
      recalledNodeIds = retrieved.recalledNodeIds;
      selectedNodeIds = retrieved.selectedNodeIds;
      emitTrace({
        type: 'observation',
        content: `Recalled ${recalledNodeIds.length} context candidate(s); selected ${selectedNodeIds.length} relevant node(s). ${retrieved.rationale}`,
      });
    } catch (e: any) {
      console.error('Context retrieval failed:', e);
      context = 'Failed to retrieve context from Mempedia.';
    }

    // ── Session carry-over from previous exhausted runs ─────────────────
    const carryOver = this.sessionCompressor.getCarryOver(conversationId);
    if (carryOver) {
      context = `${context}\n\n--- CARRY-OVER FROM PREVIOUS EXHAUSTED RUN(S) ---\n${carryOver.text}\n--- END CARRY-OVER ---`;
      emitTrace({
        type: 'observation',
        content: `Injected carry-over from ${carryOver.runCount} previous exhausted run(s) (≈${carryOver.tokenEstimate} tokens). The agent should continue from these findings rather than re-exploring.`,
      });
    }

    const recentConversationMessages = selectedConversationTurns.flatMap((turn) => [
      { role: 'user', content: turn.user },
      { role: 'assistant', content: turn.assistant }
    ]);
    const originalUserRequest = this.extractOriginalUserRequest(input);
    const asksAboutLocalSkills = /技能/.test(originalUserRequest) || /\bskills?\b/i.test(originalUserRequest);
    const asksAboutMempedia = /mempedia/i.test(originalUserRequest) || /记忆库|知识库/.test(originalUserRequest);

    const toolCatalog = TOOLS.map((tool) => {
      const fn = (tool as any).function;
      return `- ${fn.name}: ${fn.description}\n  params: ${JSON.stringify(fn.parameters)}`;
    }).join('\n');

    const alwaysIncludeBlock = alwaysIncludeSkills.length > 0
      ? `\n\n--- ALWAYS-ACTIVE SKILL POLICIES (pre-loaded, binding on every turn) ---\n${renderSkillGuidance(alwaysIncludeSkills)}\n--- END ALWAYS-ACTIVE SKILLS ---`
      : '';

    const systemPrompt = `You are a branching ReAct agent.
Treat ReAct as a functional loop. A branch is an independent child loop with its own thought -> action -> observation state.

You only use five top-level tools:
  read   -> read semantic targets such as workspace files, memory nodes, preferences, or skills
  search -> search semantic targets such as workspace, memory, preferences, or skills
  edit   -> edit semantic targets such as workspace files, memory nodes, preferences, or skills
  bash   -> run sandboxed shell commands
  web    -> search the web (returns title+url+snippet) or fetch a page; use for external/current information

Tool target guidance:
  target=workspace   -> normal repository files
  target=memory      -> Mempedia nodes and episodic memory search
  target=preferences -> project preference markdown
  target=skills      -> local SKILL.md guidance files

Local skills come from workspace SKILL.md files under ./skills/*/SKILL.md and ./.github/skills/*/SKILL.md.
${localSkillsIndex ? `${localSkillsIndex}\n` : ''}Skills are guidance documents, not tool names. Never emit skill names such as project-discovery in tool_calls.name.
If skill guidance has been injected for the turn, treat it as internal policy only. Do not inspect the skills directory, verify skill files, or summarize skill text back to the user unless the user explicitly asked about skills.
When you save knowledge, prefer markdown-first notes that preserve concrete facts, numbers, data points, historical changes, viewpoints, evidence, and explicit uncertainties instead of terse summaries.
    Separate facts from opinions or viewpoints. Never fabricate facts; if something is uncertain, attribute it or mark it as uncertain.


Allowed JSON schema:
{
  "kind": "tool" | "branch" | "final" | "skills",
  "thought": "string",
  "skills_to_load": ["skill-name"],
  "tool_calls": [{ "name": "tool_name", "arguments": {}, "goal": "optional" }],
  "branches": [{ "label": "short label", "goal": "what this child branch should try", "why": "optional", "priority": 0.0 }],
  "final_answer": "string",
  "completion_summary": "optional short summary"
}

STRICT OUTPUT CONTRACT:
- Return exactly one JSON object.
- Do not wrap the JSON in Markdown fences.
- Do not add explanation text before or after the JSON object.
- Do not output raw shell commands, pseudo-code, or checklists as the top-level reply.
- If you need to show commands to the user, place them inside "final_answer" as a JSON string.

Rules:
1. Prefer kind="tool" when one next action is clearly best.
2. Use kind="branch" in TWO situations:
   a) Multiple materially distinct strategies — different hypotheses, search paths, or execution strategies worth exploring in parallel.
   b) Multiple independent sub-tasks — when the user's request contains 2 or more clearly separable actions that do not depend on each other's output (e.g. "fetch X and also save Y", "research A and update B"). Spawn one branch per independent sub-task so they execute in parallel instead of serially in root.
3. A branch must represent either a genuinely different strategy OR a distinct independent sub-task. Never bundle unrelated work into a single branch.
4. Never create more than ${this.branchMaxWidth} child branches in one step.
5. Prefer search before edit when the correct answer may already exist in repository files.
6. For repository discovery, prefer read and search on workspace evidence before using web.
7. Prefer relative file paths rooted at the current project. Use absolute paths only when necessary for clarity or when the tool requires them.
8. The skill catalog is lightweight; request full skill guidance only when it materially changes the next step.
9. Do not answer a project-overview question by listing local skills unless the user explicitly asked about the skill system.
10. Use bash whenever shell is the practical tool, but remember dangerous shell operations are sandboxed and should require confirmation.
11. Use web for questions about external topics, current events, APIs, libraries, or anything not answerable from workspace files. Prefer workspace evidence only for project-specific questions. When web search returns results, read the snippet field first — only fetch a page if the snippet is insufficient.
11a. web mode=search returns {title, url, snippet} — use snippets to decide which results are relevant before fetching full pages.
11b. When you need current/external information, use web proactively — do not guess or rely on training data when a quick search would give a definitive answer.
12. When writing markdown knowledge, prefer structured sections such as Facts, Data, History, Viewpoints, Relations, and Evidence when the material supports them.
13. When you finish, return kind="final" with a direct user-facing answer.
18. NEVER use \`edit target=workspace\` to write directly into raw \`.mempedia\` storage. Use semantic targets instead: \`target=memory\`, \`target=preferences\`, or \`target=skills\`.
14. SKILL REVIEW: Always-active skills are pre-loaded in this system prompt and are already binding — do NOT request them again via kind="skills". For other skills in the catalog whose description overlaps with the task, return kind="skills" to load their full guidance before the first tool call. If no additional skill is needed (or the catalog is empty), proceed directly to kind="tool" or kind="branch".
15. After loading additional skills, follow their guidance strictly alongside the pre-loaded always-active skills. Skills are binding operational policy, not optional suggestions.
16. For questions about local Mempedia contents, workspace memory, preferences, or local skills, inspect semantic local evidence first. Do not answer from generic model background knowledge.
17. If the user asks what is in Mempedia or what skills are available, prefer \`search/read\` with \`target=memory\` or \`target=skills\` before final_answer.
19. NEVER save trivial conversational exchanges (greetings, thank-you messages, short pleasantries, error/failure responses) to Mempedia core knowledge via agent_upsert_markdown. Core knowledge is reserved for durable, factual, or structured content that has lasting value. Saving a greeting or one-liner response wastes storage and pollutes search results.
${alwaysIncludeBlock ? `
${alwaysIncludeBlock}

IMPORTANT: The bash command patterns shown in the skill policies above are for use INSIDE bash tool_calls.arguments.command — they are NOT the planner response format. Your planner response must always follow the JSON schema above (kind/thought/tool_calls/etc.).
` : ''}
Available tools:
${toolCatalog}

Shared context for this request:
${context || '(no context found)'}

Selected context node ids:
${selectedNodeIds.length > 0 ? selectedNodeIds.join(', ') : '(none)'}

${soulsGuidance ? `Global souls.md guidance:
${soulsGuidance}
` : ''}`;

    // ── Dynamic context budget computation ──────────────────────────────
    const systemPromptTokens = estimateTokens(systemPrompt);
    const conversationContextTokens = estimateTranscriptTokens(recentConversationMessages);
    const memoryContextTokens = estimateTokens(context);
    const ctxBudget = computeContextBudget({
      model: this.model,
      systemPromptTokens,
      conversationContextTokens,
      memoryContextTokens,
    });
    emitTrace({
      type: 'observation',
      content: `Context budget: model=${this.model} limit=${ctxBudget.modelLimit} committed=${ctxBudget.committedTokens} residual=${ctxBudget.residualBudget} → maxSteps=${ctxBudget.maxSteps} depth=${ctxBudget.maxBranchDepth} width=${ctxBudget.maxBranchWidth} totalSteps=${ctxBudget.maxTotalSteps} transcriptChars=${ctxBudget.transcriptBudgetChars} beamDepth=${ctxBudget.beamMaxDepth} beamWidth=${ctxBudget.beamWidth}`,
    });

    // Effective parameters: take the minimum of env-configured and budget-computed values.
    const effectiveMaxSteps = Math.min(this.branchMaxSteps, ctxBudget.maxSteps);
    const effectiveMaxDepth = Math.min(this.branchMaxDepth, ctxBudget.maxBranchDepth);
    const effectiveMaxWidth = Math.min(this.branchMaxWidth, ctxBudget.maxBranchWidth);
    const effectiveMaxTotalSteps = Math.min(
      this.branchMaxSteps * this.branchMaxWidth * (this.branchMaxDepth + 1),
      ctxBudget.maxTotalSteps,
    );
    const effectiveTranscriptBudgetChars = ctxBudget.transcriptBudgetChars;
    const effectiveBeamMaxDepth = Math.min(this.beamMaxDepth, ctxBudget.beamMaxDepth);
    const effectiveBeamWidth = Math.min(this.beamWidth, ctxBudget.beamWidth);
    const effectiveBeamExpansionFactor = Math.min(this.beamExpansionFactor, ctxBudget.beamExpansionFactor);

    const extractText = (content: any): string => {
      if (typeof content === 'string') {
        return content;
      }
      if (Array.isArray(content)) {
        return content.map((item) => {
          if (typeof item === 'string') {
            return item;
          }
          if (item && typeof item === 'object' && typeof item.text === 'string') {
            return item.text;
          }
          return JSON.stringify(item);
        }).join('\n');
      }
      if (content == null) {
        return '';
      }
      return String(content);
    };

    const loadedSkillMarker = 'SKILL ROUTER LOAD:';

    const getLoadedSkillNames = (transcript: Array<{ role: string; content: string }>) => {
      // Always-active skills are pre-loaded in the system prompt; treat them as already loaded.
      const loaded = new Set<string>(alwaysIncludeSkills.map((s) => s.name.toLowerCase()));
      for (const message of transcript) {
        if (message.role !== 'assistant' || typeof message.content !== 'string') {
          continue;
        }
        if (!message.content.startsWith(loadedSkillMarker)) {
          continue;
        }
        const match = message.content.match(/Loaded skills:\s*([^\n]+)/i);
        if (!match) {
          continue;
        }
        for (const part of match[1].split(',')) {
          const normalized = part.trim().toLowerCase();
          if (normalized) {
            loaded.add(normalized);
          }
        }
      }
      return loaded;
    };

    const appendSkillGuidance = (
      transcript: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
      skillsToInject: SkillRecord[],
    ) => {
      if (skillsToInject.length === 0) {
        return transcript;
      }
      const names = skillsToInject.map((skill) => skill.name).join(', ');
      return [
        ...transcript,
        { role: 'assistant' as const, content: `${loadedSkillMarker} Loaded skills: ${names}` },
        {
          role: 'user' as const,
          content: `Skill Router loaded the following full guidance because you requested it:\n\n${renderSkillGuidance(skillsToInject)}\n\nContinue the current ReAct step and return exactly one JSON object.`
        },
      ];
    };

    const buildEvidenceFirstFallbackDecision = (): PlannerDecision | null => {
      const toolCalls: Array<z.infer<typeof PlannerToolCallSchema>> = [];
      if (asksAboutLocalSkills) {
        toolCalls.push({
          name: 'search',
          arguments: { target: 'workspace', mode: 'glob', pattern: 'skills/**/SKILL.md', limit: 20 },
          goal: 'Inspect local workspace skills before answering.',
        });
        toolCalls.push({
          name: 'search',
          arguments: { target: 'workspace', mode: 'glob', pattern: '.github/skills/**/SKILL.md', limit: 20 },
          goal: 'Inspect any GitHub-scoped workspace skills before answering.',
        });
      }
      if (asksAboutMempedia) {
        toolCalls.push({
          name: 'search',
          arguments: { target: 'workspace', mode: 'glob', pattern: '.mempedia/memory/knowledge/nodes/**/*.md', limit: 20 },
          goal: 'Inspect local Mempedia knowledge nodes before answering.',
        });
        toolCalls.push({
          name: 'search',
          arguments: { target: 'workspace', mode: 'glob', pattern: '.mempedia/memory/index/*.json', limit: 20 },
          goal: 'Inspect local Mempedia index files before answering.',
        });
      }
      if (toolCalls.length === 0) {
        return null;
      }
      return {
        kind: 'tool',
        thought: 'Inspect local workspace evidence before answering this project-specific request.',
        tool_calls: toolCalls,
      };
    };

    const traceMeta = (branch: BranchState, extra: Record<string, unknown> = {}) => ({
      branchId: branch.id,
      parentBranchId: branch.parentId,
      branchLabel: branch.label,
      depth: branch.depth,
      step: branch.steps,
      ...extra,
    });

    const emitBranchTrace = (
      type: TraceEvent['type'],
      branch: BranchState,
      content: string,
      extra: Record<string, unknown> = {}
    ) => {
      emitTrace({ type, content, metadata: traceMeta(branch, extra) });
    };

    const parseDecision = (raw: string): PlannerDecision => {
      const rawTrimmed = raw.trim();
      const candidates = extractJsonishCandidates(raw);
      let normalized: Record<string, unknown> | null = null;

      for (const candidate of candidates) {
        let parsed: unknown;
        try {
          parsed = parseJsonish(candidate);
        } catch {
          continue;
        }
        const obj = Array.isArray(parsed) ? (parsed[0] || {}) : parsed;
        if (!obj || typeof obj !== 'object') {
          continue;
        }
        const probe: Record<string, unknown> = { ...obj as Record<string, unknown> };
        const VALID_KINDS = new Set(['tool', 'branch', 'final', 'skills']);
        const rawKind = typeof probe.kind === 'string' ? probe.kind.trim().toLowerCase() : null;
        const inferredKind = rawKind && VALID_KINDS.has(rawKind)
          ? rawKind
          : rawKind && !VALID_KINDS.has(rawKind) && (Array.isArray(probe.tool_calls) || TOOL_NAMES.includes(rawKind as any))
            ? 'tool'
          : Array.isArray(probe.skills_to_load) && probe.skills_to_load.length > 0
            ? 'skills'
          : typeof probe.final_answer === 'string' || typeof probe.finalAnswer === 'string' || typeof probe.answer === 'string' || typeof probe.content === 'string'
            ? 'final'
            : Array.isArray(probe.tool_calls)
              ? 'tool'
              : Array.isArray(probe.branches)
                ? 'branch'
                : null;
        if (!inferredKind) {
          continue;
        }
        probe.kind = inferredKind;
        if (inferredKind === 'final') {
          const finalAnswer = [probe.final_answer, probe.finalAnswer, probe.answer, probe.content]
            .find((value) => typeof value === 'string' && value.trim().length > 0);
          if (typeof finalAnswer !== 'string') {
            continue;
          }
          probe.final_answer = this.normalizeUserFacingAnswer(finalAnswer);
        }
        if (inferredKind === 'tool' && (!Array.isArray(probe.tool_calls) || probe.tool_calls.length === 0)) {
          continue;
        }
        if (inferredKind === 'branch' && (!Array.isArray(probe.branches) || probe.branches.length === 0)) {
          continue;
        }
        if (inferredKind === 'skills' && (!Array.isArray(probe.skills_to_load) || probe.skills_to_load.length === 0)) {
          continue;
        }
        normalized = probe;
        break;
      }

      if (!normalized) {
        const looksLikePromptEcho = /Original user request:|Current branch state:|MEMPEDIA_BINARY_PATH|list_skills|record_episodic/i.test(rawTrimmed)
          || (/agent_upsert_markdown/i.test(rawTrimmed) && !/"name"\s*:\s*"bash"/i.test(rawTrimmed));
        // Log raw output to stderr for diagnosis.
        process.stderr.write(`[parseDecision] no valid structured decision (looksLikePromptEcho=${looksLikePromptEcho}). Raw preview: ${rawTrimmed.slice(0, 400)}\n`);
        // Detect mempedia-action JSON output: model confused by skill examples and
        // output {"action":"agent_upsert_markdown",...} instead of planner JSON.
        // Wrap it as a bash tool call so the action still executes.
        if (!looksLikePromptEcho && rawTrimmed.startsWith('{')) {
          try {
            const maybeAction = JSON.parse(rawTrimmed) as Record<string, unknown>;
            if (typeof maybeAction.action === 'string' && maybeAction.action.length > 0) {
              return PlannerDecisionSchema.parse({
                kind: 'tool',
                thought: `Model output a mempedia action directly; wrapping as bash CLI call.`,
                tool_calls: [{
                  name: 'bash',
                  arguments: {
                    command: `BIN="\${MEMPEDIA_BINARY_PATH:-./target/debug/mempedia}"\n[[ -x "$BIN" ]] || BIN=./target/release/mempedia\nprintf '%s' ${JSON.stringify(rawTrimmed)} | "$BIN" --project "$PWD" --stdin`,
                  },
                }],
              });
            }
          } catch { /* not JSON, fall through */ }
        }
        const evidenceFirstDecision = buildEvidenceFirstFallbackDecision();
        if (evidenceFirstDecision && !looksLikePromptEcho) {
          emitTrace({
            type: 'observation',
            content: 'Planner returned unstructured output for a project-specific request; forcing local evidence inspection before answering.',
          });
          return PlannerDecisionSchema.parse(evidenceFirstDecision);
        }
        return PlannerDecisionSchema.parse({
          kind: 'final',
          thought: 'Planner returned no valid structured decision; using fallback final answer.',
          final_answer: this.buildPlannerFallbackFinalAnswer(rawTrimmed, looksLikePromptEcho),
        });
      }

      const normalizedRecord: Record<string, unknown> = normalized;

      if (typeof normalizedRecord.kind !== 'string') {
        if (typeof normalizedRecord.final_answer === 'string') {
          normalizedRecord.kind = 'final';
        } else if (Array.isArray(normalizedRecord.tool_calls)) {
          normalizedRecord.kind = 'tool';
        } else if (Array.isArray(normalizedRecord.skills_to_load)) {
          normalizedRecord.kind = 'skills';
        } else if (Array.isArray(normalizedRecord.branches)) {
          normalizedRecord.kind = 'branch';
        } else {
          normalizedRecord.kind = 'final';
        }
      }

      if (typeof normalizedRecord.thought !== 'string' || !normalizedRecord.thought.trim()) {
        const kind = String(normalizedRecord.kind || 'tool');
        if (kind === 'tool' && Array.isArray(normalizedRecord.tool_calls)) {
          const names = normalizedRecord.tool_calls
            .map((call: any) => (call && typeof call.name === 'string' ? call.name : 'tool'))
            .filter((name: string) => name.length > 0)
            .slice(0, 3)
            .join(', ');
          normalizedRecord.thought = names
            ? `Call tools for progress: ${names}`
            : 'Call tool to gather required context.';
        } else if (kind === 'branch') {
          normalizedRecord.thought = 'Split into distinct strategies to improve solution quality.';
        } else if (kind === 'skills') {
          normalizedRecord.thought = 'Load additional skill guidance before the next planning step.';
        } else {
          normalizedRecord.thought = 'Provide the final answer for the user.';
        }
      }

      if (normalizedRecord.kind === 'skills' && (!Array.isArray(normalizedRecord.skills_to_load) || normalizedRecord.skills_to_load.length === 0)) {
        normalizedRecord.kind = 'final';
        normalizedRecord.final_answer = '抱歉，当前没有生成有效回答。请重试一次。';
      }

      if (normalizedRecord.kind === 'final' && (typeof normalizedRecord.final_answer !== 'string' || !normalizedRecord.final_answer.trim())) {
        const fallback = [normalizedRecord.answer, normalizedRecord.content, normalizedRecord.finalAnswer]
          .find((value) => typeof value === 'string' && value.trim().length > 0);
        if (typeof fallback === 'string') {
          normalizedRecord.final_answer = fallback;
        } else {
          normalizedRecord.final_answer = '抱歉，当前没有生成有效回答。请重试一次。';
        }
      }

      // Deduplicate tool_calls: some models (e.g. MiniMax via Anthropic format)
      // occasionally return each tool call twice in the JSON response.
      if (Array.isArray(normalizedRecord.tool_calls) && normalizedRecord.tool_calls.length > 1) {
        const seen = new Set<string>();
        normalizedRecord.tool_calls = (normalizedRecord.tool_calls as Array<Record<string, unknown>>).filter((call) => {
          const key = JSON.stringify({ n: call.name, a: call.arguments });
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
      }

      return PlannerDecisionSchema.parse(normalizedRecord);
    };

    const planWithSkillRouting = async (
      initialTranscript: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
      invoke: (transcript: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>) => Promise<string>,
    ): Promise<PlannerDecision> => {
      let workingTranscript = initialTranscript;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const raw = await invoke(workingTranscript);
        const decision = parseDecision(raw);
        if (decision.kind !== 'skills') {
          return decision;
        }
        const requestedNames = Array.isArray(decision.skills_to_load) ? decision.skills_to_load : [];
        const loadedSkillNames = getLoadedSkillNames(workingTranscript);
        const skillsToInject = resolveSkillsByName(availableSkills, requestedNames, 2)
          .filter((skill) => !loadedSkillNames.has(skill.name.toLowerCase()));
        if (skillsToInject.length === 0) {
          emitTrace({
            type: 'observation',
            content: `Skill router could not resolve requested skills: ${requestedNames.join(', ') || '(none)'}.`,
          });
          return {
            kind: 'final',
            thought: 'Skill router request could not be resolved to new skills; using fallback final answer.',
            final_answer: '抱歉，当前请求的技能不可用。请重试或手动指定技能。',
          } as PlannerDecision;
        }
        emitTrace({
          type: 'observation',
          content: `Skill router loaded local guidance: ${skillsToInject.map((skill) => skill.name).join(', ')}.`,
        });
        workingTranscript = appendSkillGuidance(workingTranscript, skillsToInject);
      }
      return {
        kind: 'final',
        thought: 'Skill router exceeded load attempts; using fallback final answer.',
        final_answer: '抱歉，技能路由未能收敛为有效回答。请重试一次。',
      } as PlannerDecision;
    };

    const planWithSkillRoutingObject = async (
      initialTranscript: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
      invoke: (transcript: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>) => Promise<unknown>,
    ): Promise<PlannerDecision> => {
      let workingTranscript = initialTranscript;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const rawDecisionUnknown = await invoke(workingTranscript);
        const rawDecision = (rawDecisionUnknown && typeof rawDecisionUnknown === 'object'
          ? rawDecisionUnknown
          : {}) as Record<string, unknown>;
        const decision = PlannerDecisionSchema.parse({
          ...rawDecision,
          tool_calls: Array.isArray((rawDecision as any).tool_calls)
            ? (rawDecision as any).tool_calls.map((call: any) => ({
                ...call,
                goal: call?.goal ?? undefined,
              }))
            : undefined,
          branches: Array.isArray((rawDecision as any).branches)
            ? (rawDecision as any).branches.map((branch: any) => ({
                ...branch,
                why: branch?.why ?? undefined,
                priority: branch?.priority ?? undefined,
              }))
            : undefined,
          skills_to_load: Array.isArray((rawDecision as any).skills_to_load)
            ? (rawDecision as any).skills_to_load
            : undefined,
          final_answer: (rawDecision as any).final_answer ?? undefined,
          completion_summary: (rawDecision as any).completion_summary ?? undefined,
        });
        if (decision.kind !== 'skills') {
          return decision;
        }
        const requestedNames = Array.isArray(decision.skills_to_load) ? decision.skills_to_load : [];
        const loadedSkillNames = getLoadedSkillNames(workingTranscript);
        const skillsToInject = resolveSkillsByName(availableSkills, requestedNames, 2)
          .filter((skill) => !loadedSkillNames.has(skill.name.toLowerCase()));
        if (skillsToInject.length === 0) {
          emitTrace({
            type: 'observation',
            content: `Skill router could not resolve requested skills: ${requestedNames.join(', ') || '(none)'}.`,
          });
          return {
            kind: 'final',
            thought: 'Skill router request could not be resolved to new skills; using fallback final answer.',
            final_answer: '抱歉，当前请求的技能不可用。请重试或手动指定技能。',
          } as PlannerDecision;
        }
        emitTrace({
          type: 'observation',
          content: `Skill router loaded local guidance: ${skillsToInject.map((skill) => skill.name).join(', ')}.`,
        });
        workingTranscript = appendSkillGuidance(workingTranscript, skillsToInject);
      }
      return {
        kind: 'final',
        thought: 'Skill router exceeded load attempts; using fallback final answer.',
        final_answer: '抱歉，技能路由未能收敛为有效回答。请重试一次。',
      } as PlannerDecision;
    };

    const buildBranchMemoryJob = (
      branch: BranchState,
      reason: string,
      focus?: string,
      flags?: { savePreferences?: boolean; saveSkills?: boolean; saveAtomic?: boolean; saveEpisodic?: boolean }
    ): MemorySaveJob => {
      const branchTraces = traceBuffer.filter((event) => {
        const eventBranchId = event.metadata?.branchId;
        return typeof eventBranchId === 'string' ? eventBranchId.startsWith(branch.id) : branch.id === 'B0';
      });
      const branchSummary = branch.finalAnswer || branch.completionSummary || branch.transcript.slice(-6).map((item) => item.content).join('\n\n');
      return {
        input: `Original user request:\n${input}\n\nActive branch: ${branch.id} (${branch.label})\nBranch goal: ${branch.goal}`,
        traces: branchTraces,
        answer: branchSummary,
        reason,
        focus: focus?.trim() || branch.goal,
        savePreferences: flags?.savePreferences ?? false,
        saveSkills: flags?.saveSkills ?? false,
        saveAtomic: flags?.saveAtomic ?? false,
        saveEpisodic: flags?.saveEpisodic ?? true,
        branchId: branch.id,
      };
    };

    const buildMessages = (branch: BranchState) => {
      const compressed = compressTranscript(
        branch.transcript,
        effectiveTranscriptBudgetChars,
      );
      return [
        { role: 'system', content: systemPrompt },
        ...recentConversationMessages,
        ...compressed,
        {
          role: 'user',
          content: `Current branch state:\n- branch_id: ${branch.id}\n- parent_branch_id: ${branch.parentId || 'none'}\n- depth: ${branch.depth}/${effectiveMaxDepth}\n- label: ${branch.label}\n- goal: ${branch.goal}\n- step_budget: ${branch.steps}/${effectiveMaxSteps}\n- context_budget_residual: ${ctxBudget.residualBudget} tokens\n\nReturn exactly one JSON object. Branch (kind="branch") when (a) you have materially distinct strategies to try in parallel, or (b) the remaining work has 2+ independent sub-tasks that do not depend on each other's output. Always provide at least 2 branches when branching, or use kind="tool" / kind="final" instead.`,
        },
      ];
    };

    const executePlannerTool = async (fnName: string, args: Record<string, unknown>): Promise<string> => {
      if (!TOOL_NAMES.includes(fnName as any)) {
        return `Unknown tool: ${fnName}`;
      }
      return plannerToolAdapter.execute(fnName as (typeof TOOL_NAMES)[number], args);
    };

    const executeToolCall = async (branch: BranchState, toolCall: z.infer<typeof PlannerToolCallSchema>): Promise<string> => {
      const args = toolCall.arguments || {};
      const fnName = toolCall.name;
      emitBranchTrace('action', branch, `Calling ${fnName}${toolCall.goal ? ` — ${toolCall.goal}` : ''}`, {
        toolName: fnName,
        args,
      });

      const toolStart = Date.now();
      let result = '';
      try {
        result = await executePlannerTool(fnName, args as Record<string, unknown>);
      } catch (error: any) {
        result = `Error executing tool: ${error.message}`;
      }

      if (perfEntries) {
        perfEntries.push({ label: `tool_${branch.id}_${fnName}`, ms: Date.now() - toolStart });
      }

      const clipped = this.clipText(String(result), fnName === 'web' ? 3000 : 7000);
      const savedNodeId = fnName === 'edit' && String(args.target || '') === 'memory'
        ? this.extractSavedNodeId(clipped)
        : null;
      if (savedNodeId && !branch.savedNodeIds.includes(savedNodeId)) {
        branch.savedNodeIds.push(savedNodeId);
      }
      emitBranchTrace('observation', branch, clipped, { toolName: fnName });
      return clipped;
    };

    const finalizeFromBranch = async (branch: BranchState, reason: string): Promise<string> => {
      emitBranchTrace('thought', branch, `Forcing finalization for ${branch.id}: ${reason}`);
      const { text: _finalizeText } = await this.measure(perfEntries, `finalize_${branch.id}`, async () =>
        this.withTimeout(
          generateText({
            model: this.openai,
            messages: ([
              { role: 'system', content: `${systemPrompt}\nYou must now finish. Do not branch. Do not call tools. Return plain text only.` },
              ...recentConversationMessages,
              ...branch.transcript,
              { role: 'user', content: `Finalize branch ${branch.id}. User request:\n${input}\n\nReason: ${reason}` },
            ] as any),
          }),
          this.agentLlmTimeoutMs,
          `finalize_${branch.id} llm`
        )
      );
      return extractText(_finalizeText).trim();
    };

    const synthesizeCompletedBranches = async (branches: BranchState[]): Promise<string> => {
      if (branches.length === 1 && branches[0].finalAnswer) {
        return branches[0].finalAnswer;
      }
      emitTrace({ type: 'thought', content: `Synthesizing ${branches.length} completed branches into one final answer...` });
      const branchSummary = branches
        .map((branch) => [
          `Branch ${branch.id}`,
          `label: ${branch.label}`,
          `goal: ${branch.goal}`,
          `saved_nodes: ${branch.savedNodeIds.length ? branch.savedNodeIds.join(', ') : '(none)'}`,
          `summary: ${branch.completionSummary || '(none)'}`,
          `answer:\n${branch.finalAnswer || ''}`,
        ].join('\n'))
        .join('\n\n---\n\n');

      const { text: _synthesisText } = await this.measure(perfEntries, 'branch_synthesis', async () =>
        this.withTimeout(
          generateText({
            model: this.openai,
            messages: [
              {
                role: 'system',
                content: `You are the synthesis stage for a branching ReAct agent. Merge completed branches into the best possible final answer. Prefer correctness and directness. Mention uncertainty only when branches genuinely disagree. If you mention saved node ids, use only ids explicitly listed in saved_nodes. Do not invent node ids.${soulsGuidance ? `\n\n${soulsGuidance}` : ''}`,
              },
              {
                role: 'user',
                content: `User request:\n${input}\n\nShared context:\n${this.clipText(context || '(no shared context)', 4000)}\n\nCompleted branches:\n${branchSummary}`,
              },
            ],
          }),
          this.agentLlmTimeoutMs,
          'branch_synthesis llm'
        )
      );

      return extractText(_synthesisText).trim();
    };
    let completedBranchesForRun: Array<BranchSynthesisInput['branches'][number]> = [];
    let finalAnswer: string;

    if (this.useBeamSearch) {
      // ── Beam Search ReAct mode ──────────────────────────────────────────
      emitTrace({ type: 'thought', content: `Using beam-search mode (width=${effectiveBeamWidth}, depth=${effectiveBeamMaxDepth}, expansion=${effectiveBeamExpansionFactor}).` });
      const beamPlanner = {
        plan: async (transcript: Array<{ role: string; content: string }>) => {
          const baseTranscript = transcript as Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;

          const planFromText = async (workingTranscript: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>) =>
            planWithSkillRouting(
              workingTranscript,
              async (textTranscript) => this.measure(perfEntries, `beam_llm_plan_fallback_text`, async () =>
                this.generateJsonPromptText({
                  model: this.openai,
                  messages: textTranscript as any,
                  timeoutMs: this.agentLlmTimeoutMs,
                  timeoutLabel: 'beam_plan_fallback llm',
                  onFallback: () => emitTrace({
                    type: 'observation',
                    content: `Model ${this.model} does not support response_format=json_object; switching planner JSON prompts to plain-text mode.`,
                  }),
                })
              )
            );

          const decision = await (async () => {
            try {
              return await planWithSkillRoutingObject(
                baseTranscript,
                async (workingTranscript) => {
              const runTextFallback = async () => {
                return planFromText(workingTranscript);
              };

              const runCompatObjectMode = async () => {
                try {
                  const { object } = await this.measure(perfEntries, `beam_llm_plan_compat`, async () =>
                    this.withTimeout(
                      generateObject({
                        model: this.beamPlannerCompatOpenai,
                        messages: workingTranscript as any,
                        mode: 'json',
                        schema: PlannerDecisionStructuredSchema,
                      }),
                      this.agentLlmTimeoutMs,
                      'beam_plan_compat llm'
                    )
                  );
                  return object;
                } catch (error: any) {
                  if (this.isJsonResponseFormatUnsupported(error)) {
                    this.jsonObjectResponseFormatSupported = false;
                    emitTrace({
                      type: 'observation',
                      content: `Model ${this.model} does not support response_format=json_object; switching planner JSON prompts to plain-text mode.`,
                    });
                    return runTextFallback();
                  }
                  const shouldFallback = NoObjectGeneratedError.isInstance(error)
                    || /No object generated/i.test(String(error?.message || error));
                  if (!shouldFallback) {
                    throw error;
                  }
                  emitTrace({
                    type: 'observation',
                    content: `Beam planner compat object mode returned no object; falling back to text JSON planning. ${String(error?.message || error)}`,
                  });
                  return runTextFallback();
                }
              };

              if (this.beamStructuredResponseFormatSupported) {
                try {
                  const { object } = await this.measure(perfEntries, `beam_llm_plan`, async () =>
                    this.withTimeout(
                      generateObject({
                        model: this.beamPlannerStructuredOpenai,
                        messages: workingTranscript as any,
                        mode: 'json',
                        schema: PlannerDecisionStructuredSchema,
                      }),
                      this.agentLlmTimeoutMs,
                      'beam_plan llm'
                    )
                  );
                  return object;
                } catch (error: any) {
                  if (this.isJsonResponseFormatUnsupported(error)) {
                    this.beamStructuredResponseFormatSupported = false;
                    emitTrace({
                      type: 'observation',
                      content: `Model ${this.model} rejected schema-based structured outputs; retrying beam planner with json_object mode.`,
                    });
                    if (!this.jsonObjectResponseFormatSupported) {
                      return runTextFallback();
                    }
                    return runCompatObjectMode();
                  }
                  const shouldFallback = NoObjectGeneratedError.isInstance(error)
                    || /No object generated/i.test(String(error?.message || error));
                  if (!shouldFallback) {
                    throw error;
                  }
                  emitTrace({
                    type: 'observation',
                    content: `Beam planner object mode returned no object; falling back to text JSON planning. ${String(error?.message || error)}`,
                  });
                  return runTextFallback();
                }
              }

              if (!this.jsonObjectResponseFormatSupported) {
                return runTextFallback();
              }

              return runCompatObjectMode();
                }
              );
            } catch (error: any) {
              emitTrace({
                type: 'observation',
                content: `Beam planner object pipeline failed unexpectedly; falling back to text JSON planning. ${String(error?.message || error)}`,
              });
              return planFromText(baseTranscript);
            }
          })();

          if (decision.kind === 'tool') {
            return { kind: 'tool' as const, thought: decision.thought, toolCalls: decision.tool_calls || [] };
          }
          return { kind: 'final' as const, thought: decision.thought, content: String(decision.final_answer || ''), completionSummary: decision.completion_summary };
        },
      };

      const beamRuntime = new BeamSearchAgentRuntime({
        planner: beamPlanner,
        toolRuntime: {
          execute: async (toolName: string, args: Record<string, unknown>) => {
            const startedAt = Date.now();
            try {
              const result = await executePlannerTool(toolName, args);
              const success = !/^Error[:\s]|^ERROR:/.test(String(result));
              return {
                success,
                result,
                error: success ? undefined : String(result).replace(/^Error:\s*/i, ''),
                durationMs: Date.now() - startedAt,
              };
            } catch (error: any) {
              return {
                success: false,
                error: String(error?.message || error),
                durationMs: Date.now() - startedAt,
              };
            }
          },
          resetSession: () => runRuntimeHandle.toolRuntime.resetSession(),
        },
        beamWidth: effectiveBeamWidth,
        maxDepth: effectiveBeamMaxDepth,
        expansionFactor: effectiveBeamExpansionFactor,
        onTrace: (event) => {
          if (event.type === 'final') return;
          emitTrace({ type: event.type, content: event.content, metadata: event.metadata });
        },
      });

      finalAnswer = await beamRuntime.run(
        `Original user request:\n${input}\n\nExplore multiple strategies to solve this.`,
      );
    } else {
      // ── Branching ReAct mode (original) ─────────────────────────────────
    const runtime = new AgentRuntime({
      planner: { plan: async (transcript) => ({ kind: 'final', content: transcript.at(-1)?.content || '' }) },
      toolRuntime: {
        execute: async () => ({ success: false, error: 'unreachable tool runtime fallback', durationMs: 0 }),
        resetSession: () => runRuntimeHandle.toolRuntime.resetSession(),
      },
      maxSteps: effectiveMaxSteps,
      maxBranchDepth: effectiveMaxDepth,
      maxBranchWidth: effectiveMaxWidth,
      maxCompletedBranches: this.branchMaxCompleted,
      branchConcurrency: this.branchConcurrency,
      maxTotalSteps: effectiveMaxTotalSteps,
      planBranch: async ({ branch }) => {
        const decision = await planWithSkillRouting(
          buildMessages(branch as BranchState) as Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
          async (workingTranscript) => {
            return this.measure(perfEntries, `llm_${branch.id}_step_${branch.steps + 1}`, async () =>
              this.generateJsonPromptText({
                model: this.openai,
                messages: workingTranscript as any,
                timeoutMs: this.agentLlmTimeoutMs,
                timeoutLabel: `planBranch_${branch.id} llm`,
                onFallback: () => emitTrace({
                  type: 'observation',
                  content: `Model ${this.model} does not support response_format=json_object; switching planner JSON prompts to plain-text mode.`,
                }),
              })
            );
          }
        );
        return decision.kind === 'tool'
          ? { kind: 'tool', thought: decision.thought, toolCalls: decision.tool_calls || [] }
          : decision.kind === 'branch'
            ? { kind: 'branch', thought: decision.thought, branches: decision.branches || [] }
            : { kind: 'final', thought: decision.thought, content: String(decision.final_answer || ''), completionSummary: decision.completion_summary };
      },
      executeToolCall: async ({ branch, toolCall }) => {
        const result = await executeToolCall(branch as BranchState, toolCall as z.infer<typeof PlannerToolCallSchema>);
        return {
          toolName: toolCall.name,
          result,
          success: !/^Error[:\s]|^ERROR:/.test(result),
        };
      },
      finalizeBranch: async ({ branch, reason }) => finalizeFromBranch(branch as BranchState, reason),
      synthesizeFinal: async (inputData) => {
        completedBranchesForRun = inputData.branches;
        const branches = inputData.branches.map((branch) => ({
          id: branch.id,
          parentId: null,
          depth: 0,
          label: branch.label,
          goal: branch.goal,
          priority: 1,
          steps: 0,
          transcript: [],
          savedNodeIds: branch.savedNodeIds,
          completionSummary: branch.completionSummary,
          finalAnswer: branch.finalAnswer,
        })) as BranchState[];
        return synthesizeCompletedBranches(branches);
      },
      onTrace: (event) => {
        if (event.type === 'final') {
          return;
        }
        emitTrace({
          type: event.type,
          content: event.content,
          metadata: event.metadata,
        });
      }
    });
    finalAnswer = await runtime.run(`Original user request:\n${input}\n\nStart with the root loop. Branch (kind="branch") when (a) multiple distinct approaches are worth exploring in parallel, or (b) the request has 2 or more clearly independent sub-tasks. Always provide at least 2 branches, or do not branch.`);
    }

    // ── Session carry-over: detect exhaustion and record/clear ──────────
    if (isRunExhausted(traceBuffer, finalAnswer)) {
      const record = this.sessionCompressor.recordExhaustedRun(
        conversationId,
        input,
        traceBuffer,
        finalAnswer,
      );
      emitTrace({
        type: 'observation',
        content: `Run exhausted step budget. Compressed carry-over recorded (≈${record.tokenEstimate} tokens, ${record.toolTrace.length} tools traced). Next turn for this conversation will continue from these findings.`,
      });
    } else {
      // Normal completion — clear any accumulated carry-over.
      if (this.sessionCompressor.hasCarryOver(conversationId)) {
        this.sessionCompressor.clearCarryOver(conversationId);
        emitTrace({
          type: 'observation',
          content: 'Run completed normally. Cleared session carry-over.',
        });
      }
    }

    if (this.hadExplicitMemoryWrite(traceBuffer)) {
      memoryQueuedThisRun = true;
    }
    if (autoQueueMemorySave && !memoryQueuedThisRun) {
      const bestBranch = completedBranchesForRun[0];
      const autoBranch = bestBranch
        ? {
          id: bestBranch.id,
          parentId: null,
          depth: 0,
          label: bestBranch.label,
          goal: bestBranch.goal,
          priority: 1,
          steps: 0,
          transcript: [],
          savedNodeIds: bestBranch.savedNodeIds,
          completionSummary: bestBranch.completionSummary,
          finalAnswer: bestBranch.finalAnswer,
        }
        : {
          id: 'B0',
          parentId: null,
          depth: 0,
          label: 'root',
          goal: 'Solve the user request end-to-end.',
          priority: 1,
          steps: 0,
          transcript: [],
          savedNodeIds: [],
          finalAnswer,
        };
      const autoJob = buildBranchMemoryJob(
        autoBranch,
        'automatic post-turn four-layer memory classification',
        this.firstSentence(finalAnswer),
        {
          savePreferences: true,
          saveSkills: true,
          saveAtomic: this.shouldAutoSaveAtomicKnowledge(input, traceBuffer, finalAnswer),
          saveEpisodic: this.shouldAutoSaveEpisodic(input, finalAnswer),
        }
      );
      this.scheduleMemorySave(autoJob);
      memoryQueuedThisRun = true;
      emitTrace({
        type: 'observation',
        content: `Auto-queued independent four-layer memory classification from branch ${(bestBranch?.id) || 'B0'}.`,
      });
    }
    finalAnswer = this.normalizeUserFacingAnswer(finalAnswer);
    this.appendConversationTurn(conversationId, { user: input, assistant: finalAnswer });
    if (perfEntries && perfEntries.length > 0) {
      const totalMs = perfEntries.reduce((sum, item) => sum + item.ms, 0);
      const top = [...perfEntries]
        .sort((a, b) => b.ms - a.ms)
        .slice(0, 8)
        .map((item) => `${item.label}:${item.ms}ms`)
        .join(' | ');
      emitTrace({
        type: 'observation',
        content: `Perf total=${totalMs}ms; top=${top}`
      });
    }
    return finalAnswer;
  }
}
