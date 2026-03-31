import {
  buildLanguageModel,
  buildAnthropicLanguageModel,
  generateText,
  generateToolCalls,
  type ParseableFunctionTool,
  type LanguageModelV1,
  type ChatMessage,
  type ChatMessageContent,
  type AnthropicMessageContentBlock,
  type AnthropicToolResultContentBlock,
  type OpenAIAssistantReplayContent,
  type OpenAIToolResultReplayContent,
} from './llm.js';
import { MempediaClient } from '../mempedia/client.js';
import { ToolAction } from '../mempedia/types.js';
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { z } from 'zod';
import { resolveCodeCliRoot } from '../config/projectPaths.js';
import { AgentRuntime, createRuntime, CallbackApprovalEngine } from '../runtime/index.js';
import type { ApprovalCallback } from '../runtime/index.js';
import type { AgentBranchState, BranchSynthesisInput } from '../runtime/agent/AgentRuntime.js';
import type {
  AgentStep,
  BranchKanbanCard,
  BranchKanbanSnapshot,
  CanonicalPlanState,
  BranchDisposition,
  BranchHandoff,
  BranchOutcome,
  PlannedBranch,
  SynthesisResult,
  SynthesisSharedHandoff,
} from '../runtime/agent/types.js';
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
  compressBranchTranscript,
  checkContextAndCompress,
  getCompressionLevel,
  type ContextBudgetResult,
} from './contextBudget.js';
import { SessionCompressor, isRunExhausted } from './sessionCompressor.js';
import { PlannerToolAdapter } from './PlannerToolAdapter.js';
import { RpmLimiter, getGlobalRpmLimiter } from '../utils/RpmLimiter.js';
import { logError } from '../utils/errorLogger.js';
import { SubagentRegistry } from './subagents/registry.js';
import { runSubagent } from './subagents/runner.js';
import { planSubagentHandler } from './subagents/plan.js';
import { researchSubagentHandler } from './subagents/research.js';
import { WorkspaceManager } from '../persistence/WorkspaceManager.js';
import type {
  PlanSubagentInvocation,
  ResearchSubagentInvocation,
  SubagentInvocation,
} from './subagents/types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const PlannerToolNameSchema = z.enum(TOOL_NAMES);

const PlannerToolCallSchema = z.object({
  name: PlannerToolNameSchema,
  arguments: z.record(z.any()).default({}),
  goal: z.string().trim().min(1).max(240).optional(),
});

const PlannerBranchSchema = z.object({
  label: z.string().trim().min(1).max(80),
  goal: z.string().trim().min(1).max(240),
  group: z.number().int().min(1).max(8).optional().default(1),
  depends: z.array(z.string().trim().min(1).max(80)).min(0).max(7).optional(),
});

const PlannerThoughtSchema = z.string().trim().min(1);

const ExecutionDisciplineDecisionSchema = z.object({
  mode: z.enum(['branching', 'sequential']),
  reason: z.string().trim().min(1).max(240).optional(),
});

const PlannerToolDecisionSchema = z.object({
  kind: z.literal('tool'),
  thought: PlannerThoughtSchema,
  tool_calls: z.array(PlannerToolCallSchema).min(1).max(5),
});

const PlannerBranchDecisionBaseSchema = z.object({
  kind: z.literal('branch'),
  thought: PlannerThoughtSchema,
  branches: z.array(PlannerBranchSchema).min(1).max(8),
});

const PlannerBranchDecisionSchema = PlannerBranchDecisionBaseSchema.superRefine((decision, ctx) => {
  const labelToIndex = new Map<string, number>();
  const normalizedGroups = decision.branches.map((branch) => branch.group || 1);

  decision.branches.forEach((branch, index) => {
    if (labelToIndex.has(branch.label)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['branches', index, 'label'],
        message: 'Branch labels must be unique within one planner_branch decision.',
      });
      return;
    }
    labelToIndex.set(branch.label, index);
  });

  const outgoing = Array.from({ length: decision.branches.length }, () => [] as number[]);
  const indegree = Array.from({ length: decision.branches.length }, () => 0);

  decision.branches.forEach((branch, index) => {
    const dependencies = [...new Set((branch.depends || []).map((label) => label.trim()).filter(Boolean))];
    dependencies.forEach((dependencyLabel, dependencyIndex) => {
      if (dependencyLabel === branch.label) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['branches', index, 'depends', dependencyIndex],
          message: 'A branch cannot depend on itself.',
        });
        return;
      }

      const dependencyBranchIndex = labelToIndex.get(dependencyLabel);
      if (dependencyBranchIndex === undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['branches', index, 'depends', dependencyIndex],
          message: 'depends must reference labels from the same planner_branch call.',
        });
        return;
      }

      if (normalizedGroups[dependencyBranchIndex] > normalizedGroups[index]) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['branches', index, 'depends', dependencyIndex],
          message: 'depends cannot point to a sibling in a later execution_group.',
        });
        return;
      }

      outgoing[dependencyBranchIndex].push(index);
      indegree[index] += 1;
    });
  });

  const ready: number[] = [];
  indegree.forEach((count, index) => {
    if (count === 0) {
      ready.push(index);
    }
  });

  let processed = 0;
  while (ready.length > 0) {
    const current = ready.shift()!;
    processed += 1;
    for (const target of outgoing[current]) {
      indegree[target] -= 1;
      if (indegree[target] === 0) {
        ready.push(target);
      }
    }
  }

  if (processed < decision.branches.length) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['branches'],
      message: 'depends must form an acyclic dependency graph among sibling branch labels.',
    });
  }
});

const PlannerFinalDecisionSchema = z.object({
  kind: z.literal('final'),
  answer: z.string().trim().min(1),
  status: z.enum(['success', 'failed', 'partial', 'blocked']),
  reason: z.string().trim().max(300).optional(),
});

const PlannerSkillsDecisionSchema = z.object({
  kind: z.literal('skills'),
  skills: z.array(z.string().trim().min(1)).min(1).max(2),
});

const PlannerSubagentDecisionSchema = z.object({
  kind: z.literal('subagent'),
  subagent: z.enum(['plan', 'research', 'crawler']),
  task: z.string().trim().min(1).max(800),
  context: z.string().trim().max(400).optional(),
});

const PlannerDecisionSchema = z.union([
  PlannerToolDecisionSchema,
  PlannerBranchDecisionBaseSchema,
  PlannerFinalDecisionSchema,
  PlannerSkillsDecisionSchema,
  PlannerSubagentDecisionSchema,
]);

type PlannerDecision = z.infer<typeof PlannerDecisionSchema>;
type ExecutablePlannerDecision = Exclude<PlannerDecision, { kind: 'skills' }>;
interface PlannerDecisionResult {
  decision: PlannerDecision;
  anthropicAssistantContent?: AnthropicMessageContentBlock[];
  anthropicToolUseIds?: string[];
  openaiAssistantMessage?: Record<string, unknown>;
}
interface ExecutablePlannerDecisionResult extends Omit<PlannerDecisionResult, 'decision'> {
  decision: ExecutablePlannerDecision;
}
type PlannerSubagentDecision = z.infer<typeof PlannerSubagentDecisionSchema>;
type PlannerInvocation =
  | { kind: 'work_tool'; tool_call: z.infer<typeof PlannerToolCallSchema> }
  | { kind: 'control'; decision: PlannerDecision };

const ContextSelectionSchema = z.object({
  relevant_node_ids: z.array(z.string()).max(4).default([]),
  rationale: z.string().trim().min(1).max(280).optional(),
});

type ContextSelection = z.infer<typeof ContextSelectionSchema>;

interface BranchTranscriptMessage {
  role: 'user' | 'assistant';
  content: ChatMessageContent;
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
  inheritedMessageCount: number;
  transcript: BranchTranscriptMessage[];
  savedNodeIds: string[];
  pendingAnthropicAssistantContent?: AnthropicMessageContentBlock[] | null;
  pendingAnthropicToolUseIds?: string[];
  pendingOpenAIAssistantMessage?: Record<string, unknown> | null;
  completionSummary?: string;
  finalAnswer?: string;
  outcome?: BranchOutcome;
  outcomeReason?: string;
  disposition?: BranchDisposition;
  finalizationMode?: 'natural' | 'planner_fallback';
  handoff?: BranchHandoff;
  sharedHandoff?: SynthesisSharedHandoff;
  planVersionSeen?: number;
  planExcerpt?: string;
  alignmentChecks?: string[];
  ancestorBranchIds?: string[];
  ancestorBranchLabels?: string[];
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

export type AgentMode = 'branching' | 'react';
type AgentRunPhase = 'plan' | 'execute';

export interface AgentRunOptions {
  conversationId?: string;
  agentId?: string;
  sessionId?: string;
  /** Interactive approval callback for governance `ask` decisions. */
  onApproval?: ApprovalCallback;
  /** Agent execution mode. Defaults to 'branching'. */
  agentMode?: AgentMode;
  /** Internal plan-and-execute stage marker. */
  runPhase?: AgentRunPhase;
}

interface PerfEntry {
  label: string;
  ms: number;
}

interface ConversationTurn {
  user: string;
  assistant: string;
}

interface PersistedConversationState {
  conversation_id: string;
  recent_turns: ConversationTurn[];
  active_topic: string;
  current_goal: string;
  status: 'active' | 'blocked' | 'completed' | 'failed';
  open_loops: string[];
  referenced_entities: string[];
  last_user_request: string;
  last_assistant_summary: string;
  updated_at: string;
}

interface TurnSummaryRecord {
  ts: string;
  conversation_id: string;
  user_input: string;
  user_intent: string;
  assistant_outcome: string;
  status: 'active' | 'blocked' | 'completed' | 'failed';
  next_expected_user_action: string;
  referenced_entities: string[];
  tool_findings: string[];
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
  private memoryOpenai: LanguageModelV1;
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
  private readonly memoryJobLogDir: string;
  private readonly nodeConversationMapPath: string;
  private readonly persistedConversationStateDir: string;
  private readonly turnSummaryLogPath: string;
  private readonly relationSearchMinScore: number;
  private readonly relationSearchLimit: number;
  private readonly relationMax: number;
  private readonly branchMaxWidth: number;
  private readonly branchMaxCompleted: number;
  private readonly branchConcurrency: number;
  private readonly rebranchEnabled: boolean;
  private readonly llmRpmLimiter: RpmLimiter;
  private readonly agentLlmTimeoutMs: number;
  private readonly plannerTemperature: number;
  private readonly responseTemperature: number;
  private readonly memoryClassifier: MemoryClassifierAgent;
  private readonly llmLimiterCircuitCooldownMs: number;
  /** Per-session compressor for exhausted-run carry-over. */
  private readonly sessionCompressor: SessionCompressor;
  private readonly subagentRegistry: SubagentRegistry;
  private readonly workspaceManager: WorkspaceManager;
  /** In-process caches to avoid re-reading disk on every request. */
  private cachedWorkspaceSkills: SkillRecord[] | null = null;
  private cachedSoulsMarkdown: string | null = null;

  constructor(config: AgentConfig, projectRoot: string, binaryPath?: string) {
    this.projectRoot = projectRoot;
    this.codeCliRoot = resolveCodeCliRoot(__dirname);
    this.model = config.anthropicModel || config.model || 'gpt-4o';
    this.memoryModel = config.memoryModel || this.model;

    // ── Primary LLM: prefer MiniMax OpenAI-compat when available ───────
    const useAnthropic = Boolean(config.anthropicAuthToken);
    const primaryOpenAICompatBaseURL = this.deriveOpenAICompatBaseURL(config.baseURL)
      || this.deriveOpenAICompatBaseURL(config.anthropicBaseURL);
    const useOpenAICompat = Boolean(config.anthropicAuthToken && primaryOpenAICompatBaseURL);
    if (useOpenAICompat) {
      this.openai = buildLanguageModel({
        model: this.model,
        apiKey: config.apiKey || config.anthropicAuthToken!,
        baseURL: primaryOpenAICompatBaseURL,
      });
    } else if (useAnthropic) {
      const anthropicModel = buildAnthropicLanguageModel({
        model: this.model,
        authToken: config.anthropicAuthToken!,
        baseURL: config.anthropicBaseURL,
      });
      this.openai = anthropicModel;
    } else {
      this.openai = buildLanguageModel({
        model: this.model,
        apiKey: config.apiKey,
        baseURL: config.baseURL,
        hmacAccessKey: config.hmacAccessKey,
        hmacSecretKey: config.hmacSecretKey,
        gatewayApiKey: config.gatewayApiKey,
      });
    }

    // ── Memory LLM: reuse Anthropic if no separate memory config ───────
    const memoryBaseURL = config.memoryBaseURL || config.baseURL;
    const memoryOpenAICompatBaseURL = this.deriveOpenAICompatBaseURL(config.memoryBaseURL)
      || primaryOpenAICompatBaseURL;
    const memoryAccessKey = config.memoryHmacAccessKey || config.hmacAccessKey;
    const memorySecretKey = config.memoryHmacSecretKey || config.hmacSecretKey;
    const memoryGatewayKey = config.memoryGatewayApiKey || config.gatewayApiKey;
    if (useOpenAICompat) {
      this.memoryOpenai = buildLanguageModel({
        model: this.memoryModel,
        apiKey: config.memoryApiKey || config.apiKey || config.anthropicAuthToken!,
        baseURL: memoryOpenAICompatBaseURL || primaryOpenAICompatBaseURL,
      });
    } else if (useAnthropic && !config.memoryApiKey) {
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
    const rawBranchMaxWidth = Number(process.env.REACT_BRANCH_MAX_WIDTH ?? 5);
    this.branchMaxWidth = Number.isFinite(rawBranchMaxWidth) ? Math.max(1, Math.min(10, Math.floor(rawBranchMaxWidth))) : 5;
    const rawBranchMaxCompleted = Number(process.env.REACT_BRANCH_MAX_COMPLETED ?? 4);
    this.branchMaxCompleted = Number.isFinite(rawBranchMaxCompleted) ? Math.max(1, Math.min(8, Math.floor(rawBranchMaxCompleted))) : 4;
    const rawBranchConcurrency = Number(process.env.REACT_BRANCH_CONCURRENCY ?? 0);
    this.branchConcurrency = Number.isFinite(rawBranchConcurrency) ? Math.max(0, Math.min(64, Math.floor(rawBranchConcurrency))) : 0;
    const rawRebranchEnabled = String(process.env.REACT_REBRANCH_ENABLED ?? '1').toLowerCase();
    this.rebranchEnabled = rawRebranchEnabled !== '0' && rawRebranchEnabled !== 'false' && rawRebranchEnabled !== 'off';
    const rawLlmRpm = Number(process.env.LLM_RPM_LIMIT ?? 0);
    const normalizedLlmRpm = Number.isFinite(rawLlmRpm) ? Math.max(0, Math.floor(rawLlmRpm)) : 0;
    this.llmRpmLimiter = getGlobalRpmLimiter('llm', normalizedLlmRpm);
    const rawAgentLlmTimeoutMs = Number(process.env.AGENT_LLM_TIMEOUT_MS ?? 120000);
    this.agentLlmTimeoutMs = Number.isFinite(rawAgentLlmTimeoutMs) ? Math.max(5000, Math.floor(rawAgentLlmTimeoutMs)) : 120000;
    const rawLlmCircuitCooldownMs = Number(process.env.LLM_LIMIT_CIRCUIT_COOLDOWN_MS ?? 120000);
    this.llmLimiterCircuitCooldownMs = Number.isFinite(rawLlmCircuitCooldownMs)
      ? Math.max(1000, Math.floor(rawLlmCircuitCooldownMs))
      : 120000;
    const rawPlannerTemperature = Number(process.env.PLANNER_TEMPERATURE ?? 0.1);
    this.plannerTemperature = Number.isFinite(rawPlannerTemperature) ? Math.max(0, Math.min(1, rawPlannerTemperature)) : 0.1;
    const rawResponseTemperature = Number(process.env.RESPONSE_TEMPERATURE ?? 0.3);
    this.responseTemperature = Number.isFinite(rawResponseTemperature) ? Math.max(0, Math.min(1, rawResponseTemperature)) : 0.3;
    this.memoryLogPath = path.join(projectRoot, '.mitosis', 'logs', 'mitosis_save.log');
    this.conversationLogDir = path.join(projectRoot, '.mitosis', 'conversations');
    this.memoryJobLogDir = path.join(projectRoot, '.mitosis', 'memory_jobs');
    this.nodeConversationMapPath = path.join(projectRoot, '.mitosis', 'logs', 'node_conversations.jsonl');
    this.persistedConversationStateDir = path.join(projectRoot, '.mitosis', 'conversation_state');
    this.turnSummaryLogPath = path.join(projectRoot, '.mitosis', 'logs', 'thread_turn_summaries.jsonl');

    this.memoryClassifier = new MemoryClassifierAgent({
      chatClient: this.memoryOpenai,
      codeCliRoot: this.codeCliRoot,
      extractionMaxChars: this.extractionMaxChars,
      memoryExtractTimeoutMs: this.memoryExtractTimeoutMs,
      memoryActionTimeoutMs: this.memoryActionTimeoutMs,
      autoLinkEnabled: this.autoLinkEnabled,
      autoLinkMaxNodes: this.autoLinkMaxNodes,
      autoLinkLimit: this.autoLinkLimit,
      rpmLimiter: this.llmRpmLimiter,
    });
    this.sessionCompressor = new SessionCompressor();
    this.subagentRegistry = new SubagentRegistry([
      planSubagentHandler,
      researchSubagentHandler,
    ], projectRoot);
    this.workspaceManager = new WorkspaceManager(projectRoot);
  }

  private deriveOpenAICompatBaseURL(baseURL?: string): string | undefined {
    const normalized = String(baseURL || '').trim();
    if (!normalized) {
      return undefined;
    }
    if (/\/v1\/?$/i.test(normalized)) {
      return normalized.replace(/\/+$/g, '');
    }
    if (/api\.minimaxi\.com/i.test(normalized) && /\/anthropic\/?$/i.test(normalized)) {
      return normalized.replace(/\/anthropic\/?$/i, '/v1');
    }
    return undefined;
  }

  onBackgroundTask(callback: (task: string, status: 'started' | 'completed') => void) {
      this.onBackgroundTaskCallback = callback;
      return () => { this.onBackgroundTaskCallback = null; };
  }

  setRpmWaitCallback(callback: (waitMs: number, queueLength: number) => void) {
    this.llmRpmLimiter.setOnWaitCallback(callback);
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

  private buildPlannerWorkToolParameters(parameters: Record<string, unknown>): Record<string, unknown> {
    const schema = (parameters && typeof parameters === 'object') ? parameters as Record<string, unknown> : {};
    const rawProperties = schema.properties;
    const properties = rawProperties && typeof rawProperties === 'object'
      ? rawProperties as Record<string, unknown>
      : {};
    const required = Array.isArray(schema.required)
      ? schema.required.filter((value): value is string => typeof value === 'string' && value !== 'goal')
      : [];

    return {
      ...schema,
      type: 'object',
      additionalProperties: false,
      properties: {
        ...properties,
        goal: {
          type: 'string',
          minLength: 1,
          maxLength: 240,
        },
      },
      required,
    };
  }

  private parsePlannerWorkToolInvocation(
    toolName: z.infer<typeof PlannerToolNameSchema>,
    input: unknown,
  ): PlannerInvocation {
    const record = input && typeof input === 'object' && !Array.isArray(input)
      ? input as Record<string, unknown>
      : {};
    const { goal, ...argumentsRecord } = record;
    return {
      kind: 'work_tool',
      tool_call: PlannerToolCallSchema.parse({
        name: toolName,
        arguments: argumentsRecord,
        ...(typeof goal === 'string' && goal.trim() ? { goal: goal.trim() } : {}),
      }),
    };
  }

  private derivePlannerToolThought(text: string, toolCalls: Array<z.infer<typeof PlannerToolCallSchema>>): string {
    const normalizedText = this.clipText(String(text || '').replace(/\s+/g, ' ').trim(), 280);
    if (normalizedText) {
      return normalizedText;
    }
    const goalSummary = this.clipText(
      toolCalls
        .map((toolCall) => toolCall.goal?.trim())
        .filter((goal): goal is string => Boolean(goal))
        .join(' | '),
      280,
    );
    if (goalSummary) {
      return goalSummary;
    }
    return `Continue with ${toolCalls.map((toolCall) => toolCall.name).join(', ')}.`;
  }

  private buildPlannerDecisionTools(allowBranching: boolean): Array<ParseableFunctionTool<PlannerInvocation>> {
    const tools: Array<ParseableFunctionTool<PlannerInvocation>> = TOOLS.map((tool) => {
      const toolName = tool.function.name as z.infer<typeof PlannerToolNameSchema>;
      return {
        name: toolName,
        description: tool.function.description,
        parameters: this.buildPlannerWorkToolParameters(tool.function.parameters),
        parse: (input) => this.parsePlannerWorkToolInvocation(toolName, input),
      };
    });

    tools.push(
      {
        name: 'planner_final',
        description: 'Finish the run with a direct user-facing answer.',
        parameters: {
          type: 'object',
          additionalProperties: false,
          properties: {
            answer: {
              type: 'string',
              minLength: 1,
              maxLength: 5000,
              description: 'User-facing markdown answer. Reference created file paths instead of repeating full code listings.',
            },
            status: {
              type: 'string',
              enum: ['success', 'failed', 'partial', 'blocked'],
              description: 'Result status: success if goal fully achieved, partial if some progress made, failed if goal not met, blocked if externally blocked.',
            },
            reason: {
              type: 'string',
              maxLength: 300,
              description: 'Brief explanation when status is partial, failed, or blocked (e.g. error message, blocker description).',
            },
          },
          required: ['answer', 'status'],
        },
        parse: (input) => {
          const record = (input && typeof input === 'object' && !Array.isArray(input))
            ? input as Record<string, unknown>
            : {};
          // Provide fallbacks so a non-compliant LLM omitting required fields
          // does not throw a Zod validation error that aborts the whole branch.
          const patched = {
            kind: 'final' as const,
            answer: '抱歉，我未能生成答案，请重试一次。',
            status: 'failed' as const,
            ...record,
          };
          return {
            kind: 'control',
            decision: PlannerFinalDecisionSchema.parse(patched),
          };
        },
      },
      {
        name: 'planner_skills',
        description: 'Load one or two local skills before making the next planning decision.',
        parameters: {
          type: 'object',
          additionalProperties: false,
          properties: {
            skills: {
              type: 'array',
              minItems: 1,
              maxItems: 2,
              items: {
                type: 'string',
                minLength: 1,
              },
            },
          },
          required: ['skills'],
        },
        parse: (input) => ({
          kind: 'control',
          decision: PlannerSkillsDecisionSchema.parse({ kind: 'skills', ...(input as Record<string, unknown>) }),
        }),
      },
    );

    if (allowBranching) {
      tools.push({
        name: 'planner_subagent',
        description: 'Request a subagent for specialized work. Use subagent=plan for branching/planning, subagent=research for research tasks, subagent=crawler for web scraping.',
        parameters: {
          type: 'object',
          additionalProperties: false,
          properties: {
            subagent: {
              type: 'string',
              enum: ['plan', 'research', 'crawler'],
            },
            task: {
              type: 'string',
              minLength: 1,
              maxLength: 800,
              description: 'The task for the subagent to perform.',
            },
            context: {
              type: 'string',
              maxLength: 400,
              description: 'Optional additional context for the subagent.',
            },
          },
          required: ['subagent', 'task'],
        },
        parse: (input) => ({
          kind: 'control',
          decision: PlannerSubagentDecisionSchema.parse({ kind: 'subagent', ...(input as Record<string, unknown>) }),
        }),
      });
    }

    return tools;
  }

  private async generatePlannerDecision(options: {
    messages: ChatMessage[];
    timeoutMs: number;
    timeoutLabel: string;
    temperature?: number;
    allowBranching: boolean;
    maxTokens?: number;
  }): Promise<PlannerDecisionResult> {
    const { calls, text, providerMessage } = await this.llmRpmLimiter.run(() => this.withTimeout(
      generateToolCalls({
        model: this.openai,
        messages: options.messages,
        tools: this.buildPlannerDecisionTools(options.allowBranching),
        temperature: options.temperature,
        maxTokens: options.maxTokens ?? 1600,
      }),
      options.timeoutMs,
      options.timeoutLabel,
    ));
    const controlDecisions = calls
      .filter((call): call is { name: string; input: Extract<PlannerInvocation, { kind: 'control' }> } => call.input.kind === 'control')
      .map((call) => call.input.decision);
    const workToolCalls = calls
      .filter((call): call is { name: string; input: Extract<PlannerInvocation, { kind: 'work_tool' }> } => call.input.kind === 'work_tool')
      .map((call) => call.input.tool_call);

    if (controlDecisions.length > 0 && workToolCalls.length > 0) {
      throw new Error('Planner mixed control tools with direct work tools in one response.');
    }
    if (controlDecisions.length > 1) {
      throw new Error('Planner emitted multiple control tools in one response.');
    }
    if (controlDecisions.length === 1) {
      return { decision: controlDecisions[0] };
    }
    if (workToolCalls.length > 0) {
      return {
        decision: {
          kind: 'tool',
          thought: this.derivePlannerToolThought(text, workToolCalls),
          tool_calls: workToolCalls,
        },
        anthropicAssistantContent: providerMessage?.anthropicAssistantContent,
        anthropicToolUseIds: calls
          .map((call) => call.providerToolUseId)
          .filter((value): value is string => typeof value === 'string' && value.trim().length > 0),
      };
    }
    throw new Error('Planner emitted no actionable tool calls.');
  }

  private async generateJsonPromptText(options: {
    model: LanguageModelV1;
    messages: any;
    timeoutMs: number;
    timeoutLabel: string;
    temperature?: number;
    onFallback?: () => void | Promise<void>;
  }): Promise<string> {
    const run = async (messages: any, useJsonObject: boolean) => {
      return this.llmRpmLimiter.run(() => this.withTimeout(
        generateText({
          model: options.model,
          messages,
          temperature: options.temperature ?? this.plannerTemperature,
          maxTokens: 6000,
          ...(useJsonObject
            ? { providerOptions: { openai: { responseFormat: { type: 'json_object' as const } } } }
            : {}),
        }),
        options.timeoutMs,
        options.timeoutLabel,
      ));
    };

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

    // Strip leaked XML tool call blocks that some models (MiniMax, Qwen, etc.)
    // emit even in text-only mode.
    current = current
      .replace(/<[a-z_][a-z0-9_-]*:tool_call>[\s\S]*?<\/[a-z_][a-z0-9_-]*:tool_call>/gi, '')
      .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '')
      .replace(/<invoke\s+name=[\s\S]*?<\/invoke>/gi, '')
      .replace(/<\|tool_call\|>[\s\S]*?<\|\/tool_call\|>/gi, '')
      .trim();

    return current;
  }

  private containsInternalToolMarkup(raw: string): boolean {
    const trimmed = String(raw || '').trim();
    if (!trimmed) {
      return false;
    }
    return /<Function::[a-z0-9_:-]+>/i.test(trimmed)
      || /<\/Function>/i.test(trimmed)
      || /\[TOOL_CALL\]/i.test(trimmed)
      || /\[\/TOOL_CALL\]/i.test(trimmed)
      || /```tool_call/i.test(trimmed)
      || /^tool_call\s*$/im.test(trimmed)
      // MiniMax XML tool call format
      || /<minimax:tool_call>/i.test(trimmed)
      || /<\/minimax:tool_call>/i.test(trimmed)
      // Generic vendor-prefixed XML tool call (e.g. <vendor:tool_call>)
      || /<[a-z_][a-z0-9_-]*:tool_call[\s>]/i.test(trimmed)
      // Generic <invoke name="..."> / </invoke> (Anthropic-compat XML)
      || /<invoke\s+name=/i.test(trimmed)
      || /<\/invoke>/i.test(trimmed)
      // Qwen / GLM special-token style
      || /<\|tool_call\|>/i.test(trimmed)
      || /<\|\/tool_call\|>/i.test(trimmed)
      // <tool_call> without vendor prefix
      || /<tool_call>/i.test(trimmed)
      || /<\/tool_call>/i.test(trimmed);
  }

  private sanitizeUserFacingAnswer(answer: string, fallback = ''): string {
    const normalized = this.normalizeUserFacingAnswer(answer);
    if (!normalized) {
      return fallback.trim();
    }
    if (this.containsInternalToolMarkup(normalized) || this.looksLikeNonUserFacingPlannerOutput(normalized)) {
      return fallback.trim();
    }
    return normalized;
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
    if (this.containsInternalToolMarkup(trimmed)) {
      return true;
    }
    if (/^```[\s\S]*```$/u.test(trimmed)) {
      return true;
    }
    if (/^\s*[{[][\s\S]*[}\]]\s*$/u.test(trimmed) && !this.hasParseableJsonishPayload(trimmed)) {
      return true;
    }
    if (/^PLANNER (?:TOOL|BRANCH|FINAL|SKILLS)\b/im.test(trimmed)) {
      return true;
    }
    // Strip code-fence blocks before checking for bare shell commands so that
    // legitimate markdown answers containing ```bash ... ``` sections are not
    // incorrectly classified as internal planner output.
    const withoutCodeFences = trimmed.replace(/```[\s\S]*?```/g, '');
    return /(^|\n)\s*(find|ls|cat|grep|rg|echo|printf|cd|pwd|npm|pnpm|yarn|cargo|python|python3|node|git|curl|BIN=|\[\[)\b/m.test(withoutCodeFences);
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

  private sanitizeConversationId(conversationId: string): string {
    const cleaned = String(conversationId || 'default').replace(/[^a-zA-Z0-9_-]+/g, '_');
    return cleaned || 'default';
  }

  private getPersistedConversationStatePath(conversationId: string): string {
    return path.join(this.persistedConversationStateDir, `${this.sanitizeConversationId(conversationId)}.json`);
  }

  private classifyTurnStatus(answer: string, traces: TraceEvent[]): 'active' | 'blocked' | 'completed' | 'failed' {
    const normalizedAnswer = answer.replace(/\s+/g, ' ').trim();
    if (/抱歉|请重试|没有生成有效回答|没有找到之前的具体上下文/i.test(normalizedAnswer)) {
      return 'failed';
    }
    if (this.looksLikeNonUserFacingPlannerOutput(answer)) {
      return 'blocked';
    }
    if (traces.some((trace) => /^Error:/i.test(trace.content) || /permission denied|not found|binary not found|governance deny|timeout/i.test(trace.content))) {
      return 'blocked';
    }
    if (/需要确认|请描述一下|请选择|二选一|blocked|受阻|无法|不可用/i.test(normalizedAnswer)) {
      return 'blocked';
    }
    if (/已完成|完成了|保存成功|done|success|成功/i.test(normalizedAnswer)) {
      return 'completed';
    }
    return 'active';
  }

  private deriveToolFindings(traces: TraceEvent[], limit = 3): string[] {
    const findings: string[] = [];
    const seen = new Set<string>();
    for (const trace of traces) {
      if (trace.type !== 'observation') {
        continue;
      }
      const cleaned = trace.content.replace(/\s+/g, ' ').trim();
      if (cleaned.length < 12) {
        continue;
      }
      if (/^(Selected |Recalled |Injected carry-over|Context budget|Run completed normally|Auto-queued |Perf total=|Recovered persisted thread context|Injected persisted thread working state)/i.test(cleaned)) {
        continue;
      }
      const key = cleaned.toLowerCase();
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      findings.push(cleaned.slice(0, 220));
      if (findings.length >= limit) {
        break;
      }
    }
    return findings;
  }

  private deriveConversationGoal(input: string, traces: TraceEvent[], answer: string): string {
    const thought = [...traces]
      .reverse()
      .find((trace) => trace.type === 'thought' && !/Initializing branching ReAct context|Planner returned/i.test(trace.content));
    if (thought) {
      return thought.content.replace(/\s+/g, ' ').trim().slice(0, 200);
    }
    return this.firstSentence(this.extractOriginalUserRequest(input))
      || this.firstSentence(answer)
      || 'Continue the current thread task.';
  }

  private deriveOpenLoops(status: 'active' | 'blocked' | 'completed' | 'failed', input: string, traces: TraceEvent[], answer: string): string[] {
    if (status === 'completed') {
      return [];
    }
    const loops: string[] = [];
    const push = (value: string) => {
      const cleaned = value.replace(/\s+/g, ' ').trim();
      if (!cleaned || loops.includes(cleaned)) {
        return;
      }
      loops.push(cleaned.slice(0, 200));
    };

    const lastAction = [...traces].reverse().find((trace) => trace.type === 'action');
    if (lastAction?.metadata?.toolName) {
      push(`Continue from the last attempted tool step: ${String(lastAction.metadata.toolName)}.`);
    }
    for (const finding of this.deriveToolFindings(traces, 2)) {
      push(finding);
    }
    if (status === 'failed') {
      push('Recover the previous thread context and resume the task without asking the user to restate everything.');
    } else if (status === 'blocked') {
      push('Resolve the blocking issue before continuing the original task.');
    } else {
      push(this.firstSentence(answer) || this.firstSentence(this.extractOriginalUserRequest(input)) || 'Continue the current thread task.');
    }
    return loops.slice(0, 4);
  }

  private deriveActiveTopic(input: string, answer: string, recentTurns: ConversationTurn[]): string {
    const requestSentence = this.firstSentence(this.extractOriginalUserRequest(input));
    if (requestSentence && requestSentence.length >= 8) {
      return requestSentence;
    }
    const priorUser = [...recentTurns].reverse().map((turn) => this.firstSentence(turn.user)).find((value) => value.length >= 8);
    if (priorUser) {
      return priorUser;
    }
    return this.firstSentence(answer) || 'Current conversation thread';
  }

  private buildPersistedConversationState(
    conversationId: string,
    input: string,
    answer: string,
    traces: TraceEvent[],
    recentTurns: ConversationTurn[],
  ): PersistedConversationState {
    const status = this.classifyTurnStatus(answer, traces);
    const referencedEntities = this.collectAtomicCandidates(this.extractOriginalUserRequest(input), answer).slice(0, 8);
    return {
      conversation_id: conversationId,
      recent_turns: recentTurns,
      active_topic: this.deriveActiveTopic(input, answer, recentTurns),
      current_goal: this.deriveConversationGoal(input, traces, answer),
      status,
      open_loops: this.deriveOpenLoops(status, input, traces, answer),
      referenced_entities: referencedEntities,
      last_user_request: this.extractOriginalUserRequest(input).slice(0, 600),
      last_assistant_summary: this.firstSentence(answer).slice(0, 280),
      updated_at: new Date().toISOString(),
    };
  }

  private readPersistedConversationState(conversationId: string): PersistedConversationState | null {
    try {
      const filePath = this.getPersistedConversationStatePath(conversationId);
      if (!fs.existsSync(filePath)) {
        return null;
      }
      const parsed = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.recent_turns)) {
        return null;
      }
      return parsed as PersistedConversationState;
    } catch {
      return null;
    }
  }

  private writePersistedConversationState(state: PersistedConversationState): void {
    try {
      fs.mkdirSync(this.persistedConversationStateDir, { recursive: true });
      fs.writeFileSync(
        this.getPersistedConversationStatePath(state.conversation_id),
        JSON.stringify(state, null, 2),
        'utf-8',
      );
    } catch {}
  }

  private appendTurnSummaryRecord(record: TurnSummaryRecord): void {
    try {
      fs.mkdirSync(path.dirname(this.turnSummaryLogPath), { recursive: true });
      fs.appendFileSync(this.turnSummaryLogPath, `${JSON.stringify(record)}\n`, 'utf-8');
    } catch {}
  }

  private readTurnSummaryRecords(conversationId: string, limit = 12): TurnSummaryRecord[] {
    try {
      if (!fs.existsSync(this.turnSummaryLogPath)) {
        return [];
      }
      const rows = fs.readFileSync(this.turnSummaryLogPath, 'utf-8')
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          try {
            return JSON.parse(line) as unknown;
          } catch {
            return null;
          }
        })
        .filter((row): row is TurnSummaryRecord => {
          if (!row || typeof row !== 'object') {
            return false;
          }
          const candidate = row as Partial<TurnSummaryRecord>;
          return typeof candidate.conversation_id === 'string'
            && candidate.conversation_id === conversationId
            && typeof candidate.user_input === 'string'
            && typeof candidate.assistant_outcome === 'string'
            && typeof candidate.user_intent === 'string'
            && typeof candidate.status === 'string'
            && typeof candidate.next_expected_user_action === 'string'
            && Array.isArray(candidate.referenced_entities)
            && Array.isArray(candidate.tool_findings);
        });
      return rows.slice(-Math.max(1, limit));
    } catch {
      return [];
    }
  }

  private buildStateFromTurnSummaries(
    conversationId: string,
    summaries: TurnSummaryRecord[],
  ): PersistedConversationState | null {
    if (summaries.length === 0) {
      return null;
    }
    const recent = summaries.slice(-this.maxConversationTurns);
    const latest = recent[recent.length - 1];
    const entities = Array.from(new Set(recent.flatMap((row) => Array.isArray(row.referenced_entities) ? row.referenced_entities : []))).slice(0, 8);
    const openLoops = Array.from(new Set(
      recent
        .map((row) => row.next_expected_user_action)
        .filter((value) => typeof value === 'string' && value.trim().length > 0)
    )).slice(-4);
    return {
      conversation_id: conversationId,
      recent_turns: recent.map((row) => ({
        user: row.user_input,
        assistant: row.assistant_outcome,
      })),
      active_topic: this.firstSentence(latest.user_intent || latest.user_input) || 'Recovered conversation thread',
      current_goal: this.firstSentence(latest.next_expected_user_action || latest.user_intent || latest.user_input) || 'Continue the recovered thread task.',
      status: latest.status || 'active',
      open_loops: openLoops,
      referenced_entities: entities,
      last_user_request: latest.user_intent || latest.user_input,
      last_assistant_summary: latest.assistant_outcome,
      updated_at: latest.ts || new Date().toISOString(),
    };
  }

  private selectRelevantTurnSummaryRecords(input: string, conversationId: string): TurnSummaryRecord[] {
    const rows = this.readTurnSummaryRecords(conversationId, 12);
    if (rows.length === 0) {
      return [];
    }
    const followUp = this.isLikelyFollowUp(input);
    const scored = rows
      .map((row, index) => {
        const combined = [
          row.user_input,
          row.user_intent,
          row.assistant_outcome,
          row.next_expected_user_action,
          ...(Array.isArray(row.referenced_entities) ? row.referenced_entities : []),
          ...(Array.isArray(row.tool_findings) ? row.tool_findings : []),
        ].join('\n');
        const overlap = this.lexicalOverlapScore(input, combined);
        const recency = (index + 1) / Math.max(1, rows.length) * 0.18;
        return { row, score: overlap + (followUp ? recency : recency * 0.5), index };
      })
      .sort((a, b) => b.score - a.score || b.index - a.index);

    const threshold = followUp ? 0.04 : 0.1;
    const selected = scored.filter((item) => item.score >= threshold).slice(0, followUp ? 3 : 2);
    if (selected.length > 0) {
      return selected.sort((a, b) => a.index - b.index).map((item) => item.row);
    }
    if (followUp) {
      return rows.slice(-Math.min(2, rows.length));
    }
    return [];
  }

  private renderTurnSummaryReplay(records: TurnSummaryRecord[]): string {
    if (records.length === 0) {
      return '';
    }
    return [
      'Recent turn summary replay:',
      ...records.map((row, index) => [
        `Turn ${index + 1}:`,
        `- user_intent: ${row.user_intent || row.user_input || '(none)'}`,
        `- assistant_outcome: ${row.assistant_outcome || '(none)'}`,
        `- status: ${row.status || 'active'}`,
        `- next_expected_user_action: ${row.next_expected_user_action || '(none)'}`,
        `- referenced_entities: ${Array.isArray(row.referenced_entities) && row.referenced_entities.length > 0 ? row.referenced_entities.join(', ') : '(none)'}`,
        `- tool_findings: ${Array.isArray(row.tool_findings) && row.tool_findings.length > 0 ? row.tool_findings.join(' | ') : '(none)'}`,
      ].join('\n')),
    ].join('\n');
  }

  private ensureConversationTurnsLoaded(conversationId: string): PersistedConversationState | null {
    if (this.conversationTurnsByConversation.has(conversationId)) {
      return this.readPersistedConversationState(conversationId);
    }
    const persisted = this.readPersistedConversationState(conversationId);
    if (persisted?.recent_turns?.length) {
      this.conversationTurnsByConversation.set(
        conversationId,
        persisted.recent_turns.slice(-this.maxConversationTurns),
      );
      return persisted;
    }
    const recoveredFromJournal = this.buildStateFromTurnSummaries(
      conversationId,
      this.readTurnSummaryRecords(conversationId, this.maxConversationTurns),
    );
    if (recoveredFromJournal?.recent_turns?.length) {
      this.conversationTurnsByConversation.set(
        conversationId,
        recoveredFromJournal.recent_turns.slice(-this.maxConversationTurns),
      );
      return recoveredFromJournal;
    }
    return persisted;
  }

  private renderPersistedConversationState(state: PersistedConversationState | null): string {
    if (!state) {
      return '';
    }
    const lines = [
      'Persisted thread working state:',
      `- active_topic: ${state.active_topic || '(unknown)'}`,
      `- current_goal: ${state.current_goal || '(unknown)'}`,
      `- status: ${state.status || 'active'}`,
      `- last_user_request: ${state.last_user_request || '(none)'}`,
      `- last_assistant_summary: ${state.last_assistant_summary || '(none)'}`,
      `- referenced_entities: ${state.referenced_entities.length > 0 ? state.referenced_entities.join(', ') : '(none)'}`,
      state.open_loops.length > 0
        ? `- open_loops:\n${state.open_loops.map((item) => `  - ${item}`).join('\n')}`
        : '- open_loops: (none)',
    ];
    return lines.join('\n');
  }

  private async decideExecutionDiscipline(input: string, requestedMode: AgentMode, conversationId = 'default'): Promise<{
    mode: 'branching' | 'sequential';
    reason: string;
  }> {
    if (requestedMode !== 'branching') {
      return {
        mode: 'branching',
        reason: 'Requested agent mode is already non-branching.',
      };
    }

    const override = String(process.env.REACT_ALLOW_MUTATING_BRANCHES ?? '').trim().toLowerCase();
    if (override === '1' || override === 'true' || override === 'on') {
      return {
        mode: 'branching',
        reason: 'REACT_ALLOW_MUTATING_BRANCHES override is enabled.',
      };
    }

    const request = this.extractOriginalUserRequest(input).replace(/\s+/g, ' ').trim();
    if (!request) {
      return {
        mode: 'branching',
        reason: 'No request text available for execution-discipline selection.',
      };
    }

    const persistedState = this.ensureConversationTurnsLoaded(conversationId);
    const recentUserTurns = this.getConversationTurns(conversationId)
      .slice(-3)
      .map((turn) => turn.user);
    const messages: ChatMessage[] = [
      {
        role: 'system',
        content: `You decide execution discipline for a branching agent.
Return exactly one JSON object with this shape:
{"mode":"branching"|"sequential","reason":"short reason"}

Use first-principles task reasoning, not keyword matching.
Optimize for end-to-end correctness and coordination safety, not maximum parallelism.
Both modes should begin from planning. The real choice is between one ordered plan and one parallel coordination plan.
Choose "sequential" when the request mainly produces one integrated artifact or one ordered implementation thread, especially when later steps need outputs, decisions, or constraints produced by earlier steps. This includes coordinated writes to shared workspace files/directories, non-isolated shell side effects, or one executor integrating ordered changes that sibling branches would otherwise contend on. Choosing "sequential" here means the work should converge to one ordered owner even if some planning or research could still be parallelized.
Choose "branching" when the work benefits from a parallel task plan: substantial parts can make useful progress independently without waiting for sibling outputs, and the plan can be expressed as a small coordination graph with low coordination risk.
A brief upfront plan does not by itself require "sequential"; branching is still a form of planning when the work can be decomposed safely.
Raise the priority of execution-structure planning over immediate tool use when the safer next step is not obvious.
When in doubt, prefer the mode whose plan shape is clearer; if independent progress is plausible and coordination risk is low, lean "branching".
Keep the reason short and concrete.`,
      },
      {
        role: 'user',
        content: `Current request:
${request}

Persisted thread state:
${this.renderPersistedConversationState(persistedState) || '(none)'}

Recent user turns:
${recentUserTurns.length > 0 ? recentUserTurns.map((turn) => `- ${turn}`).join('\n') : '(none)'}

Decide the safer execution discipline for this turn.`,
      },
    ];

    try {
      const run = async (useJsonObject: boolean) => this.llmRpmLimiter.run(() => this.withTimeout(
        generateText({
          model: this.openai,
          messages,
          temperature: 0,
          maxTokens: 220,
          ...(useJsonObject
            ? { providerOptions: { openai: { responseFormat: { type: 'json_object' as const } } } }
            : {}),
        }),
        Math.min(this.agentLlmTimeoutMs, 20000),
        'execution_discipline_gate llm',
      ));

      let raw = '';
      try {
        const result = await run(this.jsonObjectResponseFormatSupported);
        raw = typeof result.text === 'string' ? result.text : String(result.text || '');
        if (!raw.trim() && this.jsonObjectResponseFormatSupported) {
          this.jsonObjectResponseFormatSupported = false;
          const retry = await run(false);
          raw = typeof retry.text === 'string' ? retry.text : String(retry.text || '');
        }
      } catch (error) {
        if (this.jsonObjectResponseFormatSupported && this.isJsonResponseFormatUnsupported(error)) {
          this.jsonObjectResponseFormatSupported = false;
          const retry = await run(false);
          raw = typeof retry.text === 'string' ? retry.text : String(retry.text || '');
        } else {
          throw error;
        }
      }

      for (const candidate of extractJsonishCandidates(raw)) {
        try {
          const parsed = ExecutionDisciplineDecisionSchema.safeParse(parseJsonish(candidate));
          if (parsed.success) {
            return {
              mode: parsed.data.mode,
              reason: this.clipText(parsed.data.reason || `Model selected ${parsed.data.mode}.`, 180),
            };
          }
        } catch {
          continue;
        }
      }
    } catch {}

    return {
      mode: 'branching',
      reason: 'Execution-discipline gate defaulted to branching; decide the coordination graph before direct tool work.',
    };
  }

  private isLikelyFollowUp(input: string): boolean {
    const text = input.trim();
    if (!text) {
      return false;
    }
    const compactLength = Array.from(text.replace(/\s+/g, '')).length;
    if (compactLength <= 6) {
      return true;
    }
    const compactTokens = this.normalizeTextForSimilarity(text.toLowerCase());
    // Treat very short, context-dependent turns as follow-ups without relying on
    // any language-specific trigger-word lists.
    if (compactTokens.length <= 4) {
      return compactLength <= 24;
    }
    return compactTokens.length <= 8 && compactLength <= 12;
  }

  private getConversationTurns(conversationId = 'default'): ConversationTurn[] {
    this.ensureConversationTurnsLoaded(conversationId);
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

    // Skip the LLM call when scores are already unambiguous: the top candidate
    // leads by a wide margin and every candidate scores above the noise floor.
    const sorted = [...candidates].sort((a, b) => b.searchScore - a.searchScore);
    const topScore = sorted[0].searchScore;
    const secondScore = sorted[1]?.searchScore ?? 0;
    const noiseFloor = 0.5;
    const dominanceGap = 0.8;
    if (topScore >= noiseFloor && topScore - secondScore >= dominanceGap) {
      const heuristic = this.heuristicSelectContextCandidates(input, candidates, selectedTurns);
      return {
        selected: heuristic,
        rationale: `Heuristic selection used (top score=${topScore.toFixed(2)} dominance gap=${(topScore - secondScore).toFixed(2)} ≥ ${dominanceGap}); skipped LLM context-selection call.`,
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
              content: 'You are a context selector. Treat recalled context as noisy by default. Choose only the candidates that are directly relevant to the current user request in light of the selected recent conversation turns. Prefer candidates that clarify entities, constraints, decisions, or unresolved work carried forward from the recent turns, even if the current request is short or elliptical. Ignore candidates that only match older tangents. Return JSON only: {"relevant_node_ids":[...],"rationale":"..."}. Select at most 3 node ids.',
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
    const topHits = searchResults.results.slice(0, 5);
    const openedResults = await Promise.all(
      topHits.map((hit: any) =>
        sendAction({ action: 'open_node', node_id: String(hit.node_id), markdown: true })
          .then((opened) => ({ hit, opened }))
          .catch(() => null),
      ),
    );
    const candidates: ContextCandidate[] = [];
    for (const result of openedResults) {
      if (!result) continue;
      const { hit, opened } = result;
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

  private writeConversationPayload(
    logDir: string,
    runId: string,
    input: string,
    traces: TraceEvent[],
    answer: string,
  ): string {
    const conversationId = `conv_${runId}`;
    try {
      fs.mkdirSync(logDir, { recursive: true });
      const payload = {
        id: conversationId,
        timestamp: new Date().toISOString(),
        input,
        answer,
        traces
      };
      const filePath = path.join(logDir, `${conversationId}.json`);
      fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf-8');
    } catch {}
    return conversationId;
  }

  private appendConversationLog(runId: string, input: string, traces: TraceEvent[], answer: string): string {
    const result = this.writeConversationPayload(this.conversationLogDir, runId, input, traces, answer);
    // Remove the incremental sidecar now that the final log is written.
    try {
      const sidecarPath = path.join(this.conversationLogDir, `conv_${runId}.jsonl`);
      if (fs.existsSync(sidecarPath)) {
        fs.unlinkSync(sidecarPath);
      }
    } catch {}
    return result;
  }

  /** Write an in-progress placeholder so runs that never complete still appear on disk. */
  private openConversationLogInProgress(runId: string, input: string): void {
    try {
      const logDir = this.conversationLogDir;
      fs.mkdirSync(logDir, { recursive: true });
      const conversationId = `conv_${runId}`;
      const payload = {
        id: conversationId,
        timestamp: new Date().toISOString(),
        input,
        status: 'in_progress',
        answer: null,
        traces: []
      };
      const filePath = path.join(logDir, `${conversationId}.json`);
      fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf-8');
    } catch {}
  }

  /** Append a single trace event as a NDJSON line for incremental disk flushing. */
  private appendConversationTraceEvent(runId: string, event: TraceEvent): void {
    try {
      const logDir = this.conversationLogDir;
      const conversationId = `conv_${runId}`;
      const sidecarPath = path.join(logDir, `${conversationId}.jsonl`);
      const line = JSON.stringify({ ts: new Date().toISOString(), ...event }) + '\n';
      fs.appendFileSync(sidecarPath, line, 'utf-8');
    } catch {}
  }

  private appendMemoryJobSnapshot(runId: string, input: string, traces: TraceEvent[], answer: string): string {
    return this.writeConversationPayload(this.memoryJobLogDir, runId, input, traces, answer);
  }

  private findConversationPayloadPath(conversationId: string): string | null {
    const candidateDirs = [this.conversationLogDir, this.memoryJobLogDir];
    for (const dir of candidateDirs) {
      const filePath = path.join(dir, `${conversationId}.json`);
      if (fs.existsSync(filePath)) {
        return filePath;
      }
    }
    return null;
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
      const filePath = this.findConversationPayloadPath(row.conversation_id);
      if (!filePath) {
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
4. 不要逐字复制对话；保留抽象、可复用长期知识，并忽略寒暄、临时上下文、错误堆栈与所有调度包装文本（如“Original user request”“Active branch”“Branch goal”）。`;

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
    return {
      user_preferences: [],
      agent_skills: [],
      atomic_knowledge: [],
    };
  }
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
    const conversationId = this.appendMemoryJobSnapshot(runId, input, traces, answer);
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
    const requestedAgentMode: AgentMode = options.agentMode || 'branching';
    const runPhase = options.runPhase;
    const conversationId = options.conversationId || 'default';
    const baseSessionId = options.sessionId || `agent-run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const executionDiscipline = !runPhase
      ? await this.decideExecutionDiscipline(input, requestedAgentMode, conversationId)
      : { mode: 'branching' as const, reason: 'Execution discipline is already fixed by the current run phase.' };

    if (!runPhase && requestedAgentMode === 'branching' && executionDiscipline.mode === 'sequential') {
      onTrace({
        type: 'thought',
        content: `Using strict plan-and-execute mode. ${executionDiscipline.reason}`,
      });

      const planningInput = `Original user request:\n${input}\n\nPLAN STAGE ONLY.\nBuild the best executable plan before changing any files.\nWhen the plan is ready, call planner_final with a concise execution plan for a single sequential executor.`;
      const executionPlan = await this.run(planningInput, onTrace, {
        ...options,
        agentMode: 'branching',
        runPhase: 'plan',
        sessionId: `${baseSessionId}-plan`,
      });

      onTrace({
        type: 'observation',
        content: 'Planning stage completed. Switching to the sequential execute stage.',
      });

      const executionInput = `Original user request:\n${input}\n\nAPPROVED EXECUTION PLAN:\n${executionPlan}\n\nEXECUTE STAGE ONLY.\nFollow the approved plan sequentially.\nUse one direct work tool at a time.\nIf a tool observation disproves part of the plan, revise the plan narrowly and continue; do not branch.`;
      return this.run(executionInput, onTrace, {
        ...options,
        agentMode: 'react',
        runPhase: 'execute',
        sessionId: `${baseSessionId}-execute`,
      });
    }

    const perfEnabled = process.env.AGENT_PERF !== '0';
    const perfEntries: PerfEntry[] | null = perfEnabled ? [] : null;
    const traceBuffer: TraceEvent[] = [];
    // Generate runId early so the conversation log can be opened immediately.
    const runId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    // Write an in-progress placeholder right away so partial/crashed runs always land on disk.
    if (!runPhase) {
      this.openConversationLogInProgress(runId, input);
    }
    const runAgentId = options.agentId || 'agent-main';
    const agentMode: AgentMode = requestedAgentMode;
    const runSessionId = baseSessionId;
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
      // Incrementally flush each trace event to the sidecar NDJSON so that mid-run
      // state is recoverable even when the conversation never completes.
      if (!runPhase) {
        setImmediate(() => this.appendConversationTraceEvent(runId, event));
      }
    };
    let llmCircuitOpenedAt = 0;
    let llmCircuitReason = '';
    const isLlmLimitDetail = (detail: string): boolean =>
      /\busage limit exceeded\b/i.test(detail)
      || /\brate limit\b/i.test(detail)
      || /\bquota\b/i.test(detail)
      || /\btoo many requests\b/i.test(detail)
      || /\b429\b/i.test(detail)
      || /当前请求量较高/i.test(detail)
      || /按量付费 api/i.test(detail)
      || /更高级别套餐/i.test(detail);
    const getActiveLlmCircuitReason = (): string => {
      if (!llmCircuitReason) {
        return '';
      }
      if ((Date.now() - llmCircuitOpenedAt) > this.llmLimiterCircuitCooldownMs) {
        llmCircuitOpenedAt = 0;
        llmCircuitReason = '';
        return '';
      }
      return llmCircuitReason;
    };
    const openLlmCircuit = (detail: string, phase: string) => {
      llmCircuitOpenedAt = Date.now();
      llmCircuitReason = this.clipText(`${phase}: ${detail}`, 400);
      emitTrace({
        type: 'observation',
        content: `LLM request queue entered degraded mode after provider limit error. Remaining planner/model steps will fall back to local completion. Detail: ${llmCircuitReason}`,
      });
    };
    const buildLlmLimitFallbackDecision = (detail: string): ExecutablePlannerDecisionResult => ({
      decision: {
        kind: 'final',
        answer: 'LLM provider limit reached; finalize from gathered evidence.',
        status: 'partial',
        reason: detail,
      },
    });
    emitTrace({
      type: 'thought',
      content: runPhase === 'plan'
        ? 'Initializing plan stage (branching)...'
        : runPhase === 'execute'
          ? 'Initializing execute stage (classic ReAct, sequential)...'
          : agentMode === 'branching'
            ? 'Initializing branching ReAct context...'
            : 'Initializing classic ReAct context...',
    });

    const persistedConversationState = this.ensureConversationTurnsLoaded(conversationId);
    if (persistedConversationState?.recent_turns?.length) {
      emitTrace({
        type: 'observation',
        content: `Recovered persisted thread context with ${persistedConversationState.recent_turns.length} recent turn(s).`,
      });
    }

    const selectedConversationTurns = this.selectRelevantConversationTurns(input, conversationId);
    emitTrace({
      type: 'observation',
      content: selectedConversationTurns.length > 0
        ? `Selected ${selectedConversationTurns.length} relevant recent conversation turn(s) for follow-up grounding.`
        : 'Selected 0 recent conversation turns; treating this request as context-isolated.',
    });
    const selectedTurnSummaries = this.selectRelevantTurnSummaryRecords(input, conversationId);
    emitTrace({
      type: 'observation',
      content: selectedTurnSummaries.length > 0
        ? `Selected ${selectedTurnSummaries.length} relevant turn summary record(s) from journal replay.`
        : 'Selected 0 turn summary records from journal replay.',
    });

    let context = '';
    let recalledNodeIds: string[] = [];
    let selectedNodeIds: string[] = [];
    // Cache skill/souls reads across requests – re-read on first access only.
    if (!this.cachedSoulsMarkdown) {
      this.cachedSoulsMarkdown = this.loadSoulsMarkdown();
    }
    if (!this.cachedWorkspaceSkills) {
      this.cachedWorkspaceSkills = loadWorkspaceSkills(this.projectRoot, this.codeCliRoot);
    }
    const soulsGuidance = this.cachedSoulsMarkdown;
    const availableSkills = this.cachedWorkspaceSkills;
    const alwaysIncludeSkills = availableSkills.filter((s) => s.alwaysInclude);
    const localSkillsIndex = renderSkillCatalog(availableSkills);
    const rawAutoQueueMemorySave = String(process.env.AUTO_QUEUE_MEMORY_SAVE ?? '1').toLowerCase();
    const autoQueueMemorySave = rawAutoQueueMemorySave !== '0' && rawAutoQueueMemorySave !== 'false' && rawAutoQueueMemorySave !== 'off';
    let memoryQueuedThisRun = false;
    emitTrace({
      type: 'observation',
      content: 'Recalled 0 context candidate(s); selected 0 relevant node(s). Context retrieval is disabled for live turns.',
    });

    // ── Session carry-over from previous exhausted runs ─────────────────
    const carryOver = this.sessionCompressor.getCarryOver(conversationId);
    if (carryOver) {
      context = `${context}\n\n--- CARRY-OVER FROM PREVIOUS EXHAUSTED RUN(S) ---\n${carryOver.text}\n--- END CARRY-OVER ---`;
      emitTrace({
        type: 'observation',
        content: `Injected carry-over from ${carryOver.runCount} previous exhausted run(s) (≈${carryOver.tokenEstimate} tokens). The agent should continue from these findings rather than re-exploring.`,
      });
    }

    const persistedContextBlock = this.renderPersistedConversationState(persistedConversationState);
    if (persistedContextBlock) {
      context = `${persistedContextBlock}${context ? `\n\n${context}` : ''}`;
      emitTrace({
        type: 'observation',
        content: 'Injected persisted thread working state into the current run context.',
      });
    }
    const journalReplayBlock = this.renderTurnSummaryReplay(selectedTurnSummaries);
    if (journalReplayBlock) {
      context = `${journalReplayBlock}${context ? `\n\n${context}` : ''}`;
      emitTrace({
        type: 'observation',
        content: 'Injected journal replay summaries into the current run context.',
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
    const plannerToolSummary = [
      'Planner-visible tool families:',
      'read -> inspect workspace files, memory nodes, preferences, or skills',
      'search -> search workspace, memory, preferences, or skills',
      'edit -> modify workspace files or semantic memory/preferences/skills targets',
      'bash -> run local shell commands under governance',
      'web -> fetch a trusted URL only; do not invent URLs',
    ].join('\n');

    const alwaysIncludeBlock = alwaysIncludeSkills.length > 0
      ? `\n\n--- ALWAYS-ACTIVE SKILL POLICIES (pre-loaded, binding on every turn) ---\n${renderSkillGuidance(alwaysIncludeSkills)}\n--- END ALWAYS-ACTIVE SKILLS ---`
      : '';

    const isBranchingMode = agentMode === 'branching';
    const isPlanStage = runPhase === 'plan';
    const isExecuteStage = runPhase === 'execute';
    const joinPromptSections = (...sections: Array<string | false | null | undefined>): string =>
      sections.filter((section): section is string => typeof section === 'string' && section.trim().length > 0).join('\n\n');
    const renderPromptList = (items: string[], ordered = false): string =>
      items.map((item, index) => `${ordered ? `${index + 1}.` : '-'} ${item}`).join('\n');
    const clampNumber = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

    // ── Compact planner rules (shared between branching and sequential) ──
    const compactToolGuidance = `Tools: read, search, edit, bash, web (mode=fetch only, requires concrete URL).
Targets: workspace (files), memory (Mempedia), preferences, skills.
Never use edit target=workspace to write into raw .mempedia storage. Use semantic targets.`;
    const compactSkillGuidance = localSkillsIndex
      ? `Skills: ${localSkillsIndex}\nSkills are policy, not tool names. Pre-loaded always-active skills are binding; do not re-request them.`
      : 'Skills are guidance documents, not tool names. Pre-loaded always-active skills are binding.';
    const compactCompletionRules = `When done, call planner_final. Always set outcome (success/partial/failed) and disposition (resolved/missing_evidence/blocked_external/exhausted_search/planner_error). Include key error lines in outcome_reason when tests fail.
Do not save trivial exchanges to Mempedia core knowledge.`;
    const plainTextOnlyRule = 'Return only plain natural-language text for the user. Do not output tool calls, internal control markup, XML-like tags, or any structured wrapper markup.';

    // ── Prompt footers ──
    const sharedPromptFooter = joinPromptSections(
      alwaysIncludeBlock ? `${alwaysIncludeBlock}

IMPORTANT: The bash command patterns shown in the skill policies above are for use INSIDE bash tool-call arguments. They are not the planner response format.` : '',
      `Available tools:
${toolCatalog}`,
      `Shared context for this request:
${context || '(no context found)'}`,
      `Selected context node ids:
${selectedNodeIds.length > 0 ? selectedNodeIds.join(', ') : '(none)'}`,
      soulsGuidance ? `Global souls.md guidance:
${soulsGuidance}` : '',
    );
    const plannerPromptFooter = joinPromptSections(
      alwaysIncludeBlock ? `${alwaysIncludeBlock}

IMPORTANT: The bash command patterns shown in the skill policies above are for use INSIDE bash tool-call arguments. They are not the planner response format.` : '',
      plannerToolSummary,
    );
    const planSubagentFooter = joinPromptSections(
      alwaysIncludeBlock ? `${alwaysIncludeBlock}

IMPORTANT: The plan subagent does not execute work tools itself. It only returns canonical planning structure.` : '',
    );

    // ── Branching mode system prompt (compact) ──
    const buildBranchingModePrompt = (footer: string) => joinPromptSections(
      'You are a branching ReAct agent coordinated by MainAgent. A branch is an execution owner for one scoped slice of the canonical plan.',
      isPlanStage ? 'PLAN STAGE: produce one coherent execution plan for a single sequential executor via planner_final. Do not claim code has already been changed.' : '',
      `RULES:
${compactToolGuidance}
${compactSkillGuidance}

Control tools: planner_subagent (subagent=plan for initial branching / plan refresh / remediation), planner_final (finish), planner_skills (load skill guidance).
For work, call read/search/edit/bash/web directly. Never mix control + work tools. Never reply with plain text. Never output <think> tags or reasoning text - always call tools directly.

Execute your branch excerpt faithfully. Do not design sibling graphs — that belongs to planner_subagent with subagent=plan.
Request planner_subagent with subagent=plan when: no canonical plan exists, current plan is wrong, or a local gap needs remediation rebranch.
Respect execution_group and depends_on as binding constraints. Reuse branch_evidence_digest before issuing more search/fetch.
Prefer search before edit. Prefer workspace evidence before web.
${isPlanStage ? 'Finish by returning one executable plan through planner_final.' : isExecuteStage ? 'Follow the approved plan sequentially.' : 'Preserve the coordination graph implied by execution_group and depends_on.'}

All work artifacts are automatically persisted to .mitosis/work/ for future reference.

${compactCompletionRules}`,
      footer,
    );

    // ── Sequential mode system prompt (compact) ──
    const buildSequentialModePrompt = (footer: string) => joinPromptSections(
      'You are a classic ReAct agent. Think -> Act -> Observe -> repeat. One tool call at a time. Do NOT branch.',
      isExecuteStage ? 'EXECUTE STAGE: follow the approved plan sequentially. Revise only when evidence forces it.' : '',
      `RULES:
${compactToolGuidance}
${compactSkillGuidance}

Control: planner_final (finish), planner_skills (load guidance). NEVER call planner_subagent.
For work, call exactly one direct work tool: read/search/edit/bash/web. Never reply with plain text. Never output <think> tags or reasoning text - always call tools directly.
${isExecuteStage ? 'Treat the approved plan as binding.' : 'Stay sequential and do not branch.'}
Prefer search before edit. Prefer workspace evidence before web.

${compactCompletionRules}`,
      footer,
    );
    const buildPlanSubagentPrompt = (footer: string) => joinPromptSections(
          'You are a plan execution model. Your only job is to output valid plan_subagent_result tool calls.',
          `INPUT:
- Task state object (JSON)
- Current plan version and text
- Branch status summary

OUTPUT:
- Exactly one plan_subagent_result tool call
- No markdown, no prose, no reasoning, no <think> tags

SUCCESS CONDITION:
- Tool call matches schema exactly
- canonical_plan ≤ 4000 tokens
- Each branch has matching alignment entry
- No cycles in depends_on`,
          `PLANNING CONSTRAINTS:
${renderPromptList([
  'Reconcile user goal + current plan + branch status into coherent canonical_plan',
  'Compress aggressively: remove prose, deduplicate, stay under 4000 tokens',
  'Design only necessary branches for current scope',
  'Use execution_group for waves, depends_on for prerequisites',
  'Each branch needs alignment excerpt stating its responsibility',
  'Prefer one branch over many when scope is sequential',
], true)}`,
          footer,
        );

    const systemPrompt = isBranchingMode
      ? buildBranchingModePrompt(sharedPromptFooter)
      : buildSequentialModePrompt(sharedPromptFooter);
    const plannerSystemPrompt = isBranchingMode
      ? buildBranchingModePrompt(plannerPromptFooter)
      : buildSequentialModePrompt(plannerPromptFooter);
    const planSubagentPrompt = buildPlanSubagentPrompt(planSubagentFooter);

    // ── Stage-aware context budget computation ──────────────────────────
    const buildStageBudget = (
      prompt: string,
      conversationTokens: number,
      memoryTokens: number,
      safetyMargin = 4096,
    ) => computeContextBudget({
      model: this.model,
      systemPromptTokens: estimateTokens(prompt),
      conversationContextTokens: conversationTokens,
      memoryContextTokens: memoryTokens,
      safetyMargin,
    });
    const deriveMaxOutputTokens = (
      budget: ContextBudgetResult,
      floor: number,
      ceiling: number,
      ratio = 0.12,
    ) => clampNumber(
      Math.floor(Math.max(floor, budget.residualBudget * ratio)),
      floor,
      ceiling,
    );
    const planningCtxBudget = buildStageBudget(plannerSystemPrompt, 0, 0, 2048);
    const executionCtxBudget = buildStageBudget(
      systemPrompt,
      estimateTranscriptTokens(recentConversationMessages),
      0,
      4096,
    );
    const finalizeCtxBudget = buildStageBudget(
      systemPrompt,
      estimateTranscriptTokens(recentConversationMessages),
      0,
      3072,
    );
    const synthesisBasePrompt = `You are the synthesis stage for a branching ReAct agent. Merge completed branches into the best possible final answer. Prefer correctness and directness. If branches conflict, prefer the most directly evidenced result, then the latest successful remediation branch, then the answer with the fewest unsupported assumptions. Mention uncertainty only when meaningful conflict remains unresolved. If you mention saved node ids, use only ids explicitly listed in saved_nodes. Do not invent node ids. ${plainTextOnlyRule}${soulsGuidance ? `\n\n${soulsGuidance}` : ''}`;
    const planSubagentCtxBudget = buildStageBudget(planSubagentPrompt, 0, 0, 3072);
    const synthesisCtxBudget = buildStageBudget(synthesisBasePrompt, 0, 0, 3072);
    const plannerMaxTokens = deriveMaxOutputTokens(planningCtxBudget, 500, 1600, 0.08);
    const planSubagentMaxTokens = deriveMaxOutputTokens(planSubagentCtxBudget, 1000, 2600, 0.14);
    const finalizeMaxTokens = deriveMaxOutputTokens(finalizeCtxBudget, 500, 1400, 0.1);
    const synthesisMaxTokens = deriveMaxOutputTokens(synthesisCtxBudget, 700, 1800, 0.12);
    emitTrace({
      type: 'observation',
      content: `Context budget: model=${this.model} limit=${planningCtxBudget.modelLimit} committed=${planningCtxBudget.committedTokens} residual=${planningCtxBudget.residualBudget} transcriptChars=${planningCtxBudget.transcriptBudgetChars} agentMode=${agentMode} rpm=${this.llmRpmLimiter.disabled ? 'unlimited' : process.env.LLM_RPM_LIMIT} [plannerCommitted=${planningCtxBudget.committedTokens} planSubagentCommitted=${planSubagentCtxBudget.committedTokens} executeCommitted=${executionCtxBudget.committedTokens} finalizeCommitted=${finalizeCtxBudget.committedTokens} synthesisCommitted=${synthesisCtxBudget.committedTokens}]`,
    });

    const effectiveTranscriptBudgetChars = planningCtxBudget.transcriptBudgetChars;
    const canonicalPlanState: CanonicalPlanState = {
      version: 0,
      canonicalPlanText: '',
      tokenEstimate: 0,
      updatedByBranchId: 'B0',
      updatedAtStep: 0,
      deltaSummary: 'No canonical plan has been published yet.',
    };

    const extractText = (content: any): string => {
      if (typeof content === 'string') {
        return content;
      }
      if (content && typeof content === 'object') {
        if ((content as { type?: unknown }).type === 'openai_tool_result') {
          return typeof (content as OpenAIToolResultReplayContent).content === 'string'
            ? (content as OpenAIToolResultReplayContent).content
            : '';
        }
        if ((content as { type?: unknown }).type === 'openai_assistant_message') {
          const message = (content as OpenAIAssistantReplayContent).message || {};
          const parts: string[] = [];
          if (typeof message.content === 'string') {
            parts.push(message.content);
          }
          if (Array.isArray(message.reasoning_details)) {
            parts.push(JSON.stringify(message.reasoning_details));
          }
          if (Array.isArray(message.tool_calls)) {
            parts.push(JSON.stringify(message.tool_calls));
          }
          return parts.filter(Boolean).join('\n');
        }
      }
      if (Array.isArray(content)) {
        return content.map((item) => {
          if (typeof item === 'string') {
            return item;
          }
          if (item && typeof item === 'object') {
            if (typeof item.text === 'string') {
              return item.text;
            }
            if (typeof item.thinking === 'string') {
              return item.thinking;
            }
            if (typeof item.content === 'string') {
              return item.content;
            }
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
    const skillRouterGuidanceCharCap = 1400;

    const splitSkillSections = (content: string) => {
      const lines = content.split(/\r?\n/);
      const sections: Array<{ heading: string; body: string }> = [];
      let currentHeading = 'Overview';
      let currentBody: string[] = [];
      for (const line of lines) {
        const headingMatch = line.match(/^#{1,3}\s+(.+)$/);
        if (headingMatch) {
          if (currentBody.length > 0) {
            sections.push({ heading: currentHeading, body: currentBody.join('\n').trim() });
          }
          currentHeading = headingMatch[1].trim();
          currentBody = [];
          continue;
        }
        currentBody.push(line);
      }
      if (currentBody.length > 0) {
        sections.push({ heading: currentHeading, body: currentBody.join('\n').trim() });
      }
      return sections.filter((section) => section.body.length > 0);
    };

    const renderSkillRouterGuidance = (skills: SkillRecord[], query: string): string => {
      const queryTokens = Array.from(new Set(
        query.toLowerCase().match(/[\p{L}\p{N}_-]+/gu) || [],
      )).filter((token) => token.length >= 3);
      return skills.map((skill) => {
        const sections = splitSkillSections(skill.content);
        const matchingSections = sections.filter((section) => {
          const searchable = `${section.heading}\n${section.body}`.toLowerCase();
          return queryTokens.some((token) => searchable.includes(token));
        });
        const selectedSections = matchingSections.length > 0
          ? matchingSections.slice(0, 2)
          : sections.slice(0, 1);
        const renderedSections = selectedSections.map((section) =>
          `Section: ${section.heading}\n${this.clipText(section.body, 420)}`,
        ).join('\n\n');
        const snippet = this.clipText(
          renderedSections || this.clipText(skill.content, 420),
          skillRouterGuidanceCharCap,
        );
        return [
          `Skill: ${skill.name}`,
          `Description: ${skill.description}`,
          matchingSections.length > 0 ? 'Matched sections were selected because they overlap with the current request.' : 'Using the shortest summary-first guidance for this skill.',
          'Guidance:',
          snippet,
        ].join('\n');
      }).join('\n\n---\n\n');
    };

    const getLoadedSkillNames = (transcript: Array<{ role: string; content: ChatMessageContent }>) => {
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
      transcript: ChatMessage[],
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
          content: `Skill Router loaded summary-first guidance for the requested skills.\nOnly the most relevant sections are included by default to conserve planner context.\nIf a later step truly needs the full skill text, inspect it explicitly via the semantic skills tools.\n\n${renderSkillRouterGuidance(skillsToInject, originalUserRequest)}\n\nContinue the current ReAct step by calling tool(s), not plain text. Use read/search/edit/bash/web directly for work, and use planner_final or planner_subagent only for control flow.`
        },
      ];
    };

    const buildEvidenceFirstFallbackDecision = (): ExecutablePlannerDecision | null => {
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

    const traceMeta = (
      branch: Pick<BranchState, 'id' | 'parentId' | 'label' | 'depth' | 'steps'>,
      extra: Record<string, unknown> = {},
    ) => ({
      branchId: branch.id,
      parentBranchId: branch.parentId,
      branchLabel: branch.label,
      depth: branch.depth,
      step: branch.steps,
      ...extra,
    });

    const emitBranchTrace = (
      type: TraceEvent['type'],
      branch: Pick<BranchState, 'id' | 'parentId' | 'label' | 'depth' | 'steps'>,
      content: string,
      extra: Record<string, unknown> = {}
    ) => {
      emitTrace({ type, content, metadata: traceMeta(branch, extra) });
    };

    const summarizePlannerDecision = (decision: PlannerDecision): string => {
      if (decision.kind === 'tool') {
        const names = decision.tool_calls
          .map((toolCall: z.infer<typeof PlannerToolCallSchema>) => toolCall.name)
          .join(',');
        return `[planner] kind=tool tools=${names}`;
      }
      if (decision.kind === 'branch') {
        return `[planner] kind=branch branches=${decision.branches.length}`;
      }
      if (decision.kind === 'subagent') {
        return `[planner] kind=subagent subagent=${decision.subagent} task=${decision.task.slice(0, 50)}...`;
      }
      if (decision.kind === 'skills') {
        return `[planner] kind=skills skills=${decision.skills.join(',')}`;
      }
      return '[planner] kind=final';
    };

    const isPlannerTimeoutDetail = (detail: string): boolean =>
      /\bllm timeout after \d+ms\b/i.test(detail)
      || /\btimed out\b/i.test(detail)
      || /\btimeout\b/i.test(detail);

    const normalizePlannerResult = (
      value: PlannerDecisionResult | PlannerDecision,
    ): PlannerDecisionResult => {
      if (value && typeof value === 'object' && 'decision' in value && (value as PlannerDecisionResult).decision) {
        return value as PlannerDecisionResult;
      }
      return { decision: value as PlannerDecision };
    };

    const normalizePlannerBranches = (
      branches: Array<z.infer<typeof PlannerBranchSchema>>,
    ): PlannedBranch[] => branches.map((branch) => ({
      label: branch.label,
      goal: branch.goal,
      group: branch.group,
      depends: branch.depends,
    }));

    const trimTextToTokenBudget = (text: string, maxTokens: number, fallbackChars = 12000): string => {
      let normalized = String(text || '').trim();
      if (!normalized) {
        return '';
      }
      if (estimateTokens(normalized) <= maxTokens) {
        return normalized;
      }

      let candidate = normalized.slice(0, Math.min(normalized.length, fallbackChars));
      while (candidate.length > 200 && estimateTokens(candidate) > maxTokens) {
        candidate = candidate.slice(0, Math.max(200, Math.floor(candidate.length * 0.8))).trim();
      }
      return candidate;
    };

    const renderPlanSubagentFocus = (
      branch: BranchState,
      kanbanSnapshot?: BranchKanbanSnapshot,
    ): string => {
      const currentCard = kanbanSnapshot?.cards?.find((card) => card.branchId === branch.id);
      const blockerText = currentCard?.blockers?.length ? currentCard.blockers.join(' | ') : '(none)';
      return [
        `branch_id=${branch.id}`,
        `label=${branch.label}`,
        `goal=${branch.goal}`,
        `plan_version_seen=${branch.planVersionSeen ?? canonicalPlanState.version}`,
        `plan_excerpt=${branch.planExcerpt || '(none)'}`,
        `alignment_checks=${branch.alignmentChecks?.join(' | ') || '(none)'}`,
        `completion_summary=${branch.completionSummary || '(none)'}`,
        `outcome_reason=${branch.outcomeReason || '(none)'}`,
        `kanban_summary=${currentCard?.summary || '(none)'}`,
        `blockers=${blockerText}`,
      ].join('\n');
    };

    const planWithSkillRouting = async (
      initialTranscript: ChatMessage[],
      invoke: (transcript: ChatMessage[]) => Promise<PlannerDecisionResult | PlannerDecision>,
    ): Promise<ExecutablePlannerDecisionResult> => {
      const activeCircuitReason = getActiveLlmCircuitReason();
      if (activeCircuitReason) {
        emitTrace({
          type: 'observation',
          content: 'Planner request skipped because the LLM queue is already in degraded mode from an earlier provider limit error.',
        });
        return buildLlmLimitFallbackDecision(activeCircuitReason);
      }
      let workingTranscript = initialTranscript;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        let plannerResult: PlannerDecisionResult;
        try {
          plannerResult = normalizePlannerResult(await invoke(workingTranscript));
        } catch (error) {
          const detail = this.clipText(String((error as any)?.message || error || 'unknown planner error'), 400);
          logError(this.projectRoot, error, 'planner_decision_generation_failed');
          emitTrace({
            type: 'error',
            content: `Planner decision generation failed: ${detail}`,
          });
          if (isPlannerTimeoutDetail(detail)) {
            emitTrace({
              type: 'observation',
              content: 'Planner timed out while choosing the next step; forcing branch finalization from gathered evidence.',
            });
            return {
              decision: {
                kind: 'final',
                answer: 'Planner timed out; finalize from gathered evidence.',
                status: 'partial',
                reason: detail,
              },
            };
          }
          if (isLlmLimitDetail(detail)) {
            openLlmCircuit(detail, 'planner');
            emitTrace({
              type: 'observation',
              content: 'Planner hit a provider usage/rate limit; skipping further model planning and forcing branch finalization from gathered evidence.',
            });
            return buildLlmLimitFallbackDecision(detail);
          }
          const evidenceFirstDecision = buildEvidenceFirstFallbackDecision();
          if (evidenceFirstDecision) {
            emitTrace({
              type: 'observation',
              content: 'Planner failed before producing a structured decision; forcing local evidence inspection before answering.',
            });
            return { decision: evidenceFirstDecision };
          }
          return {
            decision: {
              kind: 'final',
              answer: '抱歉，内部规划步骤失败了，请重试一次。',
              status: 'failed',
            },
          };
        }

        const decision = plannerResult.decision;
        emitTrace({ type: 'observation', content: summarizePlannerDecision(decision) });
        if (decision.kind !== 'skills') {
          return plannerResult as ExecutablePlannerDecisionResult;
        }

        const requestedNames = decision.skills;
        const loadedSkillNames = getLoadedSkillNames(workingTranscript);
        const resolvedSkills = resolveSkillsByName(availableSkills, requestedNames, 2);
        const skillsToInject = resolvedSkills.filter((skill) => !loadedSkillNames.has(skill.name.toLowerCase()));
        if (skillsToInject.length === 0) {
          if (resolvedSkills.length > 0) {
            // All requested skills are already loaded — the LLM already has their
            // guidance in the transcript. Continue so the next attempt re-plans
            // without injecting anything new.
            emitTrace({
              type: 'observation',
              content: `Skill router: requested skills already loaded (${requestedNames.join(', ')}); continuing without re-injection.`,
            });
            continue;
          }
          emitTrace({
            type: 'observation',
            content: `Skill router could not resolve requested skills: ${requestedNames.join(', ') || '(none)'}.`,
          });
          return {
            decision: {
              kind: 'final',
              answer: '抱歉，当前请求的技能不可用。请重试或手动指定技能。',
              status: 'failed',
            },
          };
        }
        emitTrace({
          type: 'observation',
          content: `Skill router loaded local guidance: ${skillsToInject.map((skill) => skill.name).join(', ')}.`,
        });
        workingTranscript = appendSkillGuidance(workingTranscript, skillsToInject);
      }
      return {
        decision: {
          kind: 'final',
          answer: '抱歉，技能路由未能收敛为有效回答。请重试一次。',
          status: 'failed',
        },
      };
    };

    const executeSubagentDecision = async (
      branch: BranchState,
      kanbanSnapshot: BranchKanbanSnapshot | undefined,
      decision: PlannerSubagentDecision,
    ): Promise<AgentStep> => {
      const invocation: SubagentInvocation = {
        subagent: decision.subagent,
        task: decision.task,
        context: decision.context,
      };

      const result = await runSubagent(
        this.subagentRegistry,
        {
          projectRoot: this.projectRoot,
          branch,
          kanbanSnapshot,
          canonicalPlanState,
          originalUserRequest,
          input,
          plannerTemperature: this.plannerTemperature,
          planSubagentPrompt,
          planSubagentMaxTokens: planSubagentMaxTokens ?? 1600,
          clipText: (text, maxChars) => this.clipText(text, maxChars ?? 8000),
          estimateTokens,
          normalizePlannerBranches,
          trimTextToTokenBudget,
          renderPlanSubagentFocus,
          generateToolCalls: async (options) => this.measure(
            perfEntries,
            options.timeoutLabel,
            async () => this.llmRpmLimiter.run(() => this.withTimeout(
              generateToolCalls({
                model: this.openai,
                messages: options.messages,
                tools: options.tools,
                temperature: options.temperature,
                maxTokens: options.maxTokens,
              }),
              this.agentLlmTimeoutMs,
              options.timeoutLabel,
            )),
          ),
        },
        invocation,
      );

      if (result.canonicalPlanUpdate) {
        canonicalPlanState.version = result.canonicalPlanUpdate.planVersion;
        canonicalPlanState.canonicalPlanText = result.canonicalPlanUpdate.canonicalPlan;
        canonicalPlanState.tokenEstimate = estimateTokens(result.canonicalPlanUpdate.canonicalPlan);
        canonicalPlanState.updatedByBranchId = branch.id;
        canonicalPlanState.updatedAtStep = branch.steps + 1;
        canonicalPlanState.deltaSummary = result.canonicalPlanUpdate.planDeltaSummary;
        branch.planVersionSeen = canonicalPlanState.version;
        emitTrace({
          type: 'observation',
          content: `Canonical plan updated to version ${canonicalPlanState.version} by ${branch.id}. ${canonicalPlanState.deltaSummary}`,
        });
      }

      if (result.traceSummary) {
        emitTrace({
          type: 'observation',
          content: result.traceSummary,
        });
      }

      if (result.nextStep) {
        return result.nextStep;
      }

      const error = new Error(`Subagent ${decision.subagent} produced no executable step: ${result.traceSummary}`);
      logError(this.projectRoot, error, `subagent_no_nextstep_${decision.subagent}`);

      return {
        kind: 'final',
        content: '抱歉，当前请求的子代理尚未启用。请改用直接工具或稍后再试。',
        outcome: 'partial',
        outcomeReason: result.traceSummary || `Subagent ${decision.subagent} produced no executable step.`,
        disposition: 'planner_error',
      };
    };

    const clearPendingProviderToolState = (branch: BranchState) => {
      branch.pendingAnthropicAssistantContent = null;
      branch.pendingAnthropicToolUseIds = [];
      branch.pendingOpenAIAssistantMessage = null;
    };

    const applyPlannerResultToBranch = (branch: BranchState, result: PlannerDecisionResult) => {
      if (result.decision.kind === 'tool' && this.openai.provider === 'anthropic') {
        branch.pendingAnthropicAssistantContent = result.anthropicAssistantContent ?? null;
        branch.pendingAnthropicToolUseIds = result.anthropicToolUseIds ?? [];
        branch.pendingOpenAIAssistantMessage = null;
        return;
      }
      if (result.decision.kind === 'tool' && this.openai.provider === 'openai') {
        branch.pendingOpenAIAssistantMessage = result.openaiAssistantMessage ?? null;
        branch.pendingAnthropicAssistantContent = null;
        branch.pendingAnthropicToolUseIds = [];
        return;
      }
      clearPendingProviderToolState(branch);
    };

    const buildAnthropicToolTranscript = (
      branch: BranchState,
      observations: Array<{ result: string; success: boolean }>,
    ): Array<{ role: 'assistant' | 'user'; content: ChatMessageContent }> | null => {
      if (this.openai.provider !== 'anthropic') {
        return null;
      }
      const assistantContent = branch.pendingAnthropicAssistantContent;
      const toolUseIds = branch.pendingAnthropicToolUseIds ?? [];
      clearPendingProviderToolState(branch);
      if (!assistantContent || assistantContent.length === 0 || toolUseIds.length !== observations.length) {
        return null;
      }

      const toolResults: AnthropicToolResultContentBlock[] = observations.map((observation, index) => ({
        type: 'tool_result',
        tool_use_id: toolUseIds[index],
        content: observation.result,
        ...(observation.success ? {} : { is_error: true }),
      }));

      return [
        { role: 'assistant', content: assistantContent },
        { role: 'user', content: toolResults },
      ];
    };

    const buildOpenAIToolTranscript = (
      branch: BranchState,
      observations: Array<{ result: string; success: boolean }>,
    ): Array<{ role: 'assistant' | 'user'; content: ChatMessageContent }> | null => {
      if (this.openai.provider !== 'openai') {
        return null;
      }
      const assistantMessage = branch.pendingOpenAIAssistantMessage;
      clearPendingProviderToolState(branch);
      const toolCallIds = Array.isArray(assistantMessage?.tool_calls)
        ? assistantMessage.tool_calls
          .map((toolCall: any) => (typeof toolCall?.id === 'string' ? toolCall.id.trim() : ''))
          .filter(Boolean)
        : [];
      if (!assistantMessage || toolCallIds.length < observations.length) {
        return null;
      }

      const assistantReplay: OpenAIAssistantReplayContent = {
        type: 'openai_assistant_message',
        message: assistantMessage,
      };
      const toolResults: OpenAIToolResultReplayContent[] = observations.map((observation, index) => ({
        type: 'openai_tool_result',
        tool_call_id: toolCallIds[index],
        content: observation.result,
      }));

      return [
        { role: 'assistant', content: assistantReplay },
        ...toolResults.map((toolResult) => ({ role: 'user' as const, content: toolResult })),
      ];
    };

    const buildProviderToolTranscript = (
      branch: BranchState,
      observations: Array<{ result: string; success: boolean }>,
    ): Array<{ role: 'assistant' | 'user'; content: ChatMessageContent }> | null =>
      buildAnthropicToolTranscript(branch, observations)
      || buildOpenAIToolTranscript(branch, observations);

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
      const branchSummary = branch.finalAnswer || branch.completionSummary || branch.transcript.slice(-6).map((item) => extractText(item.content)).join('\n\n');
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

    const buildPlannerMessages = (
      branch: BranchState,
      planningTranscript: ChatMessage[],
      kanbanSnapshot?: BranchKanbanSnapshot,
    ): ChatMessage[] => {
      const compareKanbanCards = (left: BranchKanbanCard, right: BranchKanbanCard) =>
        left.branchId.localeCompare(right.branchId, undefined, { numeric: true, sensitivity: 'base' });

      const renderPlannerKanbanCard = (card: BranchKanbanCard) => this.clipText([
        `[${card.branchId}] ${card.label}`,
        `status=${card.status}`,
        card.disposition ? `disposition=${card.disposition}` : '',
        card.outcome ? `outcome=${card.outcome}` : '',
        `goal=${card.goal}`,
        card.summary ? `summary=${card.summary}` : '',
        card.blockers?.length ? `blockers=${card.blockers.join(' | ')}` : '',
      ].filter(Boolean).join(' | '), 420);

      const renderPlannerKanbanSummary = (snapshot?: BranchKanbanSnapshot) => {
        if (!snapshot?.cards?.length) {
          return '(none)';
        }
        const cards = snapshot.cards.slice();
        const currentCard = cards.find((card) => card.branchId === branch.id);
        // Only show sibling/child cards that have meaningful results (completed or errored).
        const siblingCards = cards
          .filter((card) => card.parentBranchId === branch.parentId && card.branchId !== branch.id && (card.status === 'completed' || card.status === 'error'))
          .sort(compareKanbanCards)
          .slice(0, 4);
        const childCards = cards
          .filter((card) => card.parentBranchId === branch.id && (card.status === 'completed' || card.status === 'error'))
          .sort(compareKanbanCards)
          .slice(0, 4);
        const lines = [
          `total=${snapshot.summary.total} active=${snapshot.summary.active} completed=${snapshot.summary.completed} error=${snapshot.summary.error}`,
        ];
        if (currentCard) {
          lines.push(`self: ${renderPlannerKanbanCard(currentCard)}`);
        }
        for (const card of siblingCards) {
          lines.push(`sibling: ${renderPlannerKanbanCard(card)}`);
        }
        for (const card of childCards) {
          lines.push(`child: ${renderPlannerKanbanCard(card)}`);
        }
        return lines.join('\n');
      };

      const renderStructuredBranchHandoff = (handoff?: BranchHandoff) => {
        if (!handoff) {
          return '';
        }
        const fields = [
          `canonical_task=${handoff.canonicalTaskId}`,
          handoff.retryOfBranchId ? `retry_of=${handoff.retryOfBranchId}` : '',
          handoff.disposition ? `disposition=${handoff.disposition}` : '',
          handoff.missingFields?.length ? `missing_fields=${handoff.missingFields.join(' | ')}` : '',
          handoff.blockedBy?.length ? `blocked_by=${handoff.blockedBy.join(' | ')}` : '',
          handoff.mustNotRepeat?.length ? `must_not_repeat=${handoff.mustNotRepeat.join(' | ')}` : '',
        ].filter(Boolean);
        return fields.join('\n');
      };

      const uniqueCompactLines = (values: Array<string | undefined | null>) => {
        const seen = new Set<string>();
        return values
          .map((value) => this.clipText(String(value || '').replace(/\s+/g, ' ').trim(), 160))
          .filter((value) => {
            if (!value || seen.has(value)) {
              return false;
            }
            seen.add(value);
            return true;
          });
      };

      const renderBranchEvidenceDigest = () => {
        const currentCard = kanbanSnapshot?.cards?.find((card) => card.branchId === branch.id);
        const recentMessages = branch.transcript.slice(-4);
        const recentText = recentMessages
          .map((message) => this.clipText(extractText(message.content).replace(/\s+/g, ' ').trim(), 160))
          .filter(Boolean);
        const lastObservation = [...recentText]
          .reverse()
          .find((line) => /TOOL OBSERVATION|blocked near-duplicate web search loop|Branch completed/i.test(line));
        const verifiedFacts = uniqueCompactLines([
          currentCard?.summary,
          branch.completionSummary,
        ]).slice(0, 2);
        const blockers = uniqueCompactLines([
          ...(currentCard?.blockers || []),
          branch.outcomeReason,
        ]).slice(0, 2);
        return [
          `last_tool: ${lastObservation || '(none)'}`,
          verifiedFacts.length > 0 ? `facts: ${verifiedFacts.join(' | ')}` : '',
          blockers.length > 0 ? `blockers: ${blockers.join(' | ')}` : '',
          currentCard?.artifacts?.length ? `artifacts: ${currentCard.artifacts.slice(0, 3).join(', ')}` : '',
        ].filter(Boolean).join('\n');
      };

      // ── Compact metadata message ──
      const handoffBlock = renderStructuredBranchHandoff(branch.handoff);
      const retryBlock = branch.sharedHandoff ? `retry_round=${branch.sharedHandoff.retryIndex}` : '';
      const metadataContent = `branch=${branch.id} label=${branch.label} depth=${branch.depth} steps=${branch.steps}
goal: ${branch.goal}
plan_v${branch.planVersionSeen ?? canonicalPlanState.version}: ${branch.planExcerpt || '(none)'}
${branch.alignmentChecks?.length ? `checks: ${branch.alignmentChecks.join(' | ')}` : ''}
${handoffBlock}${retryBlock ? `\n${retryBlock}` : ''}
evidence: ${renderBranchEvidenceDigest()}
kanban: ${renderPlannerKanbanSummary(kanbanSnapshot)}
${canonicalPlanState.canonicalPlanText ? `canonical_plan (v${canonicalPlanState.version}):\n${this.clipText(canonicalPlanState.canonicalPlanText, branch.depth > 0 ? 800 : 2000)}` : 'No canonical plan yet — call planner_subagent with subagent=plan if branching is needed.'}

user_request: ${this.clipText(originalUserRequest || input, 500)}
Respond with tool calls. Use read/search/edit/bash/web for work. Use planner_subagent/planner_final/planner_skills for control. Never mix control + work tools.`.replace(/\n{3,}/g, '\n\n');

      // ── Dynamic transcript windowing ──
      // Child branches (depth > 0) omit recentConversationMessages: they execute a specific
      // subgoal and do not need prior conversation history. Including it can push the combined
      // context (conv history + canonical plan + tools) past the model's per-call limit.
      const activeConversationMessages = branch.depth > 0 ? [] : recentConversationMessages;
      // Calculate how many transcript tokens we can afford after envelope overhead.
      const envelopeTokens = estimateTokens(plannerSystemPrompt)
        + estimateTranscriptTokens(activeConversationMessages)
        + estimateTokens(metadataContent)
        + (plannerMaxTokens || 1600)
        + 2048; // safety margin
      const transcriptBudgetTokens = Math.max(0, planningCtxBudget.modelLimit - envelopeTokens);
      const transcriptBudgetChars = Math.max(4000, transcriptBudgetTokens * 4);
      const windowedTranscript = compressTranscript(planningTranscript, transcriptBudgetChars);

      return [
        { role: 'system', content: plannerSystemPrompt },
        ...activeConversationMessages,
        ...windowedTranscript,
        {
          role: 'user' as const,
          content: metadataContent,
        },
      ] as ChatMessage[];
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

    const buildSafeBranchDisplayFallback = (branch: Pick<BranchState, 'label' | 'completionSummary' | 'finalAnswer'>): string => {
      const safeSummary = this.sanitizeUserFacingAnswer(branch.completionSummary || '', '');
      if (safeSummary) {
        return safeSummary;
      }
      const safeFirstSentence = this.firstSentence(this.sanitizeUserFacingAnswer(branch.finalAnswer || '', ''));
      if (safeFirstSentence) {
        return safeFirstSentence;
      }
      return `分支 ${branch.label} 已完成，但最终内容不可安全展示。`;
    };

    const finalizeFromBranch = async (branch: BranchState, reason: string, fallbackAnswer?: string): Promise<string> => {
      emitBranchTrace('thought', branch, `Forcing finalization for ${branch.id}: ${reason}`);
      const activeCircuitReason = getActiveLlmCircuitReason();
      if (activeCircuitReason) {
        emitBranchTrace(
          'observation',
          branch,
          `Skipping branch finalization model call because the LLM queue is in degraded mode: ${activeCircuitReason}`,
        );
        return this.sanitizeUserFacingAnswer(fallbackAnswer || '', buildSafeBranchDisplayFallback(branch))
          || buildSafeBranchDisplayFallback(branch);
      }
      const compressedTranscript = compressBranchTranscript(branch.transcript, { tailKeep: 6, maxSummaryChars: 2000 });
      const { text: _finalizeText } = await this.measure(perfEntries, `finalize_${branch.id}`, async () => {
        return this.llmRpmLimiter.run(() => this.withTimeout(
          generateText({
            model: this.openai,
            temperature: this.responseTemperature,
            maxTokens: finalizeMaxTokens,
            messages: ([
              { role: 'system', content: `${systemPrompt}\nYou must now finish. Do not branch. Do not call tools. ${plainTextOnlyRule}` },
              ...recentConversationMessages,
              ...compressedTranscript,
              { role: 'user', content: `Finalize branch ${branch.id}. User request:\n${input}\n\nReason: ${reason}` },
            ] as any),
          }),
          this.agentLlmTimeoutMs,
          `finalize_${branch.id} llm`
        ));
      });
      const fallback = buildSafeBranchDisplayFallback(branch);
      return this.sanitizeUserFacingAnswer(extractText(_finalizeText).trim(), fallback)
        || fallback;
    };

    const buildDeterministicSynthesisFallback = (branches: BranchState[]): string => {
      const sections = branches
        .map((branch) => {
          const safeAnswer = this.sanitizeUserFacingAnswer(
            branch.finalAnswer || '',
            buildSafeBranchDisplayFallback(branch),
          );
          if (!safeAnswer) {
            return '';
          }
          return `## ${branch.label}\n${this.clipText(safeAnswer, 1600)}`;
        })
        .filter(Boolean);

      if (sections.length === 0) {
        return '抱歉，分支汇总阶段返回了内部格式，当前没有可安全展示的最终总结。';
      }

      return `以下是各已完成分支的整理结果：\n\n${sections.join('\n\n')}`;
    };

    const canonicalizeRemediationLabel = (label: string): string => {
      let current = String(label || '').replace(/\s+/g, ' ').trim();
      while (/^fix:\s*/i.test(current)) {
        current = current.replace(/^fix:\s*/i, '').trim();
      }
      return current || 'unnamed branch';
    };

    type SynthesisAttemptRecord = {
      index: number;
      branch: BranchState;
      canonicalLabel: string;
      disposition: BranchDisposition;
      ancestorCanonicalLabels: string[];
    };

    const normalizeSynthesisText = (value: string): string => String(value || '').replace(/\s+/g, ' ').trim();

    const splitSynthesisFragments = (value: string): string[] =>
      normalizeSynthesisText(value)
        .split(/(?<=[。！？.!?])\s+|[\n\r]+|(?<=;)\s+/u)
        .map((fragment) => fragment.trim().replace(/^[\-\d.)\s]+/, '').trim())
        .filter(Boolean);

    const uniqueFragments = (values: string[]): string[] => {
      const seen = new Set<string>();
      return values.filter((value) => {
        const normalized = normalizeSynthesisText(value);
        if (!normalized || seen.has(normalized)) {
          return false;
        }
        seen.add(normalized);
        return true;
      });
    };

    const pickFragments = (value: string, patterns: RegExp[], limit = 3): string[] =>
      uniqueFragments(
        splitSynthesisFragments(value).filter((fragment) => patterns.some((pattern) => pattern.test(fragment))),
      ).slice(0, limit);

    const extractPriorResultSnippet = (branch: BranchState): string =>
      this.clipText(
        this.sanitizeUserFacingAnswer(branch.finalAnswer || '', branch.completionSummary || '') || '(none)',
        260,
      );

    const hasNegativeSynthesisSignal = (branch: BranchState): boolean => {
      const signalText = normalizeSynthesisText([
        branch.outcomeReason || '',
        branch.completionSummary || '',
        branch.finalAnswer || '',
      ].join('\n'));
      if (!signalText) {
        return false;
      }
      return /planner decision generation failed|planner format fallback|schema repair|timeout|timed out|provider timeout|内部规划步骤失败|403|404|forbidden|fetch failed|access denied|访问限制|被屏蔽|无法访问|paywall|需要登录|动态js|js动态|blocked|已尝试\d+轮|历经\d+轮|多次搜索尝试均未能|连续多轮|用尽|搜索持续被.+淹没|no further .* likely|exhausted|all attempts|已用尽搜索策略|未获取到|未能获取|缺少|仍缺|remaining gap|still missing|not directly captured|无法确认|uncertainty|不确定性|未抓取到/i.test(signalText);
    };

    const isSuccessfulSynthesisBranch = (branch: BranchState): boolean => {
      if (branch.outcome === 'success') {
        return true;
      }
      if (branch.outcome === 'failed' || branch.outcome === 'partial') {
        return false;
      }
      if (!branch.finalAnswer || branch.finalAnswer.length === 0) {
        return false;
      }
      return !hasNegativeSynthesisSignal(branch);
    };

    const inferBranchDisposition = (branch: BranchState): BranchDisposition => {
      if (branch.disposition && branch.disposition !== 'unknown') {
        return branch.disposition;
      }
      if (isSuccessfulSynthesisBranch(branch)) {
        return 'resolved';
      }

      const signalText = normalizeSynthesisText([
        branch.outcomeReason || '',
        branch.completionSummary || '',
        branch.finalAnswer || '',
      ].join('\n'));

      if (!signalText) {
        return branch.outcome === 'failed' || branch.outcome === 'partial'
          ? 'missing_evidence'
          : 'unknown';
      }

      if (/planner decision generation failed|planner format fallback|schema repair|timeout|timed out|provider timeout|内部规划步骤失败/i.test(signalText)) {
        return 'planner_error';
      }
      if (/403|404|forbidden|fetch failed|access denied|访问限制|被屏蔽|无法访问|paywall|需要登录|动态js|js动态|blocked/i.test(signalText)) {
        return 'blocked_external';
      }
      if (/已尝试\d+轮|历经\d+轮|多次搜索尝试均未能|连续多轮|用尽|搜索持续被.+淹没|no further .* likely|exhausted|all attempts|已用尽搜索策略/i.test(signalText)) {
        return 'exhausted_search';
      }
      if (/未获取到|未能获取|缺少|仍缺|remaining gap|still missing|not directly captured|无法确认|uncertainty|不确定性|未抓取到/i.test(signalText)) {
        return 'missing_evidence';
      }

      if (branch.outcome === 'failed' || branch.outcome === 'partial') {
        return 'missing_evidence';
      }
      return 'unknown';
    };

    const isAncestorBranchId = (ancestorId: string, branchId: string): boolean =>
      ancestorId.length > 0 && branchId.startsWith(`${ancestorId}.`);

    const extractMissingFields = (branch: BranchState): string[] => pickFragments(
      [branch.outcomeReason, branch.completionSummary, branch.finalAnswer].filter(Boolean).join('\n'),
      [
        /未获取到/i,
        /未能获取/i,
        /缺少/i,
        /仍缺/i,
        /remaining gap/i,
        /still missing/i,
        /not directly captured/i,
        /未抓取到/i,
        /无法确认/i,
      ],
    );

    const extractBlockedBy = (branch: BranchState): string[] => pickFragments(
      [branch.outcomeReason, branch.completionSummary, branch.finalAnswer].filter(Boolean).join('\n'),
      [
        /403/i,
        /404/i,
        /forbidden/i,
        /fetch failed/i,
        /访问限制/i,
        /被屏蔽/i,
        /无法访问/i,
        /需要登录/i,
        /动态js/i,
        /js动态/i,
        /timeout/i,
      ],
    );

    const extractExhaustedStrategies = (branch: BranchState): string[] => pickFragments(
      [branch.outcomeReason, branch.completionSummary, branch.finalAnswer].filter(Boolean).join('\n'),
      [
        /已尝试/i,
        /历经/i,
        /多次搜索/i,
        /连续多轮/i,
        /用尽/i,
        /搜索持续被.+淹没/i,
        /no further .* likely/i,
        /exhausted/i,
        /all attempts/i,
      ],
    );

    const extractKnownGoodFacts = (branch: BranchState): string[] => {
      const priorResult = extractPriorResultSnippet(branch);
      if (!priorResult || priorResult === '(none)') {
        return [];
      }
      const positiveFragments = pickFragments(
        priorResult,
        [
          /成功/i,
          /已获取/i,
          /confirmed/i,
          /captured/i,
          /获取到了/i,
          /数据/i,
          /\d/,
        ],
        3,
      );
      return positiveFragments.length > 0 ? positiveFragments : [priorResult];
    };

    const buildMustNotRepeat = (disposition: BranchDisposition, branch: BranchState): string[] => {
      const instructions = [
        'Do not restart the same investigation from scratch.',
        'Reuse the previous evidence before planning fresh searches or fetches.',
      ];

      if (disposition === 'blocked_external') {
        instructions.push('Do not revisit the same blocked/paywalled/dynamic pages unless you have a genuinely new access path, API, or mirror.');
      }
      if (disposition === 'exhausted_search') {
        instructions.push('Do not issue more generic searches in the same source family without a materially new hypothesis, domain, language, or source type.');
      }
      if (disposition === 'planner_error') {
        instructions.push('Do not discard already collected evidence just because the previous planner step failed or timed out.');
      }
      if (branch.goal) {
        instructions.push(`Stay focused on the unresolved gap inside: ${this.clipText(branch.goal, 120)}`);
      }
      return uniqueFragments(instructions).slice(0, 4);
    };

    type GoalAssessment = {
      goalAchieved: boolean;
      achievedWork: string[];
      remainingWork: string[];
      unresolvedAttempts: Array<{ record: SynthesisAttemptRecord; handoff: BranchHandoff }>;
      summaryText: string;
    };

    const describeSuccessfulWork = (record: SynthesisAttemptRecord): string => this.clipText(
      `[${record.branch.id}/${record.branch.label}] ${record.branch.completionSummary || record.branch.finalAnswer || record.branch.goal || 'completed work'}`,
      260,
    );

    const describeRemainingWork = (record: SynthesisAttemptRecord, handoff: BranchHandoff): string => {
      const fragments = [
        `[${record.branch.id}/${record.branch.label}] ${record.canonicalLabel}`,
        `status=${record.disposition}`,
      ];
      if (handoff.knownGoodFacts?.length) {
        fragments.push(`verified progress: ${handoff.knownGoodFacts.join(' | ')}`);
      }
      if (handoff.missingFields?.length) {
        fragments.push(`still needed: ${handoff.missingFields.join(' | ')}`);
      } else if (handoff.priorIssue) {
        fragments.push(`remaining gap: ${handoff.priorIssue}`);
      }
      if (handoff.blockedBy?.length) {
        fragments.push(`blockers: ${handoff.blockedBy.join(' | ')}`);
      }
      return this.clipText(fragments.join(' | '), 320);
    };

    const buildGoalAssessment = (
      successfulAttempts: SynthesisAttemptRecord[],
      unresolvedAttempts: SynthesisAttemptRecord[],
      attemptsByCanonical: Map<string, SynthesisAttemptRecord[]>,
    ): GoalAssessment => {
      const unresolvedWithHandoff = unresolvedAttempts.map((record) => ({
        record,
        handoff: buildBranchHandoff(record, attemptsByCanonical),
      }));
      const achievedWork = uniqueFragments(successfulAttempts.map((record) => describeSuccessfulWork(record)));
      const remainingWork = uniqueFragments(
        unresolvedWithHandoff.map(({ record, handoff }) => describeRemainingWork(record, handoff)),
      );
      const goalAchieved = unresolvedWithHandoff.length === 0;
      return {
        goalAchieved,
        achievedWork,
        remainingWork,
        unresolvedAttempts: unresolvedWithHandoff,
        summaryText: [
          `goal_achieved=${goalAchieved ? 'yes' : 'no'}`,
          `achieved_work:\n${achievedWork.length ? achievedWork.join('\n') : '(none yet)'}`,
          `remaining_work:\n${remainingWork.length ? remainingWork.join('\n') : '(none)'}`,
        ].join('\n'),
      };
    };

    const resolveLatestSynthesisAttempts = (branches: BranchState[]) => {
      const latestByCanonical = new Map<string, SynthesisAttemptRecord>();
      const attemptsByCanonical = new Map<string, SynthesisAttemptRecord[]>();

      branches.forEach((branch, index) => {
        const canonicalLabel = canonicalizeRemediationLabel(branch.label);
        const record: SynthesisAttemptRecord = {
          index,
          branch,
          canonicalLabel,
          disposition: inferBranchDisposition(branch),
          ancestorCanonicalLabels: uniqueFragments(
            (branch.ancestorBranchLabels || []).map((label) => canonicalizeRemediationLabel(label)),
          ),
        };
        latestByCanonical.set(canonicalLabel, record);
        const existing = attemptsByCanonical.get(canonicalLabel) || [];
        existing.push(record);
        attemptsByCanonical.set(canonicalLabel, existing);
      });

      const latestAttempts = [...latestByCanonical.values()]
        .sort((left, right) => left.index - right.index);

      const frontierAttempts = latestAttempts.filter((record) =>
        !latestAttempts.some((other) =>
          other !== record
          && (
            isAncestorBranchId(record.branch.id, other.branch.id)
            || other.ancestorCanonicalLabels.includes(record.canonicalLabel)
          ),
        ),
      );

      const supersededAttempts = latestAttempts
        .filter((record) => !frontierAttempts.includes(record))
        .map((record) => ({ ...record, disposition: 'superseded' as BranchDisposition }));

      const successfulAttempts = frontierAttempts.filter((record) => record.disposition === 'resolved');
      const unresolvedAttempts = frontierAttempts.filter((record) => record.disposition !== 'resolved');

      return {
        latestAttempts,
        frontierAttempts,
        successfulAttempts,
        unresolvedAttempts,
        attemptsByCanonical,
        supersededAttempts,
      };
    };

    const buildBranchHandoff = (
      record: SynthesisAttemptRecord,
      attemptsByCanonical: Map<string, SynthesisAttemptRecord[]>,
    ): BranchHandoff => {
      const branch = record.branch;
      const priorAttempts = (attemptsByCanonical.get(record.canonicalLabel) || [record])
        .sort((left, right) => left.index - right.index)
        .map((attempt) => attempt.branch.id);

      return {
        canonicalTaskId: record.canonicalLabel,
        retryOfBranchId: branch.id,
        priorAttemptIds: priorAttempts,
        disposition: record.disposition,
        priorIssue: this.clipText(branch.outcomeReason || branch.completionSummary || 'unknown reason', 180),
        priorResultSnippet: extractPriorResultSnippet(branch),
        knownGoodFacts: extractKnownGoodFacts(branch),
        missingFields: extractMissingFields(branch),
        exhaustedStrategies: extractExhaustedStrategies(branch),
        blockedBy: extractBlockedBy(branch),
        mustNotRepeat: buildMustNotRepeat(record.disposition, branch),
      };
    };

    const synthesizeCompletedBranches = async (
      branches: BranchState[],
      archivedBranchSummary = '',
      synthesisRetry = 0,
      maxSynthesisRetries = 2,
    ): Promise<string | SynthesisResult> => {
      const {
        frontierAttempts,
        successfulAttempts,
        unresolvedAttempts,
        attemptsByCanonical,
        supersededAttempts,
      } = resolveLatestSynthesisAttempts(branches);
      const goalAssessment = buildGoalAssessment(successfulAttempts, unresolvedAttempts, attemptsByCanonical);
      const rebranchCandidates = goalAssessment.unresolvedAttempts.filter(({ record }) =>
        record.branch.finalizationMode !== 'planner_fallback',
      );

      emitTrace({
        type: 'observation',
        content: `Goal assessment before synthesis:\n${goalAssessment.summaryText}`,
      });

      if (!goalAssessment.goalAchieved && !this.rebranchEnabled) {
        emitTrace({
          type: 'observation',
          content: `Rebranch disabled by REACT_REBRANCH_ENABLED; synthesizing current frontier with ${goalAssessment.unresolvedAttempts.length} unmet workstream(s) left as-is.`,
        });
      }

      if (!goalAssessment.goalAchieved && this.rebranchEnabled && rebranchCandidates.length === 0) {
        emitTrace({
          type: 'observation',
          content: 'Skipping auto re-branch because the remaining unmet work only comes from planner fallback finalizations in this run.',
        });
      }

      if (!goalAssessment.goalAchieved && this.rebranchEnabled && synthesisRetry < maxSynthesisRetries && rebranchCandidates.length > 0) {
        emitTrace({
          type: 'thought',
          content: `Current work does not yet satisfy the user goal: ${goalAssessment.unresolvedAttempts.length} unmet workstream(s), ${successfulAttempts.length} resolved workstream(s), ${supersededAttempts.length} superseded ancestor attempt(s). Requesting remediation re-branch for ${rebranchCandidates.length} workstream(s) (retry ${synthesisRetry + 1}/${maxSynthesisRetries}).`,
        });

        const successContext = successfulAttempts
          .map((record) => `[${record.branch.id}/${record.branch.label}] (success): ${this.clipText(record.branch.finalAnswer || record.branch.completionSummary || '', 800)}`)
          .join('\n\n');

        const remediationBranches = rebranchCandidates.map(({ record, handoff }) => {
          const branch = record.branch;
          const knownGoodText = handoff.knownGoodFacts?.length ? ` Verified progress to preserve: ${handoff.knownGoodFacts.join(' | ')}` : '';
          const missingFieldsText = handoff.missingFields?.length ? ` Missing fields: ${handoff.missingFields.join(' | ')}` : '';
          const blockedByText = handoff.blockedBy?.length ? ` Blockers: ${handoff.blockedBy.join(' | ')}` : '';
          return {
            label: `fix: ${record.canonicalLabel}`,
            goal: this.clipText(
              `Judge whether "${record.canonicalLabel}" already satisfies the user goal from current results. If not, preserve reusable prior results and evidence, identify the remaining work, and complete only that gap.${knownGoodText}${missingFieldsText}${blockedByText}`,
              240,
            ),
            why: this.clipText(
              `Latest attempt ${branch.id} disposition=${record.disposition}. Re-plan from existing progress instead of restarting from scratch.`,
              220,
            ),
            priority: 0.9,
            handoff,
          };
        });

        const remediationHandoff: SynthesisSharedHandoff = {
          retryIndex: synthesisRetry + 1,
          successfulAttempts: successfulAttempts.map((record) => ({
            branchId: record.branch.id,
            label: record.branch.label,
            canonicalTaskId: record.canonicalLabel,
            summary: this.clipText(record.branch.finalAnswer || record.branch.completionSummary || '', 220),
          })),
          unresolvedAttempts: rebranchCandidates.map(({ handoff }) => handoff),
        };

        return {
          done: false,
          branches: remediationBranches,
          context: `Current goal assessment:\n${goalAssessment.summaryText}\n\nSuccessful frontier attempts:\n${successContext || '(none)'}\n\nUnmet workstreams:\n${goalAssessment.unresolvedAttempts.map(({ record, handoff }) => {
            const branch = record.branch;
            return [
              `[${branch.id}/${branch.label}] canonical=${record.canonicalLabel}`,
              `disposition=${record.disposition}`,
              `goal: ${branch.goal}`,
              `outcome=${branch.outcome}: ${branch.outcomeReason || '(no reason)'}`,
              `previous_answer: ${this.clipText(branch.finalAnswer || branch.completionSummary || '(none)', 500)}`,
              `known_good: ${(handoff.knownGoodFacts || []).join(' | ') || '(none)'}`,
              `missing_fields: ${(handoff.missingFields || []).join(' | ') || '(none)'}`,
              `blocked_by: ${(handoff.blockedBy || []).join(' | ') || '(none)'}`,
              `must_not_repeat: ${(handoff.mustNotRepeat || []).join(' | ') || '(none)'}`,
            ].join('\n');
          }).join('\n\n')}`,
          handoff: remediationHandoff,
        };
      }

      // ── Normal synthesis (all success, or retries exhausted) ──
      if (
        frontierAttempts.length === 1
        && frontierAttempts[0].branch.finalAnswer
        && (goalAssessment.goalAchieved || rebranchCandidates.length === 0)
      ) {
        const fallback = buildSafeBranchDisplayFallback(frontierAttempts[0].branch);
        return this.sanitizeUserFacingAnswer(
          frontierAttempts[0].branch.finalAnswer,
          fallback,
        ) || fallback;
      }
      emitTrace({ type: 'thought', content: `Synthesizing ${frontierAttempts.length} frontier completed branch attempt(s) into one final answer...` });
      const branchSummary = frontierAttempts
        .map((record) => [
          `Branch ${record.branch.id}`,
          `label: ${record.branch.label}`,
          `canonical_label: ${record.canonicalLabel}`,
          `plan_version_seen: ${record.branch.planVersionSeen ?? canonicalPlanState.version}`,
          `plan_excerpt: ${record.branch.planExcerpt || '(none)'}`,
          `disposition: ${record.disposition}`,
          `goal: ${record.branch.goal}`,
          `saved_nodes: ${record.branch.savedNodeIds.length ? record.branch.savedNodeIds.join(', ') : '(none)'}`,
          `outcome: ${record.branch.outcome || 'unknown'}`,
          `outcome_reason: ${record.branch.outcomeReason || '(none)'}`,
          `summary: ${record.branch.completionSummary || '(none)'}`,
          `answer:\n${record.branch.finalAnswer || ''}`,
        ].join('\n'))
        .join('\n\n---\n\n');

      emitTrace({ type: 'observation', content: `Synthesis model call started for ${frontierAttempts.length} frontier branch attempt(s).` });
      const activeCircuitReason = getActiveLlmCircuitReason();
      if (activeCircuitReason) {
        emitTrace({
          type: 'observation',
          content: `Skipping synthesis model call because the LLM queue is in degraded mode: ${activeCircuitReason}. Falling back to deterministic branch merge.`,
        });
        return buildDeterministicSynthesisFallback(frontierAttempts.map((record) => record.branch));
      }
      let _synthesisText: string;
      try {
        ({ text: _synthesisText } = await this.measure(perfEntries, 'branch_synthesis', async () => {
          return this.llmRpmLimiter.run(() => this.withTimeout(
            generateText({
              model: this.openai,
              temperature: this.responseTemperature,
              maxTokens: synthesisMaxTokens,
              messages: [
                {
                  role: 'system',
                  content: synthesisBasePrompt,
                },
                {
                  role: 'user',
                  content: `User request:\n${input}\n\nCanonical plan version: ${canonicalPlanState.version}\nCanonical plan delta summary: ${canonicalPlanState.deltaSummary || '(none)'}\n\nShared context:\n${this.clipText(context || '(no shared context)', 2400)}\n\nGoal assessment:\n${goalAssessment.summaryText}${archivedBranchSummary ? `\n\nArchived completed branch summaries:\n${this.clipText(archivedBranchSummary, 1800)}` : ''}\n\nCompleted branches:\n${branchSummary}`,
                },
              ],
            }),
            this.agentLlmTimeoutMs,
            'branch_synthesis llm'
          ));
        }));
      } catch (error: any) {
        emitTrace({
          type: 'error',
          content: `Synthesis model call failed: ${error?.message || String(error)}. Falling back to deterministic branch merge.`,
        });
        return buildDeterministicSynthesisFallback(frontierAttempts.map((record) => record.branch));
      }

      const synthesized = extractText(_synthesisText).trim();
      const sanitized = this.sanitizeUserFacingAnswer(synthesized);
      if (sanitized) {
        emitTrace({ type: 'observation', content: 'Synthesis model call completed successfully.' });
        return sanitized;
      }
      emitTrace({
        type: 'observation',
        content: 'Synthesis returned internal tool/planner markup; falling back to deterministic branch merge.',
      });
      return buildDeterministicSynthesisFallback(frontierAttempts.map((record) => record.branch));
    };
    let completedBranchesForRun: Array<BranchSynthesisInput['branches'][number]> = [];
    let finalAnswer: string;

    if (agentMode === 'react') {
      // ── Classic ReAct mode (sequential: think → act → observe → …) ────
      emitTrace({ type: 'thought', content: 'Using classic ReAct mode (sequential, no branching).' });
      const runtime = new AgentRuntime({
        planner: { plan: async (transcript) => ({ kind: 'final', content: extractText(transcript.at(-1)?.content || '') }) },
        toolRuntime: {
          execute: async () => ({ success: false, error: 'unreachable tool runtime fallback', durationMs: 0 }),
          resetSession: () => runRuntimeHandle.toolRuntime.resetSession(),
        },
        maxSteps: 60,
        maxBranchDepth: 0,
        maxBranchWidth: 1,
        maxCompletedBranches: 1,
        branchConcurrency: 1,
        maxTotalSteps: 60,
        transcriptBudgetChars: effectiveTranscriptBudgetChars,
        modelLimit: planningCtxBudget.modelLimit,
        committedTokens: planningCtxBudget.committedTokens,
        plannerEnvelopeTokens: estimateTokens(plannerSystemPrompt) + estimateTranscriptTokens(recentConversationMessages) + (plannerMaxTokens || 1600) + 2048,
        planBranch: async ({ branch, planningTranscript, kanbanSnapshot }) => {
        const effectivePlanningTranscript = planningTranscript ?? (branch as BranchState).transcript;
        const plannerMessages = buildPlannerMessages(branch as BranchState, effectivePlanningTranscript, kanbanSnapshot);
        // Pre-flight token check: if input already exceeds model limit, force-finalize
        // instead of sending a doomed LLM call that will fail and pollute the transcript.
        const preflightTokens = estimateTranscriptTokens(plannerMessages);
        if (planningCtxBudget.modelLimit > 0 && preflightTokens > planningCtxBudget.modelLimit - 2048) {
          emitBranchTrace('observation', branch, `Pre-flight: planner input ${preflightTokens} tokens exceeds model limit ${planningCtxBudget.modelLimit}. Force-finalizing branch.`);
          return {
            kind: 'final' as const,
            content: `Branch ${branch.id} stopped: planner input exceeded context window.`,
            outcome: 'partial' as const,
            disposition: 'missing_evidence' as const,
          };
        }
        const plannerResult = await planWithSkillRouting(
            plannerMessages,
            async (workingTranscript) => {
              return this.measure(perfEntries, `llm_${branch.id}_step_${branch.steps + 1}`, async () =>
                this.generatePlannerDecision({
                  messages: workingTranscript,
                  timeoutMs: this.agentLlmTimeoutMs,
                  timeoutLabel: `planBranch_${branch.id} llm`,
                  temperature: this.plannerTemperature,
                  allowBranching: false,
                  maxTokens: plannerMaxTokens,
                })
              );
            }
          );
          applyPlannerResultToBranch(branch as BranchState, plannerResult);
          const decision = plannerResult.decision;
          if (decision.kind === 'tool') {
            // In classic react, limit to one tool call at a time
            const toolCalls = decision.tool_calls.slice(0, 1);
            return { kind: 'tool' as const, thought: decision.thought, toolCalls };
          }
          if (decision.kind === 'branch') {
            // Classic react does not support branching — ask model to continue sequentially
            clearPendingProviderToolState(branch as BranchState);
            branch.transcript.push({
              role: 'user' as const,
              content: 'Branching is not available in classic ReAct mode. Please call exactly one direct work tool to proceed step by step, or call planner_final to finish.',
            });
            return { kind: 'tool' as const, thought: 'Branching not available', toolCalls: [] };
          }
          if (decision.kind === 'subagent') {
            clearPendingProviderToolState(branch as BranchState);
            branch.transcript.push({
              role: 'user' as const,
              content: 'planner_subagent is not available in classic ReAct mode. Continue with exactly one direct work tool or call planner_final.',
            });
            return { kind: 'tool' as const, thought: 'Subagent not available', toolCalls: [] };
          }
          return {
            kind: 'final' as const,
            content: decision.answer,
            outcome: decision.status === 'success' ? 'success' : decision.status === 'partial' ? 'partial' : 'failed',
            outcomeReason: decision.reason,
            disposition: decision.status === 'blocked' ? 'blocked_external' : decision.status === 'failed' ? 'planner_error' : 'resolved',
          };
        },
        buildToolTranscript: ({ branch, observations }) => buildProviderToolTranscript(branch as BranchState, observations),
        executeToolCall: async ({ branch, toolCall }) => {
          const result = await executeToolCall(branch as BranchState, toolCall as z.infer<typeof PlannerToolCallSchema>);
          return {
            toolName: toolCall.name,
            result,
            success: !/^Error[:\s]|^ERROR:/.test(result),
          };
        },
        finalizeBranch: async ({ branch, reason, fallbackAnswer }) => finalizeFromBranch(branch as BranchState, reason, fallbackAnswer),
        onTrace: (event) => {
          if (event.type === 'final') return;
          emitTrace({ type: event.type, content: event.content, metadata: event.metadata });
        },
        workspaceManager: this.workspaceManager,
      });
      finalAnswer = await runtime.run(`Original user request:\n${input}\n\nThink step by step. Use one tool at a time, observe the result, then decide the next action.`);
    } else {
      // ── Branching ReAct mode (default) ──────────────────────────────────
    const runtime = new AgentRuntime({
      planner: { plan: async (transcript) => ({ kind: 'final', content: extractText(transcript.at(-1)?.content || '') }) },
      toolRuntime: {
        execute: async () => ({ success: false, error: 'unreachable tool runtime fallback', durationMs: 0 }),
        resetSession: () => runRuntimeHandle.toolRuntime.resetSession(),
      },
      maxBranchWidth: this.branchMaxWidth,
      maxCompletedBranches: this.branchMaxCompleted,
      branchConcurrency: this.branchConcurrency,
      transcriptBudgetChars: effectiveTranscriptBudgetChars,
      modelLimit: planningCtxBudget.modelLimit,
      committedTokens: planningCtxBudget.committedTokens,
      plannerEnvelopeTokens: estimateTokens(plannerSystemPrompt) + estimateTranscriptTokens(recentConversationMessages) + (plannerMaxTokens || 1600) + 2048,
      planBranch: async ({ branch, planningTranscript, kanbanSnapshot }) => {
        const effectivePlanningTranscript = planningTranscript ?? (branch as BranchState).transcript;
        const plannerMessages = buildPlannerMessages(branch as BranchState, effectivePlanningTranscript, kanbanSnapshot);
        // Pre-flight token check: if input already exceeds model limit, force-finalize
        // instead of sending a doomed LLM call that will fail and pollute the transcript.
        const preflightTokens = estimateTranscriptTokens(plannerMessages);
        if (planningCtxBudget.modelLimit > 0 && preflightTokens > planningCtxBudget.modelLimit - 2048) {
          emitBranchTrace('observation', branch, `Pre-flight: planner input ${preflightTokens} tokens exceeds model limit ${planningCtxBudget.modelLimit}. Force-finalizing branch.`);
          return {
            kind: 'final' as const,
            content: `Branch ${branch.id} stopped: planner input exceeded context window.`,
            outcome: 'partial' as const,
            disposition: 'missing_evidence' as const,
          };
        }
        const plannerResult = await planWithSkillRouting(
          plannerMessages,
          async (workingTranscript) => {
            return this.measure(perfEntries, `llm_${branch.id}_step_${branch.steps + 1}`, async () =>
              this.generatePlannerDecision({
                messages: workingTranscript,
                timeoutMs: this.agentLlmTimeoutMs,
                timeoutLabel: `planBranch_${branch.id} llm`,
                temperature: this.plannerTemperature,
                allowBranching: true,
                maxTokens: plannerMaxTokens,
              })
            );
          }
        );
        applyPlannerResultToBranch(branch as BranchState, plannerResult);
        const decision = plannerResult.decision;
        if (decision.kind === 'tool') {
          return { kind: 'tool' as const, thought: decision.thought, toolCalls: decision.tool_calls };
        }
        if (decision.kind === 'subagent') {
          return executeSubagentDecision(branch as BranchState, kanbanSnapshot, decision);
        }
        if (decision.kind === 'branch') {
          return {
            kind: 'branch' as const,
            thought: decision.thought,
            branches: normalizePlannerBranches(decision.branches),
          };
        }
        return {
          kind: 'final' as const,
          content: decision.answer,
          outcome: decision.status === 'success' ? 'success' : decision.status === 'partial' ? 'partial' : 'failed',
          outcomeReason: decision.reason,
          disposition: decision.status === 'blocked' ? 'blocked_external' : decision.status === 'failed' ? 'planner_error' : 'resolved',
        };
      },
      buildToolTranscript: ({ branch, observations }) => buildProviderToolTranscript(branch as BranchState, observations),
      executeToolCall: async ({branch, toolCall }) => {
        const result = await executeToolCall(branch as BranchState, toolCall as z.infer<typeof PlannerToolCallSchema>);
        return {
          toolName: toolCall.name,
          result,
          success: !/^Error[:\s]|^ERROR:/.test(result),
        };
      },
      finalizeBranch: async ({ branch, reason, fallbackAnswer }) => finalizeFromBranch(branch as BranchState, reason, fallbackAnswer),
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
          inheritedMessageCount: 0,
          transcript: [],
          savedNodeIds: branch.savedNodeIds,
          completionSummary: branch.completionSummary,
          finalAnswer: branch.finalAnswer,
          outcome: branch.outcome,
          outcomeReason: branch.outcomeReason,
          disposition: branch.disposition,
          finalizationMode: branch.finalizationMode,
          planVersionSeen: branch.planVersionSeen,
          planExcerpt: branch.planExcerpt,
          alignmentChecks: branch.alignmentChecks,
          ancestorBranchIds: branch.ancestorBranchIds,
          ancestorBranchLabels: branch.ancestorBranchLabels,
        })) as BranchState[];
        return synthesizeCompletedBranches(
          branches,
          inputData.archivedBranchSummary || '',
          inputData.synthesisRetry,
          inputData.maxSynthesisRetries,
        );
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
      },
      workspaceManager: this.workspaceManager,
    });
    finalAnswer = await runtime.run(`Original user request:\n${input}\n\nStart with the root loop. Use the current canonical plan if one already exists. If the task needs initial branching or a major plan refresh, call planner_subagent with subagent=plan and a concrete proposed plan instead of inventing a sibling graph locally. Otherwise continue with local branch execution. Call planner_final when you are ready to answer.`);
    }

    finalAnswer = this.sanitizeUserFacingAnswer(
      finalAnswer,
      completedBranchesForRun.length > 0
        ? buildDeterministicSynthesisFallback(
          completedBranchesForRun.map((branch) => ({
            id: branch.id,
            parentId: null,
            depth: 0,
            label: branch.label,
            goal: branch.goal,
            priority: 1,
            steps: 0,
            inheritedMessageCount: 0,
            transcript: [],
            savedNodeIds: branch.savedNodeIds,
            completionSummary: branch.completionSummary,
            finalAnswer: branch.finalAnswer,
          })) as BranchState[],
        )
        : '抱歉，我暂时没能生成可展示的回答，请重试一次。',
    ) || '抱歉，我暂时没能生成可展示的回答，请重试一次。';
    finalAnswer = this.normalizeUserFacingAnswer(finalAnswer);

    if (runPhase === 'plan') {
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
          inheritedMessageCount: 0,
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
          inheritedMessageCount: 0,
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
          saveAtomic: true,
          saveEpisodic: true,
        }
      );
      this.scheduleMemorySave(autoJob);
      memoryQueuedThisRun = true;
      emitTrace({
        type: 'observation',
        content: `Auto-queued independent four-layer memory classification from branch ${(bestBranch?.id) || 'B0'}.`,
      });
    }
    // Persist conversation state synchronously (fast, small writes).
    this.appendConversationTurn(conversationId, { user: input, assistant: finalAnswer });
    const persistedTurns = this.getConversationTurns(conversationId);
    const nextPersistedState = this.buildPersistedConversationState(
      conversationId,
      input,
      finalAnswer,
      traceBuffer,
      persistedTurns,
    );
    this.writePersistedConversationState(nextPersistedState);
    this.appendTurnSummaryRecord({
      ts: new Date().toISOString(),
      conversation_id: conversationId,
      user_input: this.clipText(this.extractOriginalUserRequest(input), 600),
      user_intent: nextPersistedState.last_user_request,
      assistant_outcome: this.clipText(finalAnswer.replace(/\s+/g, ' ').trim(), 600),
      status: nextPersistedState.status,
      next_expected_user_action: nextPersistedState.open_loops[0] || nextPersistedState.current_goal,
      referenced_entities: nextPersistedState.referenced_entities.slice(0, 8),
      tool_findings: this.deriveToolFindings(traceBuffer, 4),
    });
    // Defer the large conversation-log write off the answer-return critical path.
    // Uses the runId generated at run start so the in-progress placeholder is overwritten.
    setImmediate(() => {
      this.appendConversationLog(runId, input, traceBuffer, finalAnswer);
    });
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
