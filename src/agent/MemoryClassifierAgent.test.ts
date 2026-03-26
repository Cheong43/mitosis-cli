import test from 'node:test';
import assert from 'node:assert/strict';

import { MemoryClassifierAgent } from './MemoryClassifierAgent.js';
import type { MemoryClassifierContext, MemoryClassifierJob } from './MemoryClassifierAgent.js';
import type { ToolAction } from '../mempedia/types.js';

test('MemoryClassifierAgent.persist writes preferences, skills, atomic memory, and episodic memory', async () => {
  const agent = new MemoryClassifierAgent({
    chatClient: {} as any,
    codeCliRoot: process.cwd(),
    extractionMaxChars: 4000,
    memoryExtractTimeoutMs: 1000,
    memoryActionTimeoutMs: 1000,
    autoLinkEnabled: true,
    autoLinkMaxNodes: 4,
    autoLinkLimit: 3,
  });

  (agent as any).extractMemoryPayload = async () => ({
    user_preferences: [
      { topic: 'response_style', preference: 'Prefer concise technical answers.', evidence: 'User asked for concise answers.' },
    ],
    agent_skills: [
      { skill_id: 'skill_cli_review', title: 'CLI review flow', content: 'Inspect, patch, test.', tags: ['cli'] },
    ],
    atomic_knowledge: [
      {
        keyword: 'PlannerToolAdapter',
        summary: 'Planner tool execution is centralized in PlannerToolAdapter.',
        description: 'PlannerToolAdapter now owns execution mapping for planner-visible tools.',
        facts: ['PlannerToolAdapter handles planner-visible tool execution.'],
        data_points: [],
        truths: [],
        viewpoints: [],
        history: [],
        uncertainties: [],
        evidence: ['repo inspection'],
        relations: ['Agent'],
      },
    ],
  });

  const actions: ToolAction[] = [];
  const mappedNodes: Array<{ nodeId: string; conversationId: string; reason: string }> = [];
  const phases: string[] = [];

  const context: MemoryClassifierContext = {
    runId: 'run-1',
    conversationId: 'conv-1',
    sendAction: async (action) => {
      actions.push(action);
      switch (action.action) {
        case 'read_user_preferences':
          return { kind: 'user_preferences', content: '# User Preferences\n' };
        case 'update_user_preferences':
          return { kind: 'user_preferences', content: action.content };
        case 'upsert_skill':
          return { kind: 'skill_result', skill_id: action.skill_id, title: action.title, content: action.content, tags: action.tags || [], updated_at: Date.now() };
        case 'agent_upsert_markdown':
          return { kind: 'version', version: { node_id: action.node_id } };
        case 'auto_link_related':
          return { kind: 'version', version: { node_id: action.node_id } };
        case 'record_episodic':
          return { kind: 'episodic_results', memories: [] };
        default:
          return { kind: 'ack', message: action.action };
      }
    },
    appendMemoryLog: (phase) => {
      phases.push(phase);
    },
    appendNodeConversationMap: (nodeId, conversationId, reason) => {
      mappedNodes.push({ nodeId, conversationId, reason });
    },
    resolveRelationTargets: async (relations) => relations.map((label) => ({ label, target: `kg_atomic_${label.toLowerCase()}` })),
    mergeUserPreferencesMarkdown: (existing, preferences) => `${existing}\n${preferences.map((item) => item.preference).join('\n')}`,
  };

  const job: MemoryClassifierJob = {
    input: 'Please keep answers concise.',
    traces: [],
    answer: 'PlannerToolAdapter is now the main planner execution layer.',
    reason: 'persist extracted memory',
    focus: 'planner architecture',
    savePreferences: true,
    saveSkills: true,
    saveAtomic: true,
    saveEpisodic: true,
    branchId: 'B1',
  };

  await agent.persist(job, context);

  assert.deepEqual(actions.map((action) => action.action), [
    'read_user_preferences',
    'update_user_preferences',
    'upsert_skill',
    'agent_upsert_markdown',
    'auto_link_related',
    'record_episodic',
  ]);
  assert.equal(mappedNodes.length, 1);
  assert.equal(mappedNodes[0]?.nodeId, 'kg_atomic_plannertooladapter');
  assert.ok(phases.includes('memory_extract_done'));
  assert.ok(phases.includes('memory_preferences_saved'));
  assert.ok(phases.includes('memory_skills_saved'));
  assert.ok(phases.includes('memory_atomic_saved'));
  assert.ok(phases.includes('memory_episodic_saved'));
});
