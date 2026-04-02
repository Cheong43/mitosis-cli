import * as fs from 'node:fs';
import * as path from 'node:path';
import { search as duckDuckGoSearch } from 'duck-duck-scrape';

import type { RuntimeHandle } from '../runtime/index.js';
import { MempediaClient } from '../mempedia/client.js';
import type { ToolAction, ToolResponse } from '../mempedia/types.js';
import {
  findSkillByName,
  loadWorkspaceSkills,
  type SkillRecord,
} from '../skills/router.js';
import { logError } from '../utils/errorLogger.js';

type MainAgentToolName = 'read' | 'search' | 'edit' | 'bash' | 'web';

interface MainAgentAdapterOptions {
  projectRoot: string;
  codeCliRoot: string;
  runtimeHandle: RuntimeHandle;
  mempediaClient?: MempediaClientLike;
}

interface MempediaClientLike {
  send(action: ToolAction): Promise<ToolResponse>;
}

interface WebPermissionPolicy {
  allowedDomains: string[];
  blockedDomains: string[];
}

function normalizeDomain(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\//, '')
    .replace(/^www\./, '')
    .replace(/\/.*$/, '');
}

function extractHostname(value: string): string {
  try {
    return normalizeDomain(new URL(value).hostname);
  } catch {
    return normalizeDomain(value);
  }
}

function parseDomainList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return [...new Set(value.map((entry) => normalizeDomain(String(entry || ''))).filter(Boolean))];
  }
  if (typeof value === 'string') {
    return [...new Set(
      value
        .split(/[,\s]+/)
        .map((entry) => normalizeDomain(entry))
        .filter(Boolean),
    )];
  }
  return [];
}

function matchesDomainRule(hostname: string, rule: string): boolean {
  return hostname === rule || hostname.endsWith(`.${rule}`);
}

export class MainAgentAdapter {
  private readonly projectRoot: string;
  private readonly codeCliRoot: string;
  private readonly runtimeHandle: RuntimeHandle;
  private readonly defaultWebPermissionPolicy: WebPermissionPolicy;
  private readonly mempediaClient: MempediaClientLike;

  constructor(options: MainAgentAdapterOptions) {
    this.projectRoot = options.projectRoot;
    this.codeCliRoot = options.codeCliRoot;
    this.runtimeHandle = options.runtimeHandle;
    this.mempediaClient = options.mempediaClient ?? new MempediaClient(options.projectRoot);
    this.defaultWebPermissionPolicy = {
      allowedDomains: parseDomainList(process.env.MITOSIS_WEB_ALLOWED_DOMAINS),
      blockedDomains: parseDomainList(process.env.MITOSIS_WEB_BLOCKED_DOMAINS),
    };
  }

  async execute(toolName: MainAgentToolName, args: Record<string, unknown>): Promise<string> {
    switch (toolName) {
      case 'read':
        return this.executeRead(args);
      case 'search':
        return this.executeSearch(args);
      case 'edit':
        return this.executeEdit(args);
      case 'bash':
        return this.executeBash(args);
      case 'web':
        return this.executeWeb(args);
      default:
        return `Unknown tool: ${toolName}`;
    }
  }

  private normalizeTarget(value: unknown): string {
    return String(value || '').trim().toLowerCase();
  }

  private getWebPermissionPolicy(args: Record<string, unknown>): WebPermissionPolicy {
    const allowedDomains = parseDomainList(args.allowed_domains);
    const blockedDomains = parseDomainList(args.blocked_domains);
    return {
      allowedDomains: allowedDomains.length > 0 ? allowedDomains : this.defaultWebPermissionPolicy.allowedDomains,
      blockedDomains: [...new Set([...this.defaultWebPermissionPolicy.blockedDomains, ...blockedDomains])],
    };
  }

  private evaluateWebAccess(url: string, policy: WebPermissionPolicy): { allowed: boolean; host: string; reason?: string } {
    const host = extractHostname(url);
    if (!host) {
      return { allowed: false, host: '', reason: 'invalid or missing hostname' };
    }
    if (policy.blockedDomains.some((domain) => matchesDomainRule(host, domain))) {
      return { allowed: false, host, reason: `domain ${host} is blocked` };
    }
    if (
      policy.allowedDomains.length > 0
      && !policy.allowedDomains.some((domain) => matchesDomainRule(host, domain))
    ) {
      return {
        allowed: false,
        host,
        reason: `domain ${host} is outside allowed_domains (${policy.allowedDomains.join(', ')})`,
      };
    }
    return { allowed: true, host };
  }

  private summarizeFetchedText(text: string): {
    summary: string;
    highlights: string[];
    contentPreview: string;
    contentLength: number;
  } {
    const normalized = text.replace(/\s+/g, ' ').trim();
    const sentences = normalized
      .split(/(?<=[。！？!?;；.])\s+/)
      .map((sentence) => sentence.trim())
      .filter((sentence) => sentence.length >= 40);
    const uniqueHighlights = [...new Set(sentences)].slice(0, 5).map((sentence) => sentence.slice(0, 240));
    const contentPreview = normalized.slice(0, 1400);
    const summaryParts = [
      normalized
        ? `Fetched page text length: ${normalized.length} characters.`
        : 'Fetched page contained little extractable text.',
      uniqueHighlights.length > 0
        ? `Top highlights: ${uniqueHighlights.map((highlight, index) => `[${index + 1}] ${highlight}`).join(' ')}`
        : '',
    ].filter(Boolean);

    return {
      summary: summaryParts.join(' '),
      highlights: uniqueHighlights,
      contentPreview,
      contentLength: normalized.length,
    };
  }

  private buildWebFetchResponse(
    url: string,
    domain: string,
    title: string,
    text: string,
    policy: WebPermissionPolicy,
  ): string {
    const fetched = this.summarizeFetchedText(text);
    return JSON.stringify({
      kind: 'web_fetch',
      url,
      domain,
      title,
      citation: '[fetch]',
      summary: fetched.summary,
      highlights: fetched.highlights,
      content_preview: fetched.contentPreview,
      content_length: fetched.contentLength,
      recommended_next_action: 'Use the highlights and content_preview first; fetch another page only if critical evidence is still missing.',
      permissions: {
        allowed_domains: policy.allowedDomains,
        blocked_domains: policy.blockedDomains,
      },
    });
  }

  private async executeRead(args: Record<string, unknown>): Promise<string> {
    const target = this.normalizeTarget(args.target);
    switch (target) {
      case 'workspace':
        return this.readWorkspaceFile(String(args.path || ''));
      case 'memory':
        return this.readMemoryNode(args);
      case 'preferences':
        return this.readPreferences();
      case 'skills':
        return this.readSkill(args);
      default:
        return 'Error: read supports target=workspace|memory|preferences|skills.';
    }
  }

  private async executeSearch(args: Record<string, unknown>): Promise<string> {
    const target = this.normalizeTarget(args.target);
    switch (target) {
      case 'workspace':
        return this.searchWorkspace(args);
      case 'memory':
        return this.searchMemory(args);
      case 'preferences':
        return this.searchPreferences(args);
      case 'skills':
        return this.searchSkills(args);
      default:
        return JSON.stringify({
          kind: 'error',
          message: 'search supports target=workspace|memory|preferences|skills.',
        });
    }
  }

  private async executeEdit(args: Record<string, unknown>): Promise<string> {
    const target = this.normalizeTarget(args.target);
    switch (target) {
      case 'workspace':
        return this.editWorkspaceFile(args);
      case 'memory':
        return this.editMemoryNode(args);
      case 'preferences':
        return this.editPreferences(args);
      case 'skills':
        return this.editSkill(args);
      default:
        return JSON.stringify({
          kind: 'error',
          message: 'edit supports target=workspace|memory|preferences|skills.',
        });
    }
  }

  private async executeBash(args: Record<string, unknown>): Promise<string> {
    const command = typeof args.command === 'string' ? args.command.trim() : '';
    if (!command) {
      const error = new Error(`bash command is empty. Received args: ${JSON.stringify(args)}`);
      logError(this.projectRoot, error, 'bash_empty_command');
      return `Error: bash requires a non-empty command parameter. Received: ${JSON.stringify(args)}`;
    }
    const toolRes = await this.runtimeHandle.executeTool('run_shell', { command });
    if (!toolRes.success) {
      return `Error: ${toolRes.error ?? 'unknown tool error'}`;
    }
    return typeof toolRes.result === 'string'
      ? toolRes.result
      : JSON.stringify(toolRes.result);
  }

  private async executeWeb(args: Record<string, unknown>): Promise<string> {
    const webTimeoutMs = Number(
      process.env.MITOSIS_WEB_TIMEOUT_MS ?? process.env.MEMPEDIA_WEB_TIMEOUT_MS ?? 15000,
    );
    const safeWebTimeout =
      Number.isFinite(webTimeoutMs) && webTimeoutMs > 0 ? webTimeoutMs : 15000;
    const mode = String(args.mode || '').trim();
    const policy = this.getWebPermissionPolicy(args);

    if (mode === 'fetch') {
      const url = String(args.url || '').trim();
      if (!url) {
        return JSON.stringify({ kind: 'error', message: 'web fetch requires url' });
      }
      const access = this.evaluateWebAccess(url, policy);
      if (!access.allowed) {
        return JSON.stringify({
          kind: 'error',
          message: `web fetch blocked: ${access.reason || 'domain not allowed'}`,
          permissions: {
            allowed_domains: policy.allowedDomains,
            blocked_domains: policy.blockedDomains,
          },
        });
      }

      const userAgent =
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
      const maxFetchRetries = 2;
      for (let attempt = 0; attempt <= maxFetchRetries; attempt += 1) {
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), safeWebTimeout);
        try {
          const response = await fetch(url, {
            headers: {
              'User-Agent': userAgent,
              Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
              'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            },
            signal: ac.signal,
            redirect: 'follow',
          });
          const html = await response.text();
          if (!response.ok) {
            if (attempt < maxFetchRetries && (response.status === 429 || response.status >= 500)) {
              clearTimeout(timer);
              await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
              continue;
            }
            return JSON.stringify({
              kind: 'error',
              message: `HTTP ${response.status} ${response.statusText}`,
            });
          }
          const title =
            html
              .match(/<title>([\s\S]*?)<\/title>/i)?.[1]
              ?.replace(/\s+/g, ' ')
              .trim() || url;
          return this.buildWebFetchResponse(
            url,
            access.host,
            title,
            this.stripHtml(html),
            policy,
          );
        } catch (err: any) {
          const isAbort = err?.name === 'AbortError';
          return JSON.stringify({
            kind: 'error',
            message: isAbort
              ? `web fetch timed out after ${safeWebTimeout}ms`
              : String(err?.message || err),
          });
        } finally {
          clearTimeout(timer);
        }
      }
    }

    if (mode === 'search') {
      const query = String(args.query || '').trim();
      if (!query) {
        return JSON.stringify({ kind: 'error', message: 'web search requires query' });
      }
      const limit = Math.min(Math.max(1, Number(args.limit || 10)), 20);

      // Try Brave Search first if API key is configured
      const braveApiKey = process.env.BRAVE_API_KEY;
      if (braveApiKey) {
        try {
          const braveUrl = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${limit}`;
          const braveResponse = await fetch(braveUrl, {
            headers: { 'X-Subscription-Token': braveApiKey, Accept: 'application/json' },
            signal: AbortSignal.timeout(safeWebTimeout),
          });
          if (braveResponse.ok) {
            const braveData = await braveResponse.json();
            const results = (braveData.web?.results || []).slice(0, limit).map((r: any) => ({
              title: r.title,
              url: r.url,
              snippet: r.description,
            }));
            return JSON.stringify({ kind: 'search_results', query, count: results.length, results, source: 'brave' });
          }
        } catch (err) {
          // Fall through to DuckDuckGo
        }
      }

      // Fallback to DuckDuckGo
      try {
        const searchResults = await duckDuckGoSearch(query, { safeSearch: 0 });
        const results = searchResults.results.slice(0, limit).map((result) => ({
          title: result.title,
          url: result.url,
          snippet: result.description,
        }));

        return JSON.stringify({
          kind: 'search_results',
          query,
          count: results.length,
          results,
          source: 'duckduckgo',
        });
      } catch (err: any) {
        return JSON.stringify({
          kind: 'error',
          message: `web search failed: ${String(err?.message || err)}`,
        });
      }
    }
    return JSON.stringify({
      kind: 'error',
      message: 'web only supports mode=fetch or mode=search.',
    });
  }

  private async readWorkspaceFile(filePath: string): Promise<string> {
    const toolRes = await this.runtimeHandle.executeTool('read_file', { path: filePath });
    if (!toolRes.success) {
      return `Error: ${toolRes.error ?? 'workspace read failed'}`;
    }
    return typeof toolRes.result === 'string'
      ? toolRes.result
      : JSON.stringify(toolRes.result);
  }

  private async searchWorkspace(args: Record<string, unknown>): Promise<string> {
    const mode = String(args.mode || '').trim();
    const limit = Math.max(
      1,
      Math.min(200, Number.isFinite(Number(args.limit)) ? Math.floor(Number(args.limit)) : 20),
    );
    if (mode === 'glob') {
      const pattern = String(args.pattern || '**/*').trim() || '**/*';
      const matcher = this.globToRegExp(pattern);
      const files = this.listWorkspaceFiles()
        .filter((filePath) => matcher.test(filePath))
        .slice(0, limit);
      return JSON.stringify({ kind: 'workspace_glob_results', pattern, results: files });
    }
    if (mode === 'grep') {
      const query = String(args.query || '').trim();
      if (!query) {
        return JSON.stringify({ kind: 'error', message: 'search grep requires query' });
      }
      const results: Array<{ path: string; line: number; text: string }> = [];
      const matcher = new RegExp(query, 'i');
      for (const relativePath of this.listWorkspaceFiles()) {
        if (results.length >= limit) break;
        const absolutePath = path.join(this.projectRoot, relativePath);
        let content = '';
        try {
          content = fs.readFileSync(absolutePath, 'utf-8');
        } catch {
          continue;
        }
        const lines = content.split(/\r?\n/);
        for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
          if (!matcher.test(lines[lineIndex])) continue;
          results.push({
            path: relativePath,
            line: lineIndex + 1,
            text: lines[lineIndex].trim().slice(0, 240),
          });
          if (results.length >= limit) break;
        }
      }
      return JSON.stringify({ kind: 'workspace_grep_results', query, results });
    }
    return JSON.stringify({
      kind: 'error',
      message: 'search target=workspace supports mode=grep|glob.',
    });
  }

  private async editWorkspaceFile(args: Record<string, unknown>): Promise<string> {
    const filePath = String(args.path || '').trim();
    const normalizedPath = filePath.replace(/\\/g, '/').replace(/^\.\//, '');
    if (normalizedPath === '.mempedia' || normalizedPath.startsWith('.mempedia/')) {
      return JSON.stringify({
        kind: 'error',
        message:
          'edit target=workspace is blocked for raw .mempedia storage. Use target=memory|preferences instead.',
      });
    }
    const toolRes = await this.runtimeHandle.executeTool('write_file', {
      path: filePath,
      content: String(args.content || ''),
    });
    if (!toolRes.success) {
      return `Error: ${toolRes.error ?? 'workspace edit failed'}`;
    }
    return typeof toolRes.result === 'string'
      ? toolRes.result
      : JSON.stringify(toolRes.result);
  }

  private async readMemoryNode(args: Record<string, unknown>): Promise<string> {
    const nodeId = String(args.node_id || '').trim();
    if (!nodeId) {
      return 'Error: read target=memory requires node_id.';
    }
    const markdown = args.markdown !== false;
    const result = await this.executeMempediaAction({
      action: 'open_node',
      node_id: nodeId,
      markdown,
      agent_id: 'planner-read',
    });
    if ((result as any)?.kind === 'markdown' && typeof (result as any).markdown === 'string') {
      return String((result as any).markdown);
    }
    return JSON.stringify(result);
  }

  private async searchMemory(args: Record<string, unknown>): Promise<string> {
    const query = String(args.query || '').trim();
    if (!query) {
      return JSON.stringify({
        kind: 'error',
        message: 'search target=memory requires query.',
      });
    }
    const limit = Number.isFinite(Number(args.limit)) ? Math.floor(Number(args.limit)) : undefined;
    const mode = String(args.mode || 'hybrid').trim().toLowerCase();
    if (mode === 'episodic') {
      const result = await this.executeMempediaAction({
        action: 'search_episodic',
        query,
        limit,
      });
      return JSON.stringify(result);
    }
    if (mode === 'keyword') {
      const result = await this.executeMempediaAction({
        action: 'search_nodes',
        query,
        limit,
      });
      return JSON.stringify(result);
    }
    const result = await this.executeMempediaAction({
      action: 'search_hybrid',
      query,
      limit,
      rrf_k: args.rrf_k,
      bm25_weight: args.bm25_weight,
      vector_weight: args.vector_weight,
      graph_weight: args.graph_weight,
      graph_depth: args.graph_depth,
      graph_seed_limit: args.graph_seed_limit,
    });
    return JSON.stringify(result);
  }

  private async editMemoryNode(args: Record<string, unknown>): Promise<string> {
    const nodeId = String(args.node_id || '').trim();
    const markdown = String(args.content ?? args.markdown ?? '').trim();
    if (!nodeId || !markdown) {
      return JSON.stringify({
        kind: 'error',
        message: 'edit target=memory requires node_id and content.',
      });
    }
    const result = await this.executeMempediaAction({
      action: 'agent_upsert_markdown',
      node_id: nodeId,
      markdown,
      importance: Number.isFinite(Number(args.importance)) ? Number(args.importance) : 1.0,
      agent_id: 'planner-edit',
      reason: String(args.reason || 'semantic tool memory edit').trim(),
      source: String(args.source || 'planner_tool_edit').trim(),
      parent_node: typeof args.parent_node === 'string' ? args.parent_node : undefined,
      node_type: typeof args.node_type === 'string' ? args.node_type : undefined,
    });
    return JSON.stringify(result);
  }

  private async readPreferences(): Promise<string> {
    const toolRes = await this.runtimeHandle.executeTool('read_file', {
      path: '.mempedia/memory/preferences.md',
    });
    if (!toolRes.success) {
      const error = String(toolRes.error || '');
      if (/no such file|enoent/i.test(error)) {
        return '';
      }
      return `Error: ${error || 'preferences read failed'}`;
    }
    return typeof toolRes.result === 'string'
      ? toolRes.result
      : JSON.stringify(toolRes.result);
  }

  private async searchPreferences(args: Record<string, unknown>): Promise<string> {
    const query = String(args.query || '').trim();
    if (!query) {
      return JSON.stringify({
        kind: 'error',
        message: 'search target=preferences requires query.',
      });
    }
    const content = await this.readPreferences();
    if (content.startsWith('Error:')) {
      return content;
    }
    const matcher = new RegExp(query, 'i');
    const results = content
      .split(/\r?\n/)
      .map((line, index) => ({ line: index + 1, text: line }))
      .filter((item) => matcher.test(item.text))
      .slice(0, Math.max(1, Math.min(50, Number.isFinite(Number(args.limit)) ? Math.floor(Number(args.limit)) : 20)))
      .map((item) => ({ ...item, text: item.text.trim().slice(0, 240) }));
    return JSON.stringify({ kind: 'preferences_search_results', query, results });
  }

  private async editPreferences(args: Record<string, unknown>): Promise<string> {
    const toolRes = await this.runtimeHandle.executeTool('write_file', {
      path: '.mempedia/memory/preferences.md',
      content: String(args.content || ''),
    });
    if (!toolRes.success) {
      return `Error: ${toolRes.error ?? 'preferences edit failed'}`;
    }
    return typeof toolRes.result === 'string'
      ? toolRes.result
      : JSON.stringify(toolRes.result);
  }

  private readLoadedSkills(): SkillRecord[] {
    return loadWorkspaceSkills(this.projectRoot, this.codeCliRoot);
  }

  private async readSkill(args: Record<string, unknown>): Promise<string> {
    const targetName = String(args.skill_id ?? args.name ?? '').trim();
    if (!targetName) {
      return 'Error: read target=skills requires skill_id or name.';
    }
    const skill = findSkillByName(this.readLoadedSkills(), targetName);
    if (!skill) {
      return `Error: skill '${targetName}' not found.`;
    }
    return this.renderSkillForRead(skill);
  }

  private async searchSkills(args: Record<string, unknown>): Promise<string> {
    const query = String(args.query || '').trim().toLowerCase();
    const limit = Math.max(
      1,
      Math.min(50, Number.isFinite(Number(args.limit)) ? Math.floor(Number(args.limit)) : 10),
    );
    const skills = this.readLoadedSkills();
    const scored = skills
      .map((skill) => {
        const haystack = `${skill.name}\n${skill.description}\n${skill.content}`.toLowerCase();
        let score = 0;
        if (!query) {
          score = 1;
        } else if (skill.name.toLowerCase().includes(query)) {
          score = 3;
        } else if (skill.description.toLowerCase().includes(query)) {
          score = 2;
        } else if (haystack.includes(query)) {
          score = 1;
        }
        return { skill, score };
      })
      .filter((item) => item.score > 0)
      .sort((left, right) => right.score - left.score || left.skill.name.localeCompare(right.skill.name))
      .slice(0, limit)
      .map((item) => ({
        name: item.skill.name,
        description: item.skill.description,
        source: item.skill.source || 'local',
        location: item.skill.location,
        score: item.score,
      }));
    return JSON.stringify({
      kind: 'skills_search_results',
      query,
      results: scored,
    });
  }

  private async editSkill(args: Record<string, unknown>): Promise<string> {
    const skillId = String(args.skill_id ?? args.name ?? '').trim();
    const content = String(args.content || '').trim();
    if (!skillId || !content) {
      return JSON.stringify({
        kind: 'error',
        message: 'edit target=skills requires skill_id/name and content.',
      });
    }
    const skillDir = path.join(this.projectRoot, 'skills', this.toSlug(skillId));
    const filePath = path.join(skillDir, 'SKILL.md');
    const description =
      String(args.description || '').trim() || this.firstSentence(content) || skillId;
    const tags = Array.isArray(args.tags)
      ? args.tags
          .map((tag) => String(tag || '').trim())
          .filter((tag) => tag.length > 0)
      : [];
    const markdown = this.buildSkillMarkdown(skillId, description, content, tags);
    const toolRes = await this.runtimeHandle.executeTool('write_file', {
      path: path.relative(this.projectRoot, filePath).replace(/\\/g, '/'),
      content: markdown,
    });
    if (!toolRes.success) {
      return `Error: ${toolRes.error ?? 'skill edit failed'}`;
    }
    return JSON.stringify({
      kind: 'skill_written',
      skill_id: skillId,
      path: path.relative(this.projectRoot, filePath).replace(/\\/g, '/'),
    });
  }

  private renderSkillForRead(skill: SkillRecord): string {
    const lines = [
      `Skill: ${skill.name}`,
      `Description: ${skill.description}`,
    ];
    if (skill.tools?.length) {
      lines.push(`Allowed tools: ${skill.tools.join(', ')}`);
    }
    if (skill.location) {
      lines.push(`Location: ${skill.location}`);
    }
    lines.push('');
    lines.push(skill.content);
    return lines.join('\n');
  }

  private buildSkillMarkdown(
    skillId: string,
    description: string,
    content: string,
    tags: string[],
  ): string {
    const escapedDescription = this.yamlEscape(description);
    const tagLine =
      tags.length > 0
        ? `tags: [${tags.map((tag) => `"${this.yamlEscape(tag)}"`).join(', ')}]\n`
        : '';
    return `---\nname: ${this.yamlEscape(skillId)}\ndescription: "${escapedDescription}"\n${tagLine}---\n\n${content.trim()}\n`;
  }

  private firstSentence(text: string): string {
    const trimmed = text.trim();
    if (!trimmed) return '';
    const match = trimmed.match(/^(.+?[。！？!?\.]|.+$)/s);
    return match ? match[1].trim() : trimmed;
  }

  private yamlEscape(value: string): string {
    return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  }

  private toSlug(value: string): string {
    const normalized = value
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 72);
    return normalized || 'empty';
  }

  private isIgnoredWorkspacePath(relativePath: string): boolean {
    return (
      relativePath.startsWith('.git/') ||
      relativePath === '.git' ||
      relativePath.startsWith('node_modules/') ||
      relativePath === 'node_modules' ||
      relativePath.startsWith('target/') ||
      relativePath === 'target' ||
      relativePath.startsWith('.mempedia/') ||
      relativePath === '.mempedia' ||
      relativePath.startsWith('.mitosis/') ||
      relativePath === '.mitosis'
    );
  }

  private listWorkspaceFiles(dir = this.projectRoot, prefix = ''): string[] {
    if (!fs.existsSync(dir)) {
      return [];
    }
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const files: string[] = [];
    for (const entry of entries) {
      const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (this.isIgnoredWorkspacePath(relativePath)) {
        continue;
      }
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        files.push(...this.listWorkspaceFiles(fullPath, relativePath));
      } else {
        files.push(relativePath.replace(/\\/g, '/'));
      }
    }
    return files;
  }

  private globToRegExp(pattern: string): RegExp {
    let regex = '^';
    for (let index = 0; index < pattern.length; index += 1) {
      const char = pattern[index];
      const next = pattern[index + 1];
      if (char === '*') {
        if (next === '*') {
          regex += '.*';
          index += 1;
        } else {
          regex += '[^/]*';
        }
      } else if (char === '?') {
        regex += '.';
      } else if ('\\^$+.|(){}[]'.includes(char)) {
        regex += `\\${char}`;
      } else {
        regex += char;
      }
    }
    regex += '$';
    return new RegExp(regex, 'i');
  }

  private stripHtml(html: string): string {
    return html
      .replace(/<script[\s\S]*?<\/script>/gi, ' ')
      .replace(/<style[\s\S]*?<\/style>/gi, ' ')
      .replace(/<[^>]+>/g, ' ')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private async executeMempediaAction(
    action: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const startedAt = Date.now();
    const actionName = String(action.action || 'unknown');
    try {
      const result = await this.mempediaClient.send(action as ToolAction);
      const elapsedMs = Date.now() - startedAt;
      if (result.kind === 'error' && /timeout|timed out/i.test(result.message || '')) {
        logError(
          this.projectRoot,
          new Error(`Mempedia action ${actionName} timed out after ${elapsedMs}ms: ${result.message}`),
          `mempedia_action_timeout_${actionName}`,
        );
      }
      return result as Record<string, unknown>;
    } catch (error) {
      const elapsedMs = Date.now() - startedAt;
      const message = error instanceof Error ? error.message : String(error);
      logError(
        this.projectRoot,
        error,
        `mempedia_action_failed_${actionName}_after_${elapsedMs}ms`,
      );
      return {
        kind: 'error',
        message: `mempedia action ${actionName} failed after ${elapsedMs}ms: ${message}`,
      };
    }
  }
}
