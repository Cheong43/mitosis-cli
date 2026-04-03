import path from 'node:path';

import type { ArtifactMetadata } from '../persistence/types.js';
import type { SharedWorkspaceFact, SharedWorkspaceFactType } from '../runtime/agent/types.js';

export interface WorkspaceToolExecutionContext {
  branchId?: string;
  step?: number;
  goal?: string;
}

export interface WorkspaceToolExecutionMetadata {
  cacheHit?: boolean;
  artifacts?: string[];
  outputs?: string[];
  workspaceFacts?: SharedWorkspaceFact[];
  duplicateInspection?: string;
}

interface CachedReadEntry {
  output: string;
  version: number;
}

interface CachedSearchEntry {
  output: string;
  version: number;
}

interface InspectionRecord {
  key: string;
  kind: 'read' | 'glob' | 'grep';
  target: string;
  branchId: string;
  step: number;
  version: number;
  count: number;
  cached: boolean;
  updatedAt: number;
}

export class WorkspaceStateTracker {
  private readonly projectRoot: string;
  private readonly factIndex = new Map<string, SharedWorkspaceFact>();
  private readonly readCache = new Map<string, CachedReadEntry>();
  private readonly searchCache = new Map<string, CachedSearchEntry>();
  private readonly scopeVersions = new Map<string, number>();
  private readonly inspections = new Map<string, InspectionRecord>();

  constructor(projectRoot: string, persistedArtifacts: ArtifactMetadata[] = []) {
    this.projectRoot = projectRoot;
    for (const artifact of persistedArtifacts) {
      const normalizedPath = this.normalizePath(artifact.path);
      if (!normalizedPath) {
        continue;
      }
      this.recordFact({
        path: normalizedPath,
        factType: 'created',
        ownerBranchId: artifact.branchId || 'persisted',
        step: 0,
        source: 'persisted_artifact',
        confidence: 0.95,
      });
    }
  }

  normalizePath(rawPath: string): string {
    const value = String(rawPath || '').trim();
    if (!value) {
      return '';
    }
    const slashNormalized = value.replace(/\\/g, '/');
    if (!path.isAbsolute(slashNormalized)) {
      return path.posix.normalize(slashNormalized).replace(/^\.\//, '').replace(/^\/+/, '');
    }
    const relative = path.relative(this.projectRoot, slashNormalized).replace(/\\/g, '/');
    if (!relative.startsWith('..') && !path.isAbsolute(relative)) {
      return path.posix.normalize(relative).replace(/^\.\//, '');
    }
    return path.posix.normalize(slashNormalized);
  }

  getReadCache(pathValue: string): string | null {
    const normalizedPath = this.normalizePath(pathValue);
    if (!normalizedPath) {
      return null;
    }
    const entry = this.readCache.get(normalizedPath);
    if (!entry) {
      return null;
    }
    return entry.version === this.getScopeVersion(this.buildPathScopeKey(normalizedPath))
      ? entry.output
      : null;
  }

  storeReadResult(
    pathValue: string,
    output: string,
    context: WorkspaceToolExecutionContext,
  ): WorkspaceToolExecutionMetadata {
    const normalizedPath = this.normalizePath(pathValue);
    if (!normalizedPath) {
      return {};
    }

    const version = this.getScopeVersion(this.buildPathScopeKey(normalizedPath));
    this.readCache.set(normalizedPath, { output, version });

    const workspaceFacts = [
      this.recordFact({
        path: normalizedPath,
        factType: 'exists',
        ownerBranchId: context.branchId || 'unknown',
        step: context.step || 0,
        source: 'workspace_read',
        confidence: 0.85,
      }),
      this.recordFact({
        path: normalizedPath,
        factType: 'inspected',
        ownerBranchId: context.branchId || 'unknown',
        step: context.step || 0,
        source: 'workspace_read',
        confidence: 0.8,
      }),
    ];

    return {
      artifacts: [normalizedPath],
      outputs: [normalizedPath],
      workspaceFacts,
      duplicateInspection: this.registerInspection({
        kind: 'read',
        target: normalizedPath,
        branchId: context.branchId || 'unknown',
        step: context.step || 0,
        version,
        cached: false,
      }),
    };
  }

  buildReadCacheHitMetadata(
    pathValue: string,
    context: WorkspaceToolExecutionContext,
  ): WorkspaceToolExecutionMetadata {
    const normalizedPath = this.normalizePath(pathValue);
    if (!normalizedPath) {
      return { cacheHit: true };
    }
    const version = this.getScopeVersion(this.buildPathScopeKey(normalizedPath));
    return {
      cacheHit: true,
      artifacts: [normalizedPath],
      outputs: [normalizedPath],
      workspaceFacts: [
        this.recordFact({
          path: normalizedPath,
          factType: 'inspected',
          ownerBranchId: context.branchId || 'unknown',
          step: context.step || 0,
          source: 'workspace_read_cache_hit',
          confidence: 0.9,
        }),
      ],
      duplicateInspection: this.registerInspection({
        kind: 'read',
        target: normalizedPath,
        branchId: context.branchId || 'unknown',
        step: context.step || 0,
        version,
        cached: true,
      }),
    };
  }

  getSearchCache(cacheKey: string, scopeKey: string): string | null {
    const entry = this.searchCache.get(cacheKey);
    if (!entry) {
      return null;
    }
    return entry.version === this.getScopeVersion(scopeKey)
      ? entry.output
      : null;
  }

  storeSearchResult(
    kind: 'glob' | 'grep',
    cacheKey: string,
    scopeKey: string,
    output: string,
    context: WorkspaceToolExecutionContext,
  ): WorkspaceToolExecutionMetadata {
    const version = this.getScopeVersion(scopeKey);
    this.searchCache.set(cacheKey, { output, version });

    const outputs = this.extractPathsFromSearchOutput(output).slice(0, 6);
    const workspaceFacts = outputs.slice(0, 6).map((resultPath) => this.recordFact({
      path: resultPath,
      factType: 'exists',
      ownerBranchId: context.branchId || 'unknown',
      step: context.step || 0,
      source: `workspace_search_${kind}`,
      confidence: 0.7,
    }));

    return {
      outputs,
      artifacts: outputs.slice(0, 3),
      workspaceFacts,
      duplicateInspection: this.registerInspection({
        kind,
        target: cacheKey,
        branchId: context.branchId || 'unknown',
        step: context.step || 0,
        version,
        cached: false,
      }),
    };
  }

  buildSearchCacheHitMetadata(
    kind: 'glob' | 'grep',
    cacheKey: string,
    output: string,
    context: WorkspaceToolExecutionContext,
    scopeKey: string,
  ): WorkspaceToolExecutionMetadata {
    const version = this.getScopeVersion(scopeKey);
    const outputs = this.extractPathsFromSearchOutput(output).slice(0, 6);
    return {
      cacheHit: true,
      outputs,
      artifacts: outputs.slice(0, 3),
      workspaceFacts: outputs.slice(0, 6).map((resultPath) => this.recordFact({
        path: resultPath,
        factType: 'inspected',
        ownerBranchId: context.branchId || 'unknown',
        step: context.step || 0,
        source: `workspace_search_${kind}_cache_hit`,
        confidence: 0.75,
      })),
      duplicateInspection: this.registerInspection({
        kind,
        target: cacheKey,
        branchId: context.branchId || 'unknown',
        step: context.step || 0,
        version,
        cached: true,
      }),
    };
  }

  recordWorkspaceEdit(
    pathValue: string,
    context: WorkspaceToolExecutionContext,
  ): WorkspaceToolExecutionMetadata {
    const normalizedPath = this.normalizePath(pathValue);
    if (!normalizedPath) {
      return {};
    }
    this.invalidatePath(normalizedPath);
    return {
      artifacts: [normalizedPath],
      outputs: [normalizedPath],
      workspaceFacts: [
        this.recordFact({
          path: normalizedPath,
          factType: 'modified',
          ownerBranchId: context.branchId || 'unknown',
          step: context.step || 0,
          source: 'workspace_edit',
          confidence: 0.95,
        }),
      ],
    };
  }

  recordBashMutations(
    command: string,
    context: WorkspaceToolExecutionContext,
  ): WorkspaceToolExecutionMetadata {
    const outputs = this.extractMutationPathsFromBash(command);
    if (outputs.length === 0) {
      return {};
    }

    const workspaceFacts: SharedWorkspaceFact[] = [];
    outputs.forEach((resultPath) => {
      this.invalidatePath(resultPath.path);
      workspaceFacts.push(this.recordFact({
        path: resultPath.path,
        factType: resultPath.factType,
        ownerBranchId: context.branchId || 'unknown',
        step: context.step || 0,
        source: 'bash_workspace_mutation',
        confidence: 0.8,
      }));
    });

    const normalizedOutputs = outputs.map((entry) => entry.path);
    return {
      artifacts: normalizedOutputs.slice(0, 6),
      outputs: normalizedOutputs.slice(0, 6),
      workspaceFacts,
    };
  }

  renderRelevantFactsDigest(branchId: string, goal: string, currentArtifacts: string[] = [], limit = 6): string {
    const goalTokens = this.tokenize(goal);
    const artifactTokens = currentArtifacts.flatMap((artifact) => this.tokenize(artifact));
    const candidates = [...this.factIndex.values()]
      .sort((left, right) => {
        const leftScore = this.scoreFact(left, branchId, goalTokens, artifactTokens);
        const rightScore = this.scoreFact(right, branchId, goalTokens, artifactTokens);
        return rightScore - leftScore || right.updatedAt - left.updatedAt;
      })
      .slice(0, limit);

    return candidates
      .map((fact) => `${fact.path} | ${fact.factType} | by=${fact.ownerBranchId} | via=${fact.source}`)
      .join('\n');
  }

  renderDuplicateInspectionDigest(branchId: string, limit = 4): string {
    return [...this.inspections.values()]
      .filter((inspection) => inspection.count > 1 && inspection.branchId === branchId)
      .sort((left, right) => right.updatedAt - left.updatedAt)
      .slice(0, limit)
      .map((inspection) =>
        `${inspection.kind}:${inspection.target} repeated ${inspection.count}x without workspace mutation${inspection.cached ? ' (cache-hit)' : ''}`)
      .join('\n');
  }

  renderSharedProgressDigest(branchId: string, goal: string, currentArtifacts: string[] = [], limit = 6): string {
    const factLines = this.renderRelevantFactsDigest(branchId, goal, currentArtifacts, limit);
    const duplicateLines = this.renderDuplicateInspectionDigest(branchId, Math.max(1, Math.floor(limit / 2)));
    return [factLines, duplicateLines ? `duplicate_inspections:\n${duplicateLines}` : '']
      .filter(Boolean)
      .join('\n');
  }

  private extractPathsFromSearchOutput(output: string): string[] {
    try {
      const parsed = JSON.parse(output);
      if (parsed?.kind === 'workspace_glob_results' && Array.isArray(parsed.results)) {
        return parsed.results
          .map((entry: unknown) => this.normalizePath(String(entry || '')))
          .filter(Boolean);
      }
      if (parsed?.kind === 'workspace_grep_results' && Array.isArray(parsed.results)) {
        return parsed.results
          .map((entry: any) => this.normalizePath(String(entry?.path || '')))
          .filter(Boolean);
      }
    } catch {
      return [];
    }
    return [];
  }

  private recordFact(input: Omit<SharedWorkspaceFact, 'updatedAt'>): SharedWorkspaceFact {
    const key = `${input.path}::${input.factType}`;
    const fact: SharedWorkspaceFact = {
      ...input,
      updatedAt: Date.now(),
    };
    this.factIndex.set(key, fact);
    return fact;
  }

  private scoreFact(
    fact: SharedWorkspaceFact,
    branchId: string,
    goalTokens: string[],
    artifactTokens: string[],
  ): number {
    let score = 0;
    if (fact.ownerBranchId !== branchId) {
      score += 10;
    }
    if (fact.source === 'persisted_artifact') {
      score += 4;
    }
    const factTokens = this.tokenize(fact.path);
    for (const token of goalTokens) {
      if (factTokens.includes(token)) {
        score += 3;
      }
    }
    for (const token of artifactTokens) {
      if (factTokens.includes(token)) {
        score += 2;
      }
    }
    score += Math.min(5, fact.confidence * 5);
    return score;
  }

  private tokenize(value: string): string[] {
    return Array.from(new Set(
      String(value || '')
        .toLowerCase()
        .split(/[^a-z0-9._/-]+/i)
        .map((token) => token.trim())
        .filter((token) => token.length >= 2),
    ));
  }

  private registerInspection(input: Omit<InspectionRecord, 'key' | 'count' | 'updatedAt'>): string {
    const key = `${input.kind}:${input.target}`;
    const current = this.inspections.get(key);
    const count = current && current.version === input.version ? current.count + 1 : 1;
    this.inspections.set(key, {
      ...input,
      key,
      count,
      updatedAt: Date.now(),
    });
    return count > 1
      ? `${input.kind}:${input.target} repeated ${count}x without workspace mutation`
      : '';
  }

  private getScopeVersion(scopeKey: string): number {
    return this.scopeVersions.get(scopeKey) || 0;
  }

  private bumpScopeVersion(scopeKey: string): void {
    this.scopeVersions.set(scopeKey, this.getScopeVersion(scopeKey) + 1);
  }

  private invalidatePath(normalizedPath: string): void {
    const parts = normalizedPath.split('/').filter(Boolean);
    this.bumpScopeVersion('workspace');
    this.bumpScopeVersion(this.buildPathScopeKey(normalizedPath));
    if (parts.length === 0) {
      this.bumpScopeVersion(this.buildDirScopeKey('.'));
      return;
    }
    for (let index = parts.length; index >= 1; index -= 1) {
      const prefix = parts.slice(0, index - 1).join('/') || '.';
      this.bumpScopeVersion(this.buildDirScopeKey(prefix));
    }
  }

  private buildPathScopeKey(normalizedPath: string): string {
    return `path:${normalizedPath}`;
  }

  buildGlobScopeKey(pattern: string): string {
    const normalizedPattern = this.normalizePath(pattern);
    const wildcardIndex = normalizedPattern.search(/[*?]/);
    const prefix = wildcardIndex >= 0 ? normalizedPattern.slice(0, wildcardIndex) : normalizedPattern;
    const dirPrefix = prefix.includes('/') ? prefix.slice(0, prefix.lastIndexOf('/')) : prefix;
    return this.buildDirScopeKey(dirPrefix || '.');
  }

  buildGrepScopeKey(): string {
    return 'workspace';
  }

  private buildDirScopeKey(normalizedDir: string): string {
    return `dir:${normalizedDir || '.'}`;
  }

  private extractMutationPathsFromBash(command: string): Array<{ path: string; factType: SharedWorkspaceFactType }> {
    const outputs: Array<{ path: string; factType: SharedWorkspaceFactType }> = [];
    const tokens = (command.match(/'[^']*'|"[^"]*"|[^\s]+/g) || [])
      .map((token) => token.replace(/^['"]|['"]$/g, ''));

    const pushPath = (rawPath: string | undefined, factType: SharedWorkspaceFactType) => {
      const normalizedPath = this.normalizePath(String(rawPath || ''));
      if (!normalizedPath || normalizedPath.startsWith('/')) {
        return;
      }
      outputs.push({ path: normalizedPath, factType });
    };

    for (let index = 0; index < tokens.length; index += 1) {
      const token = tokens[index];
      if (token === 'mkdir') {
        let cursor = index + 1;
        while (cursor < tokens.length && tokens[cursor].startsWith('-')) {
          cursor += 1;
        }
        while (cursor < tokens.length && !['&&', ';', '||', '|'].includes(tokens[cursor])) {
          pushPath(tokens[cursor], 'created');
          cursor += 1;
        }
      }
      if (token === 'touch') {
        let cursor = index + 1;
        while (cursor < tokens.length && !['&&', ';', '||', '|'].includes(tokens[cursor])) {
          pushPath(tokens[cursor], 'modified');
          cursor += 1;
        }
      }
      if (token === 'cp' || token === 'mv') {
        const target = tokens[index + 2];
        pushPath(target, token === 'mv' ? 'created' : 'modified');
      }
      if (token === '>' || token === '>>') {
        pushPath(tokens[index + 1], 'modified');
      }
      if (token.includes('>')) {
        const redirected = token.split('>').at(-1);
        if (redirected) {
          pushPath(redirected, 'modified');
        }
      }
    }

    const seen = new Set<string>();
    return outputs.filter((entry) => {
      const key = `${entry.path}:${entry.factType}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }
}
