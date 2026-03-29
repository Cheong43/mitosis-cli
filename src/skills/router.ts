import * as fs from 'node:fs';
import * as path from 'node:path';

import { resolveWorkspaceSkillRoots } from '../config/projectPaths.js';

export interface SkillRecord {
  name: string;
  description: string;
  content: string;
  tools?: string[];
  category?: string;
  priority?: number;
  alwaysInclude?: boolean;
  tags?: string[];
  source?: 'local' | 'remote' | 'claude-agent';
  location?: string;
  repository?: string;
}

function normalizePath(target: string): string {
  return target.replace(/\\/g, '/');
}

function listFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...listFiles(full));
    } else {
      files.push(full);
    }
  }
  return files;
}

function extractScalar(meta: string, key: string): string | undefined {
  return meta.match(new RegExp(`(?:^|\\n)\\s*${key}:\\s*"?([^"\\n]+)"?\\s*$`, 'im'))?.[1]?.trim();
}

function extractYamlList(meta: string, key: string): string[] {
  const inline = meta.match(new RegExp(`(?:^|\\n)\\s*${key}:\\s*\\[([^\\]]*)\\]\\s*$`, 'im'))?.[1];
  if (inline) {
    return inline
      .split(',')
      .map((value) => value.replace(/["']/g, '').trim())
      .filter(Boolean);
  }

  const block = meta.match(new RegExp(`(?:^|\\n)\\s*${key}:\\s*\\r?\\n((?:\\s*-\\s*[^\\n]+\\r?\\n?)*)`, 'im'))?.[1];
  if (block) {
    return block
      .split(/\r?\n/)
      .map((line) => line.match(/^\s*-\s*(.+)$/)?.[1]?.replace(/["']/g, '').trim() || '')
      .filter(Boolean);
  }

  const scalar = extractScalar(meta, key);
  if (!scalar) {
    return [];
  }
  return scalar
    .split(',')
    .map((value) => value.replace(/["']/g, '').trim())
    .filter(Boolean);
}

function isClaudeAgentLocation(location?: string): boolean {
  if (!location) {
    return false;
  }
  const normalized = normalizePath(location);
  return normalized.endsWith('/.claude/agents') || normalized.includes('/.claude/agents/');
}

function toDisplayLocation(projectRoot: string, filePath: string): string {
  const relative = normalizePath(path.relative(projectRoot, filePath));
  if (!relative.startsWith('..')) {
    return relative;
  }
  return normalizePath(path.resolve(filePath));
}

export function parseSkillMarkdown(raw: string, fallbackName: string, extra: Partial<SkillRecord> = {}): SkillRecord {
  const frontmatter = raw.match(/^---\s*[\r\n]+([\s\S]*?)\s*[\r\n]+---\s*[\r\n]*/);
  const body = frontmatter ? raw.slice(frontmatter[0].length).trim() : raw.trim();
  const meta = frontmatter ? frontmatter[1] : '';
  const name = extractScalar(meta, 'name') || fallbackName;
  const description = extractScalar(meta, 'description') || 'No description';
  const category = extractScalar(meta, 'category');
  const rawPriority = (extractScalar(meta, 'priority') || '').toLowerCase();
  const priority = rawPriority === 'high'
    ? 100
    : rawPriority === 'medium'
      ? 50
      : rawPriority === 'low'
        ? 10
        : Number(rawPriority || 0);
  const alwaysInclude = /(?:^|\n)\s*always_include:\s*(true|yes|1)\b/i.test(meta);
  const tags = extractYamlList(meta, 'tags');
  const tools = extractYamlList(meta, 'tools');

  return {
    name,
    description,
    content: body,
    ...(tools.length > 0 ? { tools } : {}),
    category,
    priority: Number.isFinite(priority) ? priority : 0,
    alwaysInclude,
    tags,
    ...extra,
  };
}

export function sortSkills(items: SkillRecord[]): SkillRecord[] {
  return [...items].sort((a, b) => a.name.localeCompare(b.name));
}

export function mergeSkills(...groups: SkillRecord[][]): SkillRecord[] {
  const merged = new Map<string, SkillRecord>();
  for (const group of groups) {
    for (const skill of group) {
      const key = `${skill.source || 'local'}::${skill.location || skill.name}`;
      if (!merged.has(key)) {
        merged.set(key, skill);
      }
    }
  }
  return sortSkills([...merged.values()]);
}

export function loadWorkspaceSkills(projectRoot: string, codeCliRoot: string): SkillRecord[] {
  const skillRoots = resolveWorkspaceSkillRoots(projectRoot, codeCliRoot);
  const loaded: SkillRecord[] = [];
  const seenByName = new Set<string>();
  for (const root of skillRoots) {
    if (!fs.existsSync(root)) continue;
    const skillFiles = listFiles(root).filter((filePath) => {
      if (isClaudeAgentLocation(root)) {
        return filePath.toLowerCase().endsWith('.md');
      }
      return path.basename(filePath) === 'SKILL.md';
    });
    for (const filePath of skillFiles) {
      try {
        const markdown = fs.readFileSync(filePath, 'utf-8');
        const fallbackName = isClaudeAgentLocation(root)
          ? path.basename(filePath, path.extname(filePath)) || 'unnamed-skill'
          : path.basename(path.dirname(filePath)) || 'unnamed-skill';
        const parsed = parseSkillMarkdown(markdown, fallbackName, {
          source: isClaudeAgentLocation(root) ? 'claude-agent' : 'local',
          location: toDisplayLocation(projectRoot, filePath),
        });
        const nameKey = parsed.name.trim().toLowerCase();
        if (seenByName.has(nameKey)) {
          continue;
        }
        seenByName.add(nameKey);
        loaded.push(parsed);
      } catch {}
    }
  }
  return sortSkills(loaded);
}

function tokenizeForSkillMatch(value: string): string[] {
  const matches = value.toLowerCase().match(/[\p{L}\p{N}_-]+/gu) || [];
  return [...new Set(matches.filter((token) => token.length >= 2))];
}

function scoreSkillMatch(query: string, skill: SkillRecord): number {
  const queryTokens = tokenizeForSkillMatch(query);
  if (queryTokens.length === 0) return 0;
  const skillName = skill.name.toLowerCase();
  const skillDescription = skill.description.toLowerCase();
  const skillBody = skill.content.toLowerCase().slice(0, 1600);
  let score = 0;
  for (const token of queryTokens) {
    if (skillName.includes(token)) {
      score += 3;
    } else if (skillDescription.includes(token)) {
      score += 2;
    } else if (skillBody.includes(token)) {
      score += 1;
    }
  }
  return score / queryTokens.length;
}

export function isMempediaSkill(skill: SkillRecord): boolean {
  return skill.category === 'mempedia' || (skill.tags || []).includes('mempedia');
}

export function selectAutoSkills(query: string, skills: SkillRecord[], maxCount = 3): SkillRecord[] {
  const pinned = skills
    .filter((skill) => skill.alwaysInclude && isMempediaSkill(skill))
    .sort((a, b) => (b.priority || 0) - (a.priority || 0) || a.name.localeCompare(b.name));

  const ranked = skills
    .map((skill) => {
      const score = scoreSkillMatch(query, skill);
      const priorityBonus = (skill.priority || 0) / 100;
      const mempediaBonus = isMempediaSkill(skill) ? 0.5 : 0;
      return { skill, score: score + priorityBonus + mempediaBonus };
    })
    .filter((item) => item.skill.alwaysInclude || item.score >= (isMempediaSkill(item.skill) ? 0.75 : 1))
    .sort((a, b) => b.score - a.score || a.skill.name.localeCompare(b.skill.name))
    .map((item) => item.skill);

  const merged = new Map<string, SkillRecord>();
  for (const skill of [...pinned, ...ranked]) {
    const key = `${skill.source || 'local'}::${skill.location || skill.name}`;
    if (!merged.has(key)) {
      merged.set(key, skill);
    }
  }

  return [...merged.values()].slice(0, maxCount);
}

export function findSkillByName(skills: SkillRecord[], targetName: string): SkillRecord | undefined {
  const normalized = targetName.trim().toLowerCase();
  return skills.find((skill) => {
    const name = skill.name.toLowerCase();
    const repository = skill.repository?.toLowerCase() || '';
    return name === normalized || name.endsWith(`/${normalized}`) || name.includes(normalized) || repository.includes(normalized);
  });
}

export function resolveSkillsByName(skills: SkillRecord[], requestedNames: string[], maxCount = 2): SkillRecord[] {
  const resolved: SkillRecord[] = [];
  const seen = new Set<string>();
  for (const requestedName of requestedNames) {
    const match = findSkillByName(skills, requestedName);
    if (!match) {
      continue;
    }
    const key = `${match.source || 'local'}::${match.location || match.name}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    resolved.push(match);
    if (resolved.length >= maxCount) {
      break;
    }
  }
  return resolved;
}

export function renderSkillCatalog(skills: SkillRecord[], maxCount = 24): string {
  if (skills.length === 0) {
    return '';
  }
  const lines = skills.slice(0, maxCount).map((skill) => `  - ${skill.name}: ${skill.description}`);
  return `Available local skills (names and descriptions only; request full guidance only when needed):\n${lines.join('\n')}`;
}

export function renderSkillGuidance(skills: SkillRecord[]): string {
  return skills
    .map((skill) => {
      const lines = [
        `Skill: ${skill.name}`,
        `Description: ${skill.description}`,
      ];
      if (skill.tools?.length) {
        lines.push(`Allowed tools: ${skill.tools.join(', ')}`);
      }
      lines.push('Content:');
      lines.push(skill.content);
      return lines.join('\n');
    })
    .join('\n\n---\n\n');
}
