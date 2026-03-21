import * as fs from 'node:fs';
import * as path from 'node:path';

import { resolveWorkspaceSkillRoots } from '../config/projectPaths.js';

export interface SkillRecord {
  name: string;
  description: string;
  content: string;
  category?: string;
  priority?: number;
  alwaysInclude?: boolean;
  tags?: string[];
  source?: 'local' | 'remote';
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

export function parseSkillMarkdown(raw: string, fallbackName: string, extra: Partial<SkillRecord> = {}): SkillRecord {
  const frontmatter = raw.match(/^---\s*[\r\n]+([\s\S]*?)\s*[\r\n]+---\s*[\r\n]*/);
  const body = frontmatter ? raw.slice(frontmatter[0].length).trim() : raw.trim();
  const meta = frontmatter ? frontmatter[1] : '';
  const name = meta.match(/name:\s*"?([^"\n]+)"?/i)?.[1]?.trim() || fallbackName;
  const description = meta.match(/description:\s*"?([^"\n]+)"?/i)?.[1]?.trim() || 'No description';
  const category = meta.match(/category:\s*"?([^"\n]+)"?/i)?.[1]?.trim();
  const rawPriority = meta.match(/priority:\s*"?([^"\n]+)"?/i)?.[1]?.trim().toLowerCase();
  const priority = rawPriority === 'high'
    ? 100
    : rawPriority === 'medium'
      ? 50
      : rawPriority === 'low'
        ? 10
        : Number(rawPriority || 0);
  const alwaysInclude = /always_include:\s*(true|yes|1)/i.test(meta);
  const tagBlock = meta.match(/tags:\s*\[([^\]]*)\]/i)?.[1] || '';
  const tags = tagBlock
    .split(',')
    .map((value) => value.replace(/["']/g, '').trim())
    .filter(Boolean);

  return {
    name,
    description,
    content: body,
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
  for (const root of skillRoots) {
    if (!fs.existsSync(root)) continue;
    const skillFiles = listFiles(root).filter((filePath) => path.basename(filePath) === 'SKILL.md');
    for (const filePath of skillFiles) {
      try {
        const markdown = fs.readFileSync(filePath, 'utf-8');
        const fallbackName = path.basename(path.dirname(filePath)) || 'unnamed-skill';
        loaded.push(parseSkillMarkdown(markdown, fallbackName, {
          source: 'local',
          location: normalizePath(path.relative(projectRoot, filePath)),
        }));
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
    .map((skill) => `Skill: ${skill.name}\nDescription: ${skill.description}\nContent:\n${skill.content}`)
    .join('\n\n---\n\n');
}