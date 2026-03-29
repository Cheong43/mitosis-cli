import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

function findNearestAncestor(startDir: string, predicate: (dir: string) => boolean): string | null {
  let current = path.resolve(startDir);
  while (true) {
    if (predicate(current)) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return null;
    }
    current = parent;
  }
}

function isCodeCliRepoRoot(dir: string): boolean {
  const packagePath = path.join(dir, 'package.json');
  if (!fs.existsSync(packagePath) || !fs.existsSync(path.join(dir, 'src'))) {
    return false;
  }

  try {
    const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf-8'));
    if (pkg && pkg.name === 'mitosis-cli') {
      return true;
    }
  } catch {}

  return fs.existsSync(path.join(dir, 'souls.md'))
    && fs.existsSync(path.join(dir, 'src', 'index.tsx'));
}

export function resolveWorkspaceSkillRoots(projectRoot: string, codeCliRoot: string): string[] {
  const userClaudeAgentsRoot = path.join(
    process.env.HOME?.trim() || os.homedir(),
    '.claude',
    'agents',
  );
  const roots = [
    path.join(projectRoot, '.claude', 'agents'),
    path.join(projectRoot, 'skills'),
    path.join(projectRoot, '.github', 'skills'),
  ];
  const normalizedProjectRoot = path.resolve(projectRoot);
  const normalizedCodeCliRoot = path.resolve(codeCliRoot);
  if (normalizedCodeCliRoot !== normalizedProjectRoot) {
    roots.push(
      path.join(codeCliRoot, '.claude', 'agents'),
      path.join(codeCliRoot, 'skills'),
      path.join(codeCliRoot, '.github', 'skills'),
    );
  }
  roots.push(userClaudeAgentsRoot);
  return [...new Set(roots.map((root) => path.resolve(root)))];
}

export function resolveCodeCliRoot(moduleDir: string): string {
  const searchStarts = [process.cwd(), moduleDir, path.resolve(moduleDir, '..')];
  for (const start of searchStarts) {
    const codeCliRoot = findNearestAncestor(start, isCodeCliRepoRoot);
    if (codeCliRoot) {
      return codeCliRoot;
    }
  }

  return path.resolve(moduleDir, '../..');
}

export function resolveProjectRoot(moduleDir: string, envProjectRoot = process.env.MITOSIS_PROJECT_ROOT ?? process.env.MEMPEDIA_PROJECT_ROOT): string {
  if (envProjectRoot && envProjectRoot.trim()) {
    return path.resolve(envProjectRoot);
  }

  return resolveCodeCliRoot(moduleDir);
}
