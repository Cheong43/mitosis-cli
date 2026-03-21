import * as fs from 'fs';
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
  return fs.existsSync(path.join(dir, 'package.json'))
    && fs.existsSync(path.join(dir, 'src'));
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