import * as fs from 'fs';
import * as path from 'path';

const LOG_DIR = '.mitosis/logs';
const ERROR_LOG_FILE = 'error.log';

function ensureLogDir(projectRoot: string): string {
  const logDir = path.join(projectRoot, LOG_DIR);
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  return path.join(logDir, ERROR_LOG_FILE);
}

export function logError(projectRoot: string, error: unknown, context?: string): void {
  const logPath = ensureLogDir(projectRoot);
  const timestamp = new Date().toISOString();
  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : '';

  const logEntry = `[${timestamp}]${context ? ` [${context}]` : ''} ${message}\n${stack ? `${stack}\n` : ''}\n`;

  fs.appendFileSync(logPath, logEntry, 'utf8');
}
