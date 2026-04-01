/**
 * Tool-specific result compression.
 *
 * Compresses tool observations before they are added to the transcript,
 * keeping the most informative content while dramatically reducing token
 * usage for verbose outputs.
 */

const DEFAULT_MAX_CHARS = 800;

export function compressToolResult(
  toolName: string,
  result: string,
  maxChars: number = DEFAULT_MAX_CHARS,
): string {
  if (result.length <= maxChars) return result;

  switch (toolName) {
    case 'read':
      return compressFileContent(result, maxChars);
    case 'bash':
      return compressShellOutput(result, maxChars);
    case 'search':
    case 'grep':
      return compressSearchResults(result, maxChars);
    case 'web':
      return compressWebContent(result, maxChars);
    default:
      return truncateWithContext(result, maxChars);
  }
}

/**
 * Compress file content: keep head + tail, omit middle with line count.
 */
export function compressFileContent(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content;

  const lines = content.split('\n');
  if (lines.length <= 10) return truncateWithContext(content, maxChars);

  const headBudget = Math.floor(maxChars * 0.4);
  const tailBudget = Math.floor(maxChars * 0.4);

  const headLines: string[] = [];
  let headLen = 0;
  for (const line of lines) {
    if (headLen + line.length + 1 > headBudget) break;
    headLines.push(line);
    headLen += line.length + 1;
  }

  const tailLines: string[] = [];
  let tailLen = 0;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (tailLen + lines[i].length + 1 > tailBudget) break;
    tailLines.unshift(lines[i]);
    tailLen += lines[i].length + 1;
  }

  const omitted = lines.length - headLines.length - tailLines.length;
  if (omitted <= 0) return truncateWithContext(content, maxChars);

  return [
    ...headLines,
    `\n[... ${omitted} lines omitted ...]\n`,
    ...tailLines,
  ].join('\n');
}

/**
 * Compress shell output: prioritize errors, warnings, and final status.
 */
export function compressShellOutput(output: string, maxChars: number): string {
  if (output.length <= maxChars) return output;

  const lines = output.split('\n');

  // Extract important lines (errors, warnings, exit status)
  const importantPatterns = /^(error|warning|fatal|fail|exception|panic|traceback|exit|status|npm ERR|ENOENT|EPERM)/i;
  const importantLines = lines.filter((line) => importantPatterns.test(line.trim()));

  // Keep last N lines for final status
  const tailCount = Math.min(10, Math.floor(lines.length * 0.2));
  const tailLines = lines.slice(-tailCount);

  // Build compressed output
  const parts: string[] = [];

  if (importantLines.length > 0) {
    parts.push('Key output:');
    const errorBudget = Math.floor(maxChars * 0.5);
    let errorLen = 0;
    for (const line of importantLines) {
      if (errorLen + line.length > errorBudget) break;
      parts.push(line);
      errorLen += line.length;
    }
  }

  parts.push(`\n[... ${lines.length} total lines, showing tail ...]\n`);
  const tailBudget = maxChars - parts.join('\n').length;
  let tailLen = 0;
  for (const line of tailLines) {
    if (tailLen + line.length > tailBudget) break;
    parts.push(line);
    tailLen += line.length;
  }

  return parts.join('\n');
}

/**
 * Compress search results: keep match headers and limit context lines.
 */
export function compressSearchResults(results: string, maxChars: number): string {
  if (results.length <= maxChars) return results;

  const lines = results.split('\n');
  const compressed: string[] = [];
  let totalLen = 0;

  for (const line of lines) {
    // Prioritize match lines (typically contain file:line: match)
    const isMatch = /^\S+:\d+:/.test(line) || /^(Match|Result|Found)/i.test(line);
    if (isMatch || totalLen < maxChars * 0.8) {
      if (totalLen + line.length + 1 > maxChars) break;
      compressed.push(line);
      totalLen += line.length + 1;
    }
  }

  if (compressed.length < lines.length) {
    compressed.push(`[... ${lines.length - compressed.length} more results omitted ...]`);
  }

  return compressed.join('\n');
}

/**
 * Compress web content: strip boilerplate, keep core text.
 */
export function compressWebContent(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content;

  // Remove common boilerplate patterns
  let cleaned = content
    .replace(/\n{3,}/g, '\n\n')                    // Multiple blank lines
    .replace(/^\s*(cookie|privacy|terms|copyright|©).*$/gmi, '')  // Legal/cookie notices
    .replace(/\[.*?(sign up|log in|subscribe).*?\]/gi, '')        // CTA buttons
    .trim();

  return truncateWithContext(cleaned, maxChars);
}

/**
 * Generic truncation that keeps head and tail with an omission marker.
 */
export function truncateWithContext(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;

  const headLen = Math.floor(maxChars * 0.6);
  const tailLen = Math.floor(maxChars * 0.3);
  const head = text.slice(0, headLen);
  const tail = text.slice(-tailLen);
  const omitted = text.length - headLen - tailLen;

  return `${head}\n[... ${omitted} chars omitted ...]\n${tail}`;
}
