/** Minimal shape matching the OpenAI function-calling tool format. */
interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export const TOOLS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'read',
      description: 'Read from a semantic target. Use target=workspace for repo files, target=memory for Mempedia nodes, target=preferences for project preferences, and target=skills for local skill guidance.',
      parameters: {
        type: 'object',
        properties: {
          target: { type: 'string', enum: ['workspace', 'memory', 'preferences', 'skills'], description: 'Semantic read target.' },
          path: { type: 'string', description: 'Workspace-relative file path when target=workspace.' },
          node_id: { type: 'string', description: 'Memory node id when target=memory.' },
          markdown: { type: 'boolean', description: 'When target=memory, prefer markdown output.' },
          skill_id: { type: 'string', description: 'Skill id or skill name when target=skills.' },
        },
        required: ['target'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search',
      description: 'Search a semantic target. Use workspace with grep/glob, memory with hybrid/keyword/episodic search, preferences with query match, or skills with local skill lookup.',
      parameters: {
        type: 'object',
        properties: {
          target: { type: 'string', enum: ['workspace', 'memory', 'preferences', 'skills'], description: 'Semantic search target.' },
          mode: { type: 'string', enum: ['grep', 'glob', 'keyword', 'hybrid', 'episodic'], description: 'Search mode; valid values depend on target.' },
          query: { type: 'string', description: 'Search query for grep, memory, preferences, or skills.' },
          pattern: { type: 'string', description: 'Glob pattern when target=workspace and mode=glob.' },
          limit: { type: 'number', description: 'Maximum number of results.' },
          rrf_k: { type: 'number', description: 'Optional hybrid retrieval parameter for target=memory.' },
          bm25_weight: { type: 'number', description: 'Optional hybrid retrieval parameter for target=memory.' },
          vector_weight: { type: 'number', description: 'Optional hybrid retrieval parameter for target=memory.' },
          graph_weight: { type: 'number', description: 'Optional hybrid retrieval parameter for target=memory.' },
          graph_depth: { type: 'number', description: 'Optional hybrid retrieval parameter for target=memory.' },
          graph_seed_limit: { type: 'number', description: 'Optional hybrid retrieval parameter for target=memory.' },
        },
        required: ['target', 'mode'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'edit',
      description: 'Edit a semantic target. Use workspace for repo files, memory for Mempedia markdown nodes, preferences for project preference markdown, and skills for local SKILL.md guidance.',
      parameters: {
        type: 'object',
        properties: {
          target: { type: 'string', enum: ['workspace', 'memory', 'preferences', 'skills'], description: 'Semantic edit target.' },
          path: { type: 'string', description: 'Workspace-relative file path when target=workspace.' },
          node_id: { type: 'string', description: 'Memory node id when target=memory.' },
          skill_id: { type: 'string', description: 'Skill id or skill name when target=skills.' },
          title: { type: 'string', description: 'Optional title when editing skills or memory.' },
          description: { type: 'string', description: 'Optional skill description when target=skills.' },
          content: { type: 'string', description: 'Full content to write. For memory this is markdown; for preferences it is the whole file; for skills it is SKILL.md body content.' },
          reason: { type: 'string', description: 'Optional memory write reason when target=memory.' },
          source: { type: 'string', description: 'Optional memory write source when target=memory.' },
          importance: { type: 'number', description: 'Optional memory importance when target=memory.' },
          parent_node: { type: 'string', description: 'Optional memory hierarchy field when target=memory.' },
          node_type: { type: 'string', description: 'Optional memory semantic type when target=memory.' },
          tags: { type: 'array', items: { type: 'string' }, description: 'Optional skill tags when target=skills.' },
        },
        required: ['target'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'bash',
      description: 'Run a shell command inside the local sandbox. Dangerous operations must stay sandboxed and require confirmation.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The shell command to run.' },
        },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'web',
      description: 'Search the web with budget/permission-aware citation results, or fetch a specific web page into a citation summary with highlights and a short preview. Use for external/current information not in the workspace.',
      parameters: {
        type: 'object',
        properties: {
          mode: { type: 'string', enum: ['search', 'fetch'], description: 'Whether to run a web search or fetch a page.' },
          query: { type: 'string', description: 'Search query when mode=search.' },
          url: { type: 'string', description: 'URL to fetch when mode=fetch.' },
          limit: { type: 'number', description: 'Maximum number of search results.' },
          allowed_domains: {
            type: 'array',
            items: { type: 'string' },
            description: 'Optional domain allowlist. When set, only these domains/subdomains may appear in results or be fetched.',
          },
          blocked_domains: {
            type: 'array',
            items: { type: 'string' },
            description: 'Optional domain blocklist. Matching domains/subdomains are removed from results and cannot be fetched.',
          },
        },
        required: ['mode'],
      },
    },
  },
];

export const TOOL_NAMES = TOOLS.map((tool) => tool.function.name) as ['read', 'search', 'edit', 'bash', 'web'];
