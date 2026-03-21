---
skill_id: "enterprise-kb-routing"
title: "Enterprise KB Routing"
tags: ["mempedia", "enterprise-kb", "routing"]
updated_at: 1773976542
---

---
name: enterprise-kb-routing
description: "Use on every run inside Mempedia so the agent behaves like an enterprise knowledge-base operator: search evidence first, update the correct memory layer, and treat skills as reusable operating procedures."
category: mempedia
priority: high
always_include: true
tags: [mempedia, enterprise-kb, routing]
---

# Enterprise KB Routing

## Goal

Keep the agent operating as part of an enterprise knowledge system rather than as an isolated assistant.

## Core Behavior

1. Search before asserting when repository or memory evidence may exist.
2. Prefer the five top-level tools only: `read`, `search`, `edit`, `bash`, `web`.
3. Treat `read/search/edit` as workspace-only tools. All Mempedia Layer 1/2/3/4 operations must go through `bash` by calling the `mempedia` CLI.
4. Prefer the narrowest valid memory layer instead of over-promoting information.
5. Let the independent post-turn memory agent classify the full turn across all four layers.

## CLI Pattern

- Run from the project root; `bash` already starts there.
- Resolve the binary before issuing actions:

```bash
BIN="${MEMPEDIA_BINARY_PATH:-./target/debug/mempedia}"
[[ -x "$BIN" ]] || BIN=./target/release/mempedia
```

- For simple actions with no text content (search, list, read), use a single-line heredoc:

```bash
cat <<'JSON' | "$BIN" --project "$PWD" --stdin
{"action":"list_skills"}
JSON
```

- **CRITICAL — Writing rich content (Chinese, quotes, newlines, web-sourced text):**
  Heredoc JSON is INVALID when a string value contains literal newlines or unescaped double-quotes.
  Any content that includes multi-line markdown, non-ASCII text, or external webpage content **MUST be written via Python** so that `json.dumps()` handles all escaping:

```bash
BIN="${MEMPEDIA_BINARY_PATH:-./target/debug/mempedia}"
[[ -x "$BIN" ]] || BIN=./target/release/mempedia
python3 - <<'PYEOF'
import json, subprocess, os
BIN = os.environ.get('MEMPEDIA_BINARY_PATH', './target/debug/mempedia')
if not os.path.isfile(BIN):
    BIN = './target/release/mempedia'

markdown = """\
---
node_id: "example_node"
title: "Example Title"
---

# Example Title

Full markdown body here — can contain Chinese, quotes, newlines, anything.
"""

payload = json.dumps({
    "action": "agent_upsert_markdown",
    "node_id": "example_node",
    "markdown": markdown,
    "importance": 1.9,
    "agent_id": "agent-main",
    "reason": "User-requested knowledge ingestion",
    "source": "web",
})
result = subprocess.run(
    [BIN, '--project', os.getcwd(), '--stdin'],
    input=payload, capture_output=True, text=True
)
print(result.stdout or result.stderr)
PYEOF
```

  - Never manually escape Chinese characters or newlines in heredoc JSON.
  - Never embed a multi-line string value inside `<<'JSON'...JSON` — JSON does not allow literal newlines in string values.
  - The `agent_upsert_markdown` action accepts a full markdown string (YAML frontmatter + body) and correctly derives `summary`, `title`, and structured fields from it.

## When To Escalate To Memory

- Stable project facts: Layer 1 core knowledge.
- Short-lived chronology: Layer 2 episodic memory.
- Persistent user constraints: Layer 3 preferences.
- Reusable procedures: Layer 4 skills.

## Avoid

- Treating skills as answer content.
- Treating a skill name as a tool name.
- Skipping repository evidence when the question is project-specific.