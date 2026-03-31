---
name: duckduckgo-search
description: "Use DuckDuckGo for web search when external information is needed. Privacy-focused search without API key requirements."
always_include: true
metadata:
  category: research
  priority: high
  tags: [duckduckgo, search, external, privacy]
tools: [bash, web]
---

# DuckDuckGo Search

## Goal

Use DuckDuckGo as a privacy-focused search provider for external web research.

## Current Constraints

- The built-in `web` tool only supports `mode=fetch` on known URLs.
- DuckDuckGo HTML search can be used without API keys via bash/curl.
- Results should be parsed and converted to candidate URLs for fetching.

## Preferred Behavior

1. When external search is needed, use DuckDuckGo HTML search via curl.
2. Parse search results to extract titles, snippets, and URLs.
3. Convert results into a citation-first shortlist of candidate URLs.
4. Use `web` tool with `mode=fetch` on the most relevant URLs.

## Tool Guidance

Use bash with curl to query DuckDuckGo:
```bash
curl -s "https://html.duckduckgo.com/html/?q=YOUR_QUERY" | grep -oP '(?<=href=")[^"]*(?=")' | grep -v "duckduckgo.com" | head -10
```

Or use a more detailed parser to extract titles and snippets.

## Avoid

- Do not use `web` as a search engine.
- Do not fabricate search results.
- Do not turn low-quality snippets into final claims without verification.
