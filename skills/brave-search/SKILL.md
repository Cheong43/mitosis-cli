---
name: brave-search
description: "Use when external web search is needed and a Brave Search channel is available. Prefer Brave-derived candidate URLs before calling web fetch."
always_include: true
metadata:
  category: research
  priority: high
  tags: [brave, search, external]
tools: [bash, web]
---

# Brave Search

## Goal

Use Brave Search as the preferred external search provider when the runtime has a real Brave Search channel available.

## Current Constraints

- The built-in `web` tool no longer performs search. It only supports `mode=fetch` on a known URL.
- Do not pretend Brave Search is available if no Brave API key, MCP server, or other Brave-backed integration exists in the current runtime.
- If no Brave channel exists, do not fabricate search results. Ask for a URL, use other local evidence, or state the limitation clearly.

## Preferred Behavior

1. When external search is necessary, prefer Brave Search over generic scraping or ad hoc search-engine HTML parsing.
2. Convert Brave results into a citation-first shortlist of candidate URLs.
3. After search, use `web` only for `mode=fetch` on the most relevant candidate URLs.
4. Keep source quality high with domain allow/block controls whenever possible.

## Tool Guidance

- Use `bash` for Brave Search only when the runtime has a real Brave-backed path, such as a configured API key or MCP connector.
- Use `web` only as `mode=fetch` after a trustworthy URL is already known.
- Prefer high-signal sources such as official docs, company sites, standards bodies, maintainer repos, and primary-source reporting.

## Avoid

- Do not use `web` as a search engine.
- Do not continue broad external research if there is no Brave channel and no candidate URL.
- Do not turn low-quality search snippets into final claims without fetching or otherwise verifying the underlying page.
