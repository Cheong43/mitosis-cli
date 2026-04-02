---
name: google-search
description: "Use when real Google search results are needed. Provides Google Search, Google News, Google Images, Google Scholar, Google Shopping, Google Flights, Google Hotels, Google Maps, Google Trends, Wikipedia, live feeds (RSS/Reddit/HN/YouTube/GitHub/arXiv), and more — all via local headless Chromium, no API key required."
always_include: false
metadata:
  category: research
  priority: high
  tags: [google, search, web, news, images, scholar, shopping, flights, maps, feeds, youtube, wikipedia]
tools: [bash]
---

# Google Search (noapi-google-search-mcp)

## Goal

Use the `noapi-google-search-mcp` MCP server to perform real Google searches and access 38 web research tools locally via headless Chromium. No API key is required.

## Server Location

Installed at: `/opt/homebrew/Caskroom/miniconda/base/bin/noapi-google-search-mcp`

The server speaks the MCP stdio protocol. Invoke tools by sending JSON-RPC requests via bash.

## Preferred Invocation Pattern

Use a Python one-liner to call the MCP server tools via its stdio interface:

```bash
python3 -c "
import subprocess, json, sys

def call_mcp(tool_name, arguments):
    proc = subprocess.Popen(
        ['noapi-google-search-mcp'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Initialize
    init_req = json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'mitosis','version':'1.0'}}})
    proc.stdin.write((init_req + '\n').encode())
    proc.stdin.flush()
    proc.stdout.readline()  # read init response
    # Send initialized notification
    notif = json.dumps({'jsonrpc':'2.0','method':'notifications/initialized','params':{}})
    proc.stdin.write((notif + '\n').encode())
    proc.stdin.flush()
    # Call tool
    req = json.dumps({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':tool_name,'arguments':arguments}})
    proc.stdin.write((req + '\n').encode())
    proc.stdin.flush()
    result = proc.stdout.readline()
    proc.terminate()
    data = json.loads(result)
    content = data.get('result',{}).get('content',[])
    for item in content:
        print(item.get('text',''))

call_mcp('TOOL_NAME', ARGUMENTS_DICT)
"
```

Replace `TOOL_NAME` and `ARGUMENTS_DICT` with the appropriate values from the tool list below.

## Core Search Tools

### `google_search` — Real Google Web Search
```python
call_mcp('google_search', {
    'query': 'your search query',
    'num_results': 5,           # 1-10, default 5
    'time_range': 'past_week',  # optional: past_hour, past_day, past_week, past_month, past_year
    'site': 'github.com',       # optional: restrict to domain
    'language': 'en',           # optional
    'region': 'us',             # optional
    'page': 1                   # optional: 1-10
})
```

### `fetch_page` — Fetch and render a URL (JS-capable)
```python
call_mcp('fetch_page', {'url': 'https://example.com'})
```

### `google_news` — Google News search
```python
call_mcp('google_news', {'query': 'AI news', 'num_results': 5})
```

### `google_images` — Google Image search (returns image URLs)
```python
call_mcp('google_images', {'query': 'cats', 'num_results': 5})
```

### `google_scholar` — Academic paper search
```python
call_mcp('google_scholar', {'query': 'transformer architecture', 'num_results': 5})
```

### `wikipedia` — Wikipedia article lookup
```python
call_mcp('wikipedia', {'query': 'quantum computing'})
```

### `google_trends` — Search trend data
```python
call_mcp('google_trends', {'query': 'python programming'})
```

## Commerce & Travel Tools

### `google_shopping` — Product search
```python
call_mcp('google_shopping', {'query': 'Sony WH-1000XM5', 'num_results': 5})
```

### `google_flights` — Flight search
```python
call_mcp('google_flights', {'origin': 'New York', 'destination': 'London', 'date': 'March 15'})
```

### `google_hotels` — Hotel search
```python
call_mcp('google_hotels', {'location': 'Paris', 'check_in': 'March 15', 'check_out': 'March 18'})
```

### `google_maps` — Maps and location info
```python
call_mcp('google_maps', {'query': 'coffee shops near Times Square'})
```

### `google_finance` — Stock/financial data
```python
call_mcp('google_finance', {'query': 'AAPL'})
```

## Feed Subscription Tools

### `subscribe` — Subscribe to a content source
```python
call_mcp('subscribe', {'source_type': 'news', 'identifier': 'bbc'})
# source_type: news, reddit, hackernews, github, arxiv, youtube, podcast, twitter
# Pre-configured news: bbc, cnn, nyt, guardian, npr, techcrunch, ars, verge, wired, reuters
# arXiv shortcuts: ai, ml, cv, nlp, robotics
```

### `check_feeds` — Poll all subscriptions for new content
```python
call_mcp('check_feeds', {})
```

### `search_feeds` — Full-text search across feed content
```python
call_mcp('search_feeds', {'query': 'transformer architecture'})
```

### `list_subscriptions` — List active subscriptions
```python
call_mcp('list_subscriptions', {})
```

## Preferred Behavior

1. For general web research, use `google_search` first to get candidate URLs, then `fetch_page` on the most relevant ones.
2. For news, use `google_news` or subscribe to news feeds with `subscribe`.
3. For academic research, use `google_scholar`.
4. For product/travel queries, use the specialized tools (`google_shopping`, `google_flights`, `google_hotels`).
5. Always extract and cite source URLs from results.
6. If the server fails to start, check that `noapi-google-search-mcp` is on PATH and Chromium is installed (`playwright install chromium`).

## Avoid

- Do not fabricate search results if the server is unavailable.
- Do not use `web` tool as a search engine — use this skill instead.
- Do not call `google_search` for queries that can be answered from local context.
- Do not make more than 3 sequential search calls without synthesizing results for the user.
