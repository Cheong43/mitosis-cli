import json, sys
from collections import Counter

for fname in sys.argv[1:]:
    with open(fname) as f:
        data = json.load(f)
    print(f"=== {fname.split('/')[-1]} ===")
    print(f"Input: {data.get('input','')[:200]}")
    print(f"Answer: {data.get('answer','')[:200]}")
    traces = data.get('traces', [])
    actions = [t for t in traces if t['type'] == 'action']
    errors = [t for t in traces if t['type'] == 'error']
    action_keys = []
    for a in actions:
        m = a.get('metadata', {})
        key = f"{m.get('branchId')}:{m.get('step')}:{m.get('toolName')}:{json.dumps(m.get('args',''), sort_keys=True)}"
        action_keys.append(key)
    dupes = {k: v for k, v in Counter(action_keys).items() if v > 1}
    print(f"Total traces: {len(traces)}, actions: {len(actions)}, errors: {len(errors)}, dupes: {len(dupes)}")
    for k, v in dupes.items():
        print(f"  {v}x: {k[:140]}")
    for e in errors:
        print(f"ERROR [{e.get('metadata',{}).get('branchId')} step {e.get('metadata',{}).get('step')}]: {e['content'][:300]}")
    print()
