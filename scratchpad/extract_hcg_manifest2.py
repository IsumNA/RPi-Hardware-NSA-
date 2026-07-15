import json
d = json.load(open("/tmp/hcg_candidate_manifest.json"))
for tag, items in d.items():
    print(f"=== {tag} ({len(items)} frames) ===")
    groups = {}
    for it in items:
        g = round(it["gain"], 2) if it["gain"] is not None else None
        groups.setdefault(g, []).append(it)
    for g in sorted(groups, key=lambda x: (x is None, x)):
        grp = groups[g]
        print(f"  gain={g}: n={len(grp)}  {grp[0]['captured_at']} -> {grp[-1]['captured_at']}  "
              f"first={grp[0]['filename']} last={grp[-1]['filename']}")
