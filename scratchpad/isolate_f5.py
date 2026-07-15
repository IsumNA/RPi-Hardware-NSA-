import json
d = json.load(open("/tmp/hcg_candidate_manifest.json"))
items = d["imx662_5000k_1l_"]
# F_5 gt.png written 11:34:52 BST = 10:34:52 UTC. Look at the window just before,
# say 10:00-10:35 UTC, for a clean contiguous gain-sweep sub-block.
window = [it for it in items if "2026-07-10T10:0" in it["captured_at"]
          or "2026-07-10T10:1" in it["captured_at"]
          or "2026-07-10T10:2" in it["captured_at"]
          or "2026-07-10T10:3" in it["captured_at"]]
window.sort(key=lambda x: x["captured_at"])
print(f"{len(window)} frames in window 10:00-10:39 UTC")
groups = {}
for it in window:
    g = round(it["gain"], 2) if it["gain"] is not None else None
    groups.setdefault(g, []).append(it)
for g in sorted(groups, key=lambda x: (x is None, x)):
    grp = groups[g]
    print(f"  gain={g}: n={len(grp)}  {grp[0]['captured_at']} -> {grp[-1]['captured_at']}  "
          f"first={grp[0]['filename']} last={grp[-1]['filename']}")
