"""Run ON THE PI. Extract per-frame metadata for the unsynced low-lux capture
groups (candidate HCG sessions) from project.json, to reconstruct gain-sweep
bursts without guesswork."""
import json

PROJECT = "/home/pi/ctt-server-workspace/imx662/project.json"
TAGS = ("imx662_5000k_1l_", "imx662_5000k_5l_",
        "imx662_5000k_373l_", "imx662_5000k_398l_")

d = json.load(open(PROJECT))
caps = d["captures"]
by_tag = {t: [] for t in TAGS}
for c in caps:
    fn = c.get("filename", "")
    for t in TAGS:
        if fn.startswith(t):
            ctrl = c.get("controls") or {}
            by_tag[t].append({
                "filename": fn,
                "gain": ctrl.get("gain"),
                "exposure": ctrl.get("exposure"),
                "lux": ctrl.get("lux"),
                "captured_at": c.get("captured_at"),
            })
            break

out = {}
for t, items in by_tag.items():
    items.sort(key=lambda x: x["captured_at"] or "")
    out[t] = items
    print(f"{t}: {len(items)} frames, "
          f"gains seen: {sorted(set(round(i['gain'],2) for i in items if i['gain'] is not None))}")

json.dump(out, open("/tmp/hcg_candidate_manifest.json", "w"))
print("wrote /tmp/hcg_candidate_manifest.json")
