"""Run ON THE PI. Produce an exact (scene, ag_tag) -> [filenames] manifest for
the 3 confirmed HCG capture sessions, from project.json's real per-frame gain
metadata. Writes /tmp/hcg_sort_manifest.json for the sync step to consume."""
import json

PROJECT = "/home/pi/ctt-server-workspace/imx662/project.json"

# actual reported gain -> requested ag tag (measured on this rig; ag1 is not
# reachable in HCG mode, confirmed absent from every clean sweep).
GAIN_TO_TAG = {
    3.24: "ag2", 3.98: "ag4", 7.94: "ag8", 15.85: "ag16", 31.62: "ag32",
    63.1: "ag64", 125.89: "ag128", 251.19: "ag256", 501.19: "ag512",
}

d = json.load(open(PROJECT))
caps = d["captures"]


def frames_with_prefix(prefix):
    out = []
    for c in caps:
        fn = c.get("filename", "")
        if fn.startswith(prefix):
            ctrl = c.get("controls") or {}
            out.append({"filename": fn, "gain": ctrl.get("gain"),
                       "captured_at": c.get("captured_at")})
    out.sort(key=lambda x: x["captured_at"] or "")
    return out


def assign(items, gain_round=2):
    by_tag = {}
    for it in items:
        g = it["gain"]
        if g is None:
            continue
        tag = GAIN_TO_TAG.get(round(g, gain_round))
        if tag is None:
            continue
        by_tag.setdefault(tag, []).append(it["filename"])
    return by_tag


manifest = {}

# cabinet_H_2: whole '5l' group (clean single sweep)
manifest["cabinet_H_2"] = assign(frames_with_prefix("imx662_5000k_5l_"))

# cabinet_D_10: whole '398l' group (clean single sweep)
manifest["cabinet_D_10"] = assign(frames_with_prefix("imx662_5000k_398l_"))

# cabinet_F_5: the isolated sub-block inside the entangled '1l' bucket,
# 2026-07-10T10:14:55Z .. 10:31:00Z (confirmed against its gt.png mtime).
f5_items = [it for it in frames_with_prefix("imx662_5000k_1l_")
           if "2026-07-10T10:1" in (it["captured_at"] or "")
           or "2026-07-10T10:2" in (it["captured_at"] or "")
           or "2026-07-10T10:3" in (it["captured_at"] or "")]
manifest["cabinet_F_5"] = assign(f5_items)

json.dump(manifest, open("/tmp/hcg_sort_manifest.json", "w"), indent=2)
for scene, tags in manifest.items():
    print(scene, {t: len(v) for t, v in tags.items()})

all_files = sorted({f for tags in manifest.values() for v in tags.values() for f in v})
with open("/tmp/hcg_files_to_sync.txt", "w") as f:
    f.write("\n".join(all_files) + "\n")
print(f"total unique files to sync: {len(all_files)}")
