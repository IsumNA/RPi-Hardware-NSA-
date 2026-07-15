import json

path = "/home/pi/ctt-server-workspace/imx662/project.json"
with open(path) as f:
    head = f.read(2000)
print("HEAD:")
print(head)
print("...")

try:
    with open(path) as f:
        d = json.load(f)
    print("type:", type(d))
    if isinstance(d, dict):
        print("keys:", list(d.keys())[:30])
    elif isinstance(d, list):
        print("len:", len(d))
        print("first item:", d[0] if d else None)
except Exception as e:
    print("json.load failed:", e)
