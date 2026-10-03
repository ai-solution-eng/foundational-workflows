import importlib.util, sys, os, time
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("t3", "tests/full_pipeline/test_s3_sync.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

client = m._client()
calls = {"n": 0}
orig = client.scroll
def counted(*a, **k):
    calls["n"] += 1
    if calls["n"] % 200 == 0:
        print("scroll call", calls["n"], "offset=", k.get("offset"), flush=True)
    if calls["n"] > 2000:
        raise RuntimeError("INFINITE SCROLL LOOP DETECTED")
    return orig(*a, **k)
client.scroll = counted

print("points in collection:", client.count(m.COLL).count, flush=True)
dm, rec = m._dm_with(client)
m._upsert(client, m.EXPANDED[0]); m._upsert(client, m.EXPANDED[1])
m._upsert(client, m.GONE); m._upsert(client, m.OUTSIDE)
t = time.time()
try:
    res = dm._prune_sources("ds", [m.PFX], expected={m.EXPANDED[0], m.EXPANDED[1], m.EXPANDED[2]})
    print("prune OK in %.2fs" % (time.time()-t), res, "scroll_calls=", calls["n"], flush=True)
except Exception as e:
    print("EXC after %.2fs:" % (time.time()-t), type(e).__name__, e, "scroll_calls=", calls["n"], flush=True)
