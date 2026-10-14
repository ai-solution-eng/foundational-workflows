import difflib, hashlib, sys, yaml

root = "/home/andrew/Code/HPE/MultimodalRAG/stmem"
src = open(root + "/stmem.py").read()
docs = list(yaml.safe_load_all(open(root + "/k8s/cronjob-stmem-reaper.yaml")))
kinds = [d.get("kind") for d in docs if d]
cm = [d for d in docs if d and d.get("kind") == "ConfigMap"][0]
cj = [d for d in docs if d and d.get("kind") == "CronJob"][0]
embedded = cm["data"]["stmem.py"]

# YAML block scalars strip the trailing blank line, so compare content-stripped.
same = src.rstrip("\n") == embedded.rstrip("\n")
print("docs parsed:", kinds)
print("embedded script matches source:", same)
if not same:
    d = list(difflib.unified_diff(src.splitlines(), embedded.splitlines(), lineterm="", n=1))
    print("DIFF (first 30):")
    print("\n".join(d[:30]))

spec = cj["spec"]
container = spec["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
init = spec["jobTemplate"]["spec"]["template"]["spec"]["initContainers"][0]
print("schedule:", spec["schedule"], "| concurrency:", spec["concurrencyPolicy"])
print("activeDeadlineSeconds:", spec["jobTemplate"]["spec"].get("activeDeadlineSeconds"))
print("init command:", init["command"])
print("reaper command:", container["command"])
print("mountPath:", container["volumeMounts"][0]["mountPath"])
print("secretRef:", container["envFrom"][0]["secretRef"]["name"])
print("runAsNonRoot:", spec["jobTemplate"]["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"])
print("readOnlyRootFilesystem:", container["securityContext"]["readOnlyRootFilesystem"])
print("VERDICT:", "PASS" if same else "FAIL")
sys.exit(0 if same else 1)
