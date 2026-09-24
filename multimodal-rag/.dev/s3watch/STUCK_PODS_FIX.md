# Undeletable watched-sources pods — final customer-safe cleanup (rev 6)

All commands copy-paste-ready with the G2 proxy + kubeconfig. NO policy is
ever disabled; the shared Kyverno webhook is never touched.

## Why the previous attempt failed (2026-09-24)

The finalizer+label patch ran while our live `add-vendor-app-labels-*`
policy still matched Pod kind: Kyverno re-stamped `hpe-ezua/type` mid-patch
(rule order add-vendor < assign-custom), so the platform policy matched again
and rewrote schedulerName → same Forbidden diff. The fix: make our policy
stop matching pods FIRST, at the cluster (the chart fix alone cannot do this
— the policy is a pre-install-only helm hook and `helm upgrade` never re-runs
it; a future `helm install` from the fixed chart renders the corrected
policy).

## The sequence

```bash
# 1. Patch the LIVE policy: kinds Deployment+Service only (pods no longer
#    match ANY mutating policy after this; Kyverno hot-reloads policies).
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml \
  patch clusterpolicy add-vendor-app-labels-rag-mcp-scale-large-rag-mcp-scale-large \
  --type=json \
  -p='[{"op":"replace","path":"/spec/rules/0/match/any/0/resources/kinds","value":["Deployment","Service"]}]'

# 2. Stuck Succeeded pods: strip finalizer AND the hpe-ezua/type label in ONE
#    patch. The incoming object now matches no mutating policy (our policy:
#    kind-restricted; platform policy: label gone) → no schedulerName rewrite
#    → pod-update validation passes → pending deletion completes.
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml -n mm-rag \
  patch pod rag-mcp-server-watched-sources-29835495-gnjd5 --type=json \
  -p='[{"op":"remove","path":"/metadata/finalizers"},{"op":"remove","path":"/metadata/labels/hpe-ezua~1type"}]'

https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml -n mm-rag \
  patch pod rag-mcp-server-watched-sources-29836185-wgqzz --type=json \
  -p='[{"op":"remove","path":"/metadata/finalizers"},{"op":"remove","path":"/metadata/labels/hpe-ezua~1type"}]'

# 3. Deploy the fixed chart at leisure (helm upgrade with this repo). Future
#    fresh installs render the corrected policy; existing pods (cron every
#    15 min) are no longer labelled and self-clean normally.

# 4. Verify
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml \
  -n mm-rag get pods -l app=rag-mcp-server
```

Already resolved by TTL (2026-09-24): error pods 29837625-2tjmt/-kr698 and
job rag-mcp-server-watched-sources-29837625 (404 on delete = success).

## Code changes shipped in this session

- **MultimodalRAG** (all three chart variants): `templates/kyverno.yaml`
  match kinds → Deployment+Service only;
  `tests/full_pipeline/test_kyverno_vendor_labels_render.py` (12 assertions,
  green); helm lint clean ×3; CHANGELOG entry under the 4.3.1 cut.
- **pcai-fleet-lib** (`pcai-helm-lib/chart/pcai-fleet-lib/templates/_kyverno.tpl`):
  same kind restriction + incident docstring. Covers the ~100 sibling
  `add-vendor-app-labels-*` policies fleet-wide on future installs.

## Blocked (delivery-twin RO mount — same item as the pending hardlinker sync)

- `/home/andrew/Code/HPE/SQLhandler/...` is a kernel READ-ONLY filesystem:
  the vendored `helm/charts/pcai-fleet-lib/templates/_kyverno.tpl` (hardlink
  pair with pcai-solutions, inode 27052761) and SQLhandler's chart-native
  divergent variant (`helm/templates/kyverno.yaml`, still matches Pod) could
  not be edited. When the tree is writable again: `cp` the fixed lib file
  over the vendored pair and drop `Pod` from the native variant's kinds.

## Diagnostic kubectl replication

```bash
# wedge: pod Succeeded long ago but Job not Complete + Forbid events
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml -n mm-rag get jobs,cronjobs
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml -n mm-rag \
  get events --field-selector reason=JobAlreadyActive
# mechanism: unstamped scheduler + finalizer + deletionTimestamp; any update
# shows the Forbidden schedulerName diff
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml -n mm-rag \
  get pod <POD> -o jsonpath='{.spec.schedulerName}{" fin="}{.metadata.finalizers}{" del="}{.metadata.deletionTimestamp}{"\n"}'
# policies feeding it: match selectors + patchStrategicMerge; webhook ops
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml \
  get clusterpolicies -o yaml | grep -B10 -A16 schedulerName
https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml \
  get mutatingwebhookconfigurations \
  -o jsonpath='{range .items[*]}{.metadata.name}{"  ops="}{.webhooks[*].rules[*].operations}{"\n"}{end}'
```
