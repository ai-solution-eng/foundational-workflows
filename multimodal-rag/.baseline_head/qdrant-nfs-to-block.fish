#!/usr/bin/env fish
# ─────────────────────────────────────────────────────────────────────────────
# qdrant NFS → vastdata-block migration runbook (fish)
# Written 2026-09-11. Paste sections one at a time, in order. Stop on any error.
#
# ⛔ STATUS: BLOCKED — vastdata-block has never provisioned a volume.
#    CreateVolume against the VAST mgmt API (172.16.254.254/bucket-block?volume)
#    returns HTTP 503 {"detail":"400 Client Error...","code":"service_unavailable"}.
#    Evidence: PVC garage/meta-garage-0 has retried ~22,600+ times since 2026-08-27.
#    Section 1 re-checks this; only continue once the smoke test passes.
# ─────────────────────────────────────────────────────────────────────────────

# ── 0. Wrapper — run once per shell ──────────────────────────────────────────
alias kubectl='env https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml'

# one-off (no alias):   env https_proxy=socks5://127.0.0.1:1080 kubectl --kubeconfig ~/SSH/G2/g2_kubeconfig.yaml <args>
# helm (needs both):    env https_proxy=socks5://127.0.0.1:1080 KUBECONFIG=~/SSH/G2/g2_kubeconfig.yaml helm ...

# ── 1. Precondition: block driver must work ─────────────────────────────────
# Watch for fresh ProvisioningFailed / HTTP 503 lines:
kubectl logs -n gl4f-block-csi deploy/block-vast-controller -c csi-provisioner --tail=50 | grep -E 'CreateVolume|ProvisioningFailed'
kubectl get pvc -n garage
# If the 503s continue → STOP. Hand to the platform team:
#   "SC vastdata-block (block.csi.vastdata.com) cannot provision: every CreateVolume fails
#    HTTP 503 {400 Client Error for http://172.16.254.254/bucket-block?volume, service_unavailable}.
#    PVC garage/meta-garage-0 has been failing since 2026-08-27 (driver install)."

# Smoke test (only after platform team confirms a fix):
printf 'apiVersion: v1\nkind: PersistentVolumeClaim\nmetadata:\n  name: block-smoke-test\nspec:\n  accessModes: ["ReadWriteOnce"]\n  storageClassName: vastdata-block\n  resources:\n    requests:\n      storage: 1Gi\n' | kubectl -n mm-rag apply -f -
kubectl -n mm-rag run block-smoke --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"block-smoke","image":"alpine","command":["sh","-c","echo hi > /data/t && cat /data/t && echo SMOKE_OK"],"volumeMounts":[{"name":"d","mountPath":"/data"}]}],"volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"block-smoke-test"}}]}}'
kubectl -n mm-rag delete pvc block-smoke-test

# ── 2. Quiesce qdrant (brief total outage — RF=1, ~201MB of data) ───────────
kubectl -n mm-rag scale sts rag-mcp-server-qdrant --replicas=0
kubectl -n mm-rag wait --for=delete pod -l app=rag-mcp-server-qdrant --timeout=180s

# ── 3. Stage all three replicas onto the app data PVC (rollback insurance) ──
kubectl -n mm-rag run qdrant-stage-0 --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"stage","image":"alpine","command":["sh","-c","cp -a /old/. /stage/qdrant-0/ && du -sh /stage/qdrant-0 && echo STAGED_0"],"volumeMounts":[{"name":"o","mountPath":"/old"},{"name":"s","mountPath":"/stage"}]}],"volumes":[{"name":"o","persistentVolumeClaim":{"claimName":"data-rag-mcp-server-qdrant-0"}},{"name":"s","persistentVolumeClaim":{"claimName":"rag-mcp-server-data"}}]}}'
kubectl -n mm-rag run qdrant-stage-1 --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"stage","image":"alpine","command":["sh","-c","cp -a /old/. /stage/qdrant-1/ && du -sh /stage/qdrant-1 && echo STAGED_1"],"volumeMounts":[{"name":"o","mountPath":"/old"},{"name":"s","mountPath":"/stage"}]}],"volumes":[{"name":"o","persistentVolumeClaim":{"claimName":"data-rag-mcp-server-qdrant-1"}},{"name":"s","persistentVolumeClaim":{"claimName":"rag-mcp-server-data"}}]}}'
kubectl -n mm-rag run qdrant-stage-2 --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"stage","image":"alpine","command":["sh","-c","cp -a /old/. /stage/qdrant-2/ && du -sh /stage/qdrant-2 && echo STAGED_2"],"volumeMounts":[{"name":"o","mountPath":"/old"},{"name":"s","mountPath":"/stage"}]}],"volumes":[{"name":"o","persistentVolumeClaim":{"claimName":"data-rag-mcp-server-qdrant-2"}},{"name":"s","persistentVolumeClaim":{"claimName":"rag-mcp-server-data"}}]}}'

# ── 4. Cut over: drop old claims, recreate STS, pre-create block claims ─────
kubectl -n mm-rag delete pvc data-rag-mcp-server-qdrant-0 data-rag-mcp-server-qdrant-1 data-rag-mcp-server-qdrant-2
# helm cannot patch volumeClaimTemplates (immutable) — recreate the STS object only:
kubectl -n mm-rag delete sts rag-mcp-server-qdrant --cascade=orphan
# Pre-create the claims on block storage with the SAME names → the new STS adopts them:
for i in 0 1 2
  printf 'apiVersion: v1\nkind: PersistentVolumeClaim\nmetadata:\n  name: data-rag-mcp-server-qdrant-%d\nspec:\n  accessModes: ["ReadWriteOnce"]\n  storageClassName: vastdata-block\n  resources:\n    requests:\n      storage: 25Gi\n' $i | kubectl -n mm-rag apply -f -
end
# Upgrade the release with the qdrant SC overridden (also edit persistence.qdrant.storageClass
# in helm-scale-large/local/values.g2.yaml so the change sticks):
env https_proxy=socks5://127.0.0.1:1080 KUBECONFIG=~/SSH/G2/g2_kubeconfig.yaml helm -n mm-rag upgrade rag-mcp-scale-large ~/Code/HPE/MultimodalRAG/helm-scale-large -f ~/Code/HPE/MultimodalRAG/helm-scale-large/local/values.g2.yaml --set persistence.qdrant.storageClass=vastdata-block

# ── 5. Restore staged data into the fresh (empty) block claims ──────────────
kubectl -n mm-rag run qdrant-restore-0 --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"restore","image":"alpine","command":["sh","-c","cp -a /stage/qdrant-0/. /new/ && du -sh /new && echo RESTORED_0"],"volumeMounts":[{"name":"s","mountPath":"/stage"},{"name":"n","mountPath":"/new"}]}],"volumes":[{"name":"s","persistentVolumeClaim":{"claimName":"rag-mcp-server-data"}},{"name":"n","persistentVolumeClaim":{"claimName":"data-rag-mcp-server-qdrant-0"}}]}}'
kubectl -n mm-rag run qdrant-restore-1 --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"restore","image":"alpine","command":["sh","-c","cp -a /stage/qdrant-1/. /new/ && du -sh /new && echo RESTORED_1"],"volumeMounts":[{"name":"s","mountPath":"/stage"},{"name":"n","mountPath":"/new"}]}],"volumes":[{"name":"s","persistentVolumeClaim":{"claimName":"rag-mcp-server-data"}},{"name":"n","persistentVolumeClaim":{"claimName":"data-rag-mcp-server-qdrant-1"}}]}}'
kubectl -n mm-rag run qdrant-restore-2 --rm -it --restart=Never --image=alpine --overrides='{"spec":{"containers":[{"name":"restore","image":"alpine","command":["sh","-c","cp -a /stage/qdrant-2/. /new/ && du -sh /new && echo RESTORED_2"],"volumeMounts":[{"name":"s","mountPath":"/stage"},{"name":"n","mountPath":"/new"}]}],"volumes":[{"name":"s","persistentVolumeClaim":{"claimName":"rag-mcp-server-data"}},{"name":"n","persistentVolumeClaim":{"claimName":"data-rag-mcp-server-qdrant-2"}}]}}'

# ── 6. Bring up and verify ───────────────────────────────────────────────────
kubectl -n mm-rag scale sts rag-mcp-server-qdrant --replicas=3
kubectl -n mm-rag get pods -l app=rag-mcp-server-qdrant -w   # Ctrl-C when 3/3 Running
# Expect NO 'NFS may cause data corruption', NO 'multi-mmap' warning, no ERRORs:
kubectl -n mm-rag logs rag-mcp-server-qdrant-0 --tail=300 | grep -iE 'nfs|multi-mmap|error'; or echo CLEAN
# Functional check: run a query against francesco-memory through the MCP server.

# ── 7. Cleanup (after a few days of confidence) ──────────────────────────────
# The staged copy lives at /stage/qdrant-{0,1,2} on the rag-mcp-server-data PVC —
# remove via a helper pod (same mounts as section 3, command: rm -rf /stage/qdrant-0 /stage/qdrant-1 /stage/qdrant-2).

# Notes:
# - qdrant sizes in live values (values.g2.yaml): qdrant 25Gi RWO, app data 40Gi RWX. Leave the
#   app data PVC on NFS — it holds uploads/.hashes.json/.bm25_stats.json, which are not
#   locking-sensitive. Do NOT clear .hashes.json (re-ingest would skip everything).
# - Once on block storage, optionally raise qdrant replication_factor 1→2 for HA.
