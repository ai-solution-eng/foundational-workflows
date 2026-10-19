# Deployment Guide

> **What changed (2026-10-07 doc wave):**
> - The SSO migration is now documented values-first and ORDERED — a new §5b subsection gives the enablement chain (keys/registry → D21 `oidc.enabled` → D22 browser SSO → D23 `ssoGate` → D25 `accessSelfMint` last) with the G2-vs-Hosted edge posture, and two previously undocumented values rows were added: `security.trustProxyIdentity` (D24) and `security.oidc.sso.*` (D22).
> - §5b was re-verified key-by-key against `helm/values.yaml`, the chart templates and the shipped examples — including `values-hardened-g2.yaml`, the base chart's MCP network-zone overlay (now referenced here).
> - The `${DOMAIN_NAME}` gotcha is sharpened: raw-CLI deploys pass the placeholder through and the istio validation webhook denies the VirtualService (the values editor envsubs it; CLI must resolve it first).
> - Chart-truth fixes: packaged image default is `v5.7.1` (was v4.4.1); the MCP tool count is 23 (the `memory_status` capability probe plus the two short-term-memory tools); `mcp.apiKey.existingSecret` renders in all three charts (the v4.0.0 scale wiring); the large chart adds `security.oidc.platformCa.*`.

> **PCAI is a Kubernetes wrapper — you never run `helm` or `kubectl`.** You import the packaged chart (`rag-mcp-server` `.tar.gz`) into PCAI once, then drive the deployment by setting the chart's
> **`values.yaml`** in the PCAI *Helm Values* editor (or via the PCAI API). Every `--set` in this guide maps 1:1 to a key in `values.yaml`. There is no `envsubst` step — set the actual domain value
> directly in `values.yaml`.


<div align="center"><img src="./deployment_flow-1.png" width="900" alt="Deployment architecture: clients -> Istio gateway/oauth2-proxy -> API server + MCP sidecar + embed-batcher + Redis -> Qdrant cluster + PVC -> MLIS model endpoints, with security layers annotated"></div>

---

> **⚠ Required config: `MEDIA_TOKEN_SECRET`**
>
> Since v1.9.5, both the API and MCP containers **refuse to start** without `MEDIA_TOKEN_SECRET` set. Media URLs are served via short-lived HMAC tokens (the legacy `?password=` URLs were removed), and
> this shared secret is what signs/verifies them. Deploying without it crashes both pods.
>
> It is still **settable in one of four ways** (the chart renders the Secret `<deployment.name>-model-keys`, see §3's Secrets-mechanism note and `helm/ROTATION.md`): inline `security.mediaTokenSecret`
> (back-compat), your own Secret via `security.existingSecret`, reuse of a previous install's release Secret, or **fresh auto-generation on first install** — the last is the default, so the common
> case needs no pre-deployed Secret at all. To pin it explicitly, generate one and set it in `values.yaml` before deploying:
>
> ```bash python -c "import secrets; print(secrets.token_hex(32))" # -> values.yaml:  security.mediaTokenSecret: "<output>" ```

---

## Prerequisites

- A PCAI environment where you can import the packaged chart and edit its values.
- The image is public in `ghcr.io/ai-solution-eng/...` — you only need registry push access if you build a custom image.
- Model endpoints (embedder, reranker, VLM, ASR) deployed through MLIS, with their API tokens.

---

## 1. Import the chart in PCAI and set the image

The packaged charts ship with the chart default `image.tag` as the default image — no image build is required (the current packaged default is `ghcr.io/ai-solution-eng/multimodal-rag-mcp:v5.7.1`, per `image.repository`/`image.tag` in each chart's `values.yaml`). If you need a custom build, the Dockerfile lives at `docker/Dockerfile` and expects the repo root as the build context; push the result to your registry and override the values:

```yaml
# values.yaml
image:
  repository: ghcr.io/ai-solution-eng/multimodal-rag-mcp
  tag: v5.7.1
```

> **Note on models**: The image does not bundle any ML models. It connects to remote model endpoints configured via values; the defaults point to models hosted on the PCAI internal cluster.

---

## 2. Configure deployment values

All configuration lives in the chart's `values.yaml`. Key settings you edit in the PCAI *Helm Values* editor:

```yaml
# PVC sizes — adjust for your dataset scale
persistence:
  data:
    size: 200Gi        # Uploaded files + dataset metadata
    storageClass: gl4f-filesystem   # must be RWX-capable for the shared /data mount
    accessMode: ReadWriteMany       # RWO breaks multi-pod/multi-container /data sharing
  qdrant:
    size: 150Gi        # Qdrant vectors + index
    storageClass: gl4f-filesystem
    accessMode: ReadWriteMany       # volumeClaimTemplate access mode (base chart: RWMany)
    mountReadOnly: true             # API/MCP pods mount the Qdrant PVC read-only
                                    # (/api/admin/health disk reporting) — leave true

# Model configuration — each model has name, url, className, and
# optional extra kwargs. Set url to "" to disable a component.
models:
  embedder:
    name: "Qwen/Qwen3-VL-Embedding-8B"
    url: "https://..."              # required
    className: "MultiModalEmbeddings"
  reranker:
    name: "Qwen/Qwen3-VL-Reranker-8B"
    url: "https://..."              # optional — empty string disables
    className: "MultiModalReranker"
  vlm:
    name: "Qwen/Qwen3.8-27B-FP8"    # any OpenAI-compatible VLM
    url: "https://..."              # optional — empty string disables
  asr:
    name: "CohereLabs/cohere-transcribe-03-2026"
    url: "https://..."              # optional — empty string disables
    # Each role has its own dotted key: models.embedder.url (required),
    # models.reranker.url / models.vlm.url / models.asr.url (optional).
    # An empty url disables that component at startup (a warning, not a crash).

Only the **embedder** is required: an unreachable embedder aborts startup.
Reranker/VLM/ASR are optional — if unreachable, startup logs a warning and
continues without them (reranking/captioning/transcription degrade).

# Model API keys (stored in a Kubernetes Secret, never in the image)
modelSecrets:
  # ALTERNATIVE — bring your own Secret (key material never transits values):
  # point at a Secret you own carrying the canonical key names
  # (MODEL_EMBEDDER_API_KEY / MODEL_RERANKER_API_KEY / MODEL_VLM_API_KEY /
  # MODEL_ASR_API_KEY — optionally MODEL_*_URL endpoints, S3_ACCESS_KEY_ID /
  # S3_SECRET_ACCESS_KEY, REDIS_PASSWORD). Hard-wired (missing Secret =
  # CreateContainerConfigError). Every key it carries OVERRIDES the
  # chart-managed <name>-model-keys Secret (wired last in envFrom and
  # CONFIG_DIR); keys it omit keep the seeds below and the sticky stored
  # values. Rotation is kubectl/ESO-native and hot-reloads (<=15s).
  existingSecret: ""
  embedderApiKey: "eyJ..."
  rerankerApiKey: "eyJ..."
  vlmApiKey: "eyJ..."
  asrApiKey: "eyJ..."

# S3 / MinIO — used by batch-URL ingest, watched-sources sync and backups.
# The endpoint is non-sensitive (ConfigMap); keys go into the model-keys Secret.
# Leave both keys "" to use the default boto3 credential chain instead.
s3:
  endpointUrl: http://minio.minio.svc.cluster.local:9000
  accessKeyId: ""
  secretAccessKey: ""

# RAG pipeline defaults
rag:
  ocr: false             # OCR fallback for scanned PDFs (RAG_OCR_DEFAULT; a per-dataset flag still wins)
  captionWithAsr: true    # Transcribe video audio tracks via ASR during ingestion (auto-disables if no ASR model)
  captionWithVlm: true    # VLM-describe images/videos at ingest (auto-disables if no VLM; enables VLM-skip at retrieval)
  remote: false           # Use remote model URLs vs in-cluster .svc.cluster.local
  dedupThreshold: 0.995   # Cosine similarity above which an ingest candidate is
                          # dropped as a near-duplicate (0.0–1.0)

# MCP server (sidecar)
mcp:
  enabled: true
  port: 9090
  # OPTIONAL MCP API-key auth (fleet decision 2026-09): point at a Secret whose
  # key holds the comma-separated key list; every /mcp request then needs a
  # valid key (X-API-Key or Bearer). Leave empty for gateway-fronted
  # deployments — the server runs open with a loud startup warning. To switch
  # the gate on, pre-deploy the Secret first (the chart never creates or
  # inlines this key):
  #   kubectl -n <ns> create secret generic mcp-fleet-apikeys \
  #     --from-literal='api-keys=<key1>,<key2>'
  apiKey:
    existingSecret: ""      # e.g. "mcp-fleet-apikeys"
    existingSecretKey: api-keys

# Security — REQUIRED: a shared secret for short-lived media HMAC tokens.
# Both the API and MCP servers refuse to start without it (the legacy
# ?password= media URLs were removed). Also raises the SSRF guard defaults.
security:
  mediaTokenSecret: "<random-string-shared-by-both-containers>"
  mediaTokenTtl: 3600
  ingestAllowHosts: ""        # e.g. ".minio.svc.cluster.local" (bypasses private-block)
  blockPrivateHosts: true     # default on — blocks SSRF targets

# Metrics (optional): render a prometheus-operator ServiceMonitor for /metrics.
metrics:
  serviceMonitor: false
  interval: 30s
  # When security.metricsAuth is on, name a Secret whose <key> holds a valid
  # API key; the ServiceMonitor presents it as the scrape bearer token.
  serviceMonitorBearerSecret: ""
  serviceMonitorBearerSecretKey: RAG_API_KEY

# PCAI / EZUA (Istio-based ingress)
ezua:
  enabled: true
  virtualService:
    endpoint: "rag-mcp-server.<your-domain>"   # REQUIRED when ezua.enabled — the render fails without it
    istioGateway: "istio-system/ezaf-gateway"
    timeout: 300s          # default tier; longTimeout: 3600s covers batch uploads / SSE / MCP
  authorizationPolicy:
    namespace: "istio-system"
    providerName: "oauth2-proxy"
```

### Security knobs (`security.*`)

All of these flow into the ConfigMap as hardening env vars; defaults are shown
in `values.yaml` (both examples in `helm/values-examples/` set them too):

| Key | Default | Effect |
|---|---|---|
| `security.maxDownloadBytes` | `536870912` (512 MiB) | Per remote/S3 download cap — streams are aborted past this (SSRF blast-radius guard) |
| `security.archiveMaxTotalBytes` | `2147483648` (2 GiB) | Zip/tar/rar unpacked-size cap, nested archives included |
| `security.archiveMaxMemberBytes` | `1073741824` (1 GiB) | Per-member unpacked-size cap |
| `security.archiveMaxEntries` | `10000` | Per-archive entry count cap (zip-bomb guard) |
| `security.mediaAllowPathPrefixes` | `/data/datasets:/data/staging` | `file://` prefixes the MCP `describe_media`/`transcribe_audio`/audio-query tools may read (`:`-separated, fail-closed) |
| `security.pwMaxFailures` | `10` | Password-failure throttle: failures per identity before 429s (REST + MCP unlock) |
| `security.pwFailWindow` | `300` | Window (seconds) those failures are counted over |
| `security.trustProxyIdentity` | `false` | **D24** — trust the auth proxy's identity headers (`X-Auth-Request-*`) as a presented credential (one of the three D25 mint sources). Correct `true` behind an ENFORCING proxy that overwrites these headers on every request (the ezaf-gateway / oauth2-proxy — both site examples keep it); `false` (the raw-env default) for direct unmeshed callers, which could otherwise spoof the headers — with D25 on, a spoofed identity could mint keys. |
| `security.metricsAuth` | `false` | Require an API key on `/metrics` (`X-RAG-Api-Key`/`X-API-Key`/Bearer); pair with `metrics.serviceMonitorBearerSecret` |

### Multi-user access knobs (D15/D16/D17 — opt-in)

| Key | Default | Effect |
|---|---|---|
| `mcp.apiKeyClients` / `mcp.datasetAcls` | `""` / `""` | **D15** — per-user registry keys + dataset ACLs (`name:key;name:key` / `name:ds1,ds2;name2:*`). **Since D26** the ACL value also accepts the per-dataset mode token form (`name:ds1:ro,ds2:rw,ds3`; the legacy `name:ds1,ds2` form = rw — byte-compatible, no new env var): an ro dataset refuses every write surface (MCP write tools + REST ingest/delete + dataset-manage endpoints) while reads work. KEY MATERIAL — render only when set; provide from a Secret pipeline, never commit. See the `RAG_API_KEY_CLIENTS` row in the base README. |
| `security.accessStore` | `true` | **D16** — the `/access` checkbox model: registry keys select their own datasets (protected ones with the password, saved per identity under `/data/access`) and bind their ★ memory dataset. Effective access = operator ACL ∪ selections. |
| `security.accessDenySelect` / `security.memoryDefault` | `""` / `""` | **D16** ceiling/fallback: datasets that can never be self-selected; the deployment-wide fallback memory dataset. Render only when non-empty. |
| `rag.unlockMaxTtl` | `86400` | Bound for explicit unlock TTLs on both surfaces; `0` opts into no-expiry unlocks (the `/access` page's "No expiry (0)" option). |
| `security.oidc.*` | `enabled: false` | **D21** — OIDC JWT as a second credential for the same per-user registry identity. `enabled` (default `false`) / `issuer` (REQUIRED when enabled — the Keycloak realm URL, e.g. `https://keycloak.<your-domain>/realms/UA`) / `audience` (`ua`) / `identityClaim` (`preferred_username`) / `jwksUrl` (empty = `<issuer>/protocol/openid-connect/certs`) / `jwksRefreshSeconds` (`3600`) / `observedTtl` (`86400`) / `fetchTimeoutSeconds` (`3`) / `clockSkewSeconds` (`60`). Rendered ONLY when enabled (default render byte-identical); see the OIDC subsection in § 5b. |
| `security.oidc.sso.*` | `enabled: false` | **D22** — browser SSO, the app as an OIDC CLIENT (the Open WebUI pattern; no gateway config needed): `GET /oauth/login` → realm → `GET /oauth/oidc/callback` exchanges the code, verifies through the SAME D21 machinery, and plants an HttpOnly session cookie (lowest-priority envelope — an explicit key always wins); the /access page gains a "Sign in with SSO" button. Sub-keys (all with verified chart defaults): `clientId` (`ua`), `clientSecret` (KEY MATERIAL — prefer `clientExistingSecret` + `clientExistingSecretKey: clientSecret`), `redirectUri` (`https://<endpoint>/oauth/oidc/callback` — register it in the realm client FIRST via `scripts/configure-oidc-rag.sh`, idempotent), `providerUrl` (the IN-CLUSTER Keycloak URL, plain HTTP — sidesteps the platform-CA TLS gap; the external `issuer` stays claim-compared only), `scopes` (`openid profile email`), `cookieName` (`pcai-sso`), `cookieMaxAgeSeconds` (`43200`), `cookieSecure` (`true`). Requires `oidc.enabled` (+ issuer) — the `/oauth/*` routes 404 when unconfigured. Rendered ONLY when `sso.enabled: true` (default render byte-identical); enablement order in § 5b. |
| `security.ssoGate` | `false` | **D23** — the identity-gated unified frontend. `true` = the served HTML pages (`/`, `/manage`, `/access`) require an authenticated identity (SSO cookie / Bearer JWT / API key); anonymous visitors get the minimal 401 "Sign in required" page (a `/oauth/login` link when SSO is enabled). Probes/health, the `/oauth/*` flow and media bytes stay exempt. `false` (default) = byte-identical public pages. Meaningful only with `oidc.enabled` or API keys configured (an empty deployment gates to nothing). Read per request — flip without a restart. Rendered ONLY when `true`; see the Identity-gated frontend subsection in § 5b. |
| `security.accessSelfMint` | `false` | **D25** — SSO self-mint (chart default OFF since the 2026-10-16 audit remediation — opt in with `true`; opt-in posture, unlike `accessStore`). `true` = an SSO-authenticated identity (verified D21 Bearer JWT / D22 SSO cookie / D24 trusted-proxy identity header) may mint — and rotate — a long-lived API key for its OWN registry identity from the Access page, no admin round-trip. The minted key carries exactly the identity's existing access (never creates or widens grants); API-key-authenticated callers are always 403 (a registry key can never mint); the key is shown exactly once. Meaningful only with `oidc.enabled` (or `trustProxyIdentity` on G2) AND the access overlay on (`accessStore: true` — the minted entry needs somewhere to live); without those it is inert (no SSO session can carry the mint proof). Read per request — flip without a restart. `false` = opt-out — byte-identical pre-D25 behaviour (mint 409s, whoami `flags.self_mint` false). Rendered ONLY when `true`. See README § Security hardening (D25). |

With the access store on, an ADMIN key additionally gets the **User keys** panel on
`/access` (**D17**): mint per-user keys, set dataset grants, rotate, revoke — against a
file-backed overlay at `{DATA_PATH}/access/clients.json` (merged with the
`mcp.apiKeyClients` env; env authoritative on key conflicts). No restart needed.

**Read-only grants (D26, 2026-10).** Every grant now carries a mode — `rw` (default)
or `ro` — set in the Admin tab's User keys grants editor: a searchable checkbox
dropdown per key, one row per dataset with ro/rw toggles, 🌐 global and 🔒 password
badges, and a pinned "All datasets (*)" row (the mint form is unchanged). The overlay
stores each `datasets` element as a plain string (`"reports"` = rw — existing files
unchanged) or an object `{"name": "reports", "mode": "ro"}`. An ro dataset keeps its
read surfaces (list/read/search/unlock/select/★-bind) but refuses every write surface
(the MCP write tools, REST document ingest/delete, and the PATCH/DELETE/public-toggle
manage endpoints); 🌐 **globals are read-only by default** — a public dataset is
included for every authenticated identity at mode ro, and only its creator, an explicit
`rw` grant, or an admin can write it. Precedence across sources is
strongest-restriction-wins: env rw ∪ overlay ro ⇒ ro, an explicit per-dataset ro in any
source overrides a `*` rw, and a `*:ro` token downgrades the `*` grant. Plain-string
grants, the legacy ACL form and mode-less overlay files all mean rw — deployments that
never touch modes are byte-identical, and a write against an ro dataset is refused,
never degraded. Full semantics: README § Security hardening (D26).

### Standard Kubernetes knobs

These are plain Kubernetes fields surfaced for completeness — the chart
defaults are sensible and PCAI operators rarely touch them:

| Key | Default | Effect |
|---|---|---|
| `deployment.appName` | `rag-mcp-server` | `app` label on every rendered resource (selective delete/monitor target) |
| `image.pullPolicy` | `IfNotPresent` | Pull policy for the app image (both containers + CronJobs) |
| `persistence.data.accessMode` | `ReadWriteMany` | PVC access mode for the shared `/data` claim |
| `persistence.data.storageClass` | `gl4f-filesystem` | StorageClass for the data PVC (must be RWX-capable) |
| `persistence.qdrant.accessMode` | `ReadWriteMany` | Qdrant PVC access mode (from its StatefulSet `volumeClaimTemplates`) |
| `persistence.qdrant.storageClass` | `gl4f-filesystem` | StorageClass for the Qdrant PVC |
| `persistence.qdrant.mountReadOnly` | `true` | Mount the Qdrant PVC read-only on the API/MCP pods for disk-usage reporting (disable if the claim is not shared) |
| `resources.app.requests/limits.memory·cpu` | 2Gi/8Gi mem, 2/4 cpu | Per-container resources for `rag-api-server` and `rag-mcp-server` — the dotted paths are `resources.app.requests.memory` / `resources.app.requests.cpu` / `resources.app.limits.memory` / `resources.app.limits.cpu` |
| `resources.qdrant.requests/limits.memory·cpu` | 16Gi/32Gi mem, 4/8 cpu | Per-pod resources for the Qdrant StatefulSet — `resources.qdrant.requests.memory` / `resources.qdrant.requests.cpu` / `resources.qdrant.limits.memory` / `resources.qdrant.limits.cpu` |
| `qdrant.image.pullPolicy` | `IfNotPresent` | Pull policy for the Qdrant image |

### Qdrant runtime values

| Key | Default | Effect |
|---|---|---|
| `qdrant.image.tag` | `v1.19.0` | Qdrant server image tag (pinned; upgrade deliberately, collections must stay compatible) |
| `qdrant.httpPort` | `6333` | HTTP port the app talks to (`--qdrant-port` / `QDRANT_PORT`); also the healthz probe port |
| `qdrant.grpcPort` | `6334` | gRPC port exposed on the headless Qdrant service (used by gRPC-preferring clients on the scale charts) |
| `qdrant.telemetryDisabled` | `true` | Sets `QDRANT__TELEMETRY_DISABLED=true` — stops hourly failed phone-homes to `telemetry.qdrant.io` on egress-restricted clusters |

```yaml
# Resources — scale based on expected Qdrant load (per-container; the pod
# runs an API + MCP container, so app resources apply 2x per pod)
resources:
  app:
    requests:
      memory: 2Gi
      cpu: 2
    limits:
      memory: 8Gi
      cpu: 4
  qdrant:
    requests:
      memory: 16Gi
      cpu: '4'
    limits:
      memory: 48Gi   # ~40 GiB needed for 1M × 4096-dim vectors
      cpu: '8'
```

Two RAG keys are deliberately gated and render nothing while disabled — enable
them only with intent:

- `rag.rrfDefault` (`enabled`, `dense`, `sparse`, `k`) — stamps a per-dataset
  weighted-RRF default (`RAG_RRF_DEFAULT`, format `dense,sparse[,k]`) into every
  NEW dataset at create time. `enabled: false` (default) stamps nothing and new
  datasets inherit the global 1.0/1.0 rank order. When enabled, a `dense`/`sparse`
  weight above 1.0 makes that retrieval lane's rank positions count more in the
  fused ordering (rank-space tilt — trust the order, not the magnitude) and
  `rag.rrfDefault.k` is the RRF ranking constant (default 2). An explicit `rrf`
  object on `POST /api/datasets` always wins; existing datasets are untouched
  (PATCH them).
- `rag.contextual` — stamps `contextual: true` into NEW datasets (ingest-time
  contextual retrieval; one small LLM call per real-text chunk). Off by default;
  existing datasets are untouched.

The optional backup CronJob (`backups.enabled`) exports every
non-password-protected dataset on a schedule (`backups.schedule`), uploads the
`.tar.gz` archives to `backups.bucket` (reusing the `s3.*` credentials), and —
when `backups.retentionDays` > 0 — prunes backup objects older than that.
`backups.bucket` is a **bare bucket name** (no `s3://` scheme — it is passed
straight to boto3) and **must exist before the first run** (MinIO does not
auto-create buckets; a missing bucket fails the run until the next tick).
Use a dedicated bucket — pointing it at a watched-sources prefix makes every
backup re-ingest as documents. Create it once from any running pod (which
already carries the credentials):

```bash
kubectl -n <ns> exec deploy/<deployment.name> -- python -c "import boto3,os; boto3.client('s3', endpoint_url='http://minio.minio.svc.cluster.local:9000', aws_access_key_id=os.environ['S3_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['S3_SECRET_ACCESS_KEY']).create_bucket(Bucket='<backup-bucket>'); print('bucket created')"
```

> The `${DOMAIN_NAME}` placeholder is resolved by PCAI's deployment pipeline before helm runs — submitted values are envsubst-ed (verified: the deployed release stores the fully-resolved endpoint). Keep
> the placeholder in `ezua.virtualService.endpoint` as-is. **CLI deploys are the exception:** a raw `helm upgrade -f helm/values-examples/values.g2.yaml` passes the literal through, the istio
> validation webhook DENIES the VirtualService (invalid host), and the release marks `failed` while earlier-rendered objects have already applied — resolve the placeholder first, e.g.
> `sed "s/\${DOMAIN_NAME}/<site-domain>/g" values.g2.yaml > values.resolved.yaml` (see the fleet SSO-ROLLOUT-G2 runbook). Note that `ezua.domainName` itself is a platform-convention key no chart template reads; the VirtualService is built from
> `ezua.virtualService.endpoint`.

---

## 3. Install / update in PCAI

Import the packaged `rag-mcp-server` chart into PCAI, then set the values above (image, models + `modelSecrets`, `security.mediaTokenSecret`, persistence sizes, `ezua.*`) in the *Helm Values* editor and apply. Model URLs and API keys are required.

To change a setting later, edit the values in PCAI and apply again — that is the only "upgrade" path you need.

```yaml
# values.yaml — the keys PCAI renders from (also shown above)
image:
  repository: ghcr.io/ai-solution-eng/multimodal-rag-mcp
  tag: v5.7.1
models:
  embedder:
    name: "Qwen/Qwen3-VL-Embedding-8B"
    url: "https://..."
    className: "MultiModalEmbeddings"
  reranker:
    name: "Qwen/Qwen3-VL-Reranker-8B"
    url: "https://..."
    className: "MultiModalReranker"
  vlm:
    name: "Qwen/Qwen3.8-27B-FP8"
    url: "https://..."
  asr:
    name: "CohereLabs/cohere-transcribe-03-2026"
    url: "https://..."
modelSecrets:
  embedderApiKey: "eyJ..."
  rerankerApiKey: "eyJ..."
  vlmApiKey: "eyJ..."
  asrApiKey: "eyJ..."
persistence:
  data:
    size: 500Gi
  qdrant:
    size: 200Gi
security:
  mediaTokenSecret: "<generated>"
ezua:
  virtualService:
    endpoint: "rag-mcp-server.<your-domain>"
```

> **Tip**: keep the model URLs and API keys in a private values fragment (or the PCAI Secret) so they stay out of commit history; the chart reads them from values at apply time.
>
> The chart itself ships **no key material** (`security.apiKey` / `security.mediaTokenSecret` are empty by default — P0-7, 2026-09): keys are sourced from `security.existingSecret` (a Secret you own; key names `security.existingSecretApiKey` / `security.existingSecretMediaTokenKey`), inline values (back-compat), the release Secret from a previous install (reused on upgrade), or fresh auto-generation on first install — in that precedence order. See `helm/ROTATION.md` for the full runbook.

---

## 4. Verify the deployment

In the PCAI UI the deployment should reach **Ready**. To check locally (optional, developer convenience):

```bash
# Port-forward to test locally
kubectl port-forward deployment/rag-mcp-server 8000:8000

# Health check
curl http://localhost:8000/healthz
# → {"status": "ok"}

# List datasets (empty initially)
curl http://localhost:8000/api/datasets
# → {"datasets": []}
```

Pods and logs are visible from the PCAI workload view (`kubectl get pods -l app=rag-mcp-server` and `kubectl logs -l app=rag-mcp-server -c rag-api-server` work the same as ever for operators who have cluster access).

---

## 5. Access the web UI

If EZUA (Istio) is enabled, the service is available at the VirtualService endpoint (e.g. `https://rag-mcp-server.<your-domain>`). Authentication is handled by the `oauth2-proxy` AuthorizationPolicy.

For a local developer preview only, port-forward:

```bash
kubectl port-forward deployment/rag-mcp-server 8000:8000
# → http://localhost:8000
```

The UI lets you:
- **Create datasets** with optional video captioning toggle
- **Upload files** (PDF, images, videos, audio, text) via drag-and-drop
- **Add text** in the standardized multimodal format
- **Search** with configurable `top_k` and reranker toggle
- **Browse stored documents** in each dataset

---

## 5b. Deployment profiles: Internal G2 vs Hosted trial

Everything above is identical for both site classes — the profile only changes
five values (and one pre-created Secret). `helm/values-examples/` ships a
sanitized paste-ready values document for each profile:

| | **Internal G2** (`values-examples/values.g2.yaml`) | **Hosted trial** (`values-examples/values.hosted-trial.yaml`) |
|---|---|---|
| `ezua.virtualService.endpoint` | Literal site domain (e.g. `rag-mcp-server.pcai-se-ai-application.hst.rdlabs.hpecorp.net`) — or `rag-mcp-server.${DOMAIN_NAME}` | `rag-mcp-server.${DOMAIN_NAME}` — PCAI substitutes the placeholder before rendering; keep it as-is |
| `ezua.authorizationPolicy.enabled` | `false` — the ezaf-gateway applies HPE SSO centrally; the chart creates no AuthorizationPolicy | `true` (default) — customer SSO via oauth2-proxy at the gateway |
| Model endpoints | Real shared serving URLs (`.serving.` → kept, with `rag.remote: false` rewriting them to in-cluster `.svc.cluster.local`) | Customer MLIS endpoints; `rag.remote: true` only for endpoints outside the cluster |
| Egress | In-cluster only — model URLs resolve as `.svc.cluster.local`; no outbound proxy involved | If the customer cluster routes model traffic through the HPE/corporate proxy, set the proxy envs via `extraEnv` (e.g. `HTTPS_PROXY`); the SSRF DNS-rebinding pin is skipped when a proxy performs egress DNS |
| Kyverno ClusterPolicy | The G2 cluster runs Kyverno (EzAF) — the pre-install hook creates the vendor-label policy (Deployment/Service only) | Customer admin may need to permit ClusterPolicy creation on locked-down clusters; the `values.hosted-trial.yaml` header flags this |
| `mcp.apiKey.existingSecret` | `mcp-fleet-apikeys` (pre-created in the namespace) | `mcp-fleet-apikeys` or empty — see the Secret strategy in §3 |

Pick the matching example file, paste it into the *Helm Values* editor, adjust
the `# SITE:` lines, apply. A third shipped file serves the hardened posture:
`values-examples/values-hardened-g2.yaml` is an OVERLAY to merge into a profile
file (it carries no model/security blocks of its own) that turns the MCP
network zone ON (`networkPolicy.enabled: true` — a BASE-chart-only knob; the
scale charts ship no NetworkPolicy template). Only the listed caller
namespaces, the edge-gateway pods and the probe CIDRs reach the service; its
comments carry the two live-verified G2 traps — list the LLM-gateway relay
namespace FIRST (every unlisted in-cluster caller times out — deny by
absence), and list the namespace Prometheus actually scrapes `/metrics` from
(an unlisted scrape namespace dies SILENTLY — check targets after enabling).

The same two postures exist for the `helm-scale-medium/` and
`helm-scale-large/` variants (each chart's `values-examples/` carries
`values.g2.yaml` + `values.hosted-trial.yaml`; no hardened overlay — there is
no `networkPolicy` knob on those charts). Chart-parity notes (v4.0.0): the
`mcp.apiKey.existingSecret` → `RAG_API_KEYS` sidecar wiring renders in ALL
three charts, so the optional MCP key gate is identical at every chart size;
the scale charts expose the same `metrics.serviceMonitorBearerSecret` /
`…BearerSecretKey` keys as the base chart, so a `security.metricsAuth`-on
deployment keeps Prometheus scraping working at every chart size; and the
large chart additionally ships `security.oidc.platformCa.*` (a ConfigMap-mounted
platform root CA enabling EXTERNAL-https OIDC issuer/provider URLs — absent
from the base and medium charts).

### Enabling the SSO stack — the order matters (D21 → D22 → D23 → D25)

The SSO knobs stack; enable them in THIS order (the gated knobs render only
when true — default render byte-identical, EXCEPT D25 self-mint, which now
ships default-ON in all three charts and is inert by construction without an
identity source; the rendered envs are read per request — flipping an
already-rendered flag needs no restart, but a NEW env needs one values
apply):

1. **Keys + registry first** — the platform Secret (`security.existingSecret`)
   and the per-user key registry (D15/D17). D20 makes an unkeyed deployment
   fail-closed, so the key posture must exist before any SSO work is
   meaningful.
2. **The SSO identity source** — D21 `security.oidc.enabled: true` + `issuer`
   (REQUIRED when enabled), and/or D24 `trustProxyIdentity: true` behind an
   enforcing edge proxy. Add D22 `security.oidc.sso.*` for the browser cookie
   flow (requires D21 enabled + the client secret + the callback URI
   registered in the realm client). Flipping D24 `true` WITHOUT an enforcing
   proxy is a forgery exposure — in-cluster callers could spoof the
   `X-Auth-Request-*` headers and, once D25 is on, mint keys under a forged
   identity.
3. **D23 `security.ssoGate: true`** — gates the served HTML pages; meaningful
   only with an identity source (D21/D22) or API keys configured (an empty
   deployment gates to nothing).
4. **D25 `security.accessSelfMint`** — chart-default OFF since the 2026-10-16 audit remediation (opt in with `security.accessSelfMint: true`) in all three charts
   (the same default-on, explicit-opt-out posture as the access store). Its
   only accepted identity sources are the SSO ones (D21 JWT / D22 cookie / D24
   proxy header): without one, no session can ever carry the mint proof —
   the default-on knob is inert by construction, not a broken state (it also
   needs the access store on, so the minted entry has somewhere to live).
   `false` restores the pre-D25 inert behaviour (mint 409s). An
   API-key-authenticated caller can never mint regardless.

**Edge posture per site:** on Internal G2 the ezaf-gateway applies HPE SSO
centrally — `ezua.authorizationPolicy.enabled: false` (the chart creates no
AuthorizationPolicy) with `trustProxyIdentity: true`, and app-level OIDC (D21)
binds that edge-proven identity to the registry. On a hosted trial the
oauth2-proxy AuthorizationPolicy stays on (`enabled: true`) and is
COMPLEMENTARY, not competing — edge auth gates ingress, D21 binds app
identity — with `trustProxyIdentity: true` likewise correct behind the
enforcing proxy. One G2 edge caveat (observed 2026-10): the shared gateway
chains SSO on all paths — it 302s non-browser fetches (even the app's signed
media URLs in cross-site `<img>` tags; see the CHANGELOG's
`get_dataset_file_bytes` entry for why base64-via-MCP was the fix) — so
key-carrying REST/MCP access from OUTSIDE the cluster needs a platform-side
skip-auth decision; in-cluster callers (the LLM-gateway relay, MCP clients
behind it) reach the service directly and are unaffected.

### Secret requirements — front and center

The chart itself ships **no key material**. Before (or at) first apply, these
Secrets/values must exist — everything else is optional tuning:

1. **Platform keys — required.** `security.existingSecret` (recommended) names a
   Secret **you own** carrying BOTH keys; without them the containers refuse to
   start:
   ```bash
   kubectl -n <ns> create secret generic rag-platform-keys \
     --from-literal=RAG_API_KEY="$(openssl rand -hex 16)" \
     --from-literal=MEDIA_TOKEN_SECRET="$(openssl rand -hex 32)"
   ```
   (Alternative: leave everything empty and let the chart auto-generate both on
   first install into `<deployment.name>-model-keys`; or set
   `security.apiKey` / `security.mediaTokenSecret` inline — back-compat.
   Precedence and the optional rotation runbook: `helm/ROTATION.md`.)
2. **Model API keys** — `modelSecrets.*ApiKey` in values (seeded into the
   `<name>-model-keys` Secret on first install); paste the platform JWT per MLIS endpoint.
   SECRET-FIRST afterwards: the stored value wins on every upgrade, so a values edit to an
   already-seeded key is inert — rotate with `kubectl edit secret` (hot-reloads live, see
   §8) or delete the Secret and upgrade to reseed from values. The same sticky rule covers
   `s3.accessKeyId`/`secretAccessKey` and `redis.password`.
   **Key material out of values entirely (2026-10):** set `modelSecrets.existingSecret`
   to a Secret you own with the canonical key names — every key it carries OVERRIDES the
   chart-managed Secret (wired last in `envFrom` and `CONFIG_DIR` on the API + MCP
   containers, the embed-batcher, and both CronJobs), keys it omits keep the seeded/stored
   values, and `MODEL_<ROLE>_URL` entries there override the ConfigMap endpoints. The
   wiring is hard-wired (missing Secret = `CreateContainerConfigError`, loud). After the
   Secret is in place the values seeds are optional — blank them; the stored/sticky values
   survive (never delete the `<name>-model-keys` object itself: the REST keys live there).
   Upgrade guard: an in-cluster `helm upgrade` fails the render when NO source (user
   Secret, stored chart Secret, values seeds) provides an embedder URL or API key — the
   embedder is the only truly required model, and without it every ingest fails.
   **The broadest possible Secret (one object, every deployment credential):** a single
   Secret can carry all of it — `MODEL_<ROLE>_API_KEY` (4 keys), `MODEL_<ROLE>_URL`
   (optional endpoint overrides), `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`,
   `REDIS_PASSWORD` (scale charts) — referenced by `modelSecrets.existingSecret`; and the
   same object can serve as `security.existingSecret` (add `RAG_API_KEY` +
   `MEDIA_TOKEN_SECRET`) and even `mcp.apiKey.existingSecret` (add `api-keys`). Build it
   from **`helm/local/secrets.template.yaml`**, apply with `kubectl apply -n <ns>`, blank
   the values seeds, and rotation becomes `kubectl apply` alone (pods hot-reload ≤15s, no
   rollout, no helm involvement). `kubectl apply` does NOT envsubst — bake real domains
   into `MODEL_*_URL` (the values editor resolves `${DOMAIN_NAME}`; a Secret literal does
   not). Live example: `helm-scale-large/local/secrets.g2.yaml` (keys + endpoints + S3).
3. **MCP keys (optional)** — `mcp.apiKey.existingSecret: mcp-fleet-apikeys` +
   `kubectl -n <ns> create secret generic mcp-fleet-apikeys
   --from-literal='api-keys=<key1>,<key2>'`. The chart never creates or inlines
   this key.

### Fail-closed default (D20, 2026-09-24)

A deployment with **no** `RAG_API_KEY` and no key registry no longer runs open:
unauthenticated callers bind an **anonymous identity** whose only grant is the
deployment's memory dataset (`MEMORY_DATASET` env, when set — dataset paths and
`/api/admin/*` answer 403, dataset creation 403). Configure any key
(`security.apiKey` / `existingSecret` / the MCP keyset / registry) to get the
normal full-access behaviour — every chart-managed deployment already does.
`MEMORY_DATASET` (and `RAG_MEMORY_PASSWORD`) are settable via `extraEnv` in
values — e.g. `extraEnv: {MEMORY_DATASET: "team-memory"}`.

### OIDC identity (D21, 2026-10)

`security.oidc.*` turns a verified Keycloak/UA-realm JWT into a **second credential**
for the same per-user registry identity as the user's minted API key: a
`Authorization: Bearer <jwt>` on `/api/*` or `/mcp` (RS256 against the realm JWKS;
`iss` / `aud`-as-`azp` / `exp` validated; JWKS URL = `security.oidc.jwksUrl` or
`<issuer>/protocol/openid-connect/certs` — no OIDC discovery in v1) resolves via the
`identityClaim` (`preferred_username`, fallback `sub`) or an explicit `"oidc":
"<alias>"` overlay entry. Key semantics: API keys are unchanged and remain sufficient
alone; JWT-only sign-in works (fail-closed zero-dataset identity → self-select via the
D16 access store, chart default `security.accessStore: true`); JWT identities are
**never admin** (realm `groups` are not RAG grants); operator ACL grants apply to key
and JWT alike; key and JWT rotation/revocation are independent; a `"blocked": true`
overlay entry kills the JWT path; D19 delegation precedence is unchanged (through the
LLM gateway an explicit `X-API-Key` still outranks a co-forwarded `Authorization:
Bearer`); observed JWT-only users appear in `GET /api/admin/clients` as
`{"source": "observed"}` entries. Issuer is **REQUIRED** when enabled (the app logs a
loud startup warning without it) and every knob is read per request — no restart.
Chart wiring renders the `RAG_OIDC_*` envs into BOTH containers only when
`oidc.enabled: true` (the default render is byte-identical). Browser sign-in
(D22) rides on this same resolver via `security.oidc.sso.*` — the §2 values
row and the §5b ordered-enablement subsection cover it.

**G2 note:** the ezaf-gateway applies SSO centrally at the edge
(`ezua.authorizationPolicy.enabled: false` on this site), so app-level OIDC is the
identity source — edge SSO proves who you are at the door, D21 binds that identity to
the registry. On hosted trials the oauth2-proxy AuthorizationPolicy is complementary,
not competing: edge auth gates ingress, D21 binds app identity. Full semantics: README
§ Security hardening (D21 subsection), MCP.md (JWT-only client config), API.md § 1.

### Identity-gated frontend (D23, 2026-10)

`security.ssoGate: true` (`RAG_SSO_GATE`, read per request — no restart to flip)
folds the three HTML pages into ONE identity-gated shell: the homepage re-glasses
into a four-tab app — **Datasets** (the identity-scoped upload/search workspace),
**Access** (the D15/D16 dataset checkboxes, unlock/TTL panel and ★ memory binding),
**Admin** (the operator dashboard, rendered only for `is_admin` identities),
**Stats** (a read-only dashboard for everyone), and — since D27 — **Inspector**
(a browser playground for direct MCP tool calls, authenticated as the session;
see MCP.md § 5). `/manage` and `/access` redirect
into their tabs; anonymous visitors get a minimal 401 "Sign in required" page
(with a `/oauth/login` link when browser SSO is configured). Sign-in: SSO button
first, API-key paste fallback; the embedded admin key is used only in the
key-unconfigured legacy mode.

Ownership (RATIFIED 2026-10): dataset creation stamps `created_by` in the
dataset's `meta.json` (the identity name; an admin-key creator stamps `admin`).
DELETE is **owner-or-admin** — enforced identically on REST and MCP. Datasets
created before D23 have no owner recorded: admins delete them, and only admins
can make them global (no ownership is ever backfilled). A creator may toggle
their own dataset public (available to every identity); non-creators are denied.
Listings gain `created_by` + `owned_by_me`; `GET /api/stats` is a read-only,
identity-filtered summary (`datasets.total` / `visible_to_you` / `documents` /
`storage_bytes`, `models[]`, `jobs`, `generated_at`) — a user sees exactly their
visible world, an admin sees the deployment; it never mutates anything. Enable
the gate ONLY with `security.oidc.enabled` or API keys configured — an empty
deployment gates to nothing. Full semantics incl. the 8 ratified points: README
§ Security hardening (D23 subsection), API.md § 1.

---

## 6. Connect an MCP client

When `mcp.enabled=true` (default), the MCP server runs as a sidecar container exposing `streamable-http` transport on port 9090 at `/mcp`.

For the full tool list, connection configs (opencode, Claude Desktop, Open WebUI, stdio), and the long-term memory setup, see:

- **[MCP.md](MCP.md)** — all 23 MCP tools + connection configs for any client
- **[MEMORY.md](MEMORY.md)** — per-user long-term memory setup (opencode + Open WebUI)

### Quick reference

```json
{
  "mcpServers": {
    "multimodal-rag": {
      "url": "https://rag-mcp-server.your-domain.com/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

---

## 7. Architecture overview

```
┌─────────────────────────────────────────────────────┐
│  Pod                                                 │
│                                                      │
│  ┌──────────────────┐   ┌──────────────────┐        │
│  │  API Server       │   │  MCP Server       │       │
│  │  port 8000        │   │  port 9090        │       │
│  │  (REST + Web UI)  │   │  (MCP tools)      │       │
│  └──────┬───────────┘   └──────┬───────────┘        │
│         │                      │                     │
│         └──────────┬───────────┘                     │
│                    │                                 │
│             ┌──────┴──────┐                          │
│             │   PVC /data  │  ← datasets + files    │
│             └─────────────┘                          │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              │  Qdrant (port 6333) │  ← StatefulSet
              │  ┌──────────────┐  │
              │  │ PVC /qdrant  │  │  ← vectors + index
              │  │   storage    │  │
              │  └──────────────┘  │
              └───────────────────┘
```

Both the API server and MCP server connect to the same Qdrant instance and share the same PVC, so datasets created through the web UI are immediately searchable via MCP tools and vice versa.

### EZUA / Istio integration

When `ezua.enabled=true` (default), the chart also creates:

- **VirtualService** — routes `ezua.virtualService.endpoint` through `ezua.virtualService.istioGateway` (default `istio-system/ezaf-gateway`), with three timeout tiers: batch uploads/SSE and `/mcp` get `ezua.virtualService.longTimeout` (3600s), everything else `ezua.virtualService.timeout` (300s). The endpoint is **required** — with it unset the chart render fails with `Valid .Values.ezua.virtualService.endpoint is required !`, and the endpoint value is also what builds `MEDIA_BASE_URL` (PVC-path → HTTPS media URLs in MCP search results)
- **AuthorizationPolicy** (`ezua.authorizationPolicy.enabled`, default `true`) — enforces OAuth2 authentication at the Istio ingress gateway, via provider `ezua.authorizationPolicy.providerName` (`oauth2-proxy`) in `ezua.authorizationPolicy.namespace` (`istio-system`). Set `enabled: false` where the gateway already applies SSO centrally (SE G2)
- **Kyverno ClusterPolicy** — auto-labels **Deployments and Services** in the release namespace with `hpe-ezua/type: vendor-service` and `hpe-ezua/app: rag-mcp-server` (required for the EZUA ingress to discover the service). **The policy still matches only Deployment/Service objects** (fixed 2026-09-24): the EzAF platform ships its own admission policy that rewrites the scheduler on every pod carrying that `hpe-ezua/type` label, which made cron pods with orphaned finalizers undeletable — this chart version permanently immunizes the release against that interaction. Separately (2026-10), the **pod templates of the long-running workloads** (API Deployment, Qdrant, Redis, embed-batcher) render the same vendor labels themselves: the app-catalog card's *status* is derived from labeled pods, and an app whose pods carry no `hpe-ezua` labels shows "Unknown" in the catalog. **Cron-job pods stay unlabeled** — labeled pods are scheduler-stamped at every admission and become undeletable on pod update.

---

## 7b. Watched S3 sources (opt-in continuous sync)

For deployments with an S3/MinIO drop zone, the chart can run a reconciler CronJob that keeps
datasets continuously in sync with one or more bucket prefixes — no manual batch-URL uploads:

```yaml
# values.yaml (all three charts).  Default `enabled: false` renders NOTHING —
# deployments without S3 are byte-identical to before.
watchedSources:
  enabled: true
  schedule: "*/30 * * * *"          # cron schedule (default every 30 minutes)
  sources:
    - dataset: reports              # the dataset each prefix syncs INTO (must exist)
      prefixes:
        - s3://mm-rag-drop/reports/
    - dataset: logs
      prefixes:
        - s3://mm-rag-drop/logs/
        - s3://mm-rag-drop/trace-exports/
```

Behaviour per tick (serial per source, `concurrencyPolicy: Forbid`, fail-soft):

- POSTs the prefixes to `POST /api/datasets/{dataset}/batch-urls` with `sync: true` and the
  deployment key as `X-RAG-Api-Key` (from the model-keys Secret), then polls the ingest job to
  completion, so a slow embed batch cannot overlap the next tick.
- New and content-changed objects ingest; objects deleted upstream are pruned from the dataset.
- **Unchanged objects are skipped before any download** — a per-dataset sidecar
  (`files/.watched_state.json`) remembers the ETag+Size of the last successfully ingested
  version of each object; an object still matching it costs zero S3 I/O. Content changed under
  the same key flips the ETag and re-ingests. Delete or Recreate a dataset and the state resets.
- Config is validated at values-parse time (`helm template`): every prefix must be an `s3://`
  directory URL — single-object `s3://bucket/file.ext` URLs, wildcards, query strings and
  http(s) URLs fail the render, not the cron log. A `type:` field is reserved for future
  source kinds.
- The server-side bucket allowlist `INGEST_ALLOW_S3_BUCKETS` still applies — keep it set as the
  multi-tenant guard. The S3 credentials come from the existing `s3.*` values (the same
  `S3_ENDPOINT_URL` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` wiring batch-URL ingest uses).

> The backup CronJob (`backups.*`) presents the same API key since this release — it
> previously sent none and would 401 on every call under `security.apiKey`.

---

## 8. Upgrading

In PCAI, upgrading is just editing the values and applying again. To bump the image, change `image.tag` in the *Helm Values* editor (or a model-endpoint URL, a PVC size, etc.):

To change specific settings (e.g. storage or model endpoints), edit the corresponding keys in `values.yaml` and re-apply; the rest of the values are kept. API keys sourced from the release Secret are reused across upgrades (never rotated underneath a running deployment); the optional, operator-initiated rotation procedure — and the warning that rotated keys invalidate every issued media token — is `helm/ROTATION.md`.

Model/S3/Redis keys in the same Secret are **sticky in the other direction** (2026-09): the stored value wins, values only seed. To rotate a model JWT or S3 credential, `kubectl -n <ns> edit secret <deployment.name>-model-keys` — the kubelet refreshes the `/etc/rag/secrets` mount within ~1s and both containers' `CONFIG_DIR` watcher rebuilds the affected models within ~15s, **no rollout needed**. To rotate *from values* instead, delete the Secret first, then upgrade (pods inside the delete→upgrade window may show `CreateContainerConfigError` until the upgrade recreates it).

---

## 9. Troubleshooting

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| Pods stuck in `Pending` | PVC not binding | Check the PVC status in PCAI |
| API server crash-looping | Model connection failure | Read the API pod logs in PCAI |
| MCP tools return "dataset not found" | Dataset created on different Qdrant | Verify `QDRANT_HOST` matches |
| Search returns 0 results | Empty dataset or wrong collection | Check via web UI document list |
| Qdrant OOMKilled | Vector count exceeds memory | Increase `resources.qdrant.limits.memory` |
| Watched-sources cron pod stuck with an orphaned finalizer (old chart) | The pre-fix vendor-label policy matched Pod kind and the platform scheduler-mutation policy rejected every pod update | Deploy this chart version (policy matches Deployment/Service only); one-time cleanup for already-stuck pods: strip the finalizer AND the `hpe-ezua/type` label in one patch (see the CHANGELOG 2026-09-24 entry) |
| 404 at VirtualService endpoint | Kyverno labels not applied | Confirm the Deployment/Service carry the `hpe-ezua` labels (PCAI discovery reads Deployments/Services, not Pods) |
| App card shows **Unknown** in the EzUA catalog | No pod carries the `hpe-ezua` labels — the catalog derives card *status* from labeled pods (object labels alone are not enough) | Deploy this chart version (long-running pod templates carry the labels); verify with `kubectl get pods -n <ns> --show-labels | grep vendor-service`. Cron pods are unlabeled by design — that is not the cause |
| 401 at VirtualService endpoint | OAuth2 token missing/expired | Check `oauth2-proxy` logs in `istio-system` |
| Endpoint shows `rag-mcp-server.<domain>` unresolved | Domain value not set | Set `ezua.virtualService.endpoint` in values.yaml |
