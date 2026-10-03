# Local deployment values (gitignored · hardlink-ignored · never packaged)

This directory holds **per-cluster / per-customer values files that must never leave this machine**: client credentials, site endpoints, cluster names, tokens. Three guards are in place — keep them intact in every repo:

| Guard | File | Rule |
|---|---|---|
| git | `.gitignore` | `helm*/local/*` — only this README and `values.example.yaml` are trackable |
| chart packaging | `helm/.helmignore` | `local/` — credentials can never enter a packaged/imported chart tarball |
| hardlinker | `hardlink_config.json` | `helm*/local` — nothing here is ever mirrored into `pcai-solutions/` |

## Convention

| File | Purpose |
|---|---|
| `values.example.yaml` | Tracked, sanitized starting point — copy it. |
| `values.<site>.yaml` | One real file per deployment site / cluster (e.g. `values.customer-a.yaml`, `values.toromont.yaml`, `values.se-g2.yaml`). Never committed; keep mode `0600`. |

## Using a values file

```bash
helm template <release> . -f helm/local/values.<site>.yaml   # eyeball the render first
helm upgrade --install <release> . -n <namespace> -f helm/local/values.<site>.yaml
```

On PCAI: import the packaged chart once, then paste the same values into the PCAI *Helm Values* editor (PCAI does not run `helm install`/`envsubst`).

## Secrets hygiene

Prefer out-of-band Secrets (`credentialsSecret.create: false` + `kubectl create secret generic ...`) so credentials never pass through a values file at all. If the chart must create the Secret from values, those values live ONLY here — never in `values.yaml`, never in git, never in `pcai-solutions/`.
