# Best Practices to Deploying Custom Applications in HPE's Private Cloud AI (PCAI)

This guide walks through going from a simple Streamlit app → Docker image → Kubernetes YAML → Helm chart suitable for deployment into HPE’s Private Cloud AI (PCAI).

PCAI is a turnkey platform to build, manage, and deploy AI applications at scale. For more details, see the [HPE Private Cloud AI website](https://developer.hpe.com/platform/hpe-private-cloud-ai/home/). ([HPE Developer][1])

---

## Overall Goal

We’ll build a custom application from the ground up and deploy it in PCAI:

* A **“hello world” LLM chat application** using Streamlit.
* The app will:

  * Accept user input.
  * Connect to an **OpenAI-compatible API** (such as an MLIS endpoint in PCAI).
  * Use an **API key supplied by the user** in the UI.
  * Stream responses to the browser.
* We’ll:

  1. Containerize the app with **Docker**.
  2. Create a single **Kubernetes YAML** that defines a `Deployment`, `Service`, and `VirtualService`.
  3. Convert that YAML into a **Helm chart** that can be uploaded and deployed in PCAI.

---

## Key Ideas

* **MLIS (Machine Learning Inference Software)** is the recommended way to deploy LLMs and AI models in PCAI.

  * MLIS exposes **production-grade inference endpoints**.
  * You get an **OpenAI-compatible API** and can use a simple API key to call models from your UI.
* PCAI uses **Kubernetes** (with **Istio** for networking) to manage applications.

  * Custom applications are typically deployed as **Helm charts**.
  * A **VirtualService** bound to `istio-system/ezaf-gateway` is required for external access.

---

## Key Prerequisites

You should have:

* Basic knowledge of:

  * Docker / containerization.
  * Kubernetes and Helm.
  * Python development.
* Tools:

  * `docker`
  * `kubectl`
  * `helm`
* Access to:

  * A PCAI environment.
  * `kubectl` access to a Kubernetes cluster used by PCAI.
  * A container registry reachable from the cluster.
  * An OpenAI-compatible endpoint (e.g. MLIS) and an API key.

---

# Step 1: Initial Application

We’ll create a simple Streamlit chat app that calls an OpenAI-compatible endpoint (e.g. MLIS) and streams responses.

## 1.1 Directory layout

Create a project directory:

```bash
pcai-hello-llm/
├── app
│   └── main.py
├── requirements.txt
└── Dockerfile
```

> All paths in the rest of this section assume you’re in `pcai-hello-llm/`.

---

## 1.2 Streamlit app (`app/main.py`)

This app:

* Lets the user configure:

  * Base URL (OpenAI-compatible endpoint, e.g. MLIS).
  * API key.
  * Model name.
* Streams responses token-by-token to the UI.
* Maintains chat history via `st.session_state`.

```python
import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="PCAI Hello LLM", page_icon="🤖")

st.title("PCAI – Hello World LLM Chat")
st.write(
    "This app connects to an OpenAI-compatible endpoint (for example, an MLIS "
    "inference endpoint in HPE Private Cloud AI) and streams responses."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: connection configuration
with st.sidebar:
    st.header("LLM connection")

    default_base_url = os.getenv("OPENAI_BASE_URL", "https://YOUR-MLIS-ENDPOINT/v1")
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    base_url = st.text_input(
        "Base URL",
        value=default_base_url,
        help="OpenAI-compatible base URL (e.g. https://<your-mlis-endpoint>/v1).",
    )
    api_key = st.text_input(
        "API key",
        type="password",
        help="API key for the OpenAI-compatible endpoint.",
    )
    model = st.text_input(
        "Model name",
        value=default_model,
        help="Model identifier as exposed by your endpoint.",
    )

    st.caption(
        "Tip: In PCAI, you can point this to an MLIS inference endpoint. "
        "In production, prefer storing secrets in Kubernetes Secrets rather than typing them here."
    )

if not base_url or not api_key or not model:
    st.info("Enter base URL, API key, and model name in the sidebar to start chatting.")
    st.stop()

# Create OpenAI client (for OpenAI-compatible endpoint)
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask the model something...")

if user_input:
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Build conversation with system prompt + history
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering user questions. "
                    "Keep responses concise for demo purposes."
                ),
            },
            *st.session_state.messages,
        ]

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

---

## 1.3 Dependencies (`requirements.txt`)

```txt
streamlit>=1.39.0
openai>=1.12.0
```

> Versions are examples; feel free to adjust to your environment.

---

## 1.4 Run locally (optional sanity check)

```bash
# From pcai-hello-llm/
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run app/main.py
```

Open `http://localhost:8501` in a browser, configure the base URL, API key, and model, and send a test message.

---

## 1.5 Dockerfile

We’ll containerize the app into an image that PCAI can run.

```dockerfile
# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies if needed (uncomment if required for your environment)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY app ./app

EXPOSE 8501

# Launch Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 1.6 Build and push the Docker image

Set your registry information and build/push:

```bash
# Set these to match your environment
export IMAGE_REGISTRY=your-registry.example.com/your-namespace
export IMAGE_NAME=pcai-hello-llm
export IMAGE_TAG=0.1.0

docker build -t ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
docker push ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
```

Example final image reference:
`your-registry.example.com/your-namespace/pcai-hello-llm:0.1.0`

---

# Step 2: Develop Kubernetes YAML for Testing

We’ll create a single `app.yaml` that defines:

* A `Deployment` for the Streamlit app.
* A `Service` exposing port `8501` inside the cluster.
* An Istio `VirtualService` for external access via PCAI’s gateway.

Assumptions:

* You already created the namespace:

  ```bash
  kubectl create namespace test
  ```

* You know the external domain for your PCAI environment and will replace `<DOMAIN_NAME>` with that value for **this raw YAML test** (for example, `pcai.example.com` → `demo.pcai.example.com`).

> **Key point:** PCAI uses **Istio** for networking. A `VirtualService` referencing `istio-system/ezaf-gateway` is required for external HTTP(S) access.

Create `app.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-llm
  namespace: test
  labels:
    app: hello-llm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hello-llm
  template:
    metadata:
      labels:
        app: hello-llm
    spec:
      containers:
        - name: hello-llm
          image: your-registry.example.com/your-namespace/pcai-hello-llm:0.1.0
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8501
          env:
            # Optional defaults; override inside the app UI for development
            - name: OPENAI_BASE_URL
              value: "https://YOUR-MLIS-ENDPOINT/v1"
            - name: OPENAI_MODEL
              value: "gpt-4o-mini"
          # Add resource requests/limits for production
          resources: {}

---
apiVersion: v1
kind: Service
metadata:
  name: hello-llm
  namespace: test
  labels:
    app: hello-llm
spec:
  selector:
    app: hello-llm
  ports:
    - name: http
      port: 8501
      targetPort: 8501
  type: ClusterIP

---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: hello-llm
  namespace: test
spec:
  gateways:
    # Required in PCAI: use the central ezaf-gateway
    - istio-system/ezaf-gateway
  hosts:
    # Replace <DOMAIN_NAME> with your environment domain, e.g. demo.pcai.example.com
    - demo.<DOMAIN_NAME>
  http:
    - match:
        - uri:
            prefix: /
      rewrite:
        uri: /
      route:
        - destination:
            host: hello-llm.test.svc.cluster.local
            port:
              number: 8501
```

> **Best practice:** For production, move secrets such as API keys into Kubernetes `Secret` objects and **do not** hard-code them in the manifest. The Streamlit UI in this example expects the user to provide the API key at runtime.

---

## Test the app on the cluster

Apply the manifest:

```bash
kubectl apply -f app.yaml -n test
# or, if you use the 'k' alias:
# k apply -f app.yaml -n test
```

Verify resources:

```bash
kubectl get pods -n test
kubectl get svc -n test
kubectl get virtualservice -n test
```

Once the pod is running, browse to `https://demo.<DOMAIN_NAME>` in your browser and confirm:

1. The Streamlit UI loads.
2. You can set base URL, API key, and model.
3. You see streamed responses from your endpoint.

When this all works, you’re ready to convert the raw Kubernetes YAML into a **Helm chart**.

---

# Step 3: Helm Chart

Now we’ll develop a Helm chart from the Kubernetes YAML.

High-level steps to convert a working Kubernetes YAML into a Helm chart:

1. **Start from working YAML** (like `app.yaml` above).
2. **Identify fields** that should be configurable:

   * Image repository, tag, pull policy.
   * Replica count.
   * Service type and port.
   * VirtualService host and gateway.
   * Environment variables (base URL, model).
3. **Extract those fields into `values.yaml`**.
4. **Replace hard-coded values** in the YAML with template expressions that read from `.Values`.
5. Split the YAML into multiple files under `templates/`:

   * `deployment.yaml`
   * `service.yaml`
   * `virtualservice.yaml`
6. Add the Helm metadata files:

   * `Chart.yaml`
   * `_helpers.tpl` for common labels/names.
7. Test with `helm template` or `helm install` on a test cluster.

> In practice, many teams use a frontier LLM to help parameterize the YAML and then review/edit the generated templates.

---

## 3.1 Helm chart directory layout

Create a new directory for the chart (separate from the app code):

```bash
pcai-hello-llm-chart/
├── Chart.yaml
├── values.yaml
└── templates
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── service.yaml
    └── virtualservice.yaml
```

---

## 3.2 `_helpers.tpl`

`templates/_helpers.tpl` defines common naming and labels:

```yaml
{{- define "pcai-hello-llm.name" -}}
{{- default .Chart.name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "pcai-hello-llm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := include "pcai-hello-llm.name" . }}
{{- if ne .Release.Name $name }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "pcai-hello-llm.labels" -}}
app.kubernetes.io/name: {{ include "pcai-hello-llm.name" . }}
helm.sh/chart: {{ .Chart.name }}-{{ .Chart.version | replace "+" "_" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.appVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "pcai-hello-llm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pcai-hello-llm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

---

## 3.3 `templates/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "pcai-hello-llm.fullname" . }}
  labels:
    {{- include "pcai-hello-llm.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "pcai-hello-llm.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "pcai-hello-llm.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ include "pcai-hello-llm.name" . }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: {{ .Values.service.port }}
          env:
            - name: OPENAI_BASE_URL
              value: {{ .Values.env.openaiBaseUrl | quote }}
            - name: OPENAI_MODEL
              value: {{ .Values.env.openaiModel | quote }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

---

## 3.4 `templates/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "pcai-hello-llm.fullname" . }}
  labels:
    {{- include "pcai-hello-llm.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  selector:
    {{- include "pcai-hello-llm.selectorLabels" . | nindent 4 }}
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.port }}
```

---

## 3.5 `templates/virtualservice.yaml`

This template wires your app into the Istio gateway required by PCAI and reads its configuration from `.Values.ezua.virtualService`.

```yaml
{{- if .Values.ezua.enabled }}
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: {{ include "pcai-hello-llm.fullname" . }}
  labels:
    {{- include "pcai-hello-llm.labels" . | nindent 4 }}
spec:
  gateways:
    - {{ .Values.ezua.virtualService.istioGateway | quote }}
  hosts:
    - {{ .Values.ezua.virtualService.endpoint | quote }}
  http:
    - match:
        - uri:
            prefix: /
      rewrite:
        uri: /
      route:
        - destination:
            host: {{ include "pcai-hello-llm.fullname" . }}.{{ .Release.Namespace }}.svc.cluster.local
            port:
              number: {{ .Values.service.port }}
{{- end }}
```

> Note: PCAI will typically inject the correct domain into `ezua.virtualService.endpoint` (for example using `demo.${DOMAIN_NAME}`); see the `values.yaml` section below.

---

## 3.6 `Chart.yaml`

```yaml
apiVersion: v2
name: pcai-hello-llm
description: A hello world LLM chat application for HPE Private Cloud AI
type: application
version: 0.1.0
appVersion: "0.1.0"
```

---

## 3.7 `values.yaml`

This file defines configurable parameters for the chart.

> **Important PCAI requirement:** The `ezua` block below must be at the **root** of `values.yaml`. PCAI expects these keys to drive how it exposes the VirtualService.

```yaml
# Number of pod replicas
replicaCount: 1

image:
  # Replace with your registry/namespace
  repository: your-registry.example.com/your-namespace/pcai-hello-llm
  tag: "0.1.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8501

env:
  # Default base URL and model; can be overridden at install time
  openaiBaseUrl: "https://YOUR-MLIS-ENDPOINT/v1"
  openaiModel: "gpt-4o-mini"

resources: {}
# resources:
#   requests:
#     cpu: "250m"
#     memory: "512Mi"
#   limits:
#     cpu: "500m"
#     memory: "1Gi"

nodeSelector: {}
tolerations: []
affinity: {}

# ---------------------------------------------------------------------------
# PCAI REQUIREMENT:
# This block must be at the root of values.yaml.
# PCAI uses it to drive how the app is exposed externally via Istio.
# ---------------------------------------------------------------------------
ezua:
  enabled: true
  virtualService:
    # External hostname for the service (corresponds to VirtualService spec.hosts)
    # PCAI will typically substitute ${DOMAIN_NAME} with the environment domain.
    endpoint: "demo.${DOMAIN_NAME}"
    # Istio gateway to bind to (corresponds to VirtualService spec.gateways)
    istioGateway: "istio-system/ezaf-gateway"
```

---

## 3.8 Test the Helm chart locally (optional)

From the `pcai-hello-llm-chart/` directory:

```bash
helm lint .

helm template hello-llm . \
  --namespace test \
  --set image.repository=your-registry.example.com/your-namespace/pcai-hello-llm \
  --set image.tag=0.1.0
```

You can also do a direct install to a non-PCAI test cluster:

```bash
kubectl create namespace test || true

helm install hello-llm . \
  --namespace test \
  --set image.repository=your-registry.example.com/your-namespace/pcai-hello-llm \
  --set image.tag=0.1.0 \
  --set env.openaiBaseUrl="https://YOUR-MLIS-ENDPOINT/v1"
```

Verify pods, services, and VirtualService, then hit the external URL to confirm the app works.

---

## 3.9 Deploying into PCAI

The exact PCAI UI/flow may vary slightly, but in general:

1. Package the chart:

   ```bash
   cd pcai-hello-llm-chart
   helm package .
   # This creates pcai-hello-llm-0.1.0.tgz
   ```

2. Upload the `.tgz` into PCAI’s application catalog / UI.

3. Configure:

   * Image repository and tag.
   * Any environment-specific overrides (if needed).

4. Deploy the chart into the desired namespace.

5. Access the app via the external endpoint derived from `ezua.virtualService.endpoint` (e.g. `https://demo.<your-domain>`).

---

## Summary

You now have:

* A minimal, working **Streamlit LLM chat app** that talks to an OpenAI-compatible endpoint (such as MLIS).
* A **Docker image** for the app.
* A tested **Kubernetes manifest** with `Deployment`, `Service`, and `VirtualService` wired to `istio-system/ezaf-gateway`.
* A reusable, PCAI-compatible **Helm chart** that:

  * Uses best practices for naming and labels.
  * Exposes configuration via `values.yaml`.
  * Includes the required `ezua` configuration block for PCAI.

You can share this markdown internally or with customers as an end-to-end example for building and deploying custom applications in PCAI.

[1]: https://developer.hpe.com/platform/hpe-private-cloud-ai/home/?utm_source=chatgpt.com "HPE Private Cloud AI"
