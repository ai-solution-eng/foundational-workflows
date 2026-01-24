{{/*
Expand the name of the chart.
*/}}
{{- define "mcp-filesystem-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mcp-filesystem-server.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mcp-filesystem-server.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mcp-filesystem-server.labels" -}}
helm.sh/chart: {{ include "mcp-filesystem-server.chart" . }}
{{ include "mcp-filesystem-server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mcp-filesystem-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mcp-filesystem-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mcp-filesystem-server.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mcp-filesystem-server.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create a list of allowed hosts
*/}}
{{/* Define the full list of allowed hosts */}}
{{- define "mcp-filesystem-server.allowedHosts" -}}
{{- $hosts := .Values.extraAllowedHosts | default (list "localhost" "127.0.0.1") -}}
{{- $istio := .Values.ezua.virtualService.endpoint | required ".Values.ezua.virtualService.endpoint is required !\n" -}}
{{- $internal := printf "%s.%s.svc.cluster.local:*" (include "mcp-filesystem-server.fullname" .) .Release.Namespace -}}
{{- join "," (append $hosts $istio $internal) -}}
{{- end -}}