{{/*
Expand the name of the chart.
*/}}
{{- define "mcp-s3-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mcp-s3-server.fullname" -}}
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
{{- define "mcp-s3-server.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mcp-s3-server.labels" -}}
helm.sh/chart: {{ include "mcp-s3-server.chart" . }}
{{ include "mcp-s3-server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mcp-s3-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mcp-s3-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mcp-s3-server.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mcp-s3-server.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create a list of allowed hosts
*/}}
{{/* Define the full list of allowed hosts */}}
{{- define "mcp-s3-server.allowedHosts" -}}
{{- $hosts := .Values.extraAllowedHosts | default (list "localhost" "127.0.0.1") -}}
{{- $istio := .Values.ezua.virtualService.endpoint | required ".Values.ezua.virtualService.endpoint is required !\n" -}}
{{- $hosts = append $hosts $istio -}}
{{- $internal := printf "%s.%s.svc.cluster.local:*" (include "mcp-s3-server.fullname" .) .Release.Namespace -}}
{{- $hosts = append $hosts $internal -}}
{{- /* Return as comma-separated string */ -}}
{{- $hosts | join "," -}}
{{- end -}}