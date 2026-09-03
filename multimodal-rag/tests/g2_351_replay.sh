#!/usr/bin/env bash
# Live exploit-replay probes for the G2 v3.5.2 rollout (run of the 3.5.1 audit fixes).
# Usage: API_KEY=<value> bash tests/g2_351_replay.sh
# Safe: operates only on the throwaway dataset g2-audit-351-probe (created
# password-protected, re-imported, deleted at the end).
set -u
BASE="https://rag-mcp-server.pcai-se-ai-application.hst.rdlabs.hpecorp.net"
DS="g2-audit-351-probe"
PW="probe-pw-351"
API_KEY="${API_KEY:?set API_KEY}"
PASS=0; FAIL=0

http() { # http METHOD PATH [extra curl args...] -> sets CODE, BODY
  local m=$1 p=$2; shift 2
  local out
  out=$(curl -sk --max-time 90 -H "X-Rag-Api-Key: $API_KEY" -X "$m" \
        -w $'\n__HTTP__%{http_code}' "$@" "$BASE$p") || true
  if [[ "$out" == *$'\n__HTTP__'* ]]; then
    CODE=${out##*$'\n__HTTP__'}
    BODY=${out%$'\n'__HTTP__*}
  else
    CODE="ERR"; BODY="$out"
  fi
}
check() { # check <name> <expected-code> <expected-substr>
  local name=$1 exp_code=$2 substr=$3
  if [[ "$CODE" == "$exp_code" && "$BODY" == *"$substr"* ]]; then
    echo "PASS  $name ($CODE)"; PASS=$((PASS+1))
  else
    echo "FAIL  $name (code=$CODE, want $exp_code; body=${BODY:0:180})"; FAIL=$((FAIL+1))
  fi
}

echo "== readiness =="
http GET /healthz;                 check "healthz" 200 "ok"
http GET /readyz;                  check "readyz" 200 "ready"

echo "== throwaway (password-protected from creation) =="
http POST /api/datasets -d "{\"name\":\"$DS\",\"password\":\"$PW\",\"description\":\"3.5.1 audit probe\"}"
check "create (pw-protected)" 200 "$DS"
http POST /api/datasets/$DS/verify-password -d "{\"password\":\"$PW\"}"
check "verify-password route" 200 "valid"

echo "== exploit replays (must be refused with 400) =="
http POST /api/datasets/$DS/documents -H "X-Dataset-Password: $PW" -d '[{"text":"audit","image":"/etc/passwd"}]'
check "bare /etc/passwd refused" 400 "outside the allowed prefixes"
http POST /api/datasets/$DS/documents -H "X-Dataset-Password: $PW" -d '[{"text":"audit","image":"file:///etc/passwd"}]'
check "file:///etc/passwd refused" 400 "outside the allowed prefixes"
http POST /api/datasets/$DS/documents -H "X-Dataset-Password: $PW" -d '[{"text":"audit","image":"http://169.254.169.254/latest/meta-data/"}]'
check "metadata-IP SSRF refused" 400 "private/internal"
http POST /api/datasets/$DS/documents -H "X-Dataset-Password: $PW" -d '[{"text":"audit","image":"s3://some-bucket/k"}]'
check "s3:// payload ref refused" 400 "batch-urls"

echo "== positive path: store -> search (live in-cluster embed) =="
http POST /api/datasets/$DS/documents -H "X-Dataset-Password: $PW" \
     -d '[{"text":"The aurora borealis over a snowy mountain at night"}]'
check "clean document stored" 200 "stored_ids"
sleep 2
http GET "/api/datasets/$DS/search?q=aurora%20borealis%20snowy%20mountain" -H "X-Dataset-Password: $PW"
check "hybrid search round-trips (live embed)" 200 "aurora"

echo "== import-overwrite password gate (H-2 under real conditions) =="
http POST /api/admin/datasets/import -F "file=@tests/fixtures/g2_probe_backup.tar.gz;filename=probe.tar.gz" \
     -F "new_name=$DS" -F "overwrite=true"
check "overwrite w/o password -> refused" 401 "password"
http GET /api/datasets/$DS -H "X-Dataset-Password: $PW"
check "throwaway NOT deleted by refused attempt" 200 "$DS"
http POST /api/admin/datasets/import -H "X-Dataset-Password: $PW" \
     -F "file=@tests/fixtures/g2_probe_backup.tar.gz;filename=probe.tar.gz" \
     -F "new_name=$DS" -F "overwrite=true" -F "password=$PW"
check "overwrite WITH password -> accepted" 200 "job_id"
sleep 3
http GET /api/datasets/$DS -H "X-Dataset-Password: $PW"
check "re-imported dataset present" 200 "$DS"

echo "== cleanup =="
http DELETE /api/datasets/$DS -H "X-Dataset-Password: $PW"
check "throwaway deleted" 200 "deleted"

echo; echo "$PASS passed, $FAIL failed"
exit $(( FAIL > 0 ))
