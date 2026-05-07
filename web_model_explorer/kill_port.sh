#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./kill_port.sh [--all] [PORT...]

Behavior:
  - With no args or with --all: kills this repo's dev servers (web_model_explorer + web_human_study)
    based on ports configured in the repo root `.env`.
  - With one or more PORT args: attempts to kill *only this repo's* dev servers that are listening
    on those ports (it will skip unrelated processes).

Notes:
  - Attempts a graceful shutdown (TERM) first, then force-kills remaining processes (KILL).
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

canonical_path() {
  local path="$1"
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$path" 2>/dev/null || printf '%s\n' "$path"
  else
    (cd "$path" 2>/dev/null && pwd -P) || printf '%s\n' "$path"
  fi
}

CANONICAL_REPO_ROOT="$(canonical_path "$REPO_ROOT")"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

WEB_MODEL_EXPLORER_PORT="${WEB_MODEL_EXPLORER_PORT:-3001}"
HUMAN_STUDY_BACKEND_PORT="${HUMAN_STUDY_BACKEND_PORT:-8000}"
HUMAN_STUDY_FRONTEND_PORT="${HUMAN_STUDY_FRONTEND_PORT:-5173}"

allowed_roots=()
for maybe_root in \
  "$REPO_ROOT/web_model_explorer" \
  "$REPO_ROOT/web_human_study/frontend" \
  "$REPO_ROOT/web_human_study/backend"; do
  if [[ -d "$maybe_root" ]]; then
    allowed_roots+=("$maybe_root")
    canonical_root="$(canonical_path "$maybe_root")"
    [[ "$canonical_root" != "$maybe_root" ]] && allowed_roots+=("$canonical_root")
  fi
done
if [[ ${#allowed_roots[@]} -eq 0 ]]; then
  echo "Error: could not locate dev server directories under: $REPO_ROOT" >&2
  exit 1
fi

ports=()
if [[ $# -eq 0 || "${1:-}" == "--all" ]]; then
  ports=("$WEB_MODEL_EXPLORER_PORT" "$HUMAN_STUDY_FRONTEND_PORT" "$HUMAN_STUDY_BACKEND_PORT")
  if [[ "${1:-}" == "--all" ]]; then
    shift
    # Allow additional ports after --all.
    if [[ $# -gt 0 ]]; then
      ports+=("$@")
    fi
  fi
else
  ports=("$@")
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "Error: lsof is required but not found in PATH." >&2
  exit 1
fi

pid_belongs_to_repo() {
  local pid="$1"

  # Prefer checking the process working directory (cwd).
  local cwd=""
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  if [[ -z "$cwd" ]]; then
    cwd="$(lsof -nP -p "$pid" 2>/dev/null | awk '$4=="cwd"{print $NF; exit}')"
  fi
  if [[ -n "$cwd" ]]; then
    local canonical_cwd=""
    canonical_cwd="$(canonical_path "$cwd")"
    local root=""
    for root in "${allowed_roots[@]}"; do
      if [[ "$cwd" == "$root"* || "$canonical_cwd" == "$root"* ]]; then
        return 0
      fi
    done
  fi

  # Fallback: match command line against the repo path (less reliable).
  local cmd=""
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ -n "$cmd" && ( "$cmd" == *"$REPO_ROOT"* || "$cmd" == *"$CANONICAL_REPO_ROOT"* ) ]]; then
    return 0
  fi

  return 1
}

pid_or_parent_belongs_to_repo() {
  local pid="$1"
  local current="$pid"
  local self="$$"
  while [[ -n "$current" && "$current" != "0" && "$current" != "1" && "$current" != "$self" ]]; do
    if pid_belongs_to_repo "$current"; then
      return 0
    fi
    current="$(ps -o ppid= -p "$current" 2>/dev/null | tr -d ' ' || true)"
  done
  return 1
}

kill_pid_tree() {
  local pid="$1"

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  local pgid=""
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  local self_pgid=""
  self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)"

  # Prefer killing the process group so spawned children exit too.
  if [[ -n "$pgid" && -n "$self_pgid" && "$pgid" != "$self_pgid" ]]; then
    echo "Killing process group $pgid for PID $pid"
    kill -TERM "-$pgid" 2>/dev/null || true
  else
    echo "Killing PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
  fi

  # Give it a moment to exit cleanly.
  local _i
  for _i in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done

  if [[ -n "$pgid" && -n "$self_pgid" && "$pgid" != "$self_pgid" ]]; then
    echo "Force-killing process group $pgid for PID $pid"
    kill -KILL "-$pgid" 2>/dev/null || true
  else
    echo "Force-killing PID $pid"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

wait_for_port_free() {
  local port="$1"
  local _i
  for _i in {1..30}; do
    if [[ -z "$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]]; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

echo "Repo: $REPO_ROOT"
echo "Target ports (from .env/defaults):"
echo "  - web_model_explorer: ${WEB_MODEL_EXPLORER_PORT}"
echo "  - web_human_study frontend: ${HUMAN_STUDY_FRONTEND_PORT}"
echo "  - web_human_study backend: ${HUMAN_STUDY_BACKEND_PORT}"
echo "Allowed project roots:"
printf '  - %s\n' "${allowed_roots[@]}"
echo "Checking ports: ${ports[*]}"

killed_any=false
for port in "${ports[@]}"; do
  if ! [[ "$port" =~ ^[0-9]+$ ]]; then
    echo "Skipping invalid port: $port" >&2
    continue
  fi

  # Only kill listeners (not clients).
  pids="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "Port $port: no listening process."
    continue
  fi

  echo "Port $port: found PID(s): $pids"
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if ! pid_or_parent_belongs_to_repo "$pid"; then
      cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || lsof -nP -p "$pid" 2>/dev/null | awk '$4=="cwd"{print $NF; exit}')"
      cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
      echo "Port $port: skipping PID $pid (not in this repo). cwd='${cwd:-?}' cmd='${cmd:-?}'"
      continue
    fi
    killed_any=true
    kill_pid_tree "$pid"
  done <<<"$pids"

  if wait_for_port_free "$port"; then
    echo "Port $port: no listening process."
  else
    remaining_pids="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    echo "Port $port: still busy after kill attempt. Remaining PID(s): ${remaining_pids:-?}" >&2
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      if pid_or_parent_belongs_to_repo "$pid"; then
        kill_pid_tree "$pid"
      fi
    done <<<"$remaining_pids"
    wait_for_port_free "$port" || echo "Port $port: still busy." >&2
  fi
done

if [[ "$killed_any" == "false" ]]; then
  echo "No matching dev servers found."
fi
