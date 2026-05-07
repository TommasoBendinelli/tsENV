#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Respect explicit environment overrides (e.g. `WEB_MODEL_EXPLORER_PORT=3001 bash start.sh`)
# even if `.env` defines a different value.
_had_web_model_explorer_host="${WEB_MODEL_EXPLORER_HOST+x}"
_had_web_model_explorer_port="${WEB_MODEL_EXPLORER_PORT+x}"
_had_web_model_explorer_runs_dir_name="${WEB_MODEL_EXPLORER_RUNS_DIR_NAME+x}"
_orig_web_model_explorer_host="${WEB_MODEL_EXPLORER_HOST-}"
_orig_web_model_explorer_port="${WEB_MODEL_EXPLORER_PORT-}"
_orig_web_model_explorer_runs_dir_name="${WEB_MODEL_EXPLORER_RUNS_DIR_NAME-}"

# Load local environment overrides if present.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ -n "$_had_web_model_explorer_host" ]]; then
  WEB_MODEL_EXPLORER_HOST="$_orig_web_model_explorer_host"
fi
if [[ -n "$_had_web_model_explorer_port" ]]; then
  WEB_MODEL_EXPLORER_PORT="$_orig_web_model_explorer_port"
fi
if [[ -n "$_had_web_model_explorer_runs_dir_name" ]]; then
  WEB_MODEL_EXPLORER_RUNS_DIR_NAME="$_orig_web_model_explorer_runs_dir_name"
fi

HOST="${WEB_MODEL_EXPLORER_HOST:-0.0.0.0}"
PORT="${WEB_MODEL_EXPLORER_PORT:-3001}"

find_free_port() {
  local start_port="$1"
  local host="$2"
  HOST="$host" PORT="$start_port" node - <<'NODE'
const net = require('net');
const host = process.env.HOST || '0.0.0.0';
let port = parseInt(process.env.PORT, 10);
function tryPort(p) {
  const server = net.createServer();
  server.unref();
  server.on('error', (err) => {
    if (err.code !== 'EADDRINUSE') {
      console.error(err.message || String(err));
      process.exit(1);
    }
    port += 1;
    if (port > 65535) {
      console.error('No free ports available.');
      process.exit(1);
    }
    tryPort(port);
  });
  server.listen(p, host, () => {
    console.log(p);
    server.close();
  });
}
tryPort(port);
NODE
}

chosen_port="$(find_free_port "$PORT" "$HOST")"
if [[ "$chosen_port" != "$PORT" ]]; then
  echo "Port ${PORT} is busy; attempting to stop existing dev server..."
  if [[ -x "$SCRIPT_DIR/kill_port.sh" ]]; then
    "$SCRIPT_DIR/kill_port.sh" "$PORT" || true
    for _i in {1..20}; do
      chosen_port="$(find_free_port "$PORT" "$HOST")"
      if [[ "$chosen_port" == "$PORT" ]]; then
        break
      fi
      sleep 0.25
    done
  else
    echo "Warning: kill_port.sh not found or not executable." >&2
  fi
  if [[ "$chosen_port" != "$PORT" ]]; then
    echo "Port ${PORT} is still busy; switching to ${chosen_port}."
    PORT="$chosen_port"
  fi
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to start the Model Explorer. Install Node.js/npm first." >&2
  exit 1
fi

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
  echo "Installing web_model_explorer dependencies..."
  (cd "$SCRIPT_DIR" && npm install)
fi

has_stale_next_server_cache() {
  local runtime="$SCRIPT_DIR/.next/server/webpack-runtime.js"
  local chunks_dir="$SCRIPT_DIR/.next/server/chunks"

  [[ -f "$runtime" && -d "$chunks_dir" ]] || return 1

  # Known broken dev-cache shape: runtime resolves numeric chunks as ./997.js,
  # but Next has emitted them under ./chunks/997.js instead.
  if ! grep -Fq 'return "" + chunkId + ".js"' "$runtime"; then
    return 1
  fi

  local chunk_path
  while IFS= read -r chunk_path; do
    local chunk_name
    chunk_name="$(basename "$chunk_path")"
    [[ "$chunk_name" =~ ^[0-9]+\.js$ ]] || continue
    if [[ ! -f "$SCRIPT_DIR/.next/server/$chunk_name" ]]; then
      return 0
    fi
  done < <(find "$chunks_dir" -maxdepth 1 -type f -name '*.js' 2>/dev/null)

  return 1
}

if has_stale_next_server_cache; then
  echo "Detected stale Next.js build cache (server chunk path mismatch); removing web_model_explorer/.next"
  rm -rf "$SCRIPT_DIR/.next"
fi

echo "Starting Model Explorer on ${HOST}:${PORT}..."
(cd "$SCRIPT_DIR" && ./node_modules/.bin/next dev -H "$HOST" -p "$PORT")
