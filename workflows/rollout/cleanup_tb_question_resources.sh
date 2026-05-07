#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: workflows/rollout/cleanup_tb_question_resources.sh [--dry-run] [--run-id RUN_ID ...]

When --run-id is omitted, removes Terminal-Bench question resources matching
the documented question_0- prefix pattern.
EOF
}

DRY_RUN=0
RUN_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run-id)
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "--run-id requires at least one RUN_ID." >&2
        usage
        exit 2
      fi
      while [[ $# -gt 0 && "$1" != --* ]]; do
        RUN_IDS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      # Backward-compatible positional run IDs.
      RUN_IDS+=("$1")
      shift
      ;;
  esac
done

matches_target() {
  local name="$1"
  if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    [[ "$name" == question_0-* || "$name" == *"/question_0-"* || "$name" == *"question_0-"* ]]
    return
  fi

  local run_id
  for run_id in "${RUN_IDS[@]}"; do
    [[ -z "$run_id" ]] && continue
    if [[ "$name" == *"$run_id"* ]]; then
      return 0
    fi
  done
  return 1
}

cleanup_kind() {
  local kind="$1"
  shift
  local list_cmd=("$@")
  local removed=0
  local matched=0
  local name=""

  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if ! matches_target "$name"; then
      continue
    fi
    matched=$((matched + 1))
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf 'dry_run kind=%s name=%s\n' "$kind" "$name"
      continue
    fi
    case "$kind" in
      container)
        if docker rm -f "$name" >/dev/null 2>&1; then
          removed=$((removed + 1))
        fi
        ;;
      network)
        if docker network rm "$name" >/dev/null 2>&1; then
          removed=$((removed + 1))
        fi
        ;;
    esac
  done < <("${list_cmd[@]}")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'dry_run kind=%s matched=%s\n' "$kind" "$matched"
  else
    printf 'kind=%s removed=%s matched=%s\n' "$kind" "$removed" "$matched"
  fi
}

cleanup_kind container docker ps -a --format '{{.Names}}'
cleanup_kind network docker network ls --format '{{.Name}}'
