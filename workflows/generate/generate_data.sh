#!/usr/bin/env bash
set -euo pipefail
DATASET_PATH=tsENV_questions
COPY_TO_REMOTE=1
RUNS_DIR_NAME=
PUSH_TO_HF=
HF_REPO_ID=TommasoBendinelli/tsENV
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_DATASET_CARD_TEMPLATE="$REPO_ROOT/tsenv_questions_dataset_card.md"
COMBINATIONS_CSV="$REPO_ROOT/all_possible_combinations.csv"
cd "$REPO_ROOT"

TIMING_SUMMARY=()

_elapsed_seconds() {
  local started="$1"
  local ended="$2"
  awk -v started="$started" -v ended="$ended" 'BEGIN { printf "%.3f", (ended - started) }'
}

_record_timing() {
  local label="$1"
  local elapsed="$2"
  TIMING_SUMMARY+=("${elapsed}|${label}")
}

_print_timing_summary() {
  if [[ ${#TIMING_SUMMARY[@]} -eq 0 ]]; then
    return
  fi
  printf '[timing:summary] slowest stages\n' >&2
  printf '%s\n' "${TIMING_SUMMARY[@]}" \
    | sort -t'|' -k1,1nr \
    | head -n 20 \
    | while IFS='|' read -r elapsed label; do
        printf '[timing:summary] %ss %s\n' "$elapsed" "$label" >&2
      done
}

_run_timed() {
  local label="$1"
  shift
  local started="${EPOCHREALTIME:-0}"
  local status
  printf '[stage:start] %s\n' "$label" >&2
  set +e
  "$@"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    local ended="${EPOCHREALTIME:-0}"
    local elapsed
    elapsed="$(_elapsed_seconds "$started" "$ended")"
    _record_timing "$label" "$elapsed"
    printf '[stage:done] %s elapsed=%ss\n' "$label" "$elapsed" >&2
    return 0
  fi
  local ended="${EPOCHREALTIME:-0}"
  local elapsed
  elapsed="$(_elapsed_seconds "$started" "$ended")"
  printf '[stage:fail] %s elapsed=%ss status=%s\n' "$label" "$elapsed" "$status" >&2
  return "$status"
}

_require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$command_name" >&2
    return 1
  fi
}

_write_dataset_card() {
  local dataset_root="$1"
  local readme_path="$dataset_root/README.md"
  if [[ ! -f "$HF_DATASET_CARD_TEMPLATE" ]]; then
    printf 'Dataset card template not found: %s\n' "$HF_DATASET_CARD_TEMPLATE" >&2
    return 1
  fi
  cp "$HF_DATASET_CARD_TEMPLATE" "$readme_path"
}

_push_dataset_to_hf() {
  local dataset_root="$1"
  _require_command hf
  if [[ ! -d "$dataset_root" ]]; then
    printf 'Dataset output directory not found: %s\n' "$dataset_root" >&2
    return 1
  fi
  _write_dataset_card "$dataset_root"
  hf upload "$HF_REPO_ID" "$dataset_root" . --repo-type=dataset
}

_require_env_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    printf 'Missing required environment variable: %s\n' "$var_name" >&2
    return 1
  fi
}

_load_repo_env_var_raw() {
  local var_name="$1"
  local env_path="$2"
  python3 - "$var_name" "$env_path" <<'PY'
import sys
from pathlib import Path

var_name = sys.argv[1]
env_path = Path(sys.argv[2])
if not env_path.exists():
    raise SystemExit(0)

for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() != var_name:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(value)
    raise SystemExit(0)
PY
}

_load_generate_data_env() {
  local env_path="$REPO_ROOT/.env"
  if [[ -z "${TSENV_REMOTE_DESTINATION:-}" ]]; then
    TSENV_REMOTE_DESTINATION="$(_load_repo_env_var_raw TSENV_REMOTE_DESTINATION "$env_path")"
  fi
}

_append_all_row_slug_args() {
  local csv_path="$1"
  local row_slug
  while IFS= read -r row_slug; do
    [[ -n "$row_slug" ]] || continue
    TSENV_ROW_SLUG_ARGS+=(--row-slug "$row_slug")
  done < <(
    python3 - "$csv_path" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1]).expanduser().resolve()
with csv_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        row_slug = str(row.get("row_slug") or "").strip()
        if row_slug:
            print(row_slug)
PY
  )
}

_cleanup_model_outputs() {
  local dataset_root="$1"
  shift
  local model
  mkdir -p "$dataset_root"
  for model in "$@"; do
    [[ -n "$model" ]] || continue
    rm -rf "${dataset_root%/}/${model}"
  done
}

_copy_dataset_to_remote() {
  local dataset_root="$1"
  shift
  local dataset_name
  local remote_dir
  local remote_host
  local remote_path
  local remote_path_for_ssh
  local model
  local model_destination
  _require_command ssh
  _require_command rsync
  _require_env_var TSENV_REMOTE_DESTINATION
  if [[ ! -d "$dataset_root" ]]; then
    printf 'Dataset output directory not found: %s\n' "$dataset_root" >&2
    return 1
  fi
  if [[ "$#" -eq 0 ]]; then
    printf 'No model folders were provided for remote copy.\n' >&2
    return 1
  fi
  dataset_name="$(basename "${dataset_root%/}")"
  remote_dir="${TSENV_REMOTE_DESTINATION%/}/${dataset_name}"
  if [[ "$remote_dir" != *:* ]]; then
    printf 'TSENV_REMOTE_DESTINATION must be in host:path format; got: %s\n' "$TSENV_REMOTE_DESTINATION" >&2
    return 1
  fi
  remote_host="${remote_dir%%:*}"
  remote_path="${remote_dir#*:}"
  remote_path_for_ssh="${remote_path/#~\//\$HOME/}"
  ssh "$remote_host" "mkdir -p ${remote_path_for_ssh}"
  for model in "$@"; do
    [[ -n "$model" ]] || continue
    if [[ ! -d "${dataset_root%/}/${model}" ]]; then
      printf 'Dataset model output directory not found: %s\n' "${dataset_root%/}/${model}" >&2
      return 1
    fi
    model_destination="${remote_dir}/${model}/"
    ssh "$remote_host" "rm -rf ${remote_path_for_ssh}/${model} && mkdir -p ${remote_path_for_ssh}/${model}"
    rsync -avh "${dataset_root%/}/${model}/" "$model_destination"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs-dir-name)
      RUNS_DIR_NAME="${2:?missing value for --runs-dir-name}"
      shift 2
      ;;
    --no-copy-to-remote)
      COPY_TO_REMOTE=
      shift
      ;;
    --push-to-hf)
      PUSH_TO_HF=1
      shift
      ;;
    --hf-repo-id)
      HF_REPO_ID="${2:?missing value for --hf-repo-id}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

main() {
  _load_generate_data_env
  TSENV_MODELS=(
    DampedMassBetweenWalls
  )
  _run_timed "generate_data.cleanup_output" _cleanup_model_outputs "$DATASET_PATH" "${TSENV_MODELS[@]}"
  _run_timed "generate_data.activate_env" source "$REPO_ROOT/env/bin/activate"

  # python workflows/exam_questions/create_exam_questions_tsb_ad.py --data-type univariate --output-dir $DATASET_PATH --eval-file-list outputs/uni_hardest_few_shot_3.json $COPY_TO_REMOTE
  # python workflows/exam_questions/create_exam_questions_tsb_ad.py --data-type multivariate --output-dir $DATASET_PATH --eval-file-list outputs/mul_hardest_few_shot_3.json  $COPY_TO_REMOTE

  for model in "${TSENV_MODELS[@]}"; do
    _run_timed \
      "generate_data.train_test_selection.${model}" \
      env/bin/python workflows/generate/train_test_selection.py \
      --model "${model}" \
      --runs-dir-name "$RUNS_DIR_NAME" \
      < /dev/null
  done

  TSENV_MODEL_ARGS=()
  for model in "${TSENV_MODELS[@]}"; do
    TSENV_MODEL_ARGS+=(--model "${model}")
  done
  TSENV_ROW_SLUG_ARGS=()
  _append_all_row_slug_args "$COMBINATIONS_CSV"

  _run_timed \
    "generate_data.export_tsenv_questions" \
    env/bin/python workflows/generate/create_exam_questions_tsenv_cls_from_registry.py \
    "${TSENV_MODEL_ARGS[@]}" \
    "${TSENV_ROW_SLUG_ARGS[@]}" \
    --output-dir "$DATASET_PATH" \
    --runs-dir-name "$RUNS_DIR_NAME"

  if [[ -n "$COPY_TO_REMOTE" ]]; then
    _run_timed "generate_data.copy_to_remote" _copy_dataset_to_remote "$DATASET_PATH" "${TSENV_MODELS[@]}"
  fi

  if [[ -n "$PUSH_TO_HF" ]]; then
    _run_timed "generate_data.push_to_hf" _push_dataset_to_hf "$DATASET_PATH"
  fi
}

overall_started="${EPOCHREALTIME:-0}"
printf '[stage:start] %s\n' "generate_data.total" >&2
set +e
main
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  overall_ended="${EPOCHREALTIME:-0}"
  overall_elapsed="$(_elapsed_seconds "$overall_started" "$overall_ended")"
  _record_timing "generate_data.total" "$overall_elapsed"
  printf '[stage:done] %s elapsed=%ss\n' "generate_data.total" "$overall_elapsed" >&2
  _print_timing_summary
  exit 0
fi
overall_ended="${EPOCHREALTIME:-0}"
overall_elapsed="$(_elapsed_seconds "$overall_started" "$overall_ended")"
printf '[stage:fail] %s elapsed=%ss status=%s\n' "generate_data.total" "$overall_elapsed" "$status" >&2
_print_timing_summary
exit "$status"
