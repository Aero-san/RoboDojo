#!/usr/bin/env bash

find_posttrain_config() {
  POSTTRAIN_CONFIG_FILE=""
  local expect_path=0
  local argument
  for argument in "$@"; do
    if (( expect_path )); then
      [[ -z "${POSTTRAIN_CONFIG_FILE}" ]] || {
        echo "--config may only be specified once" >&2
        return 2
      }
      POSTTRAIN_CONFIG_FILE="${argument}"
      expect_path=0
    elif [[ "${argument}" == "--config" ]]; then
      expect_path=1
    fi
  done
  (( expect_path == 0 )) || { echo "--config requires a path" >&2; return 2; }
}

load_posttrain_config() {
  local config_path="$1"
  [[ -n "${config_path}" ]] || return 0
  [[ -f "${config_path}" ]] || { echo "Post-training config not found: ${config_path}" >&2; return 1; }
  local python_bin="${POSTTRAIN_CONFIG_PYTHON:-python3}"
  command -v "${python_bin}" >/dev/null 2>&1 || {
    echo "Post-training config Python not found: ${python_bin}" >&2
    return 1
  }
  local config_dump
  config_dump=$(mktemp "${TMPDIR:-/tmp}/robodojo-posttrain-config.XXXXXX")
  if ! "${python_bin}" "${SCRIPT_DIR}/load_posttrain_config.py" "${config_path}" >"${config_dump}"; then
    rm -f "${config_dump}"
    return 1
  fi
  local name value
  while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    # Explicit environment variables override YAML; command-line options are
    # parsed after this function and therefore override both.
    if [[ ! -v "${name}" ]]; then
      printf -v "${name}" '%s' "${value}"
      export "${name}"
    fi
  done <"${config_dump}"
  rm -f "${config_dump}"
}

load_pi05_recap_config() {
  local config_path="$1"
  [[ -n "${config_path}" ]] || {
    echo "Pi0.5 RECAP requires --config PATH" >&2
    return 2
  }
  [[ -f "${config_path}" ]] || { echo "Post-training config not found: ${config_path}" >&2; return 1; }
  local python_bin="${POSTTRAIN_CONFIG_PYTHON:-${ROOT_DIR}/external_dependencies/WCM/.venv/bin/python}"
  command -v "${python_bin}" >/dev/null 2>&1 || {
    echo "Post-training config Python not found: ${python_bin}" >&2
    return 1
  }
  local config_dump
  config_dump=$(mktemp "${TMPDIR:-/tmp}/robodojo-pi05-recap-config.XXXXXX")
  if ! "${python_bin}" "${SCRIPT_DIR}/pi05_recap_config.py" "${config_path}" >"${config_dump}"; then
    rm -f "${config_dump}"
    return 1
  fi
  local name value
  while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    # The unified YAML is the source of truth. CLI execution controls are
    # parsed after this function and may override the resolved values.
    printf -v "${name}" '%s' "${value}"
    export "${name}"
  done <"${config_dump}"
  rm -f "${config_dump}"
}

write_resolved_pi05_recap_config() {
  local config_path="$1"
  local output_path="$2"
  local python_bin="${POSTTRAIN_CONFIG_PYTHON:-${ROOT_DIR}/external_dependencies/WCM/.venv/bin/python}"
  "${python_bin}" "${SCRIPT_DIR}/pi05_recap_config.py" "${config_path}" \
    --format yaml --output "${output_path}"
}
