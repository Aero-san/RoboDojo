#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUNDLE_DIR="${REPO_ROOT}/migration_bundle"

info() { printf '>>> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash scripts/migration/create_bundle.sh [--output DIR] [--skip-large-data]

Creates an offline migration folder containing:
  - Git bundles for RoboDojo and every initialized submodule;
  - binary patches and untracked, non-ignored source files;
  - zstd archives of Assets, checkpoints, datasets, training outputs, and results;
  - metadata and SHA-256 checksums.

The script never moves or deletes source data. Existing completed archives are
kept, so rerunning the command resumes the bundle creation.
EOF
}

SKIP_LARGE_DATA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a directory"
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --skip-large-data)
      SKIP_LARGE_DATA=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

command -v git >/dev/null || die "git is required"
command -v tar >/dev/null || die "tar is required"
command -v zstd >/dev/null || die "zstd is required"
command -v sha256sum >/dev/null || die "sha256sum is required"

BUNDLE_DIR="$(realpath -m "${BUNDLE_DIR}")"
[[ "${BUNDLE_DIR}" != "${REPO_ROOT}" ]] || die "Bundle directory cannot be the repository root"
mkdir -p "${BUNDLE_DIR}"/{archives,repos,patches,metadata}

write_metadata() {
  local metadata_file="${BUNDLE_DIR}/metadata/project.txt"
  {
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'source_root=%s\n' "${REPO_ROOT}"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -srmo)"
    printf 'robodojo_head=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    printf 'robodojo_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
    printf 'xpolicylab_head=%s\n' "$(git -C "${REPO_ROOT}/XPolicyLab" rev-parse HEAD)"
    printf 'xpolicylab_branch=%s\n' "$(git -C "${REPO_ROOT}/XPolicyLab" branch --show-current)"
  } >"${metadata_file}"
  git -C "${REPO_ROOT}" status --short --branch >"${BUNDLE_DIR}/metadata/robodojo-status.txt"
  git -C "${REPO_ROOT}" submodule status >"${BUNDLE_DIR}/metadata/submodules.txt"
  git -C "${REPO_ROOT}/XPolicyLab" status --short --branch >"${BUNDLE_DIR}/metadata/xpolicylab-status.txt"
  df -h "${REPO_ROOT}" >"${BUNDLE_DIR}/metadata/disk.txt"
}

archive_name_for_path() {
  printf '%s' "$1" | sed 's#/#__#g'
}

create_archive() {
  local relative_path="$1"
  local source_path="${REPO_ROOT}/${relative_path}"
  local archive_name
  archive_name="$(archive_name_for_path "${relative_path}")"
  local output="${BUNDLE_DIR}/archives/${archive_name}.tar.zst"
  local partial="${output}.partial"

  if [[ ! -e "${source_path}" ]]; then
    warn "Skipping missing path: ${relative_path}"
    return
  fi
  if [[ -s "${output}" ]]; then
    info "Keeping existing archive: ${output}"
    return
  fi

  info "Archiving ${relative_path} ($(du -sh "${source_path}" | awk '{print $1}'))"
  tar --sparse -C "${REPO_ROOT}" -cf - "${relative_path}" \
    | zstd -T0 -1 -f -o "${partial}"
  mv "${partial}" "${output}"
}

create_git_bundle() {
  local repo_path="$1"
  local label="$2"
  local output="${BUNDLE_DIR}/repos/${label}.bundle"
  local partial="${output}.partial"

  info "Capturing Git object database: ${label}"
  git -C "${repo_path}" bundle create "${partial}" --all
  git bundle verify "${partial}" >/dev/null
  mv "${partial}" "${output}"
}

capture_worktree() {
  local repo_path="$1"
  local label="$2"
  local patch_file="${BUNDLE_DIR}/patches/${label}.patch"
  local untracked_file="${BUNDLE_DIR}/patches/${label}-untracked.tar.zst"
  local lfs_file="${BUNDLE_DIR}/patches/${label}-lfs.tar.zst"

  git -C "${repo_path}" diff --binary --ignore-submodules=all HEAD >"${patch_file}"
  git -C "${repo_path}" ls-files --others --exclude-standard -z \
    | tar --null --no-recursion -C "${repo_path}" -T - -cf - \
    | zstd -T0 -1 -f -o "${untracked_file}"
  if git -C "${repo_path}" lfs version >/dev/null 2>&1; then
    local git_dir
    git_dir="$(git -C "${repo_path}" rev-parse --absolute-git-dir)"
    if [[ -d "${git_dir}/lfs/objects" ]]; then
      tar -C "${git_dir}" -cf - lfs/objects \
        | zstd -T0 -1 -f -o "${lfs_file}"
    fi
  fi
}

write_metadata
create_git_bundle "${REPO_ROOT}" RoboDojo
capture_worktree "${REPO_ROOT}" RoboDojo

while IFS=$'\t' read -r sub_path; do
  [[ -n "${sub_path}" ]] || continue
  if [[ ! -e "${REPO_ROOT}/${sub_path}/.git" ]]; then
    warn "Submodule is not initialized, skipping Git bundle: ${sub_path}"
    continue
  fi
  label="$(archive_name_for_path "${sub_path}")"
  create_git_bundle "${REPO_ROOT}/${sub_path}" "${label}"
  capture_worktree "${REPO_ROOT}/${sub_path}" "${label}"
done < <(git -C "${REPO_ROOT}" config --file .gitmodules --get-regexp path | awk '{print $2}')

if [[ "${SKIP_LARGE_DATA}" -eq 0 ]]; then
  LARGE_PATHS=(
    Assets
    XPolicyLab/policy/Pi_05/checkpoints
    data
    outputs
    eval_result
    wandb
    smoke_results
  )
  for relative_path in "${LARGE_PATHS[@]}"; do
    create_archive "${relative_path}"
  done
fi

cp "${SCRIPT_DIR}/restore_bundle.sh" "${BUNDLE_DIR}/restore.sh"
chmod +x "${BUNDLE_DIR}/restore.sh"

(
  cd "${BUNDLE_DIR}"
  find archives repos patches metadata -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)

info "Bundle ready: ${BUNDLE_DIR}"
du -sh "${BUNDLE_DIR}"
info "Verify after transfer: cd '${BUNDLE_DIR}' && sha256sum -c SHA256SUMS"
