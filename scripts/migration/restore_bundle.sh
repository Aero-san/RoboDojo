#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${SCRIPT_DIR}"
DEST_DIR="${1:-$(cd "${BUNDLE_DIR}/.." && pwd)/RoboDojo-restored}"

info() { printf '>>> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"
command -v tar >/dev/null || die "tar is required"
command -v zstd >/dev/null || die "zstd is required"
command -v sha256sum >/dev/null || die "sha256sum is required"
[[ -f "${BUNDLE_DIR}/SHA256SUMS" ]] || die "SHA256SUMS not found in ${BUNDLE_DIR}"
[[ ! -e "${DEST_DIR}" ]] || die "Destination already exists: ${DEST_DIR}"

info "Verifying migration files"
(cd "${BUNDLE_DIR}" && sha256sum -c SHA256SUMS)

info "Cloning RoboDojo from the offline Git bundle"
GIT_LFS_SKIP_SMUDGE=1 git clone "${BUNDLE_DIR}/repos/RoboDojo.bundle" "${DEST_DIR}"

restore_submodule() {
  local name="$1"
  local path="$2"
  local label
  label="$(printf '%s' "${path}" | sed 's#/#__#g')"
  local bundle="${BUNDLE_DIR}/repos/${label}.bundle"
  [[ -f "${bundle}" ]] || die "Missing submodule bundle: ${bundle}"
  git -C "${DEST_DIR}" config "submodule.${name}.url" "${bundle}"
  git -C "${DEST_DIR}" config "submodule.${name}.active" true
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DEST_DIR}" \
    -c protocol.file.allow=always submodule update --init "${path}"
}

while read -r key path; do
  name="${key#submodule.}"
  name="${name%.path}"
  restore_submodule "${name}" "${path}"
done < <(git -C "${DEST_DIR}" config --file .gitmodules --get-regexp '^submodule\..*\.path$')

if git -C "${DEST_DIR}/XPolicyLab" show-ref --verify --quiet refs/heads/mingyang/robodojo-migration; then
  git -C "${DEST_DIR}/XPolicyLab" switch mingyang/robodojo-migration
elif git -C "${DEST_DIR}/XPolicyLab" show-ref --verify --quiet refs/remotes/origin/mingyang/robodojo-migration; then
  git -C "${DEST_DIR}/XPolicyLab" switch --track origin/mingyang/robodojo-migration
fi

apply_worktree() {
  local repo_path="$1"
  local label="$2"
  local patch_file="${BUNDLE_DIR}/patches/${label}.patch"
  local untracked_file="${BUNDLE_DIR}/patches/${label}-untracked.tar.zst"
  local lfs_file="${BUNDLE_DIR}/patches/${label}-lfs.tar.zst"

  if [[ -s "${lfs_file}" ]]; then
    local git_dir
    git_dir="$(git -C "${repo_path}" rev-parse --absolute-git-dir)"
    tar --zstd -xf "${lfs_file}" -C "${git_dir}"
    git -C "${repo_path}" lfs checkout
  fi
  if [[ -s "${patch_file}" ]]; then
    git -C "${repo_path}" apply --binary "${patch_file}"
  fi
  if [[ -s "${untracked_file}" ]]; then
    tar --zstd -xf "${untracked_file}" -C "${repo_path}"
  fi
}

apply_worktree "${DEST_DIR}" RoboDojo
while read -r _key path; do
  label="$(printf '%s' "${path}" | sed 's#/#__#g')"
  apply_worktree "${DEST_DIR}/${path}" "${label}"
done < <(git -C "${DEST_DIR}" config --file .gitmodules --get-regexp '^submodule\..*\.path$')

shopt -s nullglob
for archive in "${BUNDLE_DIR}"/archives/*.tar.zst; do
  info "Extracting $(basename "${archive}")"
  tar --zstd -xf "${archive}" -C "${DEST_DIR}"
done

info "Offline restore completed: ${DEST_DIR}"
info "Next: follow docs/MIGRATION_WITH_CODEX.md to install environments, rewrite asset paths, and validate."
