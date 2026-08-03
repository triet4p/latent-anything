#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: remote_cuda_test.sh --remote USER@HOST [--ref BRANCH] [--extra EXTRA]
       [--test-command COMMAND] [--gpu-command COMMAND] [--repo-url URL]
EOF
}

remote=""
ref="main"
repo_url="$(git remote get-url origin)"
test_command='uv run pytest -q -m "not network and not large_download"'
gpu_command=""
extras=()

while (($#)); do
  case "$1" in
    --remote) remote="$2"; shift 2 ;;
    --ref) ref="$2"; shift 2 ;;
    --repo-url) repo_url="$2"; shift 2 ;;
    --extra) extras+=("$2"); shift 2 ;;
    --test-command) test_command="$2"; shift 2 ;;
    --gpu-command) gpu_command="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$remote" ]] || { echo "--remote USER@HOST is required" >&2; exit 2; }
command -v ssh >/dev/null || { echo "ssh is required; run this script from Bash, WSL, or Git Bash" >&2; exit 2; }
command -v git >/dev/null || { echo "git is required" >&2; exit 2; }

root="$(git rev-parse --show-toplevel)"
cd "$root"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Local worktree is dirty. Commit intended changes before remote testing:" >&2
  git status --short >&2
  exit 1
fi

local_sha="$(git rev-parse HEAD)"
git push origin "HEAD:refs/heads/$ref"
remote_sha="$(git ls-remote origin "refs/heads/$ref" | awk '{print $1}')"
[[ "$remote_sha" == "$local_sha" ]] || {
  echo "GitHub branch SHA $remote_sha does not match local SHA $local_sha" >&2
  exit 1
}

remote_args=("$repo_url" "$ref" "$local_sha" "$test_command" "$gpu_command")
remote_args+=("$(IFS=,; echo "${extras[*]}")")

remote_command="bash -s -- $(printf '%q ' "${remote_args[@]}")"
ssh "$remote" "$remote_command" <<'REMOTE_RUNNER'
set -Eeuo pipefail
repo_url="$1"; ref="$2"; expected_sha="$3"; test_command="$4"; gpu_command="$5"; extras_csv="$6"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/remote-cuda-test.XXXXXX")"
cleanup_status=0
cleanup() { rm -rf -- "$tmp_dir" || cleanup_status=$?; }
trap cleanup EXIT INT TERM
export UV_CACHE_DIR="$tmp_dir/uv-cache"
export TORCH_EXTENSIONS_DIR="$tmp_dir/torch-extensions"
export CUDA_CACHE_PATH="$tmp_dir/cuda-cache"

git clone --depth 1 --branch "$ref" "$repo_url" "$tmp_dir/repo"
cd "$tmp_dir/repo"
actual_sha="$(git rev-parse HEAD)"
[[ "$actual_sha" == "$expected_sha" ]] || { echo "SHA mismatch: $actual_sha != $expected_sha" >&2; exit 1; }

extra_args=()
if [[ -n "$extras_csv" ]]; then
  IFS=',' read -r -a extras <<< "$extras_csv"
  for extra in "${extras[@]}"; do extra_args+=(--extra "$extra"); done
fi
uv sync "${extra_args[@]}"

if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
  export CC=gcc-12
  export CXX=g++-12
  export CUDAHOSTCXX=g++-12
fi

hostname_value="$(hostname)"
gpu_status="unavailable"
gpu_name="none"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_status="available"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

torch_status="unknown"
if uv run python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  torch_status="available"
else
  torch_status="unavailable"
fi

if [[ -n "$gpu_command" ]]; then
  bash -lc "$gpu_command"
fi
bash -lc "$test_command"

cat <<REPORT
Remote CUDA test
- Source: $repo_url@$actual_sha
- Server: $USER@$hostname_value
- GPU/CUDA: $gpu_name ($gpu_status), torch CUDA $torch_status
- GPU test: $([[ -n "$gpu_command" ]] && echo PASS || echo NOT_REQUESTED)
- Broader tests: PASS ($test_command)
- Cleanup: scheduled for temporary directory
REPORT
REMOTE_RUNNER
