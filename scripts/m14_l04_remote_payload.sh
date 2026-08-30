#!/usr/bin/env bash
set -euo pipefail

UseCase="${1:-}"
CodeSha="${2:-}"
RepoUrl="${3:-}"

if [[ ! "$UseCase" =~ ^(IntegratedGradients|TCAV|DirectLogitLens|TunedLogitLens|Disentanglement|TrueActivationPatching|AdditiveSteering)$ ]]; then
    printf '%s\n' 'L04_STATUS=INVALID_USE_CASE' >&2
    exit 64
fi
if [[ ! "$CodeSha" =~ ^[0-9a-fA-F]{40}$ ]]; then
    printf '%s\n' 'L04_STATUS=INVALID_CODE_SHA' >&2
    exit 64
fi
if [[ ! "$RepoUrl" =~ ^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(\.git)?$ ]]; then
    printf '%s\n' 'L04_STATUS=INVALID_REPO_URL' >&2
    exit 64
fi

workdir=''
repo_dir=''
cache_root=''
preflight_file=''
bundle_file=''
before_members_file=''
cleanup_status=0
status_emitted=0

emit_status() {
    printf 'L04_STATUS=%s\n' "$1"
    status_emitted=1
}

bundle_gate_failure() {
    local reason="$1"
    local cli_status="$2"
    local final_status=66
    printf '%s\n' "$reason" >&2
    printf 'L04_BUNDLE_STATUS=66\n'
    if [[ "$cli_status" -ne 0 ]]; then
        final_status="$cli_status"
    fi
    emit_status "$final_status"
    exit "$final_status"
}

cleanup() {
    prior_exit=$?
    trap - EXIT HUP INT TERM
    cleanup_status=0
    if [[ "$status_emitted" -eq 0 ]]; then
        emit_status "$prior_exit"
    fi
    if [[ -n "$workdir" && -e "$workdir" ]]; then
        rm -rf -- "$workdir" || cleanup_status=1
    fi
    if [[ -n "$workdir" && -e "$workdir" ]]; then
        cleanup_status=1
    fi
    if [[ "$cleanup_status" -eq 0 ]]; then
        printf '%s\n' 'L04_CLEANUP=PASS'
    else
        printf '%s\n' 'L04_CLEANUP=FAIL' >&2
    fi
    if [[ "$prior_exit" -ne 0 ]]; then
        exit "$prior_exit"
    fi
    exit "$cleanup_status"
}
trap cleanup EXIT HUP INT TERM

workdir=$(mktemp -d /tmp/latent-anything-l04.XXXXXX)
if [[ ! "$workdir" =~ ^/tmp/latent-anything-l04\.[[:alnum:]]{6}$ ]] || [[ ! -d "$workdir" || -L "$workdir" ]]; then
    emit_status INVALID_WORKDIR >&2
    exit 70
fi
printf 'L04_WORKDIR=%s\n' "$workdir"
repo_dir="$workdir/repo"
cache_root="$workdir/cache"
preflight_file="$workdir/preflight.py"
before_members_file="$workdir/before-members.nul"
mkdir -p "$cache_root/uv" "$cache_root/huggingface" "$cache_root/datasets" "$cache_root/transformers"

git clone --no-checkout --quiet "$RepoUrl" "$repo_dir"
git -C "$repo_dir" checkout --quiet --detach "$CodeSha"
checked_out_sha=$(git -C "$repo_dir" rev-parse HEAD)
if [[ "${checked_out_sha,,}" != "${CodeSha,,}" ]]; then
    emit_status CODE_SHA_MISMATCH >&2
    exit 65
fi

export UV_CACHE_DIR="$cache_root/uv"
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/datasets"
export TRANSFORMERS_CACHE="$cache_root/transformers"
export LATENT_ANYTHING_RUN_NETWORK=1
export LATENT_ANYTHING_NETWORK_DEVICE=cuda
cd "$repo_dir"
printf 'L04_USE_CASE=%s\n' "$UseCase"
printf 'L04_CODE_SHA=%s\n' "${CodeSha,,}"
nvidia-smi

cat > "$preflight_file" <<'PY'
import importlib

EXPECTED = {
    "datasets": "4.8.5",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "huggingface_hub": "0.35.3",
}

for name, expected in EXPECTED.items():
    module = importlib.import_module(name)
    actual = getattr(module, "__version__", "")
    if actual != expected:
        raise RuntimeError(f"{name} version {actual!r} != {expected!r}")

torch = importlib.import_module("torch")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
PY

if uv run --locked --extra transformers \
    --with 'datasets==4.8.5' \
    --with 'transformers==4.57.6' \
    --with 'tokenizers==0.22.2' \
    --with 'huggingface-hub==0.35.3' \
    python "$preflight_file"; then
    preflight_status=0
else
    preflight_status=$?
fi
if [[ "$preflight_status" -ne 0 ]]; then
    emit_status "DEPENDENCY_PREFLIGHT_FAILED:$preflight_status" >&2
    exit "$preflight_status"
fi

find artifacts/m14 -maxdepth 1 -type f \
    -name "l04-explanations.${UseCase}.attempt*.json" -printf '%f\0' |
    sort -z > "$before_members_file"

if uv run --locked --extra transformers \
    --with 'datasets==4.8.5' \
    --with 'transformers==4.57.6' \
    --with 'tokenizers==0.22.2' \
    --with 'huggingface-hub==0.35.3' \
    python -m scripts.m14_l04_explanations \
    --run-real \
    --use-case "$UseCase" \
    --plan artifacts/m14/l04-explanations.plan.json \
    --fixture artifacts/m14/l04-prompt-factor-fixture.jsonl; then
    cli_status=0
else
    cli_status=$?
fi
printf 'L04_CLI_STATUS=%s\n' "$cli_status"

after_members_file="$workdir/after-members.nul"
find artifacts/m14 -maxdepth 1 -type f \
    -name "l04-explanations.${UseCase}.attempt*.json" -printf '%f\0' |
    sort -z > "$after_members_file"
mapfile -d '' new_members < <(comm -z -13 "$before_members_file" "$after_members_file")
if [[ "${#new_members[@]}" -ne 3 ]]; then
    bundle_gate_failure BUNDLE_INPUTS_MISSING "$cli_status"
fi

attempt=''
for member in "${new_members[@]}"; do
    if [[ ! "$member" =~ ^l04-explanations\.${UseCase}\.(attempt[0-9]+)\.(partial|run|failure)\.json$ ]]; then
        bundle_gate_failure BUNDLE_INPUTS_INVALID "$cli_status"
    fi
    if [[ -z "$attempt" ]]; then
        attempt="${BASH_REMATCH[1]}"
    elif [[ "$attempt" != "${BASH_REMATCH[1]}" ]]; then
        bundle_gate_failure BUNDLE_INPUTS_MIXED_ATTEMPTS "$cli_status"
    fi
    candidate="$repo_dir/artifacts/m14/$member"
    if [[ ! -f "$candidate" || -L "$candidate" || "$member" == */* || "$member" == *..* ]]; then
        bundle_gate_failure BUNDLE_INPUTS_INVALID "$cli_status"
    fi
done
for suffix in partial run failure; do
    expected="l04-explanations.${UseCase}.${attempt}.${suffix}.json"
    if [[ ! " ${new_members[*]} " == *" $expected "* ]]; then
        bundle_gate_failure BUNDLE_INPUTS_INCOMPLETE "$cli_status"
    fi
done

members_file="$workdir/members.nul"
for member in "${new_members[@]}"; do
    printf 'artifacts/m14/%s\0' "$member"
done | sort -z > "$members_file"
bundle_file="$workdir/l04-capture.tgz"
set +e
tar --null -czf "$bundle_file" -C "$repo_dir" --files-from="$members_file"
bundle_status=$?
set -e
printf 'L04_BUNDLE_STATUS=%s\n' "$bundle_status"
if [[ "$cli_status" -ne 0 ]]; then
    final_status="$cli_status"
elif [[ "$bundle_status" -ne 0 ]]; then
    final_status="$bundle_status"
else
    final_status=0
fi
emit_status "$final_status"
if [[ "$bundle_status" -eq 0 ]]; then
    bundle_bytes=$(wc -c < "$bundle_file")
    bundle_sha256=$(sha256sum "$bundle_file" | awk '{print $1}')
    printf 'L04_BUNDLE_BYTES=%s\n' "$bundle_bytes"
    printf 'L04_BUNDLE_SHA256=%s\n' "$bundle_sha256"
    while IFS= read -r -d '' member; do
        candidate="$repo_dir/$member"
        member_bytes=$(wc -c < "$candidate")
        member_sha256=$(sha256sum "$candidate" | awk '{print $1}')
        printf 'L04_BUNDLE_MEMBER=%s|%s|%s\n' "$member" "$member_bytes" "$member_sha256"
    done < "$members_file"
    printf '%s\n' L04_BUNDLE_B64_BEGIN
    base64 -w0 "$bundle_file"
    printf '\n%s\n' L04_BUNDLE_B64_END
fi
exit "$final_status"
