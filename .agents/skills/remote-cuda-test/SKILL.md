---
name: remote-cuda-test
description: "Run committed project tests on a remote CUDA server through Bash/SSH. Use this skill whenever the user asks to test remotely, validate a GPU/CUDA environment, run a server-side integration test, or reproduce a local test on a CUDA machine. Always push the local commit to GitHub first, clone that GitHub revision into an isolated temporary directory on the server, run the requested tests there, collect the report, and remove the temporary checkout and caches before returning. Never edit an existing server checkout or run the connection flow directly in PowerShell."
compatibility: "Bash, OpenSSH, Git, a GitHub remote, and uv on the remote host; the remote user needs a CUDA driver and a project-compatible compiler/toolchain."
---

# Remote CUDA Test

Use this workflow when a local project must be validated on a remote CUDA host. The source of truth is GitHub: the server is only a disposable test runner. This prevents an uncommitted local change or an accidental server-side edit from being mistaken for a verified result.

## Non-negotiable invariants

- Run the connection workflow from Bash, WSL, Git Bash, or another POSIX shell. Do not execute the SSH/clone/test workflow directly in PowerShell.
- Require a clean local worktree. Commit local changes with a Conventional Commit before pushing.
- Push the exact local commit to GitHub before connecting to the server.
- Clone from GitHub into a new server-side temporary directory. Do not reuse, pull into, or modify an existing checkout.
- Verify the remote clone's `HEAD` equals the pushed commit SHA before installing or testing.
- Put uv, CUDA-extension, and test caches inside the temporary directory where practical.
- Always remove the temporary clone and its generated outputs with a shell `trap`, including on failure.
- Never copy generated artifacts back into the server's persistent source tree. If an artifact is needed locally, transfer only the report after the test and record its provenance.

## Required inputs

Collect or infer:

- `REMOTE_USER@REMOTE_HOST`, for example `trietlm@192.168.30.244`.
- The GitHub `origin` URL and branch/ref to push and clone.
- A remote test command. Prefer a project test marker or a focused GPU integration test over an ad-hoc script.
- Optional extras, normally `3d` and, when the full suite needs it, `viz`.

If the local worktree is dirty, stop and report the files. The caller must decide what to commit; do not silently stage unrelated files.

## Recommended execution

Run the bundled POSIX script from Bash:

```bash
./.agents/skills/remote-cuda-test/scripts/remote_cuda_test.sh \
  --remote trietlm@192.168.30.244 \
  --ref main \
  --extra 3d \
  --extra viz \
  --test-command 'uv run pytest -q -m "not network and not large_download"' \
  --gpu-command 'uv run pytest tests/test_latent_anything/test_gaussian_3d_renderer_gpu.py -v'
```

The script pushes `HEAD` to the selected GitHub branch, resolves the branch SHA, and passes the SHA to the remote runner. It then:

1. Creates a unique temporary directory with `mktemp`.
2. Clones the selected branch from GitHub into that directory.
3. Verifies the clone SHA.
4. Runs `uv sync --extra ...` inside the clone.
5. Checks `nvidia-smi`, PyTorch CUDA availability, device name, and (when available) CUDA toolkit/compiler versions.
6. Selects `gcc-12`/`g++-12` for CUDA 12.1 when those compilers already exist; otherwise stops with an actionable toolchain message. Do not install system packages automatically unless the user explicitly authorizes it.
7. Runs the GPU command first, then the requested broader test command.
8. Prints a machine-readable summary containing commit SHA, host, GPU, CUDA/PyTorch status, commands, pass/fail status, and duration.
9. Deletes the temporary checkout, uv cache, and Torch extension cache through the exit trap.

For a one-off manual flow, the equivalent Bash sequence is:

```bash
set -Eeuo pipefail
git status --short                         # must be empty
git push origin HEAD:refs/heads/main
sha="$(git rev-parse HEAD)"

ssh trietlm@192.168.30.244 'bash -s' -- \
  "https://github.com/OWNER/REPO.git" main "$sha" <<'REMOTE'
set -Eeuo pipefail
repo_url="$1"; ref="$2"; expected_sha="$3"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/remote-cuda-test.XXXXXX")"
cleanup() { rm -rf -- "$tmp_dir"; }
trap cleanup EXIT INT TERM
export UV_CACHE_DIR="$tmp_dir/uv-cache"
export TORCH_EXTENSIONS_DIR="$tmp_dir/torch-extensions"
git clone --depth 1 --branch "$ref" "$repo_url" "$tmp_dir/repo"
cd "$tmp_dir/repo"
test "$(git rev-parse HEAD)" = "$expected_sha"
uv sync --extra 3d
nvidia-smi
uv run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
uv run pytest -m 'not network and not large_download' -q
REMOTE
```

## Toolchain and failure handling

GPU presence is not enough. A real CUDA render can fail while `nvidia-smi` and `torch.cuda.is_available()` pass because a package such as gsplat JIT-compiles CUDA code. Report these separately:

- Driver/GPU failure: `nvidia-smi` fails.
- Python CUDA failure: torch is absent or `torch.cuda.is_available()` is false.
- Extension build failure: CUDA toolkit/compiler incompatibility, missing `gcc-12`, or an API mismatch.
- Test failure: the project test itself fails after the environment is healthy.

For CUDA 12.1, GCC 13 may be rejected by nvcc. Prefer an already-installed `gcc-12`/`g++-12` pair via `CC`, `CXX`, and `CUDAHOSTCXX`. If it is absent, stop and ask before changing system packages. Do not hide the error with `-allow-unsupported-compiler`; that can turn a clear toolchain mismatch into invalid binaries.

If the server already contains a checkout, leave it unchanged. Use the disposable clone even when it is slower. If cleanup fails, report the exact temporary path so the user can remove only that path after inspection.

## Report format

Return:

```text
Remote CUDA test
- Source: <GitHub URL>@<SHA>
- Server: <user>@<host> (<hostname>)
- GPU/CUDA: <device and status>
- Environment: <torch, gsplat, compiler summary>
- GPU test: PASS/FAIL/SKIPPED (<command>)
- Broader tests: PASS/FAIL/SKIPPED (<command and counts>)
- Cleanup: PASS/FAIL (<temporary path only if cleanup failed>)
```

Never claim the server validated a commit unless the clone SHA was printed and matched the pushed local SHA.

## Bundled resources

- `scripts/remote_cuda_test.sh` — deterministic Bash runner.
- `evals/evals.json` — trigger and workflow test prompts for this skill.
