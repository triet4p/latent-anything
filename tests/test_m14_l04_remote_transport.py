"""Offline contract tests for the reusable L04 transport boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
HELPER = ROOT / "scripts" / "m14_l04_remote_transport.ps1"
PAYLOAD = ROOT / "scripts" / "m14_l04_remote_payload.sh"
SEAM = ROOT / "scripts" / "_m14_l04_transport_seam.psm1"


def _pwsh() -> str:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is required for the transport build-only contract")
    return executable


def _native_windows_powershell() -> str:
    executable = (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not executable.is_file():
        pytest.skip("native Windows PowerShell 5.1 is required for this compatibility regression")
    return str(executable)


def _ssh_path() -> str:
    executable = shutil.which("ssh.exe") or shutil.which("ssh")
    if executable is None:
        pytest.skip("ssh.exe is required to validate the explicit executable parameter")
    return executable


def _build_only(
    payload_path: Path,
    raw_capture_path: Path,
    timeout_seconds: int | None = None,
    executable: str | None = None,
) -> dict[str, object]:
    command = [
        _pwsh() if executable is None else executable,
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshPath",
        _ssh_path(),
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(payload_path),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "a" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(raw_capture_path),
        *([] if timeout_seconds is None else ["-TransportTimeoutSeconds", str(timeout_seconds)]),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def _build_only_v2(
    tmp_path: Path,
    *,
    use_case: str = "L049V2StageB",
    train: str = "C:/owner/train.jsonl",
    holdout: str = "C:/owner/holdout.jsonl",
    seed: str = "C:/owner/holdout.seed",
    candidate: str = "C:/owner/candidate.json",
    output: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _pwsh(),
            "-NoProfile",
            "-File",
            str(HELPER),
            "-SshPath",
            _ssh_path(),
            "-RemoteTarget",
            "user@example.com",
            "-PayloadPath",
            str(PAYLOAD),
            "-UseCase",
            use_case,
            "-CodeSha",
            "a" * 40,
            "-RepoUrl",
            "https://github.com/example/repo.git",
            "-RawCapturePath",
            str(tmp_path / "raw.capture"),
            "-V2TrainFixturePath",
            train,
            "-V2HoldoutFixturePath",
            holdout,
            "-V2HoldoutSeedPath",
            seed,
            "-V2CandidateManifestPath",
            candidate,
            "-V2OutputPath",
            output,
            "-BuildOnly",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_v2_stage_b_build_manifest_requires_and_redacts_owner_paths(tmp_path: Path) -> None:
    result = _build_only_v2(tmp_path)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["v2_inputs"] == {
        "train_fixture": "<owner-provisioned-path>",
        "holdout_fixture": "<owner-provisioned-path>",
        "holdout_seed": "<owner-provisioned-path>",
        "candidate_manifest": "<owner-provisioned-path>",
        "output": "<fresh-clone>/artifacts/m14/l04-l049-v2-stage-b.json",
        "contents": "redacted",
    }
    serialized = json.dumps(manifest)
    for value in ("C:/owner/train.jsonl", "C:/owner/holdout.jsonl", "C:/owner/holdout.seed", "C:/owner/candidate.json"):
        assert value not in serialized


def test_v2_stage_a_build_manifest_derives_clone_output(tmp_path: Path) -> None:
    result = _build_only_v2(tmp_path, use_case="L049V2StageA", holdout="", seed="", candidate="")
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["v2_inputs"]["output"] == "<fresh-clone>/artifacts/m14/l04-l049-v2-stage-a.json"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train": ""},
        {"holdout": ""},
        {"seed": ""},
        {"candidate": ""},
    ],
)
def test_v2_stage_b_build_manifest_rejects_missing_owner_path(tmp_path: Path, kwargs: dict[str, str]) -> None:
    result = _build_only_v2(tmp_path, **kwargs)
    assert result.returncode != 0
    assert "requires" in result.stderr


def test_v2_stage_a_rejects_stage_b_only_paths(tmp_path: Path) -> None:
    result = _build_only_v2(tmp_path, use_case="L049V2StageA", holdout="C:/owner/holdout.jsonl")
    assert result.returncode != 0
    assert "rejects Stage B-only" in result.stderr


def test_v2_output_path_cannot_be_overridden(tmp_path: Path) -> None:
    result = _build_only_v2(tmp_path, output="/tmp/outside.json")
    assert result.returncode != 0
    assert "cannot be overridden" in result.stderr


def test_build_only_normalizes_crlf_and_redacts_operational_values(tmp_path: Path) -> None:
    source = b"\xef\xbb\xbf#!/usr/bin/env bash\r\n# UTF-8: cafe\xc3\xa9\r\nexit 0\r"
    payload_path = tmp_path / "payload.sh"
    payload_path.write_bytes(source)
    raw_capture_path = tmp_path / "raw.capture"

    manifest = _build_only(payload_path, raw_capture_path)
    normalized = source.decode("utf-8")[1:].replace("\r\n", "\n").replace("\r", "\n").encode()
    payload_metadata = manifest["payload"]
    assert payload_metadata == {
        "sha256": hashlib.sha256(normalized).hexdigest(),
        "bytes": len(normalized),
    }
    assert manifest["mode"] == "build-only"
    assert manifest["use_case"] == "Disentanglement"
    assert manifest["secrets_redacted"] is True
    assert manifest["raw_capture_path_redacted"] == "<raw-capture-path>"
    assert manifest["transport_timeout_seconds"] == 3600
    assert manifest["ssh_connect_timeout_seconds"] == 15
    assert manifest["ssh_connection_attempts"] == 1
    assert manifest["ssh_batch_mode"] is True
    assert manifest["command_args_redacted"] == [
        "<ssh.exe>",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=1",
        "<remote-target>",
        "bash",
        "-s",
        "--",
        "<use-case>",
        "<code-sha>",
        "<repo-url>",
    ]
    assert manifest["kill_grace_seconds"] == 30
    serialized = json.dumps(manifest)
    for value in (str(payload_path), str(raw_capture_path), "example.com", "github.com/example"):
        assert value not in serialized
    assert base64.b64decode(base64.b64encode(normalized)) == normalized


def test_build_only_runs_under_native_windows_powershell_51(tmp_path: Path) -> None:
    manifest = _build_only(
        PAYLOAD,
        tmp_path / "raw.capture",
        executable=_native_windows_powershell(),
    )
    assert manifest["mode"] == "build-only"
    payload = manifest["payload"]
    assert isinstance(payload, dict)
    assert payload["bytes"] == len(PAYLOAD.read_bytes())


def test_build_only_accepts_dry_run_alias(tmp_path: Path) -> None:
    raw_capture_path = tmp_path / "raw.capture"
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshPath",
        _ssh_path(),
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "b" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(raw_capture_path),
        "-DryRun",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "build-only"


@pytest.mark.parametrize("timeout_seconds", [2400, 7200])
def test_build_only_accepts_transport_timeout_bounds(tmp_path: Path, timeout_seconds: int) -> None:
    manifest = _build_only(PAYLOAD, tmp_path / "raw.capture", timeout_seconds)
    assert manifest["transport_timeout_seconds"] == timeout_seconds


@pytest.mark.parametrize("timeout_seconds", [2399, 7201])
def test_transport_timeout_bounds_are_rejected(tmp_path: Path, timeout_seconds: int) -> None:
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshPath",
        _ssh_path(),
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "a" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(tmp_path / "raw.capture"),
        "-TransportTimeoutSeconds",
        str(timeout_seconds),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0


@pytest.mark.parametrize("timeout_seconds", [1, 300])
def test_build_only_accepts_ssh_connect_timeout_bounds(tmp_path: Path, timeout_seconds: int) -> None:
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshPath",
        _ssh_path(),
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "a" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(tmp_path / "raw.capture"),
        "-SshConnectTimeoutSeconds",
        str(timeout_seconds),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ssh_connect_timeout_seconds"] == timeout_seconds


@pytest.mark.parametrize("timeout_seconds", [0, 301])
def test_ssh_connect_timeout_bounds_are_rejected(tmp_path: Path, timeout_seconds: int) -> None:
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshPath",
        _ssh_path(),
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "a" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(tmp_path / "raw.capture"),
        "-SshConnectTimeoutSeconds",
        str(timeout_seconds),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0


def test_invalid_parameters_are_rejected_before_build(tmp_path: Path) -> None:
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshPath",
        _ssh_path(),
        "-RemoteTarget",
        "user;touch /tmp/pwned",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "not-a-sha",
        "-RepoUrl",
        "https://user:secret@example.com/repo.git",
        "-RawCapturePath",
        str(tmp_path / "raw.capture"),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "secret" not in result.stdout + result.stderr


def test_transport_and_payload_static_contracts() -> None:
    helper = HELPER.read_text(encoding="utf-8")
    seam = SEAM.read_text(encoding="utf-8")
    payload = PAYLOAD.read_text(encoding="utf-8")
    assert "ProcessStartInfo" not in helper
    assert "ProcessStartInfo" in seam
    assert "UseShellExecute = $false" in seam
    assert "StandardInput.BaseStream.Write" in seam
    assert "ReadToEndAsync" in seam
    assert "Stopwatch]::GetTimestamp" in seam
    assert "GetMethods() | Where-Object" in seam
    assert "$process.Kill()" in seam
    assert "DisposeAsync" not in seam
    assert "HashData" not in seam
    assert "File]::Replace($temporaryPath, $fullPath, $backupPath, $true)" in seam
    assert "WaitForExit($killWaitMilliseconds)" in seam
    assert "transport_termination_incomplete" in seam
    assert "TimeoutSeconds" in seam
    assert '"BatchMode=yes"' in helper
    assert '"ConnectTimeout=$SshConnectTimeoutSeconds"' in helper
    assert '"ConnectionAttempts=1"' in helper
    assert "<<'L04_PAYLOAD_B64'" in helper
    assert "base64 -d" in helper
    assert "L04_TRANSPORT_DECODE_STATUS" in helper
    assert "L04_TRANSPORT_DECODE_MATCH=FAIL" in helper
    assert "L04_TRANSPORT_CLEANUP=PASS" in helper
    assert seam.index("Write-L04RawCapture") < seam.index("raw_capture_written_before_parse")
    assert "Start-Process" not in helper
    assert "Invoke-Expression" not in helper
    assert "wsl" not in helper.lower()
    assert "git-bash" not in helper.lower()
    assert payload.count("python -m scripts.m14_l04_explanations") == 1
    assert "git clone --no-checkout" in payload
    assert "checkout --quiet --detach" in payload
    assert "UV_CACHE_DIR" in payload
    assert "HF_DATASETS_CACHE" in payload
    assert "nvidia-smi >&2" in payload
    assert "--fixture artifacts/m14/l04-prompt-factor-fixture.jsonl >&2; then" in payload
    assert "L04_BUNDLE_B64_BEGIN" in payload
    assert "L04_BUNDLE_B64_END" in payload
    assert "L04_CLEANUP=PASS" in payload
    assert "L04_WORKDIR=%s" in payload
    assert "expected_markers" in helper
    assert "python -c" not in payload
    assert "ssh " not in payload


FAKE_SSH = r"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

mode = os.environ.get("FAKE_MODE", "success")
if mode == "never-read":
    time.sleep(60)
    raise SystemExit(0)
stdin_bytes = sys.stdin.buffer.read()
capture_path = Path(os.environ["FAKE_CAPTURE_PATH"])
capture_path.write_text(
    json.dumps({"argv": sys.argv, "stdin_b64": base64.b64encode(stdin_bytes).decode()}),
    encoding="utf-8",
)
text = stdin_bytes.decode("utf-8")
assert not text.startswith("\ufeff")
assert "\r" not in text
announced = text.split("PayloadSha256='", 1)[1].split("'", 1)[0]
blob = text.split("<<'L04_PAYLOAD_B64'\n", 1)[1].split("\nL04_PAYLOAD_B64", 1)[0]
decoded = base64.b64decode(blob, validate=True)
actual = hashlib.sha256(decoded).hexdigest()
print("L04_TRANSPORT_DECODE_STATUS=0")
print(f"L04_TRANSPORT_DECODE_SHA256={actual}")
if mode == "mismatch":
    print("L04_TRANSPORT_DECODE_MATCH=FAIL")
    print("L04_PAYLOAD_EXECUTED=FAIL")
    sys.exit(65)
if actual != announced:
    print("L04_TRANSPORT_DECODE_MATCH=FAIL")
    sys.exit(65)
print("L04_TRANSPORT_DECODE_MATCH=PASS")
if mode == "early":
    print("FAKE_EARLY_EXIT=PASS")
    print("fake early exit", file=sys.stderr)
    sys.exit(7)
if mode == "broken":
    os._exit(9)
if mode == "child":
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; print('CHILD_STDOUT', flush=True); "
                "print('CHILD_STDERR', file=sys.stderr, flush=True); time.sleep(60)"
            ),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    time.sleep(60)
if mode == "hang":
    time.sleep(60)
print("L04_PAYLOAD_EXECUTED=PASS")
print("L04_TRANSPORT_CLEANUP=PASS")
"""


def test_real_processstartinfo_fake_ssh_success_and_exact_stdin(tmp_path: Path) -> None:
    command, capture_path, raw_path, env = _fake_command(tmp_path)
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    stdin_bytes = base64.b64decode(captured["stdin_b64"])
    assert not stdin_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in stdin_bytes
    assert hashlib.sha256(stdin_bytes).hexdigest() == hashlib.sha256(_fake_bootstrap()).hexdigest()
    assert captured["argv"][0] == str(Path(env["L04_FAKE_TARGET"]))
    assert captured["argv"][1:] == [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=1",
        "user@example.com",
        "bash",
        "-s",
        "--",
        "Disentanglement",
        "c" * 40,
        "https://github.com/example/repo.git",
    ]
    raw = raw_path.read_text(encoding="utf-8")
    assert "L04_PAYLOAD_EXECUTED=PASS" in raw
    assert "L04_TRANSPORT_CLEANUP=PASS" in raw
    assert json.loads(result.stdout)["raw_capture_written_before_parse"] is True


def test_active_l04_runbooks_derive_repo_url_from_origin() -> None:
    validation = (ROOT / "docs" / "M14_REAL_SYSTEM_VALIDATION.md").read_text(encoding="utf-8")
    gap_plan = (ROOT / "docs" / "EVIDENCE_GAP_PLAN.md").read_text(encoding="utf-8")
    stale_url = "https://github.com/trietlm/latent-anything.git"
    for document in (validation, gap_plan):
        assert "$RepoUrl = (git remote get-url origin).Trim()" in document
        assert "-RepoUrl $RepoUrl" in document
        assert stale_url not in document


@pytest.mark.parametrize("mode", ["mismatch", "early", "broken"])
def test_fake_ssh_failures_retain_raw_capture(tmp_path: Path, mode: str) -> None:
    command, _capture_path, raw_path, env = _fake_command(tmp_path, mode)
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert raw_path.is_file()
    raw = raw_path.read_text(encoding="utf-8")
    assert "--- STDOUT BEGIN ---" in raw
    assert "--- STDERR BEGIN ---" in raw
    assert "L04_PAYLOAD_EXECUTED=PASS" not in raw
    report = json.loads(result.stdout)
    assert report["raw_capture_written_before_parse"] is True


def test_internal_seam_nonexistent_executable_still_writes_raw_capture(tmp_path: Path) -> None:
    command, _capture_path, raw_path, env = _fake_command(tmp_path)
    env["L04_SSH_EXECUTABLE"] = str(tmp_path / "does-not-exist.exe")
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 70
    assert raw_path.is_file()
    report = json.loads(result.stdout)
    assert report["raw_capture_written_before_parse"] is True
    assert report["exception_type"]


def test_native_windows_powershell_fake_seam_captures_raw_and_exit(tmp_path: Path) -> None:
    command, capture_path, raw_path, env = _fake_command(
        tmp_path,
        shell=_native_windows_powershell(),
        path_with_spaces=True,
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured["argv"][0] == str(Path(env["L04_FAKE_TARGET"]))
    assert " " in captured["argv"][0]
    assert raw_path.is_file()
    report = json.loads(result.stdout)
    assert report["ssh_exit"] == 0
    assert report["raw_capture_written_before_parse"] is True
    assert report["raw_capture_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("shell_factory", [_pwsh, _native_windows_powershell], ids=["pwsh", "native-winps"])
def test_sub_megabyte_no_read_timeout_is_bounded_and_retains_raw(
    tmp_path: Path,
    shell_factory: Callable[[], str],
) -> None:
    command, _capture_path, raw_path, env = _fake_command(
        tmp_path,
        "never-read",
        timeout_seconds=2,
        bootstrap_bytes=b"x" * (256 * 1024),
        shell=shell_factory(),
    )
    started = time.monotonic()
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - started
    assert elapsed < 40
    assert result.returncode == 70
    assert raw_path.is_file()
    report = json.loads(result.stdout)
    assert report["deadline_exceeded"] is True
    assert report["raw_capture_written_before_parse"] is True
    assert report["raw_capture_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert report["transport_errors"]
    assert not list(tmp_path.glob(f".{raw_path.name}.*.tmp*"))


def test_windows_argument_fallback_quotes_spaces_and_embedded_quotes(tmp_path: Path) -> None:
    probe = tmp_path / "argument probe.ps1"
    probe.write_text(
        f"""
Import-Module '{SEAM}'
$module = Get-Module -Name '_m14_l04_transport_seam'
$values = @('plain', 'with space', 'quote"inside', 'trailing\\')
$result = @()
foreach ($value in $values) {{
    $result += & $module {{ param($candidate) ConvertTo-L04WindowsArgument -Value $candidate }} $value
}}
$result | ConvertTo-Json -Compress
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [_native_windows_powershell(), "-NoProfile", "-File", str(probe)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["plain", '"with space"', '"quote\\"inside"', "trailing\\"]


def _powershell_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@pytest.mark.parametrize("shell_factory", [_pwsh, _native_windows_powershell], ids=["pwsh", "native-winps"])
def test_child_argv_roundtrip_handles_empty_quotes_backslashes_and_unicode(
    tmp_path: Path,
    shell_factory: Callable[[], str],
) -> None:
    child = tmp_path / "space dir" / "argv child.py"
    child.parent.mkdir()
    child.write_text(
        """
import json
import os
import sys
from pathlib import Path

sys.stdin.buffer.read()
Path(os.environ["L04_ARGV_CAPTURE"]).write_text(
    json.dumps(sys.argv[1:], ensure_ascii=False), encoding="utf-8"
)
print("ARGV_ROUNDTRIP=PASS")
""",
        encoding="utf-8-sig",
    )
    values = ["", "with space", 'quote"inside', r"backslash\"quote", "trailing\\", "unicode-你好-🙂"]
    driver = tmp_path / "argv roundtrip.ps1"
    argument_lines = ",\n    ".join(_powershell_single_quoted(value) for value in [str(child), *values])
    driver.write_text(
        f"""
Import-Module '{SEAM}'
$arguments = @(
    {argument_lines}
)
$result = Invoke-L04TransportProcess `
    -SshExecutable $env:L04_PYTHON `
    -ArgumentList $arguments `
    -BootstrapBytes ([byte[]](1, 2, 3)) `
    -RawCapturePath $env:L04_RAW_PATH `
    -TimeoutSeconds 30
$result | ConvertTo-Json -Compress
if ($result.transport_error -ne $null) {{ exit 70 }}
""",
        encoding="utf-8-sig",
    )
    capture = tmp_path / "argv capture.json"
    raw_path = tmp_path / "argv.raw"
    env = os.environ.copy()
    env.update(
        {
            "L04_ARGV_CAPTURE": str(capture),
            "L04_PYTHON": sys.executable,
            "L04_RAW_PATH": str(raw_path),
        }
    )
    result = subprocess.run(
        [shell_factory(), "-NoProfile", "-File", str(driver)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text(encoding="utf-8")) == values
    assert "ARGV_ROUNDTRIP=PASS" in raw_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["never-read", "hang", "child"])
def test_internal_seam_timeout_is_bounded_kills_tree_and_writes_raw(tmp_path: Path, mode: str) -> None:
    bootstrap = b"x" * (2 * 1024 * 1024) if mode == "never-read" else _fake_bootstrap()
    command, _capture_path, raw_path, env = _fake_command(
        tmp_path,
        mode,
        timeout_seconds=2,
        bootstrap_bytes=bootstrap,
    )
    started = time.monotonic()
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - started
    assert elapsed < 40
    assert result.returncode == 70
    assert raw_path.is_file()
    report = json.loads(result.stdout)
    assert report["deadline_exceeded"] is True
    assert report["raw_capture_written_before_parse"] is True
    assert report["cleanup_status"] in {"not_required", "unknown"}
    if mode == "child":
        assert report["transport_termination_incomplete"] is False


def test_raw_write_failure_never_reports_stale_target(tmp_path: Path) -> None:
    command, _capture_path, raw_path, env = _fake_command(tmp_path)
    raw_path.mkdir()
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 70
    report = json.loads(result.stdout)
    assert report["raw_capture_write_succeeded"] is False
    assert report["raw_capture_written_before_parse"] is False
    assert report["raw_capture_sha256"] is None
    assert raw_path.is_dir()


@pytest.mark.parametrize("bad_executable", [sys.executable, "fake_ssh.py"])
def test_production_entry_rejects_non_ssh_executable(tmp_path: Path, bad_executable: str) -> None:
    raw_path = tmp_path / "raw.capture"
    executable = bad_executable if Path(bad_executable).is_absolute() else str(tmp_path / bad_executable)
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshExecutable",
        executable,
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "d" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(raw_path),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert not raw_path.exists()


def test_production_entry_rejects_python_target(tmp_path: Path) -> None:
    fake_target = tmp_path / "fake_ssh.py"
    fake_target.write_text("", encoding="utf-8")
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshExecutable",
        _ssh_path(),
        "-RemoteTarget",
        str(fake_target),
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "d" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(tmp_path / "raw.capture"),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0


@pytest.mark.parametrize(
    "repo_url",
    [
        "https:///owner/repo.git",
        "https://user:secret@github.com/owner/repo.git",
        "https://github.com/owner/repo.git?x=1",
        "https://github.com/owner/repo.git#fragment",
        "https://gitlab.com/owner/repo.git",
        "https://github.com/owner/repo.git;touch",
    ],
)
def test_adversarial_repo_urls_are_rejected(tmp_path: Path, repo_url: str) -> None:
    command = [
        _pwsh(),
        "-NoProfile",
        "-File",
        str(HELPER),
        "-SshExecutable",
        _ssh_path(),
        "-RemoteTarget",
        "user@example.com",
        "-PayloadPath",
        str(PAYLOAD),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "e" * 40,
        "-RepoUrl",
        repo_url,
        "-RawCapturePath",
        str(tmp_path / "raw.capture"),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "secret" not in result.stdout + result.stderr


def test_bundle_contract_excludes_history_and_requires_exact_attempt_set() -> None:
    payload = PAYLOAD.read_text(encoding="utf-8")
    assert "before-members.nul" in payload
    assert "after-members.nul" in payload
    assert "comm -z -13" in payload
    assert "sort -z" in payload
    assert 'tar --null -czf "$bundle_file" -C "$repo_dir" --files-from="$members_file"' in payload
    assert "set +e" in payload
    assert "set -e" in payload
    assert "L04_CLI_STATUS" in payload
    assert "L04_BUNDLE_STATUS" in payload
    assert payload.index("L04_CLI_STATUS") < payload.index("after_members_file=")
    assert payload.index("L04_BUNDLE_STATUS") < payload.index("L04_BUNDLE_B64_BEGIN")
    assert "BUNDLE_INPUTS_MIXED_ATTEMPTS" in payload
    assert "BUNDLE_INPUTS_INCOMPLETE" in payload
    assert "l04-explanations.${UseCase}.attempt*.json" in payload
    assert "l04-prompt-factor-fixture" not in payload.split("after_members_file=", 1)[-1]

    before = {
        "l04-explanations.Disentanglement.attempt1.partial.json",
        "l04-explanations.Disentanglement.attempt1.run.json",
        "l04-explanations.Disentanglement.attempt1.failure.json",
    }
    after = before | {
        "l04-explanations.Disentanglement.attempt2.partial.json",
        "l04-explanations.Disentanglement.attempt2.run.json",
        "l04-explanations.Disentanglement.attempt2.failure.json",
        "l04-explanations.TCAV.attempt9.run.json",
    }
    new_members = sorted(after - before)
    current = re.compile(r"^l04-explanations\.Disentanglement\.(attempt[0-9]+)\.(partial|run|failure)\.json$")
    assert {member for member in new_members if current.fullmatch(member)} == {
        "l04-explanations.Disentanglement.attempt2.partial.json",
        "l04-explanations.Disentanglement.attempt2.run.json",
        "l04-explanations.Disentanglement.attempt2.failure.json",
    }
    assert not current.fullmatch("../l04-explanations.Disentanglement.attempt2.run.json")
    assert not current.fullmatch("l04-explanations.Disentanglement.attempt2.partial.jsonl")


@pytest.mark.parametrize(
    ("cli_status", "bundle_status", "expected_final"),
    ((1, 66, 1), (0, 66, 66)),
)
def test_bundle_gate_exit_precedence_is_explicit_and_marker_values_remain_distinct(
    cli_status: int, bundle_status: int, expected_final: int
) -> None:
    payload = PAYLOAD.read_text(encoding="utf-8")
    assert 'local cli_status="$2"' in payload
    assert "printf 'L04_BUNDLE_STATUS=66\\n'" in payload
    assert payload.count("bundle_gate_failure BUNDLE_INPUTS_") == 5
    assert all(
        '"$cli_status"' in payload[line_start : line_start + 100]
        for line_start in (match.start() for match in re.finditer(r"bundle_gate_failure BUNDLE_INPUTS_", payload))
    )
    final = cli_status if cli_status != 0 else bundle_status
    assert final == expected_final
    markers = f"L04_CLI_STATUS={cli_status}\nL04_BUNDLE_STATUS={bundle_status}\nL04_STATUS={final}\n"
    assert f"L04_CLI_STATUS={cli_status}" in markers
    assert f"L04_BUNDLE_STATUS={bundle_status}" in markers
    assert f"L04_CLI_STATUS={cli_status}" != f"L04_BUNDLE_STATUS={bundle_status}"


def test_workdir_contract_validates_normal_exact_path_before_marker() -> None:
    payload = PAYLOAD.read_text(encoding="utf-8")
    validation = re.compile(r"^/tmp/latent-anything-l04\.[A-Za-z0-9]{6}$")
    assert validation.fullmatch("/tmp/latent-anything-l04.A1b2C3")
    assert not validation.fullmatch("/tmp/latent-anything-l04.too-long")
    assert not validation.fullmatch("/tmp/latent-anything-l04.")
    assert not validation.fullmatch("/tmp/other.A1b2C3")
    validation_text = 'if [[ ! "$workdir" =~ ^/tmp/latent-anything-l04\\.[[:alnum:]]{6}$ ]]'
    assert validation_text in payload
    assert payload.index(validation_text) < payload.index("L04_WORKDIR=%s")
    assert "latent-anything-l04.*" not in payload
    assert '[[ ! -d "$workdir" || -L "$workdir" ]]' in payload


def test_raw_publication_state_preserves_first_success_on_later_failure() -> None:
    seam = SEAM.read_text(encoding="utf-8")
    assert "if ($state.raw_capture_write_succeeded) { return }" in seam
    assert "raw_capture_finalization_error" in seam

    state: dict[str, object] = {
        "succeeded": False,
        "path": None,
        "digest": None,
        "finalization_error": None,
    }

    def publish(writer: Callable[[], str], phase: str) -> None:
        if state["succeeded"]:
            return
        try:
            digest = writer()
            state["succeeded"] = True
            state["path"] = "<raw-capture-path>"
            state["digest"] = digest
        except OSError as error:
            if phase == "finalization":
                state["finalization_error"] = type(error).__name__

    publish(lambda: "first-current-digest", "timeout")
    publish(lambda: (_ for _ in ()).throw(OSError("injected")), "finalization")
    assert state == {
        "succeeded": True,
        "path": "<raw-capture-path>",
        "digest": "first-current-digest",
        "finalization_error": None,
    }


def _fake_bootstrap() -> bytes:
    payload = b"fake payload\n"
    digest = hashlib.sha256(payload).hexdigest()
    blob = base64.b64encode(payload).decode("ascii")
    return (f"PayloadSha256='{digest}'\nbase64 -d <<'L04_PAYLOAD_B64'\n{blob}\nL04_PAYLOAD_B64\n").encode()


def _fake_command(
    tmp_path: Path,
    mode: str = "success",
    *,
    timeout_seconds: int = 3600,
    bootstrap_bytes: bytes | None = None,
    shell: str | None = None,
    path_with_spaces: bool = False,
) -> tuple[list[str], Path, Path, dict[str, str]]:
    fake_root = tmp_path / "space dir" if path_with_spaces else tmp_path
    fake_root.mkdir(exist_ok=True)
    fake_ssh = fake_root / ("fake ssh.py" if path_with_spaces else "fake_ssh.py")
    fake_ssh.write_text(FAKE_SSH, encoding="utf-8")
    capture_path = tmp_path / "fake-capture.json"
    bootstrap_path = tmp_path / "bootstrap.bin"
    bootstrap_path.write_bytes(_fake_bootstrap() if bootstrap_bytes is None else bootstrap_bytes)
    raw_path = tmp_path / f"raw-{mode}.capture"
    driver = tmp_path / "invoke-seam.ps1"
    driver.write_text(
        f"""
Import-Module '{SEAM}'
$bootstrap = [System.IO.File]::ReadAllBytes($env:L04_BOOTSTRAP_PATH)
$arguments = @(
    $env:L04_FAKE_TARGET, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'ConnectionAttempts=1', 'user@example.com', 'bash', '-s', '--',
    'Disentanglement', ('c' * 40), 'https://github.com/example/repo.git'
)
$result = Invoke-L04TransportProcess `
    -SshExecutable $env:L04_SSH_EXECUTABLE `
    -ArgumentList $arguments `
    -BootstrapBytes $bootstrap `
    -RawCapturePath $env:L04_RAW_PATH `
    -TimeoutSeconds ([int]$env:L04_TIMEOUT_SECONDS)
$result | ConvertTo-Json -Compress
if ($result.transport_error -ne $null) {{ exit 70 }}
if ($result.ssh_exit -eq $null) {{ exit 70 }}
exit $result.ssh_exit
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "L04_BOOTSTRAP_PATH": str(bootstrap_path),
            "L04_SSH_EXECUTABLE": sys.executable,
            "L04_FAKE_TARGET": str(fake_ssh),
            "L04_RAW_PATH": str(raw_path),
            "FAKE_CAPTURE_PATH": str(capture_path),
            "FAKE_MODE": mode,
            "L04_TIMEOUT_SECONDS": str(timeout_seconds),
        }
    )
    command = [shell or _pwsh(), "-NoProfile", "-File", str(driver)]
    return command, capture_path, raw_path, env
