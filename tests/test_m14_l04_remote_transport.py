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


def _ssh_path() -> str:
    executable = shutil.which("ssh.exe") or shutil.which("ssh")
    if executable is None:
        pytest.skip("ssh.exe is required to validate the explicit executable parameter")
    return executable


def _build_only(payload_path: Path, raw_capture_path: Path) -> dict[str, object]:
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
        str(payload_path),
        "-UseCase",
        "Disentanglement",
        "-CodeSha",
        "a" * 40,
        "-RepoUrl",
        "https://github.com/example/repo.git",
        "-RawCapturePath",
        str(raw_capture_path),
        "-BuildOnly",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


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
    serialized = json.dumps(manifest)
    for value in (str(payload_path), str(raw_capture_path), "example.com", "github.com/example"):
        assert value not in serialized
    assert base64.b64decode(base64.b64encode(normalized)) == normalized


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
    assert "nvidia-smi" in payload
    assert "L04_BUNDLE_B64_BEGIN" in payload
    assert "L04_BUNDLE_B64_END" in payload
    assert "L04_CLEANUP=PASS" in payload
    assert "python -c" not in payload
    assert "ssh " not in payload


FAKE_SSH = r"""
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

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
if os.environ.get("FAKE_MODE") == "mismatch":
    print("L04_TRANSPORT_DECODE_MATCH=FAIL")
    print("L04_PAYLOAD_EXECUTED=FAIL")
    sys.exit(65)
if actual != announced:
    print("L04_TRANSPORT_DECODE_MATCH=FAIL")
    sys.exit(65)
print("L04_TRANSPORT_DECODE_MATCH=PASS")
if os.environ.get("FAKE_MODE") == "early":
    print("FAKE_EARLY_EXIT=PASS")
    print("fake early exit", file=sys.stderr)
    sys.exit(7)
if os.environ.get("FAKE_MODE") == "broken":
    os._exit(9)
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
    assert captured["argv"][1:] == [
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
    assert "--null --files-from" in payload
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


def _fake_bootstrap() -> bytes:
    payload = b"fake payload\n"
    digest = hashlib.sha256(payload).hexdigest()
    blob = base64.b64encode(payload).decode("ascii")
    return (f"PayloadSha256='{digest}'\nbase64 -d <<'L04_PAYLOAD_B64'\n{blob}\nL04_PAYLOAD_B64\n").encode()


def _fake_command(tmp_path: Path, mode: str = "success") -> tuple[list[str], Path, Path, dict[str, str]]:
    fake_ssh = tmp_path / "fake_ssh.py"
    fake_ssh.write_text(FAKE_SSH, encoding="utf-8")
    capture_path = tmp_path / "fake-capture.json"
    bootstrap_path = tmp_path / "bootstrap.bin"
    bootstrap_path.write_bytes(_fake_bootstrap())
    raw_path = tmp_path / f"raw-{mode}.capture"
    driver = tmp_path / "invoke-seam.ps1"
    driver.write_text(
        f"""
Import-Module '{SEAM}'
$bootstrap = [System.IO.File]::ReadAllBytes($env:L04_BOOTSTRAP_PATH)
$arguments = @($env:L04_FAKE_TARGET, 'bash', '-s', '--', 'Disentanglement', ('c' * 40), 'https://github.com/example/repo.git')
$result = Invoke-L04TransportProcess `
    -SshExecutable $env:L04_SSH_EXECUTABLE `
    -ArgumentList $arguments `
    -BootstrapBytes $bootstrap `
    -RawCapturePath $env:L04_RAW_PATH
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
        }
    )
    command = [_pwsh(), "-NoProfile", "-File", str(driver)]
    return command, capture_path, raw_path, env
