"""Synthetic-only tests for L04 remote evidence retention."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import _m14_l04_retention_transaction as transaction
from scripts import m14_l04_remote_postprocess as postprocess
from scripts._m14_l04_digest import canonical_digest, code_sha
from scripts._m14_l04_retention_transaction import write_atomic
from scripts._m14_l049_v2_fixture import TRAIN_FIXTURE_PATH, read_rows
from scripts._m14_l049_v2_schema import V2_ADDENDUM_PATH, canonical_json_bytes, top_level_cli_sha256
from scripts._m14_l049_v2_stage_a import build_stage_a_artifact, run_real_stage_a
from scripts.m14_l04_explanations import run_real

USE_CASE = "Disentanglement"
ROOT = Path(__file__).resolve().parents[1]


def _write_v2_stage_a_triad(directory: Path, artifact: dict[str, object], source_sha: str) -> None:
    common = {
        "schema_version": "m14-l04.9-v2-transport-envelope-v1",
        "use_case": "L049V2StageA",
        "attempt": "attempt1",
        "source_commit_sha": source_sha,
        "stage": artifact["stage"],
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
    }
    values = {
        "partial": {**common, "kind": "partial", "artifact": artifact},
        "run": {**common, "kind": "run", "artifact_name": "l04-explanations.L049V2StageA.attempt1.partial.json"},
        "failure": {
            **common,
            "kind": "failure",
            "failure_ref": "l04-explanations.L049V2StageA.attempt1.failure.json",
        },
    }
    for kind, value in values.items():
        (directory / f"l04-explanations.L049V2StageA.attempt1.{kind}.json").write_bytes(
            canonical_json_bytes(value) + b"\n"
        )


def _source_triplet(tmp_path: Path, attempt: int = 1) -> tuple[Path, dict[str, bytes], str]:
    output = tmp_path / f"source-{attempt}"
    result = run_real(use_case=USE_CASE, output_dir=output)
    names = result["paths"]
    files = {
        f"artifacts/m14/{names['partial']}": (output / names["partial"]).read_bytes(),
        f"artifacts/m14/{names['run']}": (output / names["run"]).read_bytes(),
        f"artifacts/m14/{names['failure']}": (output / names["failure"]).read_bytes(),
    }
    return output, files, code_sha()


def _archive(
    files: dict[str, bytes],
    *,
    extra: tuple[str, bytes] | None = None,
    symlink: bool = False,
    special: bool = False,
    pax_member: bool = False,
) -> bytes:
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz", format=tarfile.PAX_FORMAT) as handle:
        for path, data in sorted(files.items()):
            if symlink and path.endswith(".failure.json"):
                info = tarfile.TarInfo(path)
                info.type = tarfile.SYMTYPE
                info.linkname = "partial.json"
                handle.addfile(info)
                continue
            if special and path.endswith(".failure.json"):
                info = tarfile.TarInfo(path)
                info.type = tarfile.FIFOTYPE
                handle.addfile(info)
                continue
            info = tarfile.TarInfo(path)
            if pax_member and path.endswith(".failure.json"):
                info.pax_headers = {"comment": "synthetic-extension"}
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
        if extra is not None:
            path, data = extra
            info = tarfile.TarInfo(path)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    return target.getvalue()


def _capture(files: dict[str, bytes], source_sha: str, *, archive: bytes | None = None) -> bytes:
    archive = _archive(files) if archive is None else archive
    members = "".join(
        f"L04_BUNDLE_MEMBER={path}|{len(data)}|{hashlib.sha256(data).hexdigest()}\n"
        for path, data in sorted(files.items())
    )
    blob = base64.b64encode(archive).decode("ascii")
    lines = [
        "L04_TRANSPORT_PAYLOAD_SHA256=" + "a" * 64,
        "L04_TRANSPORT_DECODE_STATUS=0",
        "L04_TRANSPORT_DECODE_SHA256=" + "a" * 64,
        "L04_TRANSPORT_DECODE_MATCH=PASS",
        "L04_WORKDIR=/tmp/latent-anything-l04.synthetic",
        f"L04_USE_CASE={USE_CASE}",
        f"L04_CODE_SHA={source_sha}",
        "L04_CLI_STATUS=1",
        "L04_BUNDLE_STATUS=0",
        "L04_STATUS=1",
        f"L04_BUNDLE_BYTES={len(archive)}",
        f"L04_BUNDLE_SHA256={hashlib.sha256(archive).hexdigest()}",
    ]
    return (
        "\n".join(lines)
        + "\n"
        + members
        + "L04_BUNDLE_B64_BEGIN\n"
        + blob
        + "\nL04_BUNDLE_B64_END\nL04_CLEANUP=PASS\nL04_TRANSPORT_CLEANUP=PASS\n"
    ).encode()


def test_representative_marker_and_base64_capture_is_parseable(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    markers, members, archive = postprocess.parse_capture(_capture(files, source_sha))
    assert markers["L04_USE_CASE"] == USE_CASE
    assert markers["L04_CODE_SHA"] == source_sha
    assert len(members) == 3
    assert hashlib.sha256(archive).hexdigest() == markers["L04_BUNDLE_SHA256"]


def test_v2_stage_a_fake_capture_retains_and_finalizes_exact_triad(tmp_path: Path) -> None:
    """The v2 CLI triad must traverse the real retention transaction."""
    _raw, train_rows = read_rows(TRAIN_FIXTURE_PATH)
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    artifact = build_stage_a_artifact(
        train_rows,
        addendum,
        source_sha256="a" * 64,
        cli_sha256=top_level_cli_sha256("stage_a_train_selection"),
    )
    source_sha = code_sha()
    triad_dir = tmp_path / "triad"
    triad_dir.mkdir()
    _write_v2_stage_a_triad(triad_dir, artifact, source_sha)
    files = {
        f"artifacts/m14/{path.name}": path.read_bytes()
        for path in triad_dir.glob("l04-explanations.L049V2StageA.attempt1.*.json")
    }
    archive = _archive(files)
    members = "".join(
        f"L04_BUNDLE_MEMBER={path}|{len(data)}|{hashlib.sha256(data).hexdigest()}\n"
        for path, data in sorted(files.items())
    )
    bundle_sha = hashlib.sha256(archive).hexdigest()
    capture = (
        "\n".join(
            [
                f"L04_TRANSPORT_PAYLOAD_SHA256={'a' * 64}",
                "L04_TRANSPORT_DECODE_STATUS=0",
                f"L04_TRANSPORT_DECODE_SHA256={'a' * 64}",
                "L04_TRANSPORT_DECODE_MATCH=PASS",
                "L04_WORKDIR=/tmp/latent-anything-l04.synthetic",
                "L04_USE_CASE=L049V2StageA",
                f"L04_CODE_SHA={source_sha}",
                "L04_CLI_STATUS=0",
                "L04_BUNDLE_STATUS=0",
                "L04_STATUS=0",
                f"L04_BUNDLE_BYTES={len(archive)}",
                f"L04_BUNDLE_SHA256={bundle_sha}",
            ]
        )
        + "\n"
        + members
        + "L04_BUNDLE_B64_BEGIN\n"
        + base64.b64encode(archive).decode("ascii")
        + "\nL04_BUNDLE_B64_END\nL04_CLEANUP=PASS\nL04_TRANSPORT_CLEANUP=PASS\n"
    ).encode()
    raw_path = tmp_path / "raw.capture"
    raw_path.write_bytes(capture)
    artifact_dir = tmp_path / "retained"
    audit_path = tmp_path / "audit.json"
    pending = postprocess.retain_capture(
        raw_capture_path=raw_path,
        source_sha=source_sha,
        use_case="L049V2StageA",
        artifact_dir=artifact_dir,
        audit_path=audit_path,
        fixture_path=TRAIN_FIXTURE_PATH,
        retain=True,
    )
    assert pending["raw_status"] == "retained_pending_finalize"
    finalized = postprocess.finalize_delete(
        raw_capture_path=raw_path,
        source_sha=source_sha,
        use_case="L049V2StageA",
        artifact_dir=artifact_dir,
        audit_path=audit_path,
        fixture_path=TRAIN_FIXTURE_PATH,
    )
    assert finalized["raw_status"] == "deleted_verified"


@pytest.mark.parametrize("failure_mode", ["cleanup_only", "runtime_and_cleanup"])
def test_v2_stage_a_d0_cleanup_failure_retains_and_finalizes(tmp_path: Path, failure_mode: str) -> None:
    """Incomplete real attempts with cleanup failure retain the exact triad."""
    _raw, train_rows = read_rows(TRAIN_FIXTURE_PATH)
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    resources: dict[str, Any] = {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "model_revision": "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {"registered": 1, "capture_calls": 1, "removed": 0},
        "intervention": {"patch_calls": 0, "control_calls": 0, "forward_calls": 1},
        "operation_counts": {
            "candidate_evaluations": 1,
            "hooks": 1,
            "captures": 1,
            "patches": 0,
            "controls": 0,
            "forwards": 1,
        },
        "cleanup": {"hook_count": 1, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 1,
            "peak_gpu_bytes": 1,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
        },
        "no_mutation": True,
    }

    def finalize() -> dict[str, Any]:
        raise RuntimeError("cleanup prompt secret must never escape")

    resources["finalize"] = finalize

    def score(*_args: Any) -> float:
        if failure_mode == "runtime_and_cleanup":
            raise IndexError("runtime prompt secret must never escape")
        return 0.0

    artifact = run_real_stage_a(
        train_rows,
        addendum,
        source_sha256="a" * 64,
        runtime={"score": score, "resources": resources},
        cli_sha256=top_level_cli_sha256("stage_a_train_selection"),
    )
    assert artifact["status"] == "stage_a_failed"
    assert artifact["evidence_level"] == "D0"
    assert artifact["resources"]["cleanup"]["completed"] is False
    assert "prompt secret" not in json.dumps(artifact)
    source_sha = code_sha()
    triad_dir = tmp_path / "triad-d0"
    triad_dir.mkdir()
    _write_v2_stage_a_triad(triad_dir, artifact, source_sha)
    files = {
        f"artifacts/m14/{path.name}": path.read_bytes()
        for path in triad_dir.glob("l04-explanations.L049V2StageA.attempt1.*.json")
    }
    archive = _archive(files)
    members = "".join(
        f"L04_BUNDLE_MEMBER={path}|{len(data)}|{hashlib.sha256(data).hexdigest()}\n"
        for path, data in sorted(files.items())
    )
    bundle_sha = hashlib.sha256(archive).hexdigest()
    capture = (
        "\n".join(
            [
                f"L04_TRANSPORT_PAYLOAD_SHA256={'a' * 64}",
                "L04_TRANSPORT_DECODE_STATUS=0",
                f"L04_TRANSPORT_DECODE_SHA256={'a' * 64}",
                "L04_TRANSPORT_DECODE_MATCH=PASS",
                "L04_WORKDIR=/tmp/latent-anything-l04.synthetic-d0",
                "L04_USE_CASE=L049V2StageA",
                f"L04_CODE_SHA={source_sha}",
                "L04_CLI_STATUS=1",
                "L04_BUNDLE_STATUS=0",
                "L04_STATUS=1",
                f"L04_BUNDLE_BYTES={len(archive)}",
                f"L04_BUNDLE_SHA256={bundle_sha}",
            ]
        )
        + "\n"
        + members
        + "L04_BUNDLE_B64_BEGIN\n"
        + base64.b64encode(archive).decode("ascii")
        + "\nL04_BUNDLE_B64_END\nL04_CLEANUP=PASS\nL04_TRANSPORT_CLEANUP=PASS\n"
    ).encode()
    raw_path = tmp_path / "raw-d0.capture"
    raw_path.write_bytes(capture)
    artifact_dir = tmp_path / "retained-d0"
    audit_path = tmp_path / "audit-d0.json"
    pending = postprocess.retain_capture(
        raw_capture_path=raw_path,
        source_sha=source_sha,
        use_case="L049V2StageA",
        artifact_dir=artifact_dir,
        audit_path=audit_path,
        fixture_path=TRAIN_FIXTURE_PATH,
        retain=True,
    )
    assert pending["raw_status"] == "retained_pending_finalize"
    finalized = postprocess.finalize_delete(
        raw_capture_path=raw_path,
        source_sha=source_sha,
        use_case="L049V2StageA",
        artifact_dir=artifact_dir,
        audit_path=audit_path,
        fixture_path=TRAIN_FIXTURE_PATH,
    )
    assert finalized["raw_status"] == "deleted_verified"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"fixture_path": TRAIN_FIXTURE_PATH},
        {"fixture_path": TRAIN_FIXTURE_PATH, "candidate_path": TRAIN_FIXTURE_PATH},
    ],
)
def test_v2_stage_b_requires_all_owner_inputs_before_mutation(tmp_path: Path, kwargs: dict[str, Path | None]) -> None:
    raw_path = tmp_path / "raw.capture"
    raw_path.write_bytes(b"not parsed")
    artifact_dir = tmp_path / "retained"
    with pytest.raises(postprocess.RetentionError, match="requires holdout fixture"):
        postprocess.retain_capture(
            raw_capture_path=raw_path,
            source_sha="a" * 40,
            use_case="L049V2StageB",
            artifact_dir=artifact_dir,
            audit_path=tmp_path / "audit.json",
            **cast(Any, kwargs),
            retain=True,
        )
    assert not artifact_dir.exists()


def test_v2_stage_a_rejects_stage_b_inputs_before_parsing(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.capture"
    raw_path.write_bytes(b"not parsed")
    with pytest.raises(postprocess.RetentionError, match="does not accept Stage B"):
        postprocess.retain_capture(
            raw_capture_path=raw_path,
            source_sha="a" * 40,
            use_case="L049V2StageA",
            artifact_dir=tmp_path / "retained",
            audit_path=tmp_path / "audit.json",
            candidate_path=TRAIN_FIXTURE_PATH,
            retain=True,
        )


def test_unexpected_stdout_noise_is_rejected(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    capture = _capture(files, source_sha).replace(
        b"L04_TRANSPORT_DECODE_STATUS=0\n",
        b"L04_TRANSPORT_DECODE_STATUS=0\nNVIDIA-SMI diagnostic\n",
        1,
    )
    with pytest.raises(postprocess.RetentionError, match="unexpected text"):
        postprocess.parse_capture(capture)


def _write_capture(tmp_path: Path, files: dict[str, bytes], source_sha: str, *, archive: bytes | None = None) -> Path:
    path = tmp_path / "raw.capture"
    path.write_bytes(_capture(files, source_sha, archive=archive))
    return path


def _call(
    tmp_path: Path, raw: Path, source_sha: str, *, retain: bool = True, audit_name: str = "audit.json"
) -> dict[str, object]:
    return postprocess.retain_capture(
        raw_capture_path=raw,
        source_sha=source_sha,
        use_case=USE_CASE,
        artifact_dir=tmp_path / "artifacts" / "m14",
        audit_path=tmp_path / audit_name,
        retain=retain,
        validate_only=not retain,
    )


def test_success_retains_exact_triplet_and_deletes_raw(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    audit = _call(tmp_path, raw, source_sha)
    assert raw.exists()
    assert audit["mode"] == "retained_pending_finalize"
    assert audit["raw_status"] == "retained_pending_finalize"
    for path, data in files.items():
        final = tmp_path / "artifacts" / "m14" / Path(path).name
        assert final.read_bytes() == data
    audit_text = (tmp_path / "audit.json").read_text(encoding="utf-8")
    assert "Please" not in audit_text
    assert "prompt" not in audit_text.lower()
    finalized = postprocess.finalize_delete(
        raw_capture_path=raw,
        source_sha=source_sha,
        use_case=USE_CASE,
        artifact_dir=tmp_path / "artifacts" / "m14",
        audit_path=tmp_path / "audit.json",
    )
    assert finalized["raw_status"] == "deleted_verified"
    assert not raw.exists()


def test_validator_failure_keeps_raw_and_writes_no_final_payload(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    broken = dict(files)
    artifact_path = next(path for path in broken if path.endswith(".partial.json"))
    value = json.loads(broken[artifact_path])
    value["use_case"] = "TCAV"
    broken[artifact_path] = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    raw = _write_capture(tmp_path, broken, source_sha)
    with pytest.raises(postprocess.RetentionError, match="validator|use-case"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()
    assert not list((tmp_path / "artifacts" / "m14").glob("*.json"))


@pytest.mark.parametrize("variant", ["duplicate", "missing", "hash"])
def test_marker_integrity_fails_closed(tmp_path: Path, variant: str) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    text = raw.read_text(encoding="utf-8")
    if variant == "duplicate":
        text = text.replace("L04_STATUS=1\n", "L04_STATUS=1\nL04_STATUS=1\n")
    elif variant == "missing":
        text = text.replace("L04_BUNDLE_STATUS=0\n", "")
    else:
        text = text.replace(
            "L04_BUNDLE_SHA256=" + hashlib.sha256(_archive(files)).hexdigest(), "L04_BUNDLE_SHA256=" + "c" * 64
        )
    raw.write_text(text, encoding="utf-8")
    with pytest.raises(postprocess.RetentionError):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()


def test_json_duplicate_keys_at_nested_depth_fail_structured() -> None:
    with pytest.raises(postprocess.RetentionError, match="duplicate JSON object key: x"):
        postprocess.json_load(b'{"outer":{"x":1,"x":2}}', "synthetic")


def test_marker_sequence_and_status_range_fail_closed(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    lines = raw.read_text(encoding="utf-8").splitlines()
    marker_index = next(i for i, line in enumerate(lines) if line.startswith("L04_BUNDLE_STATUS="))
    lines[marker_index], lines[marker_index + 1] = lines[marker_index + 1], lines[marker_index]
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(postprocess.RetentionError, match="sequence"):
        _call(tmp_path, raw, source_sha)
    raw.write_bytes(_capture(files, source_sha).replace(b"L04_CLI_STATUS=1", b"L04_CLI_STATUS=256"))
    with pytest.raises(postprocess.RetentionError, match="0..255"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()


@pytest.mark.parametrize("bad_line", ["L04_BOGUS", "L04_BOGUS=x", "L04_STATUS"])
def test_marker_like_lines_are_rejected(tmp_path: Path, bad_line: str) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    text = raw.read_text(encoding="utf-8")
    text = text.replace("L04_BUNDLE_B64_BEGIN\n", f"{bad_line}\nL04_BUNDLE_B64_BEGIN\n")
    raw.write_text(text, encoding="utf-8")
    with pytest.raises(postprocess.RetentionError, match="marker|sequence"):
        _call(tmp_path, raw, source_sha)


def test_marker_like_stderr_and_boundary_tampering_are_rejected(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    wrapped = (
        b"--- STDOUT BEGIN ---\n"
        + raw.read_bytes()
        + b"--- STDOUT END ---\n--- STDERR BEGIN ---\nL04_BOGUS\n--- STDERR END ---\n"
    )
    raw.write_bytes(wrapped)
    with pytest.raises(postprocess.RetentionError, match="stderr"):
        _call(tmp_path, raw, source_sha)
    raw.write_bytes(wrapped.replace(b"L04_BOGUS", b"L04_BUNDLE_B64_END"))
    with pytest.raises(postprocess.RetentionError, match="boundary|marker|pair"):
        _call(tmp_path, raw, source_sha)


@pytest.mark.parametrize("kind", ["traversal", "extra", "symlink", "special"])
def test_archive_shape_is_rejected(tmp_path: Path, kind: str) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    archive = _archive(
        files,
        extra=("artifacts/m14/history.json", b"{}") if kind == "extra" else None,
        symlink=kind == "symlink",
        special=kind == "special",
    )
    if kind == "traversal":
        data = next(iter(files.values()))
        archive = _archive({"../escape.json": data, **{key: value for key, value in list(files.items())[1:]}})
    raw = _write_capture(tmp_path, files, source_sha, archive=archive)
    with pytest.raises(postprocess.RetentionError):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()


def test_archive_pax_and_extension_headers_are_rejected(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    member_pax = _archive(files, pax_member=True)
    raw = _write_capture(tmp_path, files, source_sha, archive=member_pax)
    with pytest.raises(postprocess.RetentionError, match="extension|PAX|tar"):
        _call(tmp_path, raw, source_sha)

    # A global PAX header is represented by a tar extension entry.  Flip the
    # first regular header's typeflag in a synthetic archive to exercise the
    # pre-extraction scanner without relying on a platform tar implementation.
    decompressed = bytearray(gzip.decompress(_archive(files)))
    decompressed[156] = ord("g")
    raw.write_bytes(_capture(files, source_sha, archive=gzip.compress(bytes(decompressed))))
    with pytest.raises(postprocess.RetentionError, match="extension|PAX|tar"):
        _call(tmp_path, raw, source_sha)


def test_capture_sections_and_stderr_marker_injection_fail_closed(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    wrapped = (
        b"--- STDOUT BEGIN ---\n"
        + raw.read_bytes()
        + b"--- STDOUT END ---\n--- STDERR BEGIN ---\nnoise\n--- STDERR END ---\n"
    )
    raw.write_bytes(wrapped)
    _call(tmp_path, raw, source_sha, retain=False)
    raw.write_bytes(wrapped.replace(b"noise", b"L04_STATUS=0"))
    with pytest.raises(postprocess.RetentionError, match="stderr"):
        _call(tmp_path, raw, source_sha, retain=False)


def test_mixed_attempt_member_markers_and_collision_are_rejected(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    text = raw.read_text(encoding="utf-8")
    first = next(line for line in text.splitlines() if line.startswith("L04_BUNDLE_MEMBER="))
    text = text.replace(first, first.replace("attempt1", "attempt2"))
    raw.write_text(text, encoding="utf-8")
    with pytest.raises(postprocess.RetentionError, match="attempt"):
        _call(tmp_path, raw, source_sha)
    raw.write_bytes(_capture(files, source_sha))
    destination = tmp_path / "artifacts" / "m14"
    destination.mkdir(parents=True)
    (destination / next(iter(files)).split("/")[-1]).write_bytes(b"collision")
    with pytest.raises(postprocess.RetentionError, match="collision"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()


def test_member_hash_mismatch_keeps_raw(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    lines = raw.read_text(encoding="utf-8").splitlines()
    index = next(index for index, line in enumerate(lines) if line.startswith("L04_BUNDLE_MEMBER="))
    path, size, _digest = lines[index].split("=", 1)[1].split("|")
    lines[index] = f"L04_BUNDLE_MEMBER={path}|{size}|{'c' * 64}"
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(postprocess.RetentionError, match="member"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()


def test_renamed_archive_members_with_stale_internal_links_are_rejected(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    renamed = {path.replace("attempt1", "attempt2"): data for path, data in files.items()}
    raw = _write_capture(tmp_path, renamed, source_sha)
    with pytest.raises(postprocess.RetentionError, match="filename|reference|attempt"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()


def test_atomic_install_rolls_back_on_mid_transaction_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    import os

    original_link = os.link
    calls = 0

    def fail_on_second_final(
        source: str | bytes | os.PathLike[str], destination: str | bytes | os.PathLike[str]
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected link failure")
        original_link(source, destination)

    monkeypatch.setattr(transaction.os, "link", fail_on_second_final)
    with pytest.raises(OSError, match="link failure"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()
    assert not list((tmp_path / "artifacts" / "m14").glob("l04-explanations*.json"))


def test_idempotent_exact_files_are_accepted(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha, audit_name="audit-1.json")
    raw.write_bytes(_capture(files, source_sha))
    _call(tmp_path, raw, source_sha, audit_name="audit-2.json")
    assert raw.exists()


def test_audit_failure_keeps_raw_and_no_partial_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)

    def fail_audit(*_args: object, **_kwargs: object) -> None:
        raise OSError("audit failure")

    monkeypatch.setattr(postprocess, "write_atomic", fail_audit)
    with pytest.raises(OSError, match="audit failure"):
        _call(tmp_path, raw, source_sha)
    assert raw.exists()
    assert not list((tmp_path / "artifacts" / "m14").glob("l04-explanations*.json"))


def test_final_reopen_failure_keeps_raw_and_no_partial_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)

    def fail_reopen(*_args: object) -> object:
        raise postprocess.RetentionError("reopen")

    monkeypatch.setattr(postprocess, "reopen_payloads", fail_reopen)
    with pytest.raises(postprocess.RetentionError, match="reopen"):
        _call(tmp_path, raw, source_sha, audit_name="audit-2.json")
    assert raw.exists()
    assert not list((tmp_path / "artifacts" / "m14").glob("l04-explanations*.json"))


def test_dry_run_never_writes_or_deletes(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    result = postprocess.retain_capture(
        raw_capture_path=raw,
        source_sha=source_sha,
        use_case=USE_CASE,
        artifact_dir=tmp_path / "artifacts" / "m14",
        audit_path=tmp_path / "audit.json",
        retain=True,
        dry_run=True,
    )
    assert result["mode"] == "dry-run"
    assert raw.exists()
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "artifacts").exists()


def test_validate_only_never_writes_and_keeps_raw(tmp_path: Path) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    audit = _call(tmp_path, raw, source_sha, retain=False)
    assert raw.exists()
    assert audit["mode"] == "validate-only"
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "artifacts").exists()


def test_finalize_audit_failure_restores_exact_raw_and_pending_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha)
    raw_before = raw.read_bytes()
    audit_before = (tmp_path / "audit.json").read_bytes()
    original_write = write_atomic
    failed = False

    def fail_final_audit(path: Path, data: bytes, *, replace: bool) -> None:
        nonlocal failed
        if json.loads(data).get("raw_status") == "deleted_verified" and not failed:
            failed = True
            raise OSError("final audit failure")
        original_write(path, data, replace=replace)

    monkeypatch.setattr(postprocess, "write_atomic", fail_final_audit)
    with pytest.raises(postprocess.RetentionError, match="pending audit restored"):
        postprocess.finalize_delete(
            raw_capture_path=raw,
            source_sha=source_sha,
            use_case=USE_CASE,
            artifact_dir=tmp_path / "artifacts" / "m14",
            audit_path=tmp_path / "audit.json",
        )
    assert raw.read_bytes() == raw_before
    assert (tmp_path / "audit.json").read_bytes() == audit_before
    assert all((tmp_path / "artifacts" / "m14" / Path(path).name).read_bytes() == data for path, data in files.items())
    finalized = postprocess.finalize_delete(
        raw_capture_path=raw,
        source_sha=source_sha,
        use_case=USE_CASE,
        artifact_dir=tmp_path / "artifacts" / "m14",
        audit_path=tmp_path / "audit.json",
    )
    assert finalized["raw_status"] == "deleted_verified"
    assert not raw.exists()


def test_finalize_double_failure_publishes_fatal_state_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha)

    def fail_restore(*_args: object, **_kwargs: object) -> None:
        raise OSError("snapshot restore failure")

    monkeypatch.setattr(postprocess, "restore_raw_snapshot", fail_restore)
    original_write = write_atomic

    def fail_final_audit(path: Path, data: bytes, *, replace: bool) -> None:
        if json.loads(data).get("raw_status") == "deleted_verified":
            raise OSError("final audit failure")
        original_write(path, data, replace=replace)

    monkeypatch.setattr(postprocess, "write_atomic", fail_final_audit)
    with pytest.raises(postprocess.RetentionError, match="raw_restore_failed"):
        postprocess.finalize_delete(
            raw_capture_path=raw,
            source_sha=source_sha,
            use_case=USE_CASE,
            artifact_dir=tmp_path / "artifacts" / "m14",
            audit_path=tmp_path / "audit.json",
        )
    current_audit = json.loads((tmp_path / "audit.json").read_bytes())
    assert current_audit["raw_status"] == "raw_restore_failed"
    assert current_audit["raw_capture"]["quarantine"]["status"] == "absent_restore_failed"
    assert current_audit["raw_status"] != "deleted_verified"
    assert not raw.exists()


@pytest.mark.parametrize(
    "field",
    [
        "source_sha",
        "attempt",
        "raw_capture",
        "marker_exits",
        "transport",
        "bundle",
        "validation",
        "promoted",
        "mode",
    ],
)
def test_finalize_rejects_any_pending_audit_subtree_tamper(tmp_path: Path, field: str) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha)
    audit_path = tmp_path / "audit.json"
    audit = json.loads(audit_path.read_bytes())
    if field == "source_sha":
        audit[field] = "b" * 40
    elif field == "attempt":
        audit[field] = "attempt99"
    elif field == "raw_capture":
        audit[field]["bytes"] += 1
    elif field == "marker_exits":
        audit[field]["cli"] = 0
    elif field == "transport":
        audit[field]["decode_match"] = "FAIL"
    elif field == "bundle":
        audit[field]["sha256"] = "c" * 64
    elif field == "validation":
        audit[field]["archive"] = "FAIL"
    elif field == "mode":
        audit[field] = "retain"
    else:
        audit[field] = True
    audit_path.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
    with pytest.raises(postprocess.RetentionError, match="audit|match|provenance"):
        postprocess.finalize_delete(
            raw_capture_path=raw,
            source_sha=source_sha,
            use_case=USE_CASE,
            artifact_dir=tmp_path / "artifacts" / "m14",
            audit_path=audit_path,
        )
    assert raw.exists()
    assert all((tmp_path / "artifacts" / "m14" / Path(path).name).read_bytes() == data for path, data in files.items())


def test_finalize_quarantine_audit_failure_restores_raw_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha)

    def fail_quarantine_audit(path: Path, data: bytes, *, replace: bool) -> None:
        if json.loads(data).get("raw_status") == "quarantined_pending_delete":
            raise OSError("quarantine audit failure")
        write_atomic(path, data, replace=replace)

    monkeypatch.setattr(postprocess, "write_atomic", fail_quarantine_audit)
    with pytest.raises(OSError, match="quarantine audit failure"):
        postprocess.finalize_delete(
            raw_capture_path=raw,
            source_sha=source_sha,
            use_case=USE_CASE,
            artifact_dir=tmp_path / "artifacts" / "m14",
            audit_path=tmp_path / "audit.json",
        )
    assert raw.exists()
    assert not list(tmp_path.glob(".*.quarantine"))


def test_finalize_restore_rename_failure_keeps_quarantine_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha)

    def fail_restore(*_args: object, **_kwargs: object) -> None:
        raise OSError("restore rename failure")

    monkeypatch.setattr(postprocess, "restore_quarantine", fail_restore)

    def fail_quarantine_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("quarantine audit failure")

    monkeypatch.setattr(postprocess, "write_atomic", fail_quarantine_write)
    with pytest.raises(postprocess.RetentionError, match="quarantine"):
        postprocess.finalize_delete(
            raw_capture_path=raw,
            source_sha=source_sha,
            use_case=USE_CASE,
            artifact_dir=tmp_path / "artifacts" / "m14",
            audit_path=tmp_path / "audit.json",
        )
    assert not raw.exists()
    quarantines = list(tmp_path.glob(".*.quarantine"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes()


def test_finalize_delete_failure_keeps_quarantine_pending_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, files, source_sha = _source_triplet(tmp_path)
    raw = _write_capture(tmp_path, files, source_sha)
    _call(tmp_path, raw, source_sha)
    monkeypatch.setattr(postprocess, "delete_quarantine", lambda _path: (_ for _ in ()).throw(OSError("delete")))
    with pytest.raises(OSError, match="delete"):
        postprocess.finalize_delete(
            raw_capture_path=raw,
            source_sha=source_sha,
            use_case=USE_CASE,
            artifact_dir=tmp_path / "artifacts" / "m14",
            audit_path=tmp_path / "audit.json",
        )
    assert not raw.exists()
    assert json.loads((tmp_path / "audit.json").read_bytes())["raw_status"] == "quarantined_pending_delete"
    assert len(list(tmp_path.glob(".*.quarantine"))) == 1


def test_transport_contract_delegates_retention_without_helper_deletion() -> None:
    helper = (ROOT / "scripts/m14_l04_remote_transport.ps1").read_text(encoding="utf-8")
    payload = (ROOT / "scripts/m14_l04_remote_payload.sh").read_text(encoding="utf-8")
    assert "m14_l04_remote_postprocess" in helper
    assert "--retain" in helper
    assert "--raw-capture $RawCapturePath" in helper
    assert "--raw-capture $capture.raw_capture_path" not in helper
    assert "L04_BUNDLE_BYTES" in helper
    assert "L04_BUNDLE_SHA256" in payload
    assert "L04_BUNDLE_MEMBER" in payload
    assert "Remove-Item" not in helper


def test_historical_d9_audit_is_immutable_and_sidecar_is_sanitized() -> None:
    audit_path = (
        ROOT / "artifacts/m14/l04-explanations.ssh.Disentanglement.d9b16923eb9108c7bcc8e6bc12ace4ebd16ff506.audit.json"
    )
    sidecar_path = ROOT / (
        "artifacts/m14/l04-explanations.ssh.Disentanglement.d9b16923eb9108c7bcc8e6bc12ace4ebd16ff506."
        "retention-failure.json"
    )
    assert (
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
        == "1273aa6c20d305193f5f0b0848e332c5a4ddc44938b4eb9b8ebbb1700fcb1003"
    )
    sidecar = json.loads(sidecar_path.read_bytes())
    assert canonical_digest(sidecar, "sidecar_sha256") == sidecar["sidecar_sha256"]
    assert sidecar["observed_remote"]["promotion_claim"] is True
    assert sidecar["repository_promotion"] is False
    assert sidecar["reason"] == "payload_triplet_absent/raw_absent_nonreconstructable"
    assert "prompt" not in sidecar_path.read_text(encoding="utf-8").lower()
