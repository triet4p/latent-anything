"""Pure parsing and validation for an L04 remote capture."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import io
import json
import re
import tarfile
from typing import Any

from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
USE_CASES = (
    "IntegratedGradients",
    "TCAV",
    "DirectLogitLens",
    "TunedLogitLens",
    "Disentanglement",
    "TrueActivationPatching",
    "AdditiveSteering",
)
MEMBER_RE = re.compile(
    r"^artifacts/m14/l04-explanations\.(?P<use_case>[A-Za-z]+)\.(?P<attempt>attempt[0-9]+)\.(?P<kind>partial|run|failure)\.json$"
)
REQUIRED_SINGLETON_MARKERS = (
    "L04_TRANSPORT_PAYLOAD_SHA256",
    "L04_TRANSPORT_DECODE_STATUS",
    "L04_TRANSPORT_DECODE_SHA256",
    "L04_TRANSPORT_DECODE_MATCH",
    "L04_TRANSPORT_CLEANUP",
    "L04_WORKDIR",
    "L04_USE_CASE",
    "L04_CODE_SHA",
    "L04_CLI_STATUS",
    "L04_BUNDLE_STATUS",
    "L04_STATUS",
    "L04_BUNDLE_BYTES",
    "L04_BUNDLE_SHA256",
    "L04_CLEANUP",
)
EXPECTED_MEMBER_KINDS = {"partial", "run", "failure"}
KNOWN_MARKERS = set(REQUIRED_SINGLETON_MARKERS) | {"L04_BUNDLE_MEMBER"}
MARKER_SEQUENCE = (
    "L04_TRANSPORT_PAYLOAD_SHA256",
    "L04_TRANSPORT_DECODE_STATUS",
    "L04_TRANSPORT_DECODE_SHA256",
    "L04_TRANSPORT_DECODE_MATCH",
    "L04_WORKDIR",
    "L04_USE_CASE",
    "L04_CODE_SHA",
    "L04_CLI_STATUS",
    "L04_BUNDLE_STATUS",
    "L04_STATUS",
    "L04_BUNDLE_BYTES",
    "L04_BUNDLE_SHA256",
    "L04_BUNDLE_MEMBER",
    "L04_BUNDLE_B64_BEGIN",
    "L04_BUNDLE_B64_END",
    "L04_CLEANUP",
    "L04_TRANSPORT_CLEANUP",
)


class RetentionError(ValueError):
    """A capture failed a fail-closed retention gate."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def json_load(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RetentionError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RetentionError(f"{label} must be a JSON object")
    return value


def parse_capture(raw: bytes, *, max_base64_bytes: int = 64 * 1024 * 1024) -> tuple[dict[str, str], list[str], bytes]:
    """Parse the exact ordered marker protocol and decode its bounded bundle."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RetentionError("raw capture must be strict UTF-8") from exc
    lines = text.splitlines()
    stdout_begin = [index for index, line in enumerate(lines) if line == "--- STDOUT BEGIN ---"]
    stdout_end = [index for index, line in enumerate(lines) if line == "--- STDOUT END ---"]
    stderr_begin = [index for index, line in enumerate(lines) if line == "--- STDERR BEGIN ---"]
    stderr_end = [index for index, line in enumerate(lines) if line == "--- STDERR END ---"]
    if stdout_begin or stdout_end or stderr_begin or stderr_end:
        if len(stdout_begin) != 1 or stdout_begin != [0] or len(stdout_end) != 1:
            raise RetentionError("raw capture section boundaries are invalid")
        if len(stderr_begin) != 1 or len(stderr_end) != 1:
            raise RetentionError("raw capture section boundaries are invalid")
        if stdout_end[0] <= stdout_begin[0] or stderr_begin[0] != stdout_end[0] + 1:
            raise RetentionError("raw capture section boundaries are invalid")
        if stderr_end[0] <= stderr_begin[0] or stderr_end[0] != len(lines) - 1:
            raise RetentionError("raw capture section boundaries are invalid")
        lines = lines[1 : stdout_end[0]]
        stderr_lines = text.splitlines()[stderr_begin[0] + 1 : stderr_end[0]]
        if any(line.startswith("L04_") for line in stderr_lines):
            raise RetentionError("stdout markers are not allowed in stderr")
    marker_records: list[tuple[str, str]] = []
    begin_indexes = [index for index, line in enumerate(lines) if line == "L04_BUNDLE_B64_BEGIN"]
    end_indexes = [index for index, line in enumerate(lines) if line == "L04_BUNDLE_B64_END"]
    if len(begin_indexes) != 1 or len(end_indexes) != 1 or begin_indexes[0] >= end_indexes[0]:
        raise RetentionError("bundle Base64 markers must be one ordered pair")
    begin, end = begin_indexes[0], end_indexes[0]
    body_lines = lines[begin + 1 : end]
    body = "".join(body_lines)
    if not body or len(body.encode("ascii", errors="ignore")) > max_base64_bytes:
        raise RetentionError("bundle Base64 body is empty or exceeds the safety bound")
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for character in body):
        raise RetentionError("bundle Base64 body contains non-Base64 data")
    for _index, line in enumerate(lines):
        if _index <= begin or _index >= end:
            unexpected = (
                bool(line)
                and not line.startswith("L04_")
                and line
                not in {
                    "L04_BUNDLE_B64_BEGIN",
                    "L04_BUNDLE_B64_END",
                }
            )
        else:
            unexpected = False
        if unexpected:
            raise RetentionError("stdout contains unexpected text outside declared markers")
        if line.startswith("L04_") and _index not in {begin, end} and "=" not in line:
            raise RetentionError("marker-like stdout line must contain exactly one assignment")
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.startswith("L04_"):
            if line.count("=") != 1:
                raise RetentionError(f"marker {name} has malformed assignment")
            if begin <= _index <= end:
                raise RetentionError("markers are not allowed inside the bounded Base64 body")
            if name not in KNOWN_MARKERS:
                raise RetentionError(f"unknown marker {name}")
            marker_records.append((name, value))
    expected_prefix = [
        name
        for name in MARKER_SEQUENCE
        if name not in {"L04_BUNDLE_MEMBER", "L04_BUNDLE_B64_BEGIN", "L04_BUNDLE_B64_END"}
    ]
    # Remove repeatable member names from the singleton sequence and verify all
    # non-members are in exact order, with the three member markers contiguous.
    names = [name for name, _ in marker_records]
    member_positions = [index for index, name in enumerate(names) if name == "L04_BUNDLE_MEMBER"]
    if len(member_positions) != 3 or member_positions != list(range(member_positions[0], member_positions[0] + 3)):
        raise RetentionError("exactly three contiguous bundle member markers are required")
    singleton_names = [name for name in names if name != "L04_BUNDLE_MEMBER"]
    if singleton_names != expected_prefix:
        raise RetentionError("stdout marker sequence is missing, duplicated, or out of order")
    marker_map: dict[str, str] = {}
    members: list[str] = []
    for name, value in marker_records:
        if name == "L04_BUNDLE_MEMBER":
            members.append(value)
        elif name in marker_map:
            raise RetentionError(f"duplicate marker {name}")
        else:
            marker_map[name] = value
    for name in REQUIRED_SINGLETON_MARKERS:
        if name not in marker_map:
            raise RetentionError(f"missing marker {name}")
    try:
        archive = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RetentionError("bundle Base64 is invalid") from exc
    return marker_map, members, archive


def int_marker(markers: dict[str, str], name: str) -> int:
    value = markers[name]
    if not re.fullmatch(r"[0-9]+", value):
        raise RetentionError(f"marker {name} is not an integer")
    parsed = int(value)
    if parsed > 255:
        raise RetentionError(f"marker {name} is outside 0..255")
    return parsed


def size_marker(markers: dict[str, str], name: str) -> int:
    """Parse an announced byte count without applying the 8-bit exit bound."""
    value = markers[name]
    if not re.fullmatch(r"[0-9]+", value):
        raise RetentionError(f"marker {name} is not a byte count")
    return int(value)


def parse_member_markers(values: list[str], use_case: str, attempt: str) -> dict[str, tuple[int, str]]:
    if len(values) != 3:
        raise RetentionError("exactly three bundle member markers are required")
    result: dict[str, tuple[int, str]] = {}
    paths: list[str] = []
    for value in values:
        pieces = value.split("|")
        if len(pieces) != 3:
            raise RetentionError("bundle member marker has invalid fields")
        path, size_text, digest = pieces
        match = MEMBER_RE.fullmatch(path)
        if match is None or match.group("use_case") != use_case or match.group("attempt") != attempt:
            raise RetentionError("bundle member marker has invalid use-case or attempt")
        if not re.fullmatch(r"[0-9]+", size_text) or SHA256_RE.fullmatch(digest) is None:
            raise RetentionError("bundle member marker has invalid size or SHA-256")
        if path in result:
            raise RetentionError("duplicate bundle member marker")
        result[path] = (int(size_text), digest)
        paths.append(path)
    if paths != sorted(paths):
        raise RetentionError("bundle member markers are not in canonical order")
    return result


def _raw_tar_headers(archive: bytes) -> None:
    try:
        stream = gzip.decompress(archive)
    except (OSError, EOFError) as exc:
        raise RetentionError("bundle is not valid gzip") from exc
    offset = 0
    zero_blocks = 0
    while offset + 512 <= len(stream):
        header = stream[offset : offset + 512]
        offset += 512
        if header == b"\0" * 512:
            zero_blocks += 1
            if zero_blocks == 2:
                if any(stream[offset:]):
                    raise RetentionError("archive has trailing bytes after end-of-archive")
                return
            continue
        zero_blocks = 0
        typeflag = header[156:157]
        if typeflag not in {b"", b"0"}:
            raise RetentionError("archive contains a tar extension, link, or special member")
        size_field = header[124:136].rstrip(b"\0 ")
        try:
            size = int(size_field or b"0", 8)
        except ValueError as exc:
            raise RetentionError("archive member size is invalid") from exc
        offset += ((size + 511) // 512) * 512
    raise RetentionError("archive has no complete end-of-archive marker")


def inspect_archive(
    archive: bytes, use_case: str, announced_members: dict[str, tuple[int, str]]
) -> tuple[str, dict[str, bytes], dict[str, tuple[int, str]]]:
    """Inspect all headers before reading validated regular JSON members."""
    _raw_tar_headers(archive)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz", errorlevel=2) as handle:
            if getattr(handle, "pax_headers", {}) != {}:
                raise RetentionError("archive/global PAX headers must be empty")
            members = handle.getmembers()
            names = [member.name for member in members]
            matches = [MEMBER_RE.fullmatch(name) for name in names]
            if any(match is None for match in matches):
                raise RetentionError("archive contains an invalid member path")
            typed = [match for match in matches if match is not None]
            if any(match.group("use_case") != use_case for match in typed):
                raise RetentionError("archive contains a different use-case")
            attempts = {match.group("attempt") for match in typed}
            kinds = {match.group("kind") for match in typed}
            if len(attempts) != 1 or kinds != EXPECTED_MEMBER_KINDS or len(names) != 3 or len(set(names)) != 3:
                raise RetentionError("archive must contain exactly one attempt's partial, run, and failure JSON")
            attempt = attempts.pop()
            if set(announced_members) != set(names):
                raise RetentionError("announced bundle members do not match archive")
            extracted: dict[str, bytes] = {}
            observed: dict[str, tuple[int, str]] = {}
            for member in members:
                if (
                    not member.isreg()
                    or member.linkname
                    or member.pax_headers != {}
                    or member.sparse is not None
                    or member.name.startswith("/")
                    or "\\" in member.name
                    or any(part in {"", ".", ".."} for part in member.name.split("/"))
                ):
                    raise RetentionError("archive contains a non-regular, linked, PAX, or unsafe member")
                source = handle.extractfile(member)
                if source is None:
                    raise RetentionError("archive regular member has no readable payload")
                data = source.read()
                expected_size, expected_digest = announced_members[member.name]
                if len(data) != member.size or len(data) != expected_size or sha256(data) != expected_digest:
                    raise RetentionError(f"member hash/size mismatch: {member.name}")
                json_load(data, member.name)
                extracted[member.name] = data
                observed[member.name] = (len(data), sha256(data))
            return attempt, extracted, observed
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise RetentionError("bundle archive is not a readable tar") from exc


def validate_triplet(
    files: dict[str, bytes], plan: dict[str, Any], source_sha: str, use_case: str, attempt: str
) -> dict[str, dict[str, Any]]:
    expected = {kind: f"l04-explanations.{use_case}.{attempt}.{kind}.json" for kind in EXPECTED_MEMBER_KINDS}
    values: dict[str, dict[str, Any]] = {}
    for kind, basename in expected.items():
        path = f"artifacts/m14/{basename}"
        if path not in files:
            raise RetentionError(f"validated bundle member missing: {path}")
        values[kind] = json_load(files[path], path)
    artifact = values["partial"]
    run = values["run"]
    failure = values["failure"]
    artifact_git = artifact.get("provenance", {}).get("git_sha")
    if artifact_git != source_sha or run.get("code_sha") != source_sha or failure.get("code_sha") != source_sha:
        raise RetentionError("envelope source SHA does not match expected source SHA")
    if (
        run.get("artifact_name") != expected["partial"]
        or failure.get("failure_ref") != expected["failure"]
        or not isinstance(failure.get("run_record"), dict)
        or failure["run_record"].get("artifact_name") != expected["partial"]
    ):
        raise RetentionError("envelope filenames do not match bundle attempt/use-case")
    executions = artifact.get("executions")
    active = (
        next((entry for entry in executions if isinstance(entry, dict) and entry.get("use_case") == use_case), None)
        if isinstance(executions, list)
        else None
    )
    if not isinstance(active, dict) or active.get("failure_ref") != expected["failure"]:
        raise RetentionError("artifact execution failure reference does not match bundle member")
    if run.get("use_case") != use_case or failure.get("use_case") != use_case or artifact.get("use_case") != use_case:
        raise RetentionError("envelope use-case linkage does not match bundle")
    artifact_errors = validate_artifact(artifact, plan)
    run_errors = validate_run_record(run, artifact, plan)
    failure_errors = validate_failure(failure, plan, artifact)
    errors = {
        "artifact": artifact_errors,
        "run_record": run_errors,
        "failure": failure_errors,
    }
    nonempty = {label: items for label, items in errors.items() if items}
    if nonempty:
        raise RetentionError(
            "envelope validators failed: " + "; ".join(f"{key}: {value}" for key, value in nonempty.items())
        )
    return values


def member_attempt(values: list[str], use_case: str) -> str:
    paths = [value.split("|", 1)[0] for value in values]
    matches = [MEMBER_RE.fullmatch(path) for path in paths]
    if any(match is None or match.group("use_case") != use_case for match in matches):
        raise RetentionError("member markers contain an invalid use-case/path")
    attempts = {match.group("attempt") for match in matches if match is not None}
    if len(attempts) != 1:
        raise RetentionError("member markers contain mixed attempts")
    return attempts.pop()


__all__ = [
    "EXPECTED_MEMBER_KINDS",
    "MEMBER_RE",
    "RetentionError",
    "SHA1_RE",
    "SHA256_RE",
    "USE_CASES",
    "int_marker",
    "inspect_archive",
    "json_load",
    "member_attempt",
    "parse_capture",
    "parse_member_markers",
    "sha256",
    "size_marker",
    "validate_triplet",
]
