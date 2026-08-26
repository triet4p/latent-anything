"""Private bounded validation, canonicalization, and artifact I/O helpers."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NoReturn

MAX_MAPPING_ENTRIES = 256
MAX_KEY_LENGTH = 128
MAX_STRING_LENGTH = 4096
MAX_CONFIG_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_METRIC_EVENTS = 4096
MAX_TAGS = 128
MAX_SEQUENCE_ITEMS = 4096
MAX_NESTING_DEPTH = 16

_SENSITIVE_KEY = re.compile(
    r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|credential|authorization|auth)"
)
_SENSITIVE_VALUE = re.compile(
    r"(?:^sk-[A-Za-z0-9]|^gh[pousr]_[A-Za-z0-9]|^xox[baprs]-|^bearer\s|BEGIN [A-Z ]*PRIVATE KEY)",
    re.I,
)
ErrorFactory = type[Exception]


def _fail(error_type: ErrorFactory, message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise error_type(message)
    raise error_type(message) from cause


def normalize_json(
    value: object,
    *,
    error_type: ErrorFactory,
    active: set[int] | None = None,
    depth: int = 0,
) -> object:
    if depth > MAX_NESTING_DEPTH:
        _fail(error_type, "recorder values exceed the nesting bound")
    active_ids = set() if active is None else active
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ENTRIES:
            _fail(error_type, "recorder mappings exceed the entry bound")
        value_id = id(value)
        if value_id in active_ids:
            _fail(error_type, "recorder values must not contain cycles")
        active_ids.add(value_id)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > MAX_KEY_LENGTH:
                    _fail(error_type, "recorder mapping keys must be bounded non-empty strings")
                reject_sensitive_key(key, error_type=error_type)
                result[key] = normalize_json(item, error_type=error_type, active=active_ids, depth=depth + 1)
        finally:
            active_ids.remove(value_id)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            _fail(error_type, "recorder sequences exceed the entry bound")
        value_id = id(value)
        if value_id in active_ids:
            _fail(error_type, "recorder values must not contain cycles")
        active_ids.add(value_id)
        try:
            return [normalize_json(item, error_type=error_type, active=active_ids, depth=depth + 1) for item in value]
        finally:
            active_ids.remove(value_id)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            _fail(error_type, "recorder strings exceed the size bound")
        if isinstance(value, str) and _SENSITIVE_VALUE.search(value):
            _fail(error_type, "recorder values must not contain secret-like material")
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            _fail(error_type, "recorder values must contain finite floats")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is value:
            _fail(error_type, f"unsupported recorder value: {type(value).__name__}")
        return normalize_json(scalar, error_type=error_type, active=active_ids, depth=depth + 1)
    _fail(error_type, f"unsupported recorder value: {type(value).__name__}")


def canonical_json(value: object, *, error_type: ErrorFactory) -> bytes:
    return json.dumps(
        normalize_json(value, error_type=error_type), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def validate_name(value: str, *, field: str, error_type: ErrorFactory) -> str:
    if not value or len(value) > MAX_STRING_LENGTH:
        _fail(error_type, f"{field} must be a non-empty bounded string")
    if any(character in value for character in ("/", "\\", "\x00")):
        _fail(error_type, f"{field} must not contain path separators")
    return value


def validate_artifact_name(name: str, *, error_type: ErrorFactory) -> str:
    if type(name) is not str or not name or len(name) > MAX_STRING_LENGTH:
        _fail(error_type, "artifact name must be a non-empty bounded string")
    if "\\" in name or "\x00" in name or "%" in name or ":" in name:
        _fail(error_type, "artifact name must use canonical POSIX separators")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "/".join(path.parts) != name
    ):
        _fail(error_type, "artifact name must be a safe relative POSIX path")
    return name


def validate_mapping(value: Mapping[str, object] | None, *, field: str, error_type: ErrorFactory) -> dict[str, object]:
    if value is None:
        return {}
    if len(value) > MAX_MAPPING_ENTRIES:
        _fail(error_type, f"{field} exceeds the entry bound")
    result: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key or len(key) > MAX_KEY_LENGTH:
            _fail(error_type, f"{field} keys must be bounded non-empty strings")
        reject_sensitive_key(key, error_type=error_type)
        result[key] = normalize_json(item, error_type=error_type)
    if len(canonical_json(result, error_type=error_type)) > MAX_CONFIG_BYTES:
        _fail(error_type, f"{field} exceeds the serialized size bound")
    return result


def validate_tags(tags: Mapping[str, str] | None, *, error_type: ErrorFactory) -> dict[str, str]:
    if tags is None:
        return {}
    if len(tags) > MAX_TAGS:
        _fail(error_type, "tags exceed the entry bound")
    result: dict[str, str] = {}
    for key, value in tags.items():
        if type(key) is not str or not key or len(key) > MAX_KEY_LENGTH:
            _fail(error_type, "tag keys must be bounded non-empty strings")
        reject_sensitive_key(key, error_type=error_type)
        if type(value) is not str or len(value) > MAX_STRING_LENGTH:
            _fail(error_type, "tag values must be bounded strings")
        if _SENSITIVE_VALUE.search(value):
            _fail(error_type, "tag values must not contain secret-like material")
        result[key] = value
    if len(canonical_json(result, error_type=error_type)) > MAX_CONFIG_BYTES:
        _fail(error_type, "tags exceed the serialized size bound")
    return result


def validate_metrics(metrics: Mapping[str, float], *, error_type: ErrorFactory) -> dict[str, float]:
    if len(metrics) > MAX_MAPPING_ENTRIES:
        _fail(error_type, "metrics exceed the entry bound")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if type(key) is not str or not key or len(key) > MAX_KEY_LENGTH:
            _fail(error_type, "metric keys must be bounded non-empty strings")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            _fail(error_type, "metrics must contain finite numeric values", error)
        if type(value) is bool or not math.isfinite(numeric_value):
            _fail(error_type, "metrics must contain finite numeric values")
        result[key] = numeric_value
    return result


def reject_sensitive_key(key: str, *, error_type: ErrorFactory) -> None:
    if _SENSITIVE_KEY.search(key.lower().replace("-", "_")):
        _fail(error_type, "recorder keys must not contain secret-like names")


def validate_string_mapping(value: Mapping[str, str] | None, *, field: str, error_type: ErrorFactory) -> dict[str, str]:
    validated = validate_mapping(value, field=field, error_type=error_type)
    result: dict[str, str] = {}
    for key, item in validated.items():
        if not isinstance(item, str):
            _fail(error_type, f"{field} values must be bounded strings")
        result[key] = item
    return result


def validate_seeds(seeds: Sequence[int], *, error_type: ErrorFactory) -> tuple[int, ...]:
    if len(seeds) > MAX_SEQUENCE_ITEMS:
        _fail(error_type, "seeds exceed the entry bound")
    result: list[int] = []
    for seed in seeds:
        if type(seed) is bool or type(seed) is not int or seed < 0:
            _fail(error_type, "seeds must be non-negative integers")
        result.append(seed)
    return tuple(result)


def read_artifact(content: bytes | bytearray | memoryview | str | Path, *, error_type: ErrorFactory) -> bytes:
    if type(content) is bytes:
        if len(content) > MAX_ARTIFACT_BYTES:
            _fail(error_type, "artifact exceeds the size bound")
        return content
    if type(content) is bytearray:
        if len(content) > MAX_ARTIFACT_BYTES:
            _fail(error_type, "artifact exceeds the size bound")
        return bytes(content)
    if isinstance(content, memoryview):
        if content.nbytes > MAX_ARTIFACT_BYTES:
            _fail(error_type, "artifact exceeds the size bound")
        return content.tobytes()
    if isinstance(content, (str, Path)):
        return read_artifact_path(Path(content), error_type=error_type)
    _fail(error_type, "artifact content must be bytes, bytearray, memoryview, or a path")


def read_artifact_path(path: Path, *, error_type: ErrorFactory) -> bytes:
    if has_reparse_component(path, error_type=error_type):
        _fail(error_type, "artifact path must not traverse symlinks or reparse points")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(error_type, "artifact path cannot be opened safely", error)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_ARTIFACT_BYTES:
                _fail(error_type, "artifact must be a bounded regular file")
            data = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except Exception as error:
        if isinstance(error, error_type):
            raise
        _fail(error_type, "artifact path could not be read safely", error)
    if len(data) > MAX_ARTIFACT_BYTES:
        _fail(error_type, "artifact exceeds the size bound")
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        _fail(error_type, "artifact changed while it was being read")
    return data


def has_reparse_component(path: Path, *, error_type: ErrorFactory) -> bool:
    candidate = path if path.is_absolute() else (Path.cwd() / path)
    current = Path(candidate.anchor) if candidate.anchor else Path()
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current /= part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail(error_type, "artifact path cannot be inspected safely", error)
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            return True
    return False


def safe_artifact_path(root: str | Path, name: str, *, error_type: ErrorFactory) -> Path:
    safe_name = validate_artifact_name(name, error_type=error_type)
    if has_reparse_component(Path(root), error_type=error_type):
        _fail(error_type, "artifact root must not be a symlink or reparse point")
    try:
        canonical_root = Path(root).resolve(strict=True)
    except OSError as error:
        _fail(error_type, "artifact root cannot be resolved safely", error)
    target = canonical_root.joinpath(*safe_name.split("/"))
    try:
        resolved_target = target.resolve(strict=False)
    except OSError as error:
        _fail(error_type, "artifact path cannot be resolved safely", error)
    try:
        resolved_target.relative_to(canonical_root)
    except ValueError as error:
        _fail(error_type, "artifact path escapes its temporary root", error)
    current = canonical_root
    for part in safe_name.split("/"):
        current /= part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail(error_type, "artifact path cannot be inspected safely", error)
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            _fail(error_type, "artifact path must not traverse symlinks or reparse points")
    return target
