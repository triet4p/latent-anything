"""Small process-safe SQLite cache for checksummed portable payloads.

The cache stores opaque bytes only. Callers must use a portable artifact
payload and include every behavior-affecting component/config/checkpoint and
plugin identity in :func:`make_disk_cache_key`; fitted/state-mutating outputs
must not be cached without that complete state envelope.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from latent_anything.runtime.cache import CacheKey

_SCHEMA_VERSION = "disk-cache-v1"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class DiskCacheError(ValueError):
    """Raised for unsafe cache paths, malformed keys, or invalid payloads."""


@dataclass(frozen=True)
class DiskCacheStats:
    """Point-in-time SQLite cache statistics."""

    hits: int
    misses: int
    sets: int
    entries: int
    bytes: int


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        stat_result = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(stat_result, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_link_like_components(path: Path) -> None:
    for component in (path, *path.parents):
        if _is_link_like(component):
            raise DiskCacheError("disk cache path must not contain a symlink, junction, or reparse point")


def make_disk_cache_key(
    key: CacheKey,
    *,
    plugin_identity: str = "",
    checkpoint_identity: str = "",
    behavior_state_identity: str = "",
) -> str:
    """Hash a runtime key with provenance and complete behavior-state identities."""

    for label, value in (
        ("plugin_identity", plugin_identity),
        ("checkpoint_identity", checkpoint_identity),
        ("behavior_state_identity", behavior_state_identity),
    ):
        if not value or value != value.strip():
            raise DiskCacheError(f"{label} must be a non-empty canonical string")

    values = {
        "schema_version": _SCHEMA_VERSION,
        "runtime_key": asdict(key),
        "plugin_identity": plugin_identity,
        "checkpoint_identity": checkpoint_identity,
        "behavior_state_identity": behavior_state_identity,
    }
    try:
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiskCacheError(f"cache key is not canonical: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


class SQLiteDiskCache:
    """A bounded, deterministic-eviction SQLite cache of portable bytes."""

    def __init__(
        self, path: str | os.PathLike[str], *, max_bytes: int = 512 * 1024 * 1024, max_entries: int = 10_000
    ) -> None:
        if max_bytes < 1 or max_entries < 1:
            raise ValueError("disk cache limits must be positive")
        self.path = Path(path)
        _reject_link_like_components(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _reject_link_like_components(self.path)
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _session(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK(size >= 0),
                    created_ns INTEGER NOT NULL,
                    accessed_ns INTEGER NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS cache_lru ON cache_entries(accessed_ns, created_ns, key)")

    @staticmethod
    def _validate_key(key: str) -> None:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise DiskCacheError("disk cache keys must be 64-character lowercase SHA-256 hex strings")

    def get(self, key: str) -> bytes | None:
        """Return a defensive byte copy or miss; corrupt rows are deleted."""

        self._validate_key(key)
        now = time.time_ns()
        with self._session() as connection:
            connection.execute("BEGIN")
            try:
                row = connection.execute(
                    "SELECT size, payload_sha256 FROM cache_entries WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    connection.commit()
                    self._misses += 1
                    return None
                size = row[0]
                digest = row[1]
                if (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                    or size > self.max_bytes
                    or not isinstance(digest, str)
                ):
                    connection.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    connection.commit()
                    self._misses += 1
                    return None
                payload_row = connection.execute("SELECT payload FROM cache_entries WHERE key = ?", (key,)).fetchone()
                if payload_row is None or not isinstance(payload_row[0], (bytes, bytearray, memoryview)):
                    connection.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    connection.commit()
                    self._misses += 1
                    return None
                payload = bytes(payload_row[0])
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                    connection.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    connection.commit()
                    self._misses += 1
                    return None
                connection.execute(
                    "UPDATE cache_entries SET accessed_ns = ?, hits = hits + 1 WHERE key = ?", (now, key)
                )
                connection.commit()
                self._hits += 1
                return payload
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def set(self, key: str, payload: object) -> None:
        """Atomically store low-level bytes and evict deterministic LRU rows.

        Callers caching framework artifacts must use :meth:`set_portable` so
        the payload is validated as a portable envelope before storage.
        """

        self._validate_key(key)
        if not isinstance(payload, bytes):
            raise TypeError("disk cache payload must be bytes")
        if len(payload) > self.max_bytes:
            raise DiskCacheError("disk cache payload exceeds max_bytes")
        now = time.time_ns()
        digest = hashlib.sha256(payload).hexdigest()
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO cache_entries(key, payload, payload_sha256, size, created_ns, accessed_ns, hits)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(key) DO UPDATE SET
                    payload = excluded.payload,
                    payload_sha256 = excluded.payload_sha256,
                    size = excluded.size,
                    accessed_ns = excluded.accessed_ns,
                    hits = 0
                """,
                (key, payload, digest, len(payload), now, now),
            )
            self._evict(connection)
            connection.commit()
        self._sets += 1

    def set_portable(self, key: str, payload: bytes) -> None:
        """Validate and cache a portable envelope at the coherent seam."""

        self._validate_key(key)
        from latent_anything.portable import PortableLimits, PortableNodeError, decode_portable

        try:
            decode_portable(payload, limits=PortableLimits(max_input_bytes=min(self.max_bytes, 768 * 1024 * 1024)))
        except (PortableNodeError, TypeError, ValueError) as exc:
            raise DiskCacheError(f"disk cache portable payload is invalid: {exc}") from exc
        self.set(key, payload)

    def get_portable(self, key: str) -> bytes | None:
        """Return a cached portable envelope, deleting invalid decoded bytes."""

        payload = self.get(key)
        if payload is None:
            return None
        from latent_anything.portable import PortableLimits, PortableNodeError, decode_portable

        try:
            decode_portable(payload, limits=PortableLimits(max_input_bytes=min(self.max_bytes, 768 * 1024 * 1024)))
        except (PortableNodeError, TypeError, ValueError) as exc:
            with self._session() as connection:
                connection.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            raise DiskCacheError(f"cached portable payload is invalid: {exc}") from exc
        return payload

    def _evict(self, connection: sqlite3.Connection) -> None:
        while True:
            count, total = cast(
                tuple[int, int],
                connection.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM cache_entries").fetchone(),
            )
            if count <= self.max_entries and total <= self.max_bytes:
                return
            victim = connection.execute(
                "SELECT key FROM cache_entries ORDER BY accessed_ns ASC, created_ns ASC, key ASC LIMIT 1"
            ).fetchone()
            if victim is None:
                return
            connection.execute("DELETE FROM cache_entries WHERE key = ?", (victim[0],))

    def clear(self) -> None:
        """Remove all cache entries while preserving the database schema."""

        with self._session() as connection:
            connection.execute("DELETE FROM cache_entries")

    @property
    def stats(self) -> DiskCacheStats:
        """Return current counters and bounded occupancy."""

        with self._session() as connection:
            entries, total = cast(
                tuple[int, int],
                connection.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM cache_entries").fetchone(),
            )
        # Sets are process-local because the durable table intentionally only
        # stores portable payload state, not telemetry that affects behavior.
        return DiskCacheStats(hits=self._hits, misses=self._misses, sets=self._sets, entries=entries, bytes=total)


__all__ = ["DiskCacheError", "DiskCacheStats", "SQLiteDiskCache", "make_disk_cache_key"]
