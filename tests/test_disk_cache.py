"""Focused Sprint 74 Task 04 tests for the SQLite disk cache."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys

import pytest

from latent_anything.portable import encode_portable
from latent_anything.runtime import CacheKey
from latent_anything.runtime.disk_cache import DiskCacheError, SQLiteDiskCache, make_disk_cache_key


def _key(seed: str = "a") -> CacheKey:
    return CacheKey("runtime", "encode", "component", seed * 64, "b" * 64, "c" * 64, "0.1")


def test_state_and_provenance_are_part_of_stable_disk_key() -> None:
    first = make_disk_cache_key(
        _key(), plugin_identity="plugin@1", checkpoint_identity="ckpt-a", behavior_state_identity="state-a"
    )
    second = make_disk_cache_key(
        _key(), plugin_identity="plugin@1", checkpoint_identity="ckpt-a", behavior_state_identity="state-b"
    )
    assert first != second
    assert first == make_disk_cache_key(
        _key(), plugin_identity="plugin@1", checkpoint_identity="ckpt-a", behavior_state_identity="state-a"
    )
    with pytest.raises(DiskCacheError, match="non-empty"):
        make_disk_cache_key(_key(), plugin_identity="", checkpoint_identity="ckpt-a", behavior_state_identity="state-a")


def test_portable_cache_seam_validates_payload_and_restores_it(tmp_path: object) -> None:
    cache = SQLiteDiskCache(str(tmp_path) + "/portable.sqlite")
    key = make_disk_cache_key(
        _key(), plugin_identity="plugin@1", checkpoint_identity="ckpt-a", behavior_state_identity="state-a"
    )
    payload = encode_portable({"answer": 42})

    cache.set_portable(key, payload)

    assert cache.get_portable(key) == payload
    with pytest.raises(DiskCacheError, match="portable payload"):
        cache.set_portable(key, b"not-an-arrow-envelope")


def test_sqlite_cache_round_trip_cross_process_and_deterministic_eviction(tmp_path: object) -> None:
    path = str(tmp_path) + "/cache.sqlite"
    cache = SQLiteDiskCache(path, max_bytes=8, max_entries=2)
    first = hashlib.sha256(b"a").hexdigest()
    second = hashlib.sha256(b"b").hexdigest()
    third = hashlib.sha256(b"c").hexdigest()
    cache.set(first, b"a")
    cache.set(second, b"b")
    assert cache.get(first) == b"a"
    cache.set(third, b"ccc")
    assert cache.get(second) is None

    code = (
        "from latent_anything.runtime.disk_cache import SQLiteDiskCache; "
        f"c=SQLiteDiskCache({path!r}); print(c.get({third!r}))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert "b'ccc'" in result.stdout
    assert cache.stats.hits >= 1
    assert cache.stats.misses >= 1


def test_corrupt_rows_fail_as_misses_and_invalid_keys_fail_closed(tmp_path: object) -> None:
    path = str(tmp_path) + "/cache.sqlite"
    cache = SQLiteDiskCache(path)
    key = hashlib.sha256(b"payload").hexdigest()
    cache.set(key, b"payload")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE cache_entries SET payload = ? WHERE key = ?", (b"tampered", key))
    assert cache.get(key) is None
    with pytest.raises(DiskCacheError, match="64-character"):
        cache.get("not-a-key")


def test_oversized_corrupt_rows_are_rejected_before_payload_load(tmp_path: object) -> None:
    path = str(tmp_path) + "/oversized.sqlite"
    cache = SQLiteDiskCache(path, max_bytes=8)
    key = hashlib.sha256(b"payload").hexdigest()
    cache.set(key, b"payload")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE cache_entries SET size = ? WHERE key = ?", (9, key))

    assert cache.get(key) is None
    assert cache.stats.entries == 0


def test_cache_rejects_symlink_database_path(tmp_path: object) -> None:
    target = tmp_path / "target.sqlite"
    link = tmp_path / "link.sqlite"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this test environment")
    with pytest.raises(DiskCacheError, match="symlink|reparse"):
        SQLiteDiskCache(link)


def test_concurrent_process_writers_preserve_sqlite_integrity(tmp_path: object) -> None:
    path = str(tmp_path) + "/concurrent.sqlite"
    code = (
        "import hashlib; "
        "from latent_anything.runtime.disk_cache import SQLiteDiskCache; "
        f"c=SQLiteDiskCache({path!r}, max_entries=100); "
        "[c.set(hashlib.sha256(f'{i}'.encode()).hexdigest(), f'{i}'.encode()) for i in range(10)]"
    )
    processes = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(3)]
    assert all(process.wait(timeout=30) == 0 for process in processes)
    cache = SQLiteDiskCache(path, max_entries=100)
    assert cache.stats.entries == 10
    assert cache.get(hashlib.sha256(b"9").hexdigest()) == b"9"
