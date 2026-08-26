"""Stable facade for versioned, local run evidence.

The public classes and functions in this module retain their historical import
paths and serialization identities. Codec/migration, schema, persistence,
artifact, and comparison mechanics live in focused private modules.
"""

from __future__ import annotations

from latent_anything._run_record_codec import (
    RUN_RECORD_SCHEMA_VERSION,
    compute_run_identity,
    migrate_run_record,
    runtime_profile_metadata,
)
from latent_anything._run_record_comparison import RunComparisonReport, build_comparison_report
from latent_anything._run_record_persistence import DuplicateRunError, FileSystemRunRecorder
from latent_anything._run_record_schema import ArtifactRef, RunRecord, RunStatus

# Preserve module identities for imports, reprs, and pickle payloads created by
# the frozen public run-record contract.
for _public_type in (ArtifactRef, RunRecord, RunComparisonReport, DuplicateRunError, FileSystemRunRecorder):
    _public_type.__module__ = __name__


__all__ = [
    "ArtifactRef",
    "DuplicateRunError",
    "FileSystemRunRecorder",
    "RUN_RECORD_SCHEMA_VERSION",
    "RunComparisonReport",
    "RunRecord",
    "RunStatus",
    "build_comparison_report",
    "compute_run_identity",
    "migrate_run_record",
    "runtime_profile_metadata",
]
