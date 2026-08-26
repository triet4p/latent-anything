"""Pure comparison/report assembly for recorded runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from latent_anything._run_record_codec import json_value
from latent_anything._run_record_schema import RunRecord


@dataclass(frozen=True)
class RunComparisonReport:
    """Metric comparison across at least two recorded runs."""

    title: str
    baseline_run_id: str
    runs: tuple[Mapping[str, object], ...]
    metric_deltas: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "title": self.title,
            "baseline_run_id": self.baseline_run_id,
            "runs": [json_value(run) for run in self.runs],
            "metric_deltas": json_value(self.metric_deltas),
        }


def build_comparison_report(
    records: Sequence[RunRecord], *, title: str = "Latent Anything run comparison"
) -> RunComparisonReport:
    """Compare metrics and provenance for two or more records."""
    if len(records) < 2:
        raise ValueError("comparison requires at least two run records")
    baseline = records[0]
    baseline_metrics = dict(baseline.metrics)
    deltas: dict[str, dict[str, float]] = {}
    for record in records[1:]:
        deltas[record.run_id] = {
            metric: value - baseline_metrics[metric]
            for metric, value in record.metrics.items()
            if metric in baseline_metrics
        }
    rows: list[Mapping[str, object]] = [
        {
            "run_id": record.run_id,
            "name": record.name,
            "status": record.status,
            "identity": record.identity,
            "model_revisions": dict(record.model_revisions),
            "dataset_revisions": dict(record.dataset_revisions),
            "seeds": list(record.seeds),
            "metrics": dict(record.metrics),
            "theory_evidence_ids": list(record.theory_evidence_ids),
        }
        for record in records
    ]
    return RunComparisonReport(
        title=title,
        baseline_run_id=baseline.run_id,
        runs=tuple(rows),
        metric_deltas=deltas,
    )
