"""Runtime profiling helpers for Layer C execution paths.

Sprint 24 adds lightweight profiling hooks that runtime-aware callers can
pass into concrete pipeline and executor paths. The hooks stay concrete:
they record stage timings for today's runtime stories without freezing a
broader runtime protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, TypeVar, cast

RuntimeStage = Literal["cache", "decode", "encode", "method", "transition"]
ProfileMetadataValue = str | int | float | bool | None
_T = TypeVar("_T")


@dataclass(frozen=True)
class ProfileEvent:
    """One observed runtime stage timing."""

    stage: RuntimeStage
    duration_seconds: float
    metadata: dict[str, ProfileMetadataValue] = field(default_factory=lambda: cast(dict[str, ProfileMetadataValue], {}))


@dataclass(frozen=True)
class RuntimeProfile:
    """Immutable snapshot of recorded runtime events."""

    events: tuple[ProfileEvent, ...]

    @property
    def total_seconds(self) -> float:
        """Return the sum of all recorded durations."""
        return sum(event.duration_seconds for event in self.events)

    def stage_totals(self) -> dict[RuntimeStage, float]:
        """Aggregate total duration by stage name."""
        totals: dict[RuntimeStage, float] = {}
        for event in self.events:
            totals[event.stage] = totals.get(event.stage, 0.0) + event.duration_seconds
        return totals


class RuntimeProfiler:
    """Mutable collector for runtime stage timings."""

    def __init__(self) -> None:
        self._events: list[ProfileEvent] = []

    def record(
        self,
        stage: RuntimeStage,
        duration_seconds: float,
        **metadata: ProfileMetadataValue,
    ) -> None:
        """Record one stage duration."""
        if duration_seconds < 0.0:
            msg = f"duration_seconds must be non-negative, got {duration_seconds}"
            raise ValueError(msg)
        self._events.append(ProfileEvent(stage=stage, duration_seconds=duration_seconds, metadata=dict(metadata)))

    def measure(
        self,
        stage: RuntimeStage,
        operation: Callable[[], _T],
        **metadata: ProfileMetadataValue,
    ) -> _T:
        """Execute a synchronous operation and record its elapsed time."""
        start = perf_counter()
        result = operation()
        self.record(stage, perf_counter() - start, **metadata)
        return result

    def snapshot(self) -> RuntimeProfile:
        """Return an immutable snapshot of recorded events."""
        return RuntimeProfile(events=tuple(self._events))

    def clear(self) -> None:
        """Remove all recorded events."""
        self._events.clear()
