"""Standalone MedFact-Bench environment for Verifiers and Prime RL."""

from __future__ import annotations

from .constants import DATASET_ID, DATASET_REVISION, EXPECTED_DATASET_COUNTS
from .core import (
    INVALID_LABEL,
    LABELS,
    MedFactScoreParser,
    accuracy,
    parseable_score,
    prediction_label,
    score_to_label,
    strict_format,
)

ENVIRONMENT_ID = "medfact-bench"


def load_taskset(config):
    """Construct the typed MedFact-Bench taskset."""
    from .taskset import MedFactTaskset

    return MedFactTaskset(config=config)


def load_harness(config):
    """Construct the prompt-preserving endpoint-backed harness."""
    from .harness import MedFactHarness

    return MedFactHarness(config=config)


def load_environment(
    subset: str = "all",
    split: str = "eval",
    dataset_path: str | None = None,
    cache_dir: str | None = None,
    **kwargs,
):
    """Construct the one-turn MedFact-Bench environment (classic Verifiers v0 API).

    This is the entry point that ``verifiers.load_environment("medfact-bench")``
    and ``vf-eval`` resolve, so it returns a v0 ``SingleTurnEnv`` exposing
    ``evaluate``/``start_server``/``stop_server``. The native v1 taskset/harness
    path is resolved separately by plugin id via ``MedFactTaskset`` /
    ``MedFactHarness``; both share the same dataset loading and scoring.
    """
    from .environment import build_environment

    return build_environment(
        subset=subset,
        split=split,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        **kwargs,
    )


def __getattr__(name: str):
    """Lazily expose Verifiers-backed classes without making reporting imports heavy."""
    if name == "build_environment":
        from .environment import build_environment

        return build_environment
    if name in {"MedFactHarness", "MedFactHarnessConfig"}:
        from .harness import MedFactHarness, MedFactHarnessConfig

        return {"MedFactHarness": MedFactHarness, "MedFactHarnessConfig": MedFactHarnessConfig}[name]
    if name in {
        "MedFactSubset",
        "MedFactTask",
        "MedFactTaskData",
        "MedFactTaskset",
        "MedFactTasksetConfig",
    }:
        from .taskset import MedFactSubset, MedFactTask, MedFactTaskData, MedFactTaskset, MedFactTasksetConfig

        return {
            "MedFactSubset": MedFactSubset,
            "MedFactTask": MedFactTask,
            "MedFactTaskData": MedFactTaskData,
            "MedFactTaskset": MedFactTaskset,
            "MedFactTasksetConfig": MedFactTasksetConfig,
        }[name]
    raise AttributeError(f"module 'medfact_bench' has no attribute {name!r}")


__all__ = [
    "DATASET_ID",
    "DATASET_REVISION",
    "ENVIRONMENT_ID",
    "EXPECTED_DATASET_COUNTS",
    "INVALID_LABEL",
    "LABELS",
    "MedFactHarness",
    "MedFactHarnessConfig",
    "MedFactScoreParser",
    "MedFactSubset",
    "MedFactTask",
    "MedFactTaskData",
    "MedFactTaskset",
    "MedFactTasksetConfig",
    "accuracy",
    "build_environment",
    "load_environment",
    "load_harness",
    "load_taskset",
    "parseable_score",
    "prediction_label",
    "score_to_label",
    "strict_format",
]
