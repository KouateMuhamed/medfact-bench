"""Standalone MedFact-Bench environment for Verifiers and Prime RL."""

from __future__ import annotations

import verifiers as vf

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
from .environment import build_environment
from .harness import MedFactHarness, MedFactHarnessConfig
from .taskset import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_DATASET_COUNTS,
    MedFactSubset,
    MedFactTask,
    MedFactTaskData,
    MedFactTaskset,
    MedFactTasksetConfig,
)

ENVIRONMENT_ID = "medfact-bench"


def load_taskset(config: MedFactTasksetConfig) -> MedFactTaskset:
    """Construct the typed MedFact-Bench taskset."""
    return MedFactTaskset(config=config)


def load_harness(config: MedFactHarnessConfig) -> MedFactHarness:
    """Construct the prompt-preserving endpoint-backed harness."""
    return MedFactHarness(config=config)


def load_environment(
    subset: str = "all",
    split: str = "eval",
    dataset_path: str | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Construct the one-turn MedFact-Bench environment (classic Verifiers v0 API).

    This is the entry point that ``verifiers.load_environment("medfact-bench")``
    and ``vf-eval`` resolve, so it returns a v0 ``SingleTurnEnv`` exposing
    ``evaluate``/``start_server``/``stop_server``. The native v1 taskset/harness
    path is resolved separately by plugin id via ``MedFactTaskset`` /
    ``MedFactHarness``; both share the same dataset loading and scoring.
    """
    return build_environment(
        subset=subset,
        split=split,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        **kwargs,
    )


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
