"""Standalone MedFact-Bench environment for Verifiers and Prime RL."""

from __future__ import annotations

import verifiers.v1 as vf

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


def load_environment(config: vf.EnvConfig) -> vf.Environment:
    """Construct a one-turn MedFact-Bench environment from a strict Verifiers config."""
    taskset_config = config.taskset
    if not taskset_config.id:
        taskset_config = MedFactTasksetConfig(id=ENVIRONMENT_ID)

    harness_config = config.harness
    if harness_config.id == "default":
        harness_config = MedFactHarnessConfig(
            id=ENVIRONMENT_ID,
            runtime=harness_config.runtime,
            env=harness_config.env,
            forward_env=harness_config.forward_env,
            disabled_tools=harness_config.disabled_tools,
        )

    normalized = config.model_copy(
        update={
            "taskset": taskset_config,
            "harness": harness_config,
            "max_turns": 1,
        }
    )
    return vf.Environment(config=normalized)


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
    "load_environment",
    "load_harness",
    "load_taskset",
    "parseable_score",
    "prediction_label",
    "score_to_label",
    "strict_format",
]
