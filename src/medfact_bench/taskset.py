"""Native Verifiers 0.2 taskset for MedFact-Bench."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from datasets import Dataset, load_dataset

from .core import MedFactScoreParser, prediction_label

DATASET_ID = "ncbi/MedFact-Bench"
DATASET_REVISION = "249028caf7ad5a3e63331269a606f4b2696693ed"

EXPECTED_DATASET_COUNTS = {
    "scifact": 340,
    "healthver": 903,
    "medaesqa": 9_106,
    "pubmedqa-fact": 500,
    "bioasq-fact": 3_425,
}

LABELS = ("SUPPORT", "NEI", "CONTRADICT")
REQUIRED_COLUMNS = (
    "dataset",
    "claim",
    "source",
    "label",
    "system_prompt",
    "user_prompt",
)


class MedFactSubset(str, Enum):
    """Supported MedFact-Bench component datasets."""

    ALL = "all"
    SCIFACT = "scifact"
    HEALTHVER = "healthver"
    MEDAESQA = "medaesqa"
    PUBMEDQA_FACT = "pubmedqa-fact"
    BIOASQ_FACT = "bioasq-fact"


class MedFactTaskData(vf.TaskData):
    """One held-out MedFact-Bench verification example."""

    answer: Literal["SUPPORT", "NEI", "CONTRADICT"]
    info: dict[str, str]


class MedFactTask(vf.Task[MedFactTaskData]):
    """Single-turn classification behavior and diagnostics."""

    @vf.reward(weight=1.0)
    async def accuracy(self, trace: vf.Trace) -> float:
        return float(prediction_label(trace.last_reply) == self.data.answer)

    @vf.metric
    async def parseable_score(self, trace: vf.Trace) -> float:
        parser = MedFactScoreParser()
        return float(parser.parse_score(trace.last_reply) is not None)

    @vf.metric
    async def strict_format(self, trace: vf.Trace) -> float:
        return float(MedFactScoreParser().is_strict_format(trace.last_reply))


class MedFactTasksetConfig(vf.TasksetConfig):
    """Dataset selection for the held-out benchmark taskset."""

    subset: Literal[
        "all",
        "scifact",
        "healthver",
        "medaesqa",
        "pubmedqa-fact",
        "bioasq-fact",
    ] = "all"
    split: Literal["eval", "train"] = "eval"
    dataset_path: str | None = None
    cache_dir: str | None = None


def _load_source_dataset(dataset_path: str | None, cache_dir: str | None) -> Dataset:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MedFact-Bench dataset path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"MedFact-Bench dataset path must be a Parquet file: {path}")
        try:
            return load_dataset(
                "parquet",
                data_files=str(path),
                split="train",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise ValueError(f"Failed to load the local MedFact-Bench Parquet file at {path}: {exc}") from exc

    try:
        return load_dataset(
            DATASET_ID,
            split="train",
            revision=DATASET_REVISION,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load {DATASET_ID} at revision {DATASET_REVISION}: {exc}") from exc


def _validate_dataset(dataset: Dataset) -> str:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataset.column_names))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"MedFact-Bench dataset is missing required columns: {missing}.")
    if len(dataset) == 0:
        raise ValueError("MedFact-Bench dataset is empty.")

    for column in REQUIRED_COLUMNS:
        values = dataset[column]
        null_count = sum(value is None for value in values)
        if null_count:
            raise ValueError(f"MedFact-Bench column {column!r} contains {null_count} null value(s).")
        non_string_count = sum(not isinstance(value, str) for value in values)
        if non_string_count:
            raise ValueError(f"MedFact-Bench column {column!r} contains {non_string_count} non-string value(s).")
        empty_count = sum(not value.strip() for value in values)
        if empty_count:
            raise ValueError(f"MedFact-Bench column {column!r} contains {empty_count} empty value(s).")

    dataset_values = set(dataset["dataset"])
    unexpected_datasets = sorted(dataset_values - set(EXPECTED_DATASET_COUNTS))
    if unexpected_datasets:
        values = ", ".join(unexpected_datasets)
        raise ValueError(f"MedFact-Bench dataset contains unsupported component values: {values}.")

    label_values = set(dataset["label"])
    unexpected_labels = sorted(label_values - set(LABELS))
    if unexpected_labels:
        values = ", ".join(unexpected_labels)
        raise ValueError(f"MedFact-Bench dataset contains unsupported labels: {values}.")

    system_prompts = set(dataset["system_prompt"])
    if len(system_prompts) != 1:
        raise ValueError(
            f"MedFact-Bench dataset must contain exactly one distinct system prompt; found {len(system_prompts)}."
        )
    return next(iter(system_prompts))


def _select_dataset(dataset: Dataset, subset: MedFactSubset) -> Dataset:
    if subset is MedFactSubset.ALL:
        return dataset
    selected = dataset.filter(
        lambda row: row["dataset"] == subset.value,
        load_from_cache_file=False,
    )
    if len(selected) == 0:
        raise ValueError(f"MedFact-Bench subset {subset.value!r} contains no rows.")
    return selected


class MedFactTaskset(vf.Taskset[MedFactTask, MedFactTasksetConfig]):
    """Load MedFact-Bench as immutable evaluation tasks."""

    def load(self) -> list[MedFactTask]:
        if self.config.split != "eval":
            raise ValueError("MedFact-Bench v0.1.0 exposes only the 'eval' split; training data is not available.")

        source = _load_source_dataset(self.config.dataset_path, self.config.cache_dir)
        system_prompt = _validate_dataset(source)
        subset = MedFactSubset(self.config.subset)
        selected = _select_dataset(source, subset)

        return [
            MedFactTask(
                MedFactTaskData(
                    idx=index,
                    prompt=row["user_prompt"],
                    system_prompt=system_prompt,
                    answer=row["label"],
                    info={"dataset": row["dataset"]},
                ),
                self.config.task,
            )
            for index, row in enumerate(selected)
        ]
