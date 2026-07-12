from __future__ import annotations

import asyncio
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers as vf
from datasets import Dataset
from pydantic import ValidationError

from medfact_bench import (
    EXPECTED_DATASET_COUNTS,
    MedFactHarness,
    MedFactHarnessConfig,
    MedFactTaskset,
    MedFactTasksetConfig,
    load_environment,
)
from medfact_bench import taskset as taskset_module

REAL_DATASET_PATH = Path(
    "/Users/kouatemuhamed/Claude/Projects/MedARC Agentic Medical Fact Verifier/"
    "datasets/MedFact-Bench/MedFact-Bench.parquet"
)


def _row(
    dataset: str = "scifact",
    label: str = "SUPPORT",
    user_prompt: str = "Article:\nEvidence\n\nClaim:\nClaim",
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "claim": "Claim",
        "source": "Evidence",
        "label": label,
        "system_prompt": "Canonical system prompt",
        "user_prompt": user_prompt,
    }


def _dataset(rows: list[dict[str, str]] | None = None) -> Dataset:
    return Dataset.from_list(rows or [_row()])


def _load_tasks(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, str]], **config: object):
    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: _dataset(rows))
    return MedFactTaskset(MedFactTasksetConfig(id="medfact-bench", **config)).load()


def test_taskset_preserves_order_duplicates_and_prompt_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row(user_prompt="First prompt"),
        _row(dataset="healthver", label="NEI", user_prompt="Second prompt"),
        _row(user_prompt="First prompt"),
    ]

    tasks = _load_tasks(monkeypatch, rows)

    assert [task.data.idx for task in tasks] == [0, 1, 2]
    assert [task.data.prompt for task in tasks] == ["First prompt", "Second prompt", "First prompt"]
    assert all(task.data.system_prompt == "Canonical system prompt" for task in tasks)
    assert tasks[0].data == tasks[2].data.model_copy(update={"idx": 0})
    assert tasks[1].data.info == {"dataset": "healthver"}


def test_taskset_filters_before_constructing_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(), _row(dataset="healthver", label="NEI"), _row()]

    tasks = _load_tasks(monkeypatch, rows, subset="healthver")

    assert len(tasks) == 1
    assert tasks[0].data.idx == 0
    assert tasks[0].data.info == {"dataset": "healthver"}


def test_training_split_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: pytest.fail("dataset should not load"))
    taskset = MedFactTaskset(MedFactTasksetConfig(id="medfact-bench", split="train"))

    with pytest.raises(ValueError, match="only the 'eval' split"):
        taskset.load()


def test_invalid_subset_is_rejected_by_typed_config() -> None:
    with pytest.raises(ValidationError):
        MedFactTasksetConfig(id="medfact-bench", subset="invalid")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: [{key: value for key, value in rows[0].items() if key != "source"}], "missing required columns"),
        (lambda rows: [{**rows[0], "label": "MAYBE"}], "unsupported labels"),
        (lambda rows: [{**rows[0], "dataset": "unknown"}], "unsupported component values"),
        (lambda rows: [{**rows[0], "claim": ""}], "empty value"),
        (lambda rows: [rows[0], {**rows[0], "system_prompt": "Different"}], "exactly one distinct"),
    ],
)
def test_dataset_schema_validation(
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    message: str,
) -> None:
    rows = mutator([_row()])
    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: _dataset(rows))

    with pytest.raises(ValueError, match=message):
        MedFactTaskset(MedFactTasksetConfig(id="medfact-bench")).load()


def test_invalid_local_path_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="dataset path does not exist"):
        taskset_module._load_source_dataset(str(missing), None)


def test_task_rewards_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _load_tasks(monkeypatch, [_row()])[0]
    trace = SimpleNamespace(last_reply="<think>Supported</think><score>2</score>")

    assert asyncio.run(task.accuracy(trace)) == 1.0
    assert asyncio.run(task.parseable_score(trace)) == 1.0
    assert asyncio.run(task.strict_format(trace)) == 1.0
    assert task.accuracy._vf_weight == 1.0
    assert task.parseable_score.metric is True
    assert task.strict_format.metric is True


def test_harness_preserves_separate_system_and_user_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _load_tasks(monkeypatch, [_row(user_prompt="Exact user prompt")])[0]
    harness = MedFactHarness(MedFactHarnessConfig(id="medfact-bench"))

    system_prompt, user_prompt = harness.resolve_prompt(task.data)

    assert system_prompt == "Canonical system prompt"
    assert user_prompt == "Exact user prompt"


def test_environment_loader_returns_one_turn_singleturn_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`verifiers.load_environment("medfact-bench")` / `vf-eval` drive the classic v0 API,
    so the loader must return a one-turn `SingleTurnEnv` built over the eval dataset."""
    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: _dataset())

    environment = load_environment(subset="scifact")

    assert isinstance(environment, vf.SingleTurnEnv)
    assert environment.max_turns == 1
    assert environment.env_id == "medfact-bench"
    assert environment.env_args["subset"] == "scifact"

    eval_dataset = environment.get_eval_dataset()
    assert eval_dataset[0]["prompt"] == [
        {"role": "system", "content": "Canonical system prompt"},
        {"role": "user", "content": "Article:\nEvidence\n\nClaim:\nClaim"},
    ]
    assert eval_dataset[0]["answer"] == "SUPPORT"


def test_environment_loader_tolerates_generic_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """`vf-eval` may forward extra args (e.g. `max_turns`) via the generic bridge; they are
    recorded but must not break the single-turn protocol."""
    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: _dataset())

    environment = load_environment(max_turns=9)

    assert isinstance(environment, vf.SingleTurnEnv)
    assert environment.max_turns == 1
    assert environment.env_args["max_turns"] == 9


def test_environment_rewards_score_three_way_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v0 rubric must reward correct labels and report the parse/format diagnostics,
    matching the v1 taskset's reward/metrics."""
    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: _dataset())
    environment = load_environment(subset="scifact")

    async def score(completion: list[dict[str, str]]) -> dict[str, float]:
        state = {"completion": completion, "answer": "SUPPORT", "prompt": [], "info": {}, "task": "medfact-bench"}
        await environment.rubric.score_rollout(state)
        return {"reward": state["reward"], **state["metrics"]}

    correct = asyncio.run(score([{"role": "assistant", "content": "<think>r</think>\n<score>2</score>"}]))
    assert correct["reward"] == 1.0
    assert correct["accuracy"] == 1.0
    assert correct["parseable_score"] == 1.0
    assert correct["strict_format"] == 1.0

    wrong = asyncio.run(score([{"role": "assistant", "content": "text <score>-2</score> tail"}]))
    assert wrong["reward"] == 0.0
    assert wrong["accuracy"] == 0.0
    assert wrong["parseable_score"] == 1.0
    assert wrong["strict_format"] == 0.0

    unparseable = asyncio.run(score([{"role": "assistant", "content": "no score"}]))
    assert unparseable["accuracy"] == 0.0
    assert unparseable["parseable_score"] == 0.0


def test_environment_records_prediction_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring records the model's mapped verdict (``prediction``) and raw ``pred_score``
    next to the ground-truth ``answer``, and the env auto-persists those columns."""
    from medfact_bench.environment import PREDICTION_COLUMNS

    monkeypatch.setattr(taskset_module, "_load_source_dataset", lambda *_: _dataset())
    environment = load_environment(subset="scifact")
    assert PREDICTION_COLUMNS == ("prediction", "pred_score")

    async def score(content: str) -> dict:
        state = {
            "completion": [{"role": "assistant", "content": content}],
            "answer": "SUPPORT",
            "prompt": [],
            "info": {},
        }
        await environment.rubric.score_rollout(state)
        return state

    correct = asyncio.run(score("<think>r</think>\n<score>2</score>"))
    assert correct["answer"] == "SUPPORT"  # ground truth unchanged
    assert correct["prediction"] == "SUPPORT"
    assert correct["pred_score"] == 2

    wrong = asyncio.run(score("blah <score>-2</score>"))
    assert wrong["prediction"] == "CONTRADICT"
    assert wrong["pred_score"] == -2

    invalid = asyncio.run(score("no score here"))
    assert invalid["prediction"] == "INVALID"
    assert invalid["pred_score"] is None


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MEDFACT_BENCH_RUN_DATASET_INTEGRATION") != "1",
    reason="Set MEDFACT_BENCH_RUN_DATASET_INTEGRATION=1 to run the local Parquet integration test.",
)
def test_local_parquet_dataset_invariants(tmp_path: Path) -> None:
    if not REAL_DATASET_PATH.exists():
        pytest.fail(f"Configured MedFact-Bench Parquet file does not exist: {REAL_DATASET_PATH}")

    tasks = MedFactTaskset(
        MedFactTasksetConfig(
            id="medfact-bench",
            dataset_path=str(REAL_DATASET_PATH),
            cache_dir=str(tmp_path / "datasets-cache"),
        )
    ).load()

    assert len(tasks) == 14_274
    assert Counter(task.data.info["dataset"] for task in tasks) == Counter(EXPECTED_DATASET_COUNTS)
    assert Counter(task.data.answer for task in tasks) == {
        "SUPPORT": 10_118,
        "NEI": 2_988,
        "CONTRADICT": 1_168,
    }
    assert all(task.data.system_prompt and task.data.prompt for task in tasks)
    assert [task.data.idx for task in tasks] == list(range(14_274))

    duplicate_excess = sum(
        count - 1
        for count in Counter((task.data.prompt, task.data.answer, task.data.info["dataset"]) for task in tasks).values()
        if count > 1
    )
    assert duplicate_excess == 2_261
