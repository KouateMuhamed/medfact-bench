"""Classic Verifiers (v0) SingleTurnEnv for the Prime Environments Hub.

The native benchmark is defined as a v1 ``Taskset``/``Harness`` pair (see
``taskset.py``/``harness.py``), which is what ``uv run eval`` and Prime's v1
tooling resolve by plugin id. The Environments Hub quality-scan, however, drives
environments through the older ``vf-eval`` path, which calls
``verifiers.load_environment(env_id)`` and expects a v0 ``Environment`` exposing
``evaluate``/``start_server``/``stop_server``. This module builds that v0 view,
reusing the exact same dataset loading, parsing, and scoring as the v1 taskset so
both paths score identically.
"""

from __future__ import annotations

import verifiers as vf
from datasets import Dataset

from . import taskset as _taskset
from .core import MedFactScoreParser, score_to_label
from .core import accuracy as _score_accuracy
from .core import parseable_score as _score_parseable
from .core import strict_format as _score_strict_format

ENVIRONMENT_ID = "medfact-bench"

# Extra per-rollout columns recorded next to the ground-truth ``answer``: the model's
# mapped verdict (``prediction``) and the raw parsed integer (``pred_score``).
PREDICTION_COLUMNS = ("prediction", "pred_score")

# One shared parser; the reward wrappers ignore any parser injected by the rubric
# and always use the paper-format MedFact parser.
_PARSER = MedFactScoreParser()


def _record_prediction(state: dict) -> None:
    """Store the model's mapped prediction and raw score on the rollout state."""
    score = _PARSER.parse_completion(state.get("completion"))
    state["pred_score"] = score  # int in [-2, 2], or None when unparseable
    state["prediction"] = score_to_label(score)  # SUPPORT / NEI / CONTRADICT / INVALID


class MedFactRubric(vf.Rubric):
    """Scores accuracy/diagnostics and also records the model's mapped prediction.

    ``answer`` (the ground-truth label) is already saved; this adds the model's
    parsed verdict so results carry GT vs prediction side by side.
    """

    async def score_rollout(self, state: dict) -> None:
        await super().score_rollout(state)
        _record_prediction(state)

    async def score_group(self, states: list[dict]) -> None:
        await super().score_group(states)
        for state in states:
            _record_prediction(state)


class MedFactSingleTurnEnv(vf.SingleTurnEnv):
    """SingleTurnEnv that always persists the prediction columns.

    ``evaluate`` runs client-side and forwards ``state_columns`` to the env server,
    so appending them here makes ``prediction``/``pred_score`` appear in results
    without the caller needing ``--state-columns``.
    """

    async def evaluate(self, *args, state_columns: list[str] | None = None, **kwargs):
        cols = list(state_columns or [])
        for col in PREDICTION_COLUMNS:
            if col not in cols:
                cols.append(col)
        return await super().evaluate(*args, state_columns=cols, **kwargs)


def _accuracy(completion, answer, **kwargs) -> float:
    return _score_accuracy(completion, answer, _PARSER)


def _parseable_score(completion, **kwargs) -> float:
    return _score_parseable(completion, _PARSER)


def _strict_format(completion, **kwargs) -> float:
    return _score_strict_format(completion, _PARSER)


# Metric column names surfaced by vf-eval come from ``__name__``.
_accuracy.__name__ = "accuracy"
_parseable_score.__name__ = "parseable_score"
_strict_format.__name__ = "strict_format"


def _build_eval_dataset(
    subset: str,
    split: str,
    dataset_path: str | None,
    cache_dir: str | None,
) -> Dataset:
    if split != "eval":
        raise ValueError("MedFact-Bench v0.1.0 exposes only the 'eval' split; training data is not available.")

    # Referenced through the module so tests can monkeypatch ``_load_source_dataset``.
    source = _taskset._load_source_dataset(dataset_path, cache_dir)
    system_prompt = _taskset._validate_dataset(source)
    selected = _taskset._select_dataset(source, _taskset.MedFactSubset(subset))

    return selected.map(
        lambda row: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["user_prompt"]},
            ],
            "answer": row["label"],
            "task": ENVIRONMENT_ID,
            "info": {"dataset": row["dataset"]},
        },
        remove_columns=selected.column_names,
        load_from_cache_file=False,
    )


def build_environment(
    subset: str = "all",
    split: str = "eval",
    dataset_path: str | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build the one-turn MedFact-Bench environment for the classic ``vf-eval`` path.

    Extra ``kwargs`` (e.g. a ``max_turns`` forwarded by generic tooling) are
    recorded in ``env_args`` but do not change the single-turn protocol.
    """
    eval_dataset = _build_eval_dataset(subset, split, dataset_path, cache_dir)
    parser = vf.Parser()
    rubric = MedFactRubric(
        funcs=[_accuracy, _parseable_score, _strict_format],
        weights=[1.0, 0.0, 0.0],
        parser=parser,
    )
    return MedFactSingleTurnEnv(
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        env_id=ENVIRONMENT_ID,
        env_args={
            "subset": subset,
            "split": split,
            "dataset_path": dataset_path,
            "cache_dir": cache_dir,
            **kwargs,
        },
    )
