from __future__ import annotations

import pytest

from medfact_bench import (
    INVALID_LABEL,
    MedFactScoreParser,
    parseable_score,
    prediction_label,
    strict_format,
)


@pytest.mark.parametrize(
    ("completion", "expected_score", "expected_label"),
    [
        ("<score>-2</score>", -2, "CONTRADICT"),
        ("<score>-1</score>", -1, "CONTRADICT"),
        ("<score>0</score>", 0, "NEI"),
        ("<score>1</score>", 1, "SUPPORT"),
        ("<score>2</score>", 2, "SUPPORT"),
        ("text <score> +2 </score> trailing", 2, "SUPPORT"),
        ("<score>2</score><score>-2</score>", 2, "SUPPORT"),
        ([{"role": "assistant", "content": "<score>+1</score>"}], 1, "SUPPORT"),
        (
            [{"role": "assistant", "content": [{"type": "text", "text": "<score>-1</score>"}]}],
            -1,
            "CONTRADICT",
        ),
    ],
)
def test_parser_accepts_valid_first_score(
    completion: object,
    expected_score: int,
    expected_label: str,
) -> None:
    parser = MedFactScoreParser()

    assert parser.parse_completion(completion) == expected_score
    assert prediction_label(completion, parser) == expected_label


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "<score></score>",
        "<score>1.0</score>",
        "<score>support</score>",
        "<score>3</score>",
        "<score>-3</score>",
        "<score>−1</score>",  # noqa: RUF001 - verifies that Unicode minus remains invalid
        "<think>reasoning</think>",
        [{"role": "user", "content": "<score>1</score>"}],
        None,
    ],
)
def test_parser_rejects_invalid_scores(completion: object) -> None:
    parser = MedFactScoreParser()

    assert parser.parse_completion(completion) is None
    assert prediction_label(completion, parser) == INVALID_LABEL


def test_parseability_and_strict_format_are_independent() -> None:
    parser = MedFactScoreParser()
    strict_completion = "<think>Reasoning</think><score>-1</score>"
    relaxed_completion = "Explanation first. <score>-1</score>"
    duplicate_tag_completion = "<think>Reasoning</think><score>-1</score><score>2</score>"
    missing_think_completion = "<score>-1</score>"

    assert parseable_score(strict_completion, parser) == 1.0
    assert strict_format(strict_completion, parser) == 1.0
    assert parseable_score(relaxed_completion, parser) == 1.0
    assert strict_format(relaxed_completion, parser) == 0.0
    assert parseable_score(duplicate_tag_completion, parser) == 1.0
    assert strict_format(duplicate_tag_completion, parser) == 0.0
    assert parseable_score(missing_think_completion, parser) == 1.0
    assert strict_format(missing_think_completion, parser) == 0.0


def test_parser_uses_the_last_assistant_message() -> None:
    completion = [
        {"role": "assistant", "content": "<score>-2</score>"},
        {"role": "user", "content": "Try again."},
        {"role": "assistant", "content": "<score>2</score>"},
    ]

    assert MedFactScoreParser().parse_completion(completion) == 2
