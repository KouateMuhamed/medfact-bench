"""Score parsing and label mapping for MedFact-Bench."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

LABELS = ("SUPPORT", "NEI", "CONTRADICT")
INVALID_LABEL = "INVALID"
VALID_SCORES = frozenset({-2, -1, 0, 1, 2})

_SCORE_PATTERN = re.compile(r"<score>(.*?)</score>", re.DOTALL)
_STRICT_FORMAT_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<score>(.*?)</score>\s*\Z",
    re.DOTALL,
)


def _parse_score_value(value: str) -> int | None:
    try:
        score = int(value.strip())
    except (TypeError, ValueError):
        return None
    return score if score in VALID_SCORES else None


def _message_field(message: Any, field: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(field, default)
    return getattr(message, field, default)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        text = _message_field(part, "text")
        if text is not None:
            text_parts.append(str(text))
    return "".join(text_parts)


class MedFactScoreParser:
    """Parse the first paper-format integer score from model output."""

    @staticmethod
    def parse_score(text: str) -> int | None:
        """Return the first valid score enclosed in a score tag."""
        match = _SCORE_PATTERN.search(text)
        if match is None:
            return None
        return _parse_score_value(match.group(1))

    def parse(self, text: str) -> int | None:
        """Parse a completion string."""
        return self.parse_score(text)

    def completion_text(self, completion: Any) -> str | None:
        """Return the final assistant text from a string or message sequence."""
        if isinstance(completion, str):
            return completion
        if not isinstance(completion, list):
            return None
        assistant_messages = [message for message in completion if _message_field(message, "role") == "assistant"]
        if not assistant_messages:
            return None
        content = _message_field(assistant_messages[-1], "content", "") or ""
        return _content_to_text(content)

    def parse_completion(self, completion: Any) -> int | None:
        """Parse a string or chat-message completion."""
        text = self.completion_text(completion)
        return self.parse(text) if text is not None else None

    def is_strict_format(self, completion: Any) -> bool:
        """Check exact two-block Med-V1 formatting independently of score parsing."""
        text = self.completion_text(completion)
        if text is None:
            return False
        if any(text.count(tag) != 1 for tag in ("<think>", "</think>", "<score>", "</score>")):
            return False
        match = _STRICT_FORMAT_PATTERN.fullmatch(text)
        return match is not None and _parse_score_value(match.group(1)) is not None


def score_to_label(score: int | None) -> str:
    """Map the five-point Med-V1 score to the three benchmark labels."""
    if score in (1, 2):
        return "SUPPORT"
    if score == 0:
        return "NEI"
    if score in (-1, -2):
        return "CONTRADICT"
    return INVALID_LABEL


def prediction_label(completion: Any, parser: MedFactScoreParser | None = None) -> str:
    """Return the benchmark label predicted by a completion."""
    resolved_parser = parser or MedFactScoreParser()
    return score_to_label(resolved_parser.parse_completion(completion))


def accuracy(completion: Any, answer: str, parser: MedFactScoreParser | None = None) -> float:
    """Score exact three-way classification accuracy."""
    return float(prediction_label(completion, parser) == answer)


def parseable_score(completion: Any, parser: MedFactScoreParser | None = None) -> float:
    """Measure whether a valid score can be parsed."""
    resolved_parser = parser or MedFactScoreParser()
    return float(resolved_parser.parse_completion(completion) is not None)


def strict_format(completion: Any, parser: MedFactScoreParser | None = None) -> float:
    """Measure exact compliance with the requested think and score blocks."""
    resolved_parser = parser or MedFactScoreParser()
    return float(resolved_parser.is_strict_format(completion))
