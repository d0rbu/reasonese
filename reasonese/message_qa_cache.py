"""Readable YAML cache for message-compliance judgments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from beartype import beartype

from reasonese.conversation import GeneratedMessage, GeneratedText
from reasonese.matchup import prompt_spec_from_dict, prompt_spec_to_dict
from reasonese.message_qa import MessageQaVerdict, QaIssue
from reasonese.openrouter import JsonObject


def _verdict_from_dict(raw: object) -> MessageQaVerdict:
    if not isinstance(raw, dict):
        raise ValueError("cached message QA verdict must be a mapping")
    data = cast(dict[str, Any], raw)
    expected = {"input", "content", "complies", "issues", "response"}
    if set(data) != expected:
        raise ValueError("cached message QA verdict has invalid fields")
    complies = data["complies"]
    raw_issues = data["issues"]
    response = data["response"]
    if not isinstance(complies, bool):
        raise ValueError("cached message QA complies field must be a boolean")
    if not isinstance(raw_issues, list):
        raise ValueError("cached message QA issues field must be a list")
    if not isinstance(response, dict):
        raise ValueError("cached message QA response must be a mapping")
    return MessageQaVerdict(
        prompt_spec_from_dict(data["input"]),
        GeneratedText.parse(data["content"]),
        complies,
        tuple(QaIssue.parse(issue) for issue in raw_issues),
        cast(JsonObject, response),
    )


@beartype
def _verdict_to_dict(verdict: MessageQaVerdict) -> dict[str, object]:
    return {
        "input": prompt_spec_to_dict(verdict.spec),
        "content": str(verdict.content),
        "complies": verdict.complies,
        "issues": [str(issue) for issue in verdict.issues],
        "response": verdict.response,
    }


@beartype
@dataclass(frozen=True, slots=True)
class YamlMessageQaCache:
    """QA verdicts keyed by a datapoint and its exact materialized text."""

    path: Path

    def load(self) -> tuple[MessageQaVerdict, ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            return ()
        if not isinstance(raw, dict) or set(raw) != {"message_qa"}:
            raise ValueError(f"{self.path} must contain one 'message_qa' list")
        raw_verdicts = raw["message_qa"]
        if not isinstance(raw_verdicts, list):
            raise ValueError(f"{self.path} must contain one 'message_qa' list")
        return tuple(_verdict_from_dict(item) for item in raw_verdicts)

    @beartype
    def get(self, message: GeneratedMessage) -> MessageQaVerdict | None:
        return next((verdict for verdict in self.load() if verdict.matches(message)), None)

    @beartype
    def put_many(self, verdicts: tuple[MessageQaVerdict, ...]) -> None:
        by_spec = {verdict.spec: verdict for verdict in self.load()}
        by_spec.update({verdict.spec: verdict for verdict in verdicts})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"message_qa": [_verdict_to_dict(verdict) for verdict in by_spec.values()]},
                handle,
                sort_keys=False,
                allow_unicode=True,
                width=100,
            )
