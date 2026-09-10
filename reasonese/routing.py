"""Invocation-local collection routing, billing permission, and audit summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field

from beartype import beartype

from reasonese.axes import Assistant, Author
from reasonese.openrouter import (
    JsonObject,
    RoutePreference,
    RouteProvenance,
    completion_provenance,
    select_route,
)


@beartype
@dataclass(slots=True)
class CollectionRouting:
    """One invocation's explicit policy and observed route counts."""

    preference: RoutePreference = RoutePreference.FREE
    allow_paid: bool = False
    _counts: Counter[tuple[str, str, str, str | None, str | None, str | None]] = field(
        default_factory=Counter, init=False, repr=False
    )

    def require_paid(self, work: str) -> None:
        if not self.allow_paid:
            raise ValueError(
                f"--allow-paid is required for {work}; no provider calls were authorized"
            )

    def record(
        self,
        stage: str,
        model: Author | Assistant,
        source: str,
        provenance: RouteProvenance | None,
        response: JsonObject | None,
    ) -> None:
        reported = response.get("model") if response is not None else None
        self._counts[
            (
                stage,
                str(model),
                source,
                str(provenance.requested_model_id) if provenance else None,
                str(provenance.transport) if provenance else None,
                reported if isinstance(reported, str) else None,
            )
        ] += 1

    def summary(self) -> list[dict[str, object]]:
        return [
            dict(
                zip(
                    (
                        "stage",
                        "model",
                        "source",
                        "requested_model_id",
                        "transport",
                        "response_model",
                    ),
                    key,
                    strict=True,
                ),
                count=count,
            )
            for key, count in sorted(
                self._counts.items(),
                key=lambda item: tuple("" if value is None else value for value in item[0]),
            )
        ]

    def announce(
        self, authors: tuple[Author, ...], assistants: tuple[Assistant, ...], *, prefer_batch: bool
    ) -> None:
        selected = []
        for stage, models in (("author", authors), ("assistant", assistants)):
            for model in dict.fromkeys(models):
                if model == Author.USER:
                    continue
                route = select_route(model, self.preference)
                provenance = completion_provenance(
                    route, (), prefer_batch=prefer_batch and stage == "author"
                )
                selected.append({"stage": stage, "model": str(model), **provenance.to_dict()})
        print(
            json.dumps(
                {
                    "routes_for_missing_work": selected,
                    "allow_paid": self.allow_paid,
                    "paid_services": [
                        "message QA",
                        "response judgments (study collection)",
                        "assistant web search",
                    ],
                    "cache_policy": "reuse original provenance; missing historical provenance is unknown",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )


def add_route_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--route",
        type=RoutePreference,
        choices=tuple(RoutePreference),
        default=RoutePreference.FREE,
        help="free preferred by default; paid is synchronous; batch prefers batch authoring",
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="authorize uncached paid models, QA, judgments, and assistant web search",
    )


def routing_from_arguments(args: argparse.Namespace) -> CollectionRouting:
    if args.no_batch and args.route is RoutePreference.BATCH:
        raise ValueError("--no-batch cannot be combined with --route batch")
    return CollectionRouting(args.route, args.allow_paid)
