"""Typed boundary for a future natural-language preference interpreter."""

from __future__ import annotations

from typing import Protocol

from trustworthy_recsys.serving.contracts import RefinementRequest, RefinementResponse, UnavailableRefinement


class RefinementInterpreter(Protocol):
    name: str
    version: str

    def interpret(self, request: RefinementRequest) -> RefinementResponse: ...


class UnavailableInterpreter:
    name = "unavailable"
    version = "none"

    def interpret(self, request: RefinementRequest) -> RefinementResponse:
        return UnavailableRefinement(
            message="Natural-language refinement is not configured. Structured preference controls remain available."
        )
