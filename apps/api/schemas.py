"""Request/response shapes for the conversation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class MessageRequest(BaseModel):
    message: str
    customer_id: str | None = None


class LiveResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    gateway_reachable: bool


class KnowledgeQuestionRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    norm: str | None
    article_ref: str | None
    source_url: str | None


class KnowledgeAnswerResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    refused: bool
