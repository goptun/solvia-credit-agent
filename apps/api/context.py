"""Application-wide dependencies, built once at startup and reused per request."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.settings import Settings
from apps.agent.nodes.knowledge_agent import RetrieveFn
from apps.agent.observability.tracing import Tracer
from apps.agent.repositories.customers import CustomerRepository
from apps.agent.state import ConversationState
from rag.embeddings.port import EmbeddingsPort
from rag.settings import RagSettings


@dataclass
class AppContext:
    llm_factory: LLMFactory
    llm_settings: Settings
    customer_repository: CustomerRepository
    checkpointer: BaseCheckpointSaver[str]
    graph: CompiledStateGraph[ConversationState, None, ConversationState, ConversationState]
    tracer: Tracer
    knowledge_embeddings: EmbeddingsPort
    knowledge_retrieve: RetrieveFn
    rag_settings: RagSettings
