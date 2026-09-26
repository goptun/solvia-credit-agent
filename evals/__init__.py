"""Evaluation harness: versioned datasets, metrics with confidence
intervals, an offline regression gate and budgeted live runs.

An orchestration layer like `scripts/`: it may import `apps/` and `rag/`,
which never import it. `evals.core` is pure Python (no LangChain, psycopg
or FastAPI imports). See `openspec/changes/add-llm-evals/design.md`.
"""
