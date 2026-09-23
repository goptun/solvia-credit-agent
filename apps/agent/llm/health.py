"""LLM gateway reachability check, used by `GET /health/ready`.

See `docs/infra-assessment.md` for how this was verified against the
real `9router` gateway.
"""

from __future__ import annotations

import httpx

from apps.agent.llm.settings import Settings


async def check_gateway_reachable(settings: Settings, timeout_seconds: float = 5.0) -> bool:
    """`True` if the configured gateway's `/models` endpoint responds."""
    url = f"{settings.llm_base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, headers=headers)
        return response.status_code < 500
    except httpx.HTTPError:
        return False
