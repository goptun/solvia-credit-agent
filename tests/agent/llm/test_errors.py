"""Error classification: only transient failures are retried."""

from __future__ import annotations

import httpx
import httpx2
import openai
from google.genai import errors as genai_errors

from apps.agent.llm.errors import RetryableLLMError, is_transient


def _openai_status_error(status_code: int, cls: type[openai.APIStatusError]) -> Exception:
    # openai's installed version transports over `httpx2`, a separate
    # package from `httpx` — its exception types expect `httpx2` request/
    # response objects specifically.
    response = httpx2.Response(
        status_code, request=httpx2.Request("POST", "http://gateway.invalid")
    )
    return cls("boom", response=response, body=None)


def test_retryable_llm_error_is_transient() -> None:
    assert is_transient(RetryableLLMError("empty completion")) is True


def test_httpx_timeout_and_connect_errors_are_transient() -> None:
    assert is_transient(httpx.TimeoutException("timed out")) is True
    assert is_transient(httpx.ConnectError("refused")) is True


def test_httpx2_timeout_and_connect_errors_are_transient() -> None:
    assert is_transient(httpx2.TimeoutException("timed out")) is True
    assert is_transient(httpx2.ConnectError("refused")) is True


def test_openai_timeout_and_connection_errors_are_transient() -> None:
    request = httpx2.Request("POST", "http://gateway.invalid")
    assert is_transient(openai.APITimeoutError(request=request)) is True
    assert is_transient(openai.APIConnectionError(message="down", request=request)) is True


def test_openai_rate_limit_429_is_transient() -> None:
    error = _openai_status_error(429, openai.RateLimitError)
    assert is_transient(error) is True


def test_openai_5xx_is_transient() -> None:
    error = _openai_status_error(500, openai.InternalServerError)
    assert is_transient(error) is True


def test_openai_4xx_is_not_transient() -> None:
    error = _openai_status_error(400, openai.BadRequestError)
    assert is_transient(error) is False


def test_openai_401_is_not_transient() -> None:
    error = _openai_status_error(401, openai.AuthenticationError)
    assert is_transient(error) is False


def test_google_server_error_is_transient() -> None:
    error = genai_errors.ServerError(code=503, response_json={}, response=None)
    assert is_transient(error) is True


def test_google_client_error_429_is_transient() -> None:
    error = genai_errors.ClientError(code=429, response_json={}, response=None)
    assert is_transient(error) is True


def test_google_client_error_400_is_not_transient() -> None:
    error = genai_errors.ClientError(code=400, response_json={}, response=None)
    assert is_transient(error) is False


def test_generic_value_error_is_not_transient() -> None:
    assert is_transient(ValueError("not a real gateway error")) is False
