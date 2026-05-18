"""
Unified LLM client — supports both OpenCode Zen and Ollama backends.
Routes by model prefix: opencode/ → OpenCode Zen, ollama/ → Ollama.
"""
import os
import logging
import time
from typing import Optional

import httpx

from multimodal_ds.config import (
    OLLAMA_BASE_URL,
    LLM_TIMEOUT,
    LLM_RETRIES,
    OPENCODE_ZEN_BASE_URL,
    OPENCODE_ZEN_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)


def _call_opencode_zen(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """Call OpenCode Zen API with retry logic."""
    import json

    api_key = OPENCODE_ZEN_API_KEY
    if not api_key:
        logger.warning("[LLM] OPENCODE_ZEN_API_KEY not set, falling back to Ollama")
        return _call_ollama(messages, model, max_tokens, temperature)

    api_url = OPENCODE_ZEN_BASE_URL

    # Strip prefix from model name
    model = model.replace("opencode/", "")

    for attempt in range(LLM_RETRIES + 1):
        try:
            response = httpx.post(
                api_url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=LLM_TIMEOUT, pool=5.0),
            )
            
            logger.debug(f"[LLM] OpenCode Response HTTP {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                except ValueError as e:
                    raise ValueError(f"Invalid JSON (HTTP 200). Body: {response.text!r}") from e
                    
                logger.debug(f"[LLM] OpenCode Response Body: {response_json}")
                
                # Detect response schema based on available keys to support
                # both OpenAI-compatible and Ollama/OpenCode endpoints.
                if "choices" in response_json:
                    # OpenAI-compatible
                    message = response_json.get("choices", [{}])[0].get("message", {})
                    # Some models (like MiniMax) return content in reasoning field instead of content
                    content = message.get("content") or message.get("reasoning") or ""
                elif "message" in response_json:
                    # Ollama/OpenCode schema - also check reasoning field
                    message = response_json.get("message", {})
                    content = message.get("content") or message.get("reasoning") or ""
                elif "response" in response_json:
                    # Raw fallback
                    content = response_json.get("response", "")
                else:
                    raise ValueError(f"Unsupported response schema: {response_json}")

                if not content:
                    logger.warning(f"[LLM] Parsed content is empty. Raw response: {response_json}")
                    
                return content
                
            elif response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"[LLM] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                logger.warning(f"[LLM] OpenCode request failed: {response.status_code}")
                logger.debug(f"[LLM] OpenCode Error Response: {response.text}")
                
        except httpx.TimeoutException:
            logger.warning(f"[LLM] OpenCode request timed out after {LLM_TIMEOUT}s (attempt {attempt + 1})")
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
                continue
        except ValueError as ve:
            logger.error(f"[LLM] OpenCode parsing failed: {ve}", exc_info=True)
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            logger.warning(f"[LLM] OpenCode call failed (attempt {attempt + 1}): {e}")
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
                continue
    return f"[Error: OpenCode Zen failed after {LLM_RETRIES + 1} attempts]"


def _call_ollama(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """Call local Ollama instance."""
    # Strip prefix from model name
    model = model.replace("ollama/", "")

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=LLM_TIMEOUT, pool=5.0),
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
    except httpx.TimeoutException:
        logger.warning(f"[LLM] Ollama request timed out after {LLM_TIMEOUT}s")
    except Exception as e:
        logger.warning(f"[LLM] Ollama call failed: {e}")
    return f"[Error: Ollama request failed]"


def chat(
    messages: list[dict],
    model: str = "ollama/qwen2.5:7b",
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    if model.startswith("openrouter/"):
        return _call_openrouter(messages, model, max_tokens, temperature)
    elif model.startswith("opencode/"):
        return _call_opencode_zen(messages, model, max_tokens, temperature)
    elif model.startswith("ollama/"):
        return _call_ollama(messages, model, max_tokens, temperature)
    else:
        return _call_ollama(messages, f"ollama/{model}", max_tokens, temperature)
    
# ADD this new function after _call_ollama and before the chat() function:

def _call_openrouter(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """Call OpenRouter API — supports hundreds of models with one API key."""
    api_key = OPENROUTER_API_KEY
    if not api_key:
        logger.warning("[LLM] OPENROUTER_API_KEY not set, falling back to Ollama")
        return _call_ollama(messages, "ollama/qwen2.5:7b", max_tokens, temperature)

    # Strip openrouter/ prefix
    model = model.replace("openrouter/", "")

    for attempt in range(LLM_RETRIES + 1):
        try:
            response = httpx.post(
                OPENROUTER_BASE_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/mmads",
                    "X-Title": "MMADS Agentic DS Engine",
                },
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=LLM_TIMEOUT,
                    write=30.0,
                    pool=5.0,
                ),
            )

            if response.status_code == 200:
                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content:
                    return content
                logger.warning(f"[LLM] OpenRouter empty response: {data}")

            elif response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"[LLM] OpenRouter rate limited — retrying in {wait}s")
                time.sleep(wait)
                continue

            elif response.status_code == 402:
                logger.error("[LLM] OpenRouter: insufficient credits")
                return "[Error: OpenRouter insufficient credits]"

            else:
                logger.warning(
                    f"[LLM] OpenRouter HTTP {response.status_code}: {response.text[:200]}"
                )

        except httpx.TimeoutException:
            logger.warning(f"[LLM] OpenRouter timeout (attempt {attempt + 1})")
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            logger.warning(f"[LLM] OpenRouter error (attempt {attempt + 1}): {e}")
            if attempt < LLM_RETRIES:
                time.sleep(2 ** attempt)
                continue

    return f"[Error: OpenRouter failed after {LLM_RETRIES + 1} attempts]"


def chat_with_fallback(
    messages: list[dict],
    primary_model: str = "openai/gpt-oss-120b:free",
    fallback_model: str = "ollama/qwen2.5:7b",
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """
    Try primary model first, fall back to secondary if it fails.
    Useful for when primary API is rate-limited or unavailable.
    """
    result = chat(messages, primary_model, max_tokens, temperature)
    if result.startswith("[Error:"):
        logger.warning(f"[LLM] Primary model {primary_model} failed, trying fallback...")
        return chat(messages, fallback_model, max_tokens, temperature)
    return result