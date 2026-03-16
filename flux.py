"""Azure AI Foundry FLUX.2-Pro image generation helpers."""

from __future__ import annotations

import base64
from typing import Optional

import requests

DEFAULT_FLUX_API_VERSION = "2024-05-01-preview"


def _image_generation_url(endpoint: str, deployment: str, api_version: str) -> str:
    return (
        f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
        f"/images/generations?api-version={api_version}"
    )


def _is_foundry_provider_endpoint(endpoint: str) -> bool:
    return "/providers/" in endpoint


def _with_api_version(url: str, api_version: str) -> str:
    if "api-version=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}api-version={api_version}"


def _headers(api_key: str) -> dict[str, str]:
    # Include both auth styles for compatibility across Azure Foundry setups.
    return {
        "api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def generate_flux_image(
    endpoint: str,
    api_key: str,
    deployment: str,
    prompt: str,
    size: str = "1024x1024",
    output_format: str = "png",
    response_format: str = "b64_json",
    api_version: str = DEFAULT_FLUX_API_VERSION,
) -> dict:
    """Submit a FLUX.2-Pro image generation request and return raw JSON response."""
    endpoint = endpoint.strip()

    if "services.ai.azure.com" in endpoint and not _is_foundry_provider_endpoint(endpoint):
        raise ValueError(
            "Detected an Azure AI Foundry project endpoint. Provide the full FLUX provider URL "
            "(it should include '/providers/.../flux-2-pro')."
        )

    payload = {
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": response_format,
        "output_format": output_format,
    }

    if _is_foundry_provider_endpoint(endpoint):
        request_url = _with_api_version(endpoint, api_version)
    else:
        if not deployment:
            raise ValueError("Deployment name is required when using Azure OpenAI endpoint format.")
        request_url = _image_generation_url(endpoint, deployment, api_version)

    resp = requests.post(
        request_url,
        headers=_headers(api_key),
        json=payload,
        timeout=120,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} — response body: {resp.text}", response=resp) from exc

    return resp.json()


def extract_flux_image_bytes(result: dict) -> Optional[bytes]:
    """Extract image bytes from FLUX response (supports b64_json and signed URL)."""
    data = result.get("data") or []
    if not data:
        return None

    first = data[0]

    b64 = first.get("b64_json")
    if b64:
        return base64.b64decode(b64)

    url = first.get("url")
    if url:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content

    return None
