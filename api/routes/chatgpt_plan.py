"""Admin controls for a host-local ChatGPT plan registration.

OAuth is deliberately absent from this public web server: the documented direct
flow uses a loopback callback on the computer running the user's browser. Tokens
stay in protected host storage and never pass through dashboard requests.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agent.chatgpt_auth import ChatGPTAuthError, ChatGPTAuthStore
from agent.config import chatgpt_auth_directory, environment_values
from api.errors import envelope_response
from api.security.permissions import WorkspaceContext, require_admin
from api.settings import store

router = APIRouter(prefix="/api/settings/chatgpt", tags=["settings"])


def auth_store() -> ChatGPTAuthStore:
    """Only installation configuration chooses the credential directory."""
    return ChatGPTAuthStore(chatgpt_auth_directory())


async def _status(request: Request) -> dict[str, Any]:
    state = await asyncio.to_thread(auth_store().status)
    db = request.state.db
    environment = environment_values()
    selected = await store.get(db, "llm.chatgpt_client_id") or state.get("selected_client_id")
    accounts = [
        {
            "client_id": row["client_id"],
            "label": row["label"],
            "email": row.get("email"),
            "connected": row["connected"],
            "selected": row["client_id"] == selected,
        }
        for row in state["accounts"]
    ]
    connected = any(row["selected"] and row["connected"] for row in accounts)
    return {
        "connected": connected,
        "selected_client_id": selected,
        "accounts": accounts,
        "reason": None if connected else "chatgpt_not_connected",
        "provider": await store.get(db, "llm.provider") or environment["provider"] or None,
        "model": await store.get(db, "llm.model") or environment["model"] or None,
    }


def _auth_error() -> object:
    return envelope_response(
        status_code=409,
        code="chatgpt_not_connected",
        message="The selected ChatGPT account needs local sign-in or renewed plan permission.",
    )


@router.get("", summary="ChatGPT connection metadata, without tokens or model requests")
async def connection_status(
    request: Request, context: Annotated[WorkspaceContext, require_admin]
) -> object:
    try:
        return await _status(request)
    except ChatGPTAuthError:
        return _auth_error()


async def model_catalog(client_id: str | None) -> list[dict[str, str]]:
    """List account-specific visible models without making an inference request."""
    token = await auth_store().access_token(client_id=client_id)
    async with httpx.AsyncClient(timeout=15, follow_redirects=False, trust_env=False) as client:
        async with client.stream(
            "GET",
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            response.raise_for_status()
            body = bytearray()
            async with asyncio.timeout(30):
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > 1024 * 1024:
                        raise ValueError("ChatGPT model catalog is too large.")
                    body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        raise ValueError("Invalid ChatGPT model catalog.") from None
    rows = value.get("models") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Invalid ChatGPT model catalog.")
    return [
        {"slug": row["slug"], "display_name": row["display_name"]}
        for row in rows
        if isinstance(row, dict)
        and row.get("visibility") == "list"
        and isinstance(row.get("slug"), str)
        and isinstance(row.get("display_name"), str)
    ]


def _catalog_error() -> object:
    return envelope_response(
        status_code=502,
        code="chatgpt_unavailable",
        message=(
            "The ChatGPT model catalog is unavailable. No provider or billing path changed."
        ),
    )


@router.get("/models", summary="Read the chosen ChatGPT account's model catalog")
async def list_models(
    request: Request,
    context: Annotated[WorkspaceContext, require_admin],
    client_id: str | None = None,
) -> object:
    try:
        selected = client_id or (await _status(request))["selected_client_id"]
        if not selected:
            return _auth_error()
        return {"models": await model_catalog(selected)}
    except ChatGPTAuthError:
        return _auth_error()
    except (httpx.HTTPError, ValueError, TimeoutError):
        return _catalog_error()


class AccountSelection(BaseModel):
    client_id: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)


@router.post("/select", summary="Use a saved ChatGPT registration for this installation")
async def select_account(
    request: Request,
    payload: AccountSelection,
    context: Annotated[WorkspaceContext, require_admin],
) -> object:
    try:
        models = await model_catalog(payload.client_id)
        if payload.model not in {row["slug"] for row in models}:
            return envelope_response(
                status_code=409,
                code="chatgpt_model_unavailable",
                message="Choose a model available to the selected ChatGPT account.",
            )
        db = request.state.db
        # One database transaction switches identity, model and provider. The
        # protected credential registration is unchanged by this preference.
        await store.set_value(db, "llm.chatgpt_client_id", payload.client_id)
        await store.set_value(db, "llm.model", payload.model)
        await store.set_value(db, "llm.provider", "chatgpt_plan")
        await db.commit()
        return await _status(request)
    except ChatGPTAuthError:
        return _auth_error()
    except (httpx.HTTPError, ValueError, TimeoutError):
        return _catalog_error()
