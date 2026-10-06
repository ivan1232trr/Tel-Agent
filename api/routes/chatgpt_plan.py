"""Admin controls for a host-local ChatGPT plan registration.

OAuth is deliberately absent from this public web server: the documented direct
flow uses a loopback callback on the computer running the user's browser. A signed-in
owner can explicitly transfer one existing registration through the HTTPS import
route. Tokens stay in protected host storage and are never returned to the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response

from agent.chatgpt_auth import ChatGPTAuthError, ChatGPTAuthStore
from agent.config import chatgpt_auth_directory, environment_values
from api.errors import envelope_response
from api.security.permissions import WorkspaceContext, require_admin, require_owner
from api.settings import store

router = APIRouter(prefix="/api/settings/chatgpt", tags=["settings"])
MAX_IMPORT_BYTES = 1024 * 1024


def auth_store() -> ChatGPTAuthStore:
    """Only installation configuration chooses the credential directory."""
    return ChatGPTAuthStore(chatgpt_auth_directory())


async def _status(request: Request, *, can_import: bool = False) -> dict[str, Any]:
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
        "can_import": can_import and _https_import_configured(request),
    }


def _https_import_configured(request: Request) -> bool:
    """Use the operator's canonical HTTPS origin, never untrusted proxy headers."""
    public = request.app.state.settings.public_base_url
    if not isinstance(public, str):
        return False
    try:
        canonical = urlsplit(public)
        return (
            canonical.scheme == "https"
            and bool(canonical.hostname)
            and not (canonical.username or canonical.password)
            and canonical.hostname == request.url.hostname
            and (canonical.port or 443) == (request.url.port or 443)
        )
    except ValueError:
        return False


def _import_error(status: int, code: str, message: str) -> Response:
    response = envelope_response(status_code=status, code=code, message=message)
    response.headers["Cache-Control"] = "no-store"
    return response


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
        return await _status(request, can_import=context.outranks_or_is("owner"))
    except ChatGPTAuthError:
        return _auth_error()


@router.post("/import", summary="Owner-uploaded ChatGPT registration over HTTPS")
async def import_account(
    request: Request, context: Annotated[WorkspaceContext, require_owner]
) -> Response:
    """Accept one explicit browser upload; never parse credentials into an API model.

    Raw bytes avoid validation errors echoing token fields. This endpoint adds a
    stricter origin check than ordinary API routes, including refusing absent Origin.
    The owner's browser selects and submits the file; there is no URL/path importer.
    """
    if not _https_import_configured(request):
        return _import_error(
            409, "chatgpt_import_insecure", "Account import requires the configured HTTPS API."
        )
    origin = request.headers.get("origin", "")
    if (
        not origin.startswith("https://")
        or origin not in request.app.state.settings.cors_origins
        or request.headers.get("x-tel-agent-import") != "1"
    ):
        return _import_error(
            403, "forbidden", "Use the signed-in HTTPS dashboard to import an account."
        )
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "application/octet-stream"
        or request.url.query
    ):
        return _import_error(
            415, "chatgpt_import_invalid", "Choose one ChatGPT registration JSON file."
        )
    data = await request.body()
    if not data or len(data) > MAX_IMPORT_BYTES:
        return _import_error(
            413, "chatgpt_import_invalid", "The registration file is empty or too large."
        )

    # Keep at most one import worker per process, including after a disconnected
    # request. The auth store's OS lock additionally protects other API processes.
    lock = getattr(request.app.state, "chatgpt_import_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.chatgpt_import_lock = lock
    if lock.locked():
        return _import_error(
            409, "chatgpt_import_in_progress", "An account import is already in progress."
        )
    await lock.acquire()
    cancelled = threading.Event()

    def transfer() -> dict[str, Any]:
        return auth_store().import_registration_bytes(data, cancelled=cancelled.is_set)

    worker = asyncio.create_task(asyncio.to_thread(transfer))
    request.app.state.chatgpt_import_worker = worker

    def finished(task: asyncio.Task[dict[str, Any]]) -> None:
        # Consume background failures without printing exception text or credential
        # locals after a browser disconnect or the request middleware's timeout.
        if not task.cancelled():
            task.exception()
        request.app.state.chatgpt_import_worker = None
        lock.release()

    worker.add_done_callback(finished)
    try:
        await asyncio.shield(worker)
        state = await _status(request, can_import=True)
        return JSONResponse(state, headers={"Cache-Control": "no-store"})
    except ChatGPTAuthError as error:
        if error.code in {"import_conflict", "registration_conflict", "account_changed"}:
            return _import_error(
                409,
                "chatgpt_import_conflict",
                "A working registration already exists. It was not replaced.",
            )
        return _import_error(
            422,
            "chatgpt_import_failed",
            "The account could not be imported. Check the selected registration and refresh "
            "the connection before trying again.",
        )
    except Exception:
        # No raw upstream, JSON, filesystem or unexpected exception may expose the
        # upload in diagnostics. A lost response may follow a successful atomic write.
        return _import_error(
            503,
            "chatgpt_import_failed",
            "The import result could not be confirmed. Refresh the connection before retrying.",
        )
    finally:
        cancelled.set()


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
        return await _status(request, can_import=context.outranks_or_is("owner"))
    except ChatGPTAuthError:
        return _auth_error()
    except (httpx.HTTPError, ValueError, TimeoutError):
        return _catalog_error()
