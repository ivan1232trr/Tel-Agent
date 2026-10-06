"""Subscription provider wiring without OAuth, real credentials or model requests."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from agent.chatgpt_auth import ChatGPTAuthError
from agent.config import ENVIRONMENT_NAMES, ConfigurationError, settings_from
from agent.providers.llm.base import TextDelta
from api import llm
from api.main import create_app
from api.models import Membership, User, Workspace
from api.routes import chatgpt_plan
from api.security.password import hash_password
from api.settings import store


def test_chatgpt_configuration_never_uses_legacy_key_or_custom_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CHATGPT_AUTH_DIR", str(tmp_path))
    settings = settings_from(
        provider="chatgpt_plan",
        model="eligible-model",
        api_key="legacy-secret",
        base_url="https://untrusted.test",
        names=ENVIRONMENT_NAMES,
    )
    assert settings.provider == "chatgpt_plan"
    assert settings.api_key == ""
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.chatgpt_auth_dir == str(tmp_path)


def test_chatgpt_requires_an_explicit_model():
    with pytest.raises(ConfigurationError, match="Choose a model"):
        settings_from(
            provider="chatgpt_plan", model="", api_key="", base_url="", names=ENVIRONMENT_NAMES
        )


async def test_resolve_never_decrypts_unrelated_legacy_key(migrated, monkeypatch):
    await store.set_value(migrated, "llm.provider", "chatgpt_plan")
    await store.set_value(migrated, "llm.model", "eligible-model")
    await store.set_value(migrated, "llm.chatgpt_client_id", "oaiapp_selected")
    await migrated.commit()

    async def forbidden(_db):
        raise AssertionError("OAuth must not read the old API key")

    monkeypatch.setattr(llm, "stored_values", forbidden)
    settings = await llm.resolve(migrated)
    assert settings.chatgpt_client_id == "oaiapp_selected"
    assert settings.api_key == ""


class FakeAuth:
    def status(self):
        return {
            "connected": True,
            "selected_client_id": "oaiapp_one",
            "accounts": [
                {
                    "client_id": "oaiapp_one",
                    "label": "Personal",
                    "email": "me@example.test",
                    "connected": True,
                    "selected": True,
                    "subject": "subject",
                    "access_token": "must-never-leak",
                    "refresh_token": "must-never-leak",
                }
            ],
            "reason": None,
            "access_token": "must-never-leak",
        }

    async def access_token(self, client_id=None):
        assert client_id == "oaiapp_one"
        return "stand-in-credential"


@pytest.fixture
async def chat_clients(migrated, settings, database_url, monkeypatch):
    monkeypatch.setattr(chatgpt_plan, "auth_store", FakeAuth)
    space = Workspace(name="One installation")
    migrated.add(space)
    await migrated.flush()
    password = "a test-only password phrase"  # noqa: S105
    for role in ("owner", "admin", "viewer"):
        user = User(username=role, password_hash=hash_password(password))
        migrated.add(user)
        await migrated.flush()
        migrated.add(Membership(user_id=user.id, workspace_id=space.id, role=role))
    await migrated.commit()
    app = create_app(
        settings.model_copy(
            update={
                "database_url": database_url,
                "public_base_url": "https://localhost",
                "cors_origins": ["https://localhost"],
            }
        )
    )
    clients = {}
    async with app.router.lifespan_context(app):
        for role in ("owner", "admin", "viewer"):
            client = AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost")
            assert (
                await client.post(
                    "/api/auth/login", json={"username": role, "password": password}
                )
            ).status_code == 200
            clients[role] = client
        try:
            yield clients
        finally:
            for client in clients.values():
                await client.aclose()


async def test_status_never_returns_tokens_or_fetches_models(chat_clients, monkeypatch):
    async def forbidden(_id):
        raise AssertionError("Status must not contact OpenAI")

    monkeypatch.setattr(chatgpt_plan, "model_catalog", forbidden)
    response = await chat_clients["admin"].get("/api/settings/chatgpt")
    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert "token" not in response.text
    assert "must-never-leak" not in response.text


async def test_viewer_cannot_read_select_or_list_accounts(chat_clients):
    for path in ("/api/settings/chatgpt", "/api/settings/chatgpt/models"):
        assert (await chat_clients["viewer"].get(path)).status_code == 403
    assert (
        await chat_clients["viewer"].post(
            "/api/settings/chatgpt/select", json={"client_id": "oaiapp_one", "model": "m"}
        )
    ).status_code == 403


async def test_select_commits_identity_model_and_provider(chat_clients, migrated, monkeypatch):
    async def models(client_id):
        assert client_id == "oaiapp_one"
        return [{"slug": "eligible-model", "display_name": "Eligible"}]

    monkeypatch.setattr(chatgpt_plan, "model_catalog", models)
    response = await chat_clients["admin"].post(
        "/api/settings/chatgpt/select",
        json={"client_id": "oaiapp_one", "model": "eligible-model"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "chatgpt_plan"
    assert response.json()["model"] == "eligible-model"
    migrated.expire_all()
    settings = await llm.resolve(migrated)
    assert settings.chatgpt_client_id == "oaiapp_one"
    assert settings.api_key == ""


async def test_select_rejects_unavailable_model_without_changes(
    chat_clients, migrated, monkeypatch
):
    async def models(_id):
        return [{"slug": "eligible-model", "display_name": "Eligible"}]

    monkeypatch.setattr(chatgpt_plan, "model_catalog", models)
    response = await chat_clients["admin"].post(
        "/api/settings/chatgpt/select", json={"client_id": "oaiapp_one", "model": "wrong-model"}
    )
    assert response.status_code == 409
    assert await store.get(migrated, "llm.provider") is None


async def test_model_catalog_uses_oauth_and_account_specific_shape(monkeypatch):
    monkeypatch.setattr(chatgpt_plan, "auth_store", FakeAuth)

    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert request.headers["authorization"] == "Bearer stand-in-credential"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "second", "display_name": "Second", "visibility": "list"},
                    {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
                    {"slug": "first", "display_name": "First", "visibility": "list"},
                ]
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        chatgpt_plan.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert await chatgpt_plan.model_catalog("oaiapp_one") == [
        {"slug": "second", "display_name": "Second"},
        {"slug": "first", "display_name": "First"},
    ]


@pytest.mark.parametrize(
    "body", [b"[]", b"null", b"not json", b'{"models":{}}', b" " * (1024 * 1024 + 1)]
)
async def test_model_catalog_rejects_malformed_or_oversized_payloads(monkeypatch, body):
    monkeypatch.setattr(chatgpt_plan, "auth_store", FakeAuth)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        chatgpt_plan.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body)),
            **kwargs,
        ),
    )
    with pytest.raises(ValueError, match="catalog"):
        await chatgpt_plan.model_catalog("oaiapp_one")


@pytest.mark.parametrize("error", [ValueError("unsafe upstream body"), TimeoutError()])
async def test_catalog_failure_is_safe_and_never_changes_provider(
    chat_clients, migrated, monkeypatch, error
):
    async def unavailable(_client_id):
        raise error

    monkeypatch.setattr(chatgpt_plan, "model_catalog", unavailable)
    for response in (
        await chat_clients["admin"].get("/api/settings/chatgpt/models"),
        await chat_clients["admin"].post(
            "/api/settings/chatgpt/select",
            json={"client_id": "oaiapp_one", "model": "eligible-model"},
        ),
    ):
        assert response.status_code == 502
        assert "unsafe upstream body" not in response.text
    assert await store.get(migrated, "llm.provider") is None


async def test_test_button_consumes_chatgpt_completion(chat_clients, monkeypatch):
    import agent.providers.llm as providers
    from agent.config import LlmSettings

    async def resolve(_db):
        return LlmSettings("chatgpt_plan", "eligible-model", "", "https://api.openai.com/v1")

    finished = []

    class Provider:
        async def stream(self, _messages):
            yield TextDelta("Hello")
            finished.append(True)

    monkeypatch.setattr(llm, "resolve", resolve)
    monkeypatch.setattr(providers, "provider_for", lambda _settings: Provider())
    response = await chat_clients["admin"].post("/api/settings/llm/test")
    assert response.status_code == 200
    assert finished == [True]


IMPORT_HEADERS = {
    "Origin": "https://localhost",
    "Content-Type": "application/octet-stream",
    "X-Tel-Agent-Import": "1",
}


async def test_only_owner_has_import_capability(chat_clients):
    for role, expected in (("owner", True), ("admin", False)):
        response = await chat_clients[role].get("/api/settings/chatgpt")
        assert response.status_code == 200
        assert response.json()["can_import"] is expected


async def test_owner_upload_is_opaque_bounded_and_metadata_only(chat_clients, monkeypatch):
    calls = []

    class ImportAuth(FakeAuth):
        def import_registration_bytes(self, data, *, cancelled=None):
            assert not cancelled()
            calls.append(data)
            return {"access_token": "must-never-leak"}

    monkeypatch.setattr(chatgpt_plan, "auth_store", ImportAuth)
    response = await chat_clients["owner"].post(
        "/api/settings/chatgpt/import",
        content=b'{"synthetic":"credential"}',
        headers=IMPORT_HEADERS,
    )
    assert response.status_code == 200
    assert calls == [b'{"synthetic":"credential"}']
    assert response.json()["can_import"] is True
    assert "token" not in response.text
    assert "must-never-leak" not in response.text
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("role", ["admin", "viewer"])
async def test_non_owner_cannot_import(chat_clients, role):
    response = await chat_clients[role].post(
        "/api/settings/chatgpt/import", content=b"{}", headers=IMPORT_HEADERS
    )
    assert response.status_code == 403


async def test_unauthenticated_upload_is_rejected(chat_clients):
    client = chat_clients["owner"]
    client.cookies.clear()
    response = await client.post(
        "/api/settings/chatgpt/import", content=b"{}", headers=IMPORT_HEADERS
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Content-Type": "application/octet-stream", "X-Tel-Agent-Import": "1"}, 403),
        ({**IMPORT_HEADERS, "Origin": "https://attacker.example"}, 403),
        ({**IMPORT_HEADERS, "Origin": "http://localhost"}, 403),
        ({**IMPORT_HEADERS, "Origin": "null"}, 403),
        ({**IMPORT_HEADERS, "X-Tel-Agent-Import": "0"}, 403),
        ({**IMPORT_HEADERS, "Content-Type": "application/json"}, 415),
        ({**IMPORT_HEADERS, "Content-Type": "multipart/form-data"}, 415),
    ],
)
async def test_import_rejects_wrong_origin_or_media_type(chat_clients, headers, status):
    response = await chat_clients["owner"].post(
        "/api/settings/chatgpt/import", content=b"{}", headers=headers
    )
    assert response.status_code == status


@pytest.mark.parametrize("public_base_url", [None, "http://localhost", "https://other.example"])
async def test_import_requires_configured_canonical_https_api(chat_clients, public_base_url):
    client = chat_clients["owner"]
    client._transport.app.state.settings.public_base_url = public_base_url
    status = await client.get("/api/settings/chatgpt")
    assert status.json()["can_import"] is False
    response = await client.post(
        "/api/settings/chatgpt/import", content=b"{}", headers=IMPORT_HEADERS
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "chatgpt_import_insecure"


@pytest.mark.parametrize("body", [b"", b"x" * (1024 * 1024 + 1)])
async def test_import_rejects_empty_or_oversized_file(chat_clients, body):
    response = await chat_clients["owner"].post(
        "/api/settings/chatgpt/import", content=body, headers=IMPORT_HEADERS
    )
    assert response.status_code == 413


async def test_import_has_no_url_or_path_input(chat_clients):
    response = await chat_clients["owner"].post(
        "/api/settings/chatgpt/import?path=/etc/passwd", content=b"{}", headers=IMPORT_HEADERS
    )
    assert response.status_code == 415


@pytest.mark.parametrize(
    "error,status,code",
    [
        (
            ChatGPTAuthError("invalid_identity", "DO_NOT_EXPOSE_UPLOAD"),
            422,
            "chatgpt_import_failed",
        ),
        (
            ChatGPTAuthError("account_changed", "DO_NOT_EXPOSE_UPLOAD"),
            409,
            "chatgpt_import_conflict",
        ),
        (RuntimeError("DO_NOT_EXPOSE_UPLOAD"), 503, "chatgpt_import_failed"),
    ],
)
async def test_import_never_echoes_or_logs_exception_details(
    chat_clients, monkeypatch, caplog, error, status, code
):
    class BrokenAuth(FakeAuth):
        def import_registration_bytes(self, data, *, cancelled=None):
            raise error

    monkeypatch.setattr(chatgpt_plan, "auth_store", BrokenAuth)
    response = await chat_clients["owner"].post(
        "/api/settings/chatgpt/import", content=b"DO_NOT_EXPOSE_UPLOAD", headers=IMPORT_HEADERS
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "DO_NOT_EXPOSE_UPLOAD" not in response.text
    assert "DO_NOT_EXPOSE_UPLOAD" not in caplog.text
    assert not chat_clients["owner"]._transport.app.state.chatgpt_import_lock.locked()


async def test_import_cancellation_signals_worker_and_blocks_overlapping_upload(
    monkeypatch, caplog
):
    started = threading.Event()
    stopped = threading.Event()
    release = threading.Event()

    class SlowAuth(FakeAuth):
        def import_registration_bytes(self, data, *, cancelled=None):
            started.set()
            while not cancelled():
                time.sleep(0.001)
            stopped.set()
            release.wait(timeout=2)
            raise RuntimeError("DO_NOT_LOG_CANCELLED_CREDENTIAL")

    monkeypatch.setattr(chatgpt_plan, "auth_store", SlowAuth)
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(
                public_base_url="https://localhost", cors_origins=["https://localhost"]
            )
        )
    )

    def upload_request():
        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        return Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "https",
                "path": "/api/settings/chatgpt/import",
                "query_string": b"",
                "headers": [(b"host", b"localhost")]
                + [
                    (key.lower().encode(), value.encode())
                    for key, value in IMPORT_HEADERS.items()
                ],
                "app": app,
            },
            receive,
        )

    # Exercise cancellation at the route boundary. Auth/origin denial is covered
    # above; cancelling ASGITransport itself also cancels unrelated DB cleanup.
    context = SimpleNamespace(role="owner")
    pending = asyncio.create_task(chatgpt_plan.import_account(upload_request(), context))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        concurrent = await chatgpt_plan.import_account(upload_request(), context)
        assert concurrent.status_code == 409
        assert b"chatgpt_import_in_progress" in concurrent.body
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert await asyncio.to_thread(stopped.wait, 2)
        assert app.state.chatgpt_import_lock.locked()
    finally:
        release.set()
        worker = app.state.chatgpt_import_worker
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    assert not app.state.chatgpt_import_lock.locked()
    assert "DO_NOT_LOG_CANCELLED_CREDENTIAL" not in caplog.text
