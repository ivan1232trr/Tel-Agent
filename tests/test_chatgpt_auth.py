"""ChatGPT OAuth uses fake issuers and disposable credentials; never live OAuth."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from agent.chatgpt_auth import (
    AUTHORIZE_URL,
    DISCOVERY_URL,
    DYNAMIC_CLIENT_ID,
    ISSUER,
    RESOURCE,
    SCOPES,
    TOKEN_URL,
    ChatGPTAuthError,
    ChatGPTAuthStore,
    _atomic_private,
    _Attempt,
    _read_private,
    get_access_token,
)

CLIENT = "oaiapp_test_first"
OTHER = "oaiapp_test_second"
ACCESS = "synthetic-access-value"
REFRESH = "synthetic-refresh-value"
JWKS = f"{ISSUER}/.well-known/jwks.json"
REVOKE = f"{ISSUER}/oauth/revoke"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeIssuer:
    def __init__(self, signing_key):
        self.key = signing_key
        self.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
        self.jwk.update(kid="key1", use="sig", alg="RS256")
        self.tokens = {}
        self.requests = []
        self.token_status = 200
        self.revocation_status = 200
        self.discovery = {
            "issuer": ISSUER,
            "authorization_endpoint": AUTHORIZE_URL,
            "token_endpoint": TOKEN_URL,
            "jwks_uri": JWKS,
            "revocation_endpoint": REVOKE,
        }

    def identity(self, *, client_id=CLIENT, subject="subject-one", nonce="nonce", **claims):
        return jwt.encode(
            {
                "iss": ISSUER,
                "sub": subject,
                "aud": client_id,
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "nonce": nonce,
                "email": "same@example.test",
                **claims,
            },
            self.key,
            algorithm="RS256",
            headers={"kid": "key1"},
        )

    def response(self, *, id_token=None, scope=SCOPES):
        result = {
            "access_token": ACCESS,
            "refresh_token": REFRESH,
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        if id_token is not None:
            result["id_token"] = id_token
        if scope is not None:
            result["scope"] = scope
        return result

    def __call__(self, request):
        self.requests.append(request)
        url = str(request.url)
        if url == DISCOVERY_URL:
            return httpx.Response(200, json=self.discovery)
        if url == JWKS:
            return httpx.Response(200, json={"keys": [self.jwk]})
        if url == TOKEN_URL:
            return httpx.Response(self.token_status, json=self.tokens)
        if url == REVOKE:
            return httpx.Response(self.revocation_status)
        pytest.fail("Unexpected endpoint used")


@pytest.fixture
def issuer(signing_key):
    return FakeIssuer(signing_key)


@pytest.fixture
def store(tmp_path, issuer):
    return ChatGPTAuthStore(tmp_path / "credentials", transport=httpx.MockTransport(issuer))


def login(
    store,
    issuer,
    *,
    client_id=CLIENT,
    subject="subject-one",
    scope=SCOPES,
    returning=False,
    label=None,
):
    attempt = store._begin(
        "http://127.0.0.1:43210/auth/callback", client_id if returning else None, label
    )
    issuer.tokens = issuer.response(
        id_token=issuer.identity(
            client_id=client_id,
            subject=subject,
            nonce=attempt.nonce,
        ),
        scope=scope,
    )
    result = store._complete(
        attempt,
        urlencode(
            {
                "code": "synthetic-code",
                "state": attempt.state,
                "client_id": client_id,
            }
        ),
    )
    return result


def expire(store, client_id=CLIENT):
    path = store._record_path(client_id)
    value = _read_private(path)
    value["expires_at"] = time.time() - 1
    _atomic_private(path, value)
    return value


def test_empty_inspection_has_no_network_or_files(store, issuer):
    assert store.status() == {
        "connected": False,
        "selected_client_id": None,
        "accounts": [],
        "reason": "sign_in_required",
    }
    assert store.accounts() == []
    assert not store.directory.exists()
    assert issuer.requests == []


def test_host_is_stable_and_storage_is_owner_only(store):
    first = store.init_host()
    assert first.startswith("urn:uuid:")
    assert ChatGPTAuthStore(store.directory).init_host() == first
    assert store.directory.stat().st_mode & 0o777 == 0o700
    assert (store.directory / "host.json").stat().st_mode & 0o777 == 0o600
    assert (store.directory / ".lock").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "unsafe", ["permissions", "directory-symlink", "file-symlink", "hardlink"]
)
def test_unsafe_storage_rejected(store, tmp_path, unsafe):
    if unsafe == "directory-symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        store.directory.symlink_to(target, target_is_directory=True)
    else:
        store.init_host()
        path = store.directory / "host.json"
        if unsafe == "permissions":
            path.chmod(0o644)
        elif unsafe == "file-symlink":
            target = tmp_path / "target.json"
            path.rename(target)
            path.symlink_to(target)
        else:
            os.link(path, tmp_path / "copy.json")
    with pytest.raises(ChatGPTAuthError):
        store.init_host()


def test_pkce_dynamic_registration_and_returning_hints(store, issuer):
    pending = store._begin("http://127.0.0.1:4567/auth/callback", None, None)
    url = pending.authorization_url()
    query = parse_qs(urlsplit(url).query)
    assert url.startswith(AUTHORIZE_URL)
    assert query["client_id"] == [DYNAMIC_CLIENT_ID]
    assert query["agent_name_hint"] == ["Tel-Agent"]
    assert query["resource"] == [RESOURCE]
    assert query["scope"] == [SCOPES]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [
        base64.urlsafe_b64encode(hashlib.sha256(pending.verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    ]
    assert pending.verifier not in url
    login(store, issuer)
    returning = store._begin("http://127.0.0.1:7654/auth/callback", CLIENT, None)
    params = parse_qs(urlsplit(returning.authorization_url()).query)
    assert params["client_id"] == [CLIENT]
    assert "agent_name_hint" not in params
    assert params["id_token_hint"] == [_read_private(store._record_path(CLIENT))["id_token"]]
    assert params["login_hint"] == ["same@example.test"]
    assert returning.state != pending.state
    assert returning.nonce != pending.nonce
    assert returning.verifier != pending.verifier
    assert "prompt" not in params
    assert parse_qs(urlsplit(returning.authorization_url(consent=True)).query)["prompt"] == [
        "consent"
    ]
    assert pending.verifier not in repr(pending)


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:4321/auth/callback",
        "https://127.0.0.1:4321/auth/callback",
        "http://127.0.0.1:4321/callback",
        "https://public.example/auth/callback",
        "http://127.0.0.1:4321/auth/callback?token=test",
        "http://u:p@127.0.0.1:4321/auth/callback",
        "http://127.0.0.1:invalid/auth/callback",
        "http://127.0.0.1:99999/auth/callback",
        "http://[invalid/auth/callback",
    ],
)
def test_only_documented_loopback_redirect_allowed(uri):
    with pytest.raises(ChatGPTAuthError, match=r"127\.0\.0\.1"):
        _Attempt(None, "host", uri).authorization_url()


@pytest.mark.parametrize(
    "query",
    [
        {"state": "wrong", "code": "secret", "client_id": CLIENT},
        {"state": "é", "code": "secret", "client_id": CLIENT},
        {"code": "secret", "client_id": CLIENT},
        {"state": "correct", "error": "access_denied", "client_id": CLIENT},
        {"state": "correct", "code": "secret"},
        {"state": "correct", "code": "secret", "client_id": DYNAMIC_CLIENT_ID},
        {"state": "correct", "client_id": CLIENT},
    ],
)
def test_invalid_callbacks_never_exchange(store, issuer, query):
    attempt = store._begin("http://127.0.0.1:4321/auth/callback", None, None)
    attempt.state = "correct"
    with pytest.raises(ChatGPTAuthError):
        store._complete(attempt, urlencode(query))
    assert issuer.requests == []
    assert store.accounts() == []
    assert attempt.consumed


def test_callback_is_one_use_and_rejects_duplicates():
    pending = _Attempt(None, "host", "http://127.0.0.1:4321/auth/callback")
    query = urlencode({"state": pending.state, "code": "code", "client_id": CLIENT})
    assert pending.consume(query) == ("code", CLIENT)
    with pytest.raises(ChatGPTAuthError, match="expired"):
        pending.consume(query)
    duplicate = _Attempt(None, "host", pending.redirect_uri)
    with pytest.raises(ChatGPTAuthError):
        duplicate.consume(
            f"state={duplicate.state}&state={duplicate.state}&code=code&client_id={CLIENT}"
        )


def test_expired_attempt_never_consumes_code():
    attempt = _Attempt(None, "host", "http://127.0.0.1:4321/auth/callback", expires_at=0)
    with pytest.raises(ChatGPTAuthError, match="expired"):
        attempt.consume(
            urlencode({"state": attempt.state, "code": "code", "client_id": CLIENT})
        )


def test_successful_exchange_is_public_client_and_metadata_is_safe(store, issuer):
    account = login(store, issuer, label="Personal")
    assert account["connected"] and account["selected"]
    assert account["label"] == "Personal"
    request = next(request for request in issuer.requests if str(request.url) == TOKEN_URL)
    form = parse_qs(request.content.decode())
    assert form["client_id"] == [CLIENT]
    assert form["grant_type"] == ["authorization_code"]
    assert form["redirect_uri"] == ["http://127.0.0.1:43210/auth/callback"]
    assert form["resource"] == [RESOURCE]
    assert "client_secret" not in form
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    serialized = json.dumps(store.status())
    assert ACCESS not in serialized and REFRESH not in serialized
    assert "id_token" not in serialized
    assert store._record_path(CLIENT).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://attacker.example"},
        {"aud": OTHER},
        {"exp": 1},
        {"nonce": "wrong"},
        {"sub": ""},
        {"iat": int(time.time()) + 9999},
        {"azp": OTHER},
        {"aud": [CLIENT, OTHER]},
        {"exp": float("inf")},
        {"iat": float("nan")},
        {"exp": True},
        {"nonce": "é"},
    ],
)
def test_bad_id_claims_cannot_activate(store, issuer, claims):
    pending = store._begin("http://127.0.0.1:4321/auth/callback", None, None)
    defaults = {"nonce": pending.nonce, **claims}
    issuer.tokens = issuer.response(id_token=issuer.identity(**defaults))
    with pytest.raises(ChatGPTAuthError):
        store._complete(
            pending, urlencode({"state": pending.state, "code": "code", "client_id": CLIENT})
        )
    assert store.accounts() == []


def test_bad_signature_and_unsigned_jwt_rejected(store, issuer):
    valid = issuer.identity()
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims = jwt.decode(valid, options={"verify_signature": False})
    for bad in (
        jwt.encode(claims, other_key, algorithm="RS256", headers={"kid": "key1"}),
        jwt.encode(claims, key="", algorithm="none", headers={"kid": "key1"}),
    ):
        with pytest.raises(ChatGPTAuthError, match="verified"):
            store._validate_id_token(bad, CLIENT, nonce="nonce")


def test_unknown_signing_key_refetches_cached_jwks(store, issuer):
    store._validate_id_token(issuer.identity(), CLIENT)
    issuer.jwk["kid"] = "key2"
    claims = {
        "iss": ISSUER,
        "sub": "s",
        "aud": CLIENT,
        "exp": int(time.time()) + 60,
        "iat": int(time.time()),
    }
    token = jwt.encode(claims, issuer.key, algorithm="RS256", headers={"kid": "key2"})
    assert store._validate_id_token(token, CLIENT)["sub"] == "s"
    assert sum(str(request.url) == JWKS for request in issuer.requests) == 2


def test_identity_only_sign_in_retained_without_enabling_inference(store, issuer):
    first = login(store, issuer, scope="openid profile email offline_access")
    assert not first["connected"] and not first["plan_enabled"]
    assert not first["selected"]
    assert len(store.accounts()) == 1
    with pytest.raises(ChatGPTAuthError):
        store.select(CLIENT)


def test_identity_only_grant_without_offline_access_retains_verified_account(store, issuer):
    pending = store._begin("http://127.0.0.1:4321/auth/callback", None, None)
    issuer.tokens = issuer.response(
        id_token=issuer.identity(nonce=pending.nonce), scope="openid profile email"
    )
    issuer.tokens.pop("refresh_token")
    account = store._complete(
        pending, urlencode({"state": pending.state, "code": "code", "client_id": CLIENT})
    )
    assert not account["connected"] and not account["selected"]
    assert len(store.accounts()) == 1
    returning = store._begin("http://127.0.0.1:4321/auth/callback", CLIENT, None)
    assert returning.subject == "subject-one" and returning.id_token_hint
    with pytest.raises(ChatGPTAuthError):
        store.select(CLIENT)


async def test_scope_is_required_before_any_request(store, issuer):
    login(store, issuer, scope="openid profile email offline_access")
    before = len(issuer.requests)
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token(CLIENT)
    assert error.value.code == "plan_permission_missing"
    assert len(issuer.requests) == before


def test_same_email_registrations_stay_separate(store, issuer):
    login(store, issuer, label="First")
    login(store, issuer, client_id=OTHER, subject="different-subject", label="Second")
    assert len(store.accounts()) == 2
    assert store.status()["selected_client_id"] == OTHER
    store.select(CLIENT)
    assert store.status()["selected_client_id"] == CLIENT
    with pytest.raises(ChatGPTAuthError):
        store.select("oaiapp_absent")
    assert store.status()["selected_client_id"] == CLIENT


def test_reauthorization_rejects_changed_client_and_subject(store, issuer):
    login(store, issuer)
    original = store._record_path(CLIENT).read_bytes()
    attempt = store._begin("http://127.0.0.1:9876/auth/callback", CLIENT, None)
    with pytest.raises(ChatGPTAuthError) as error:
        store._complete(
            attempt, urlencode({"code": "code", "state": attempt.state, "client_id": OTHER})
        )
    assert error.value.code == "identity_mismatch"
    with pytest.raises(ChatGPTAuthError) as error:
        login(store, issuer, subject="other-subject", returning=True)
    assert error.value.code == "identity_mismatch"
    assert store._record_path(CLIENT).read_bytes() == original


def test_returning_callback_may_omit_client_id(store, issuer):
    login(store, issuer)
    attempt = store._begin("http://127.0.0.1:9876/auth/callback", CLIENT, None)
    issuer.tokens = issuer.response(id_token=issuer.identity(nonce=attempt.nonce))
    assert store._complete(attempt, urlencode({"state": attempt.state, "code": "code"}))[
        "connected"
    ]


async def test_unexpired_token_has_no_network(store, issuer):
    login(store, issuer)
    before = len(issuer.requests)
    assert await get_access_token(store) == ACCESS
    assert len(issuer.requests) == before


async def test_rotating_refresh_replaces_token_set_and_keeps_grant_when_omitted(store, issuer):
    login(store, issuer)
    expire(store)
    issuer.tokens = issuer.response(scope=None)
    replacement_access, replacement_refresh = "replacement-access", "replacement-refresh"
    issuer.tokens.update(access_token=replacement_access, refresh_token=replacement_refresh)
    assert await store.access_token() == replacement_access
    saved = _read_private(store._record_path(CLIENT))
    assert saved["refresh_token"] == replacement_refresh
    assert set(saved["scopes"]) == set(SCOPES.split())
    form = parse_qs(issuer.requests[-1].content.decode())
    assert form == {
        "grant_type": ["refresh_token"],
        "client_id": [CLIENT],
        "refresh_token": [REFRESH],
        "resource": [RESOURCE],
    }
    assert saved["expires_at"] > time.time() + 3500


async def test_concurrent_async_refresh_happens_once(store, issuer):
    login(store, issuer)
    expire(store)
    issuer.tokens = issuer.response()
    before = sum(str(request.url) == TOKEN_URL for request in issuer.requests)
    results = await asyncio.gather(*(store.access_token() for _ in range(6)))
    assert results == [ACCESS] * 6
    assert sum(str(request.url) == TOKEN_URL for request in issuer.requests) == before + 1


def _process_refresh(directory, counter, barrier, output):
    def fake(request):
        if str(request.url) != TOKEN_URL:
            raise RuntimeError("Unexpected endpoint")
        with counter.get_lock():
            counter.value += 1
        time.sleep(0.1)
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    store = ChatGPTAuthStore(Path(directory), transport=httpx.MockTransport(fake))
    barrier.wait(timeout=5)
    output.put(asyncio.run(store.access_token()))


def test_refresh_serialized_across_processes(store, issuer):
    login(store, issuer)
    expire(store)
    context = multiprocessing.get_context("fork")
    counter = context.Value("i", 0)
    barrier = context.Barrier(2)
    output = context.Queue()
    workers = [
        context.Process(
            target=_process_refresh, args=(str(store.directory), counter, barrier, output)
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert counter.value == 1
    assert [output.get(timeout=1) for _ in range(2)] == ["new-access"] * 2


@pytest.mark.parametrize(
    "code",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ],
)
async def test_terminal_refresh_clears_tokens_but_retains_mapping(store, issuer, code):
    login(store, issuer)
    host = store.init_host()
    expire(store)
    issuer.token_status = 400
    issuer.tokens = {"error": code, "error_description": REFRESH}
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token()
    assert error.value.code == code and REFRESH not in str(error.value)
    saved = _read_private(store._record_path(CLIENT))
    assert saved["subject"] == "subject-one" and saved["client_id"] == CLIENT
    assert not {"access_token", "refresh_token", "id_token"} & saved.keys()
    assert store.init_host() == host
    assert not store.status()["connected"]


@pytest.mark.parametrize(
    "status,code", [(503, "server_error"), (429, "rate_limit"), (400, "invalid_client")]
)
async def test_nonterminal_refresh_preserves_credentials(store, issuer, status, code):
    login(store, issuer)
    expire(store)
    original = store._record_path(CLIENT).read_bytes()
    issuer.token_status = status
    issuer.tokens = {"error": code}
    with pytest.raises(ChatGPTAuthError):
        await store.access_token()
    assert store._record_path(CLIENT).read_bytes() == original


async def test_refresh_respects_earliest_time(store, issuer):
    login(store, issuer)
    record = expire(store)
    record["earliest_refresh_at"] = time.time() + 60
    _atomic_private(store._record_path(CLIENT), record)
    before = len(issuer.requests)
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token()
    assert error.value.code == "refresh_too_early"
    assert len(issuer.requests) == before


async def test_refresh_cannot_silently_change_identity(store, issuer):
    login(store, issuer)
    expire(store)
    issuer.tokens = issuer.response(id_token=issuer.identity(subject="different-account"))
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token()
    assert error.value.code == "identity_mismatch"
    assert "refresh_token" not in _read_private(store._record_path(CLIENT))


@pytest.mark.parametrize("nonce", [None, "original", "wrong"])
async def test_refresh_identity_nonce_is_optional_but_must_match_if_present(
    store, issuer, nonce
):
    login(store, issuer)
    original = expire(store)
    claims = {
        "iss": ISSUER,
        "sub": "subject-one",
        "aud": CLIENT,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    if nonce is not None:
        claims["nonce"] = original["nonce"] if nonce == "original" else nonce
    token = jwt.encode(claims, issuer.key, algorithm="RS256", headers={"kid": "key1"})
    issuer.tokens = issuer.response(id_token=token)
    if nonce == "wrong":
        with pytest.raises(ChatGPTAuthError) as error:
            await store.access_token()
        assert error.value.code == "invalid_identity"
        assert "refresh_token" not in _read_private(store._record_path(CLIENT))
    else:
        assert await store.access_token() == ACCESS


async def test_refresh_scope_loss_is_persisted_and_blocks_inference(store, issuer):
    login(store, issuer)
    expire(store)
    issuer.tokens = issuer.response(scope="openid email")
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token()
    assert error.value.code == "plan_permission_missing"
    assert _read_private(store._record_path(CLIENT))["scopes"] == ["email", "openid"]
    assert not store.status()["connected"]


def test_sign_out_revokes_and_only_clears_selected_registration(store, issuer):
    login(store, issuer)
    login(store, issuer, client_id=OTHER, subject="subject-two")
    host_id = store.init_host()
    result = store.sign_out(CLIENT)
    assert result["revocation_confirmed"]
    assert "refresh_token" not in _read_private(store._record_path(CLIENT))
    assert _read_private(store._record_path(OTHER))["refresh_token"] == REFRESH
    assert store.status()["selected_client_id"] == OTHER
    assert store.init_host() == host_id
    form = parse_qs(issuer.requests[-1].content.decode())
    assert form == {
        "token": [REFRESH],
        "token_type_hint": ["refresh_token"],
        "client_id": [CLIENT],
    }
    returning = store._begin("http://127.0.0.1:1234/auth/callback", CLIENT, None)
    assert returning.id_token_hint is None
    assert returning.client_id == CLIENT


def test_failed_remote_revocation_retries_and_warns_but_clears_local(
    store, issuer, monkeypatch
):
    login(store, issuer)
    issuer.revocation_status = 503
    monkeypatch.setattr("agent.chatgpt_auth.time.sleep", lambda seconds: None)
    result = store.sign_out()
    assert result["signed_out"] and not result["revocation_confirmed"]
    assert "not confirmed" in result["message"]
    assert "refresh_token" not in _read_private(store._record_path(CLIENT))
    assert sum(str(request.url) == REVOKE for request in issuer.requests) == 3
    assert store.status()["selected_client_id"] is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://attacker.example/revoke",
        "http://auth.openai.com/revoke",
        "https://auth.openai.com@attacker.example/revoke",
        "https://auth.openai.com:invalid/revoke",
        "https://auth.openai.com:99999/revoke",
        "https://auth.openai.com/revoke?redirect=https://attacker.example",
        "https://auth.openai.com/revoke#fragment",
    ],
)
def test_discovery_cannot_exfiltrate_credentials(store, issuer, endpoint):
    login(store, issuer)
    issuer.discovery["revocation_endpoint"] = endpoint
    store._discovery_cache = None
    result = store.sign_out()
    assert not result["revocation_confirmed"]
    assert all(str(request.url) != endpoint for request in issuer.requests)


def test_vm_import_preserves_vm_host_identity(store, issuer, tmp_path):
    login(store, issuer)
    local_host = store.init_host()
    vm = ChatGPTAuthStore(tmp_path / "vm", transport=httpx.MockTransport(issuer))
    vm_host = vm.init_host()
    assert vm_host != local_host
    vm.import_registration(store._record_path(CLIENT))
    assert vm.init_host() == vm_host
    assert _read_private(vm._record_path(CLIENT))["ext_agent_host_id"] == vm_host
    assert vm.status()["connected"]


def test_vm_import_rejects_tampered_identity_and_unsafe_permissions(store, issuer, tmp_path):
    login(store, issuer)
    vm = ChatGPTAuthStore(tmp_path / "vm", transport=httpx.MockTransport(issuer))
    source = store._record_path(CLIENT)
    value = _read_private(source)
    value["subject"] = "tampered"
    _atomic_private(source, value)
    with pytest.raises(ChatGPTAuthError):
        vm.import_registration(source)
    source.chmod(0o644)
    with pytest.raises(ChatGPTAuthError):
        vm.import_registration(source)
    assert vm.accounts() == []


def test_login_race_cannot_restore_a_signed_out_session(store, issuer):
    login(store, issuer)
    pending = store._begin("http://127.0.0.1:4321/auth/callback", CLIENT, None)
    store.sign_out()
    issuer.tokens = issuer.response(id_token=issuer.identity(nonce=pending.nonce))
    with pytest.raises(ChatGPTAuthError) as error:
        store._complete(pending, urlencode({"state": pending.state, "code": "code"}))
    assert error.value.code == "account_changed"
    assert not store.status()["connected"]


def test_cli_status_does_not_initialize_or_leak(tmp_path, capsys):
    path = Path(__file__).parents[1] / "scripts" / "chatgpt_auth.py"
    spec = importlib.util.spec_from_file_location("chatgpt_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "missing"
    assert module.main(["--auth-dir", str(directory), "status"]) == 0
    assert not directory.exists()
    assert json.loads(capsys.readouterr().out)["connected"] is False


def test_loopback_listener_starts_before_browser_and_never_logs_code(
    store, issuer, monkeypatch, capsys
):
    from agent import chatgpt_auth

    events = []
    pending_url = []

    class Socket:
        def makefile(self, *args, **kwargs):
            params = parse_qs(urlsplit(pending_url[0]).query)
            query = urlencode(
                {"state": params["state"][0], "code": "secret-test-code", "client_id": CLIENT}
            )
            return io.BytesIO(
                f"GET /auth/callback?{query} HTTP/1.1\r\nHost: 127.0.0.1:54321\r\n\r\n".encode()
            )

        def sendall(self, data):
            assert b"secret-test-code" not in data

    class Server:
        server_port = 54321

        def __init__(self, address, handler):
            assert address[0] == "127.0.0.1"
            self.handler = handler
            events.append("listener")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def handle_request(self):
            self.handler(Socket(), ("127.0.0.1", 12345), self)

    def browser(url):
        events.append("browser")
        pending_url.append(url)
        nonce = parse_qs(urlsplit(url).query)["nonce"][0]
        issuer.tokens = issuer.response(id_token=issuer.identity(nonce=nonce))
        return True

    monkeypatch.setattr(chatgpt_auth, "HTTPServer", Server)
    assert store.sign_in(browser_open=browser)["connected"]
    assert events == ["listener", "browser", "closed"]
    captured = capsys.readouterr()
    assert "secret-test-code" not in captured.out + captured.err
    assert pending_url[0] not in captured.out + captured.err


@pytest.mark.parametrize("operation", ["accounts", "status"])
def test_metadata_filesystem_errors_are_sanitized(store, monkeypatch, operation):
    store.init_host()

    def fail(selected):
        raise PermissionError(f"private path contains {REFRESH}")

    monkeypatch.setattr(store, "_accounts", fail)
    with pytest.raises(ChatGPTAuthError) as error:
        getattr(store, operation)()
    assert error.value.code == "storage_unavailable"
    assert REFRESH not in str(error.value)


@pytest.mark.parametrize("operation", ["accounts", "status"])
def test_metadata_rejects_dangling_directory_symlinks(store, tmp_path, operation):
    store.directory.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ChatGPTAuthError) as error:
        getattr(store, operation)()
    assert error.value.code == "unsafe_storage"


@pytest.mark.parametrize("contents", [b"\xff", b"[" * 2000, b"x" * (1024 * 1024 + 1)])
def test_invalid_credential_bytes_are_sanitized(store, contents):
    store.init_host()
    (store.directory / "host.json").write_bytes(contents)
    with pytest.raises(ChatGPTAuthError) as error:
        store.status()
    assert error.value.code == "invalid_storage"


def test_credential_fifo_is_rejected_without_waiting_for_a_writer(store):
    store.init_host()
    path = store.directory / "host.json"
    path.unlink()
    os.mkfifo(path, mode=0o600)
    with pytest.raises(ChatGPTAuthError) as error:
        store.status()
    assert error.value.code == "unsafe_storage"


def test_cli_storage_error_has_no_traceback_or_private_path(store, capsys):
    path = Path(__file__).parents[1] / "scripts" / "chatgpt_auth.py"
    spec = importlib.util.spec_from_file_location("chatgpt_cli_error_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store.init_host()
    (store.directory / "host.json").unlink()
    (store.directory / "host.json").symlink_to(store.directory / ".lock")
    assert module.main(["--auth-dir", str(store.directory), "status"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "storage_unavailable" in output.err
    assert "Traceback" not in output.err and str(store.directory) not in output.err


def test_http_response_limit_stops_reading_and_closes_stream(tmp_path):
    class Oversized(httpx.SyncByteStream):
        reads = 0
        closed = False

        def __iter__(self):
            for _ in range(10):
                self.reads += 1
                yield b"x" * (1024 * 1024)

        def close(self):
            self.closed = True

    body = Oversized()
    store = ChatGPTAuthStore(
        tmp_path / "unused",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body)),
    )
    with pytest.raises(ChatGPTAuthError) as error:
        store._http("GET", DISCOVERY_URL)
    assert error.value.code == "invalid_response"
    assert body.reads == 2 and body.closed
    assert not store.directory.exists()


def test_http_redirect_is_not_followed_and_errors_are_sanitized(tmp_path):
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": f"https://attacker.example/{REFRESH}"})

    store = ChatGPTAuthStore(tmp_path / "unused", transport=httpx.MockTransport(redirect))
    with pytest.raises(ChatGPTAuthError) as error:
        store._json_response(store._http("POST", TOKEN_URL, data={"refresh_token": REFRESH}))
    assert len(requests) == 1 and str(requests[0].url) == TOKEN_URL
    assert REFRESH not in str(error.value)


async def test_refresh_transport_failure_keeps_credentials_without_leaking(store, issuer):
    login(store, issuer)
    expire(store)
    original = store._record_path(CLIENT).read_bytes()

    def fail(request):
        raise httpx.ConnectError(REFRESH, request=request)

    store._transport = httpx.MockTransport(fail)
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token()
    assert error.value.code == "network_error"
    assert REFRESH not in str(error.value)
    assert store._record_path(CLIENT).read_bytes() == original


@pytest.mark.parametrize("problem", ["missing_refresh", "expiry", "identity", "scopes"])
async def test_successful_but_invalid_refresh_never_reuses_consumed_token(
    store, issuer, problem
):
    login(store, issuer)
    expire(store)
    issuer.tokens = issuer.response()
    if problem == "missing_refresh":
        issuer.tokens.pop("refresh_token")
    elif problem == "expiry":
        issuer.tokens["expires_in"] = 10**500
    elif problem == "identity":
        issuer.tokens["id_token"] = "synthetic-invalid-identity"  # noqa: S105 - fake token
    else:
        issuer.tokens["scope"] = []
    with pytest.raises(ChatGPTAuthError):
        await store.access_token()
    saved = _read_private(store._record_path(CLIENT))
    assert "refresh_token" not in saved
    assert saved["subject"] == "subject-one"
    before = len(issuer.requests)
    with pytest.raises(ChatGPTAuthError):
        await store.access_token()
    assert len(issuer.requests) == before


@pytest.mark.parametrize("body", [b"not JSON", b"[" * 2000, b"x" * (1024 * 1024 + 1)])
async def test_invalid_success_response_clears_consumed_refresh_token(store, issuer, body):
    login(store, issuer)
    expire(store)
    store._transport = httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    with pytest.raises(ChatGPTAuthError) as error:
        await store.access_token()
    assert error.value.code == "invalid_response"
    assert "refresh_token" not in _read_private(store._record_path(CLIENT))


def test_vm_import_allows_expired_verified_identity_hint(store, issuer, tmp_path):
    login(store, issuer)
    source = store._record_path(CLIENT)
    incoming = _read_private(source)
    incoming["id_token"] = issuer.identity(exp=int(time.time()) - 60, nonce=incoming["nonce"])
    _atomic_private(source, incoming)
    vm = ChatGPTAuthStore(tmp_path / "vm", transport=httpx.MockTransport(issuer))
    assert vm.import_registration(source)["connected"]
    assert vm.status()["selected_client_id"] == CLIENT


def test_revocation_network_failures_retry_then_clear_local(store, issuer, monkeypatch):
    login(store, issuer)
    requests = []

    def fail(request):
        requests.append(request)
        raise httpx.ReadTimeout(REFRESH, request=request)

    store._transport = httpx.MockTransport(fail)
    monkeypatch.setattr("agent.chatgpt_auth.time.sleep", lambda _: None)
    result = store.sign_out()
    assert not result["revocation_confirmed"]
    assert len(requests) == 3
    assert all(str(request.url) == REVOKE for request in requests)
    assert "refresh_token" not in _read_private(store._record_path(CLIENT))
    assert REFRESH not in json.dumps(result)


def test_invalid_grant_restarts_authorization_once_with_issued_client(
    store, issuer, monkeypatch
):
    from agent import chatgpt_auth

    urls = []

    class Socket:
        def makefile(self, *args, **kwargs):
            params = parse_qs(urlsplit(urls[-1]).query)
            query = urlencode(
                {"state": params["state"][0], "code": "code", "client_id": CLIENT}
            )
            return io.BytesIO(
                f"GET /auth/callback?{query} HTTP/1.1\r\nHost: 127.0.0.1:54321\r\n\r\n".encode()
            )

        def sendall(self, data):
            pass

    class Server:
        server_port = 54321

        def __init__(self, address, handler):
            self.handler = handler

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def handle_request(self):
            self.handler(Socket(), ("127.0.0.1", 12345), self)

    def browser(url):
        urls.append(url)
        if len(urls) == 1:
            issuer.token_status = 400
            issuer.tokens = {"error": "invalid_grant"}
        else:
            issuer.token_status = 200
            nonce = parse_qs(urlsplit(url).query)["nonce"][0]
            issuer.tokens = issuer.response(id_token=issuer.identity(nonce=nonce))
        return True

    monkeypatch.setattr(chatgpt_auth, "HTTPServer", Server)
    assert store.sign_in(browser_open=browser)["connected"]
    assert len(urls) == 2
    first, second = (parse_qs(urlsplit(url).query) for url in urls)
    assert first["client_id"] == [DYNAMIC_CLIENT_ID]
    assert second["client_id"] == [CLIENT]
    assert "agent_name_hint" not in second
    for key in ("state", "nonce", "code_challenge"):
        assert first[key] != second[key]
    assert first["redirect_uri"] == second["redirect_uri"]
    assert len(store.accounts()) == 1


def registration_payload(issuer, /, **changes):
    """A disposable registration assembled without authorizing or refreshing."""
    now = time.time()
    return {
        "version": 1,
        "issuer": ISSUER,
        "client_id": CLIENT,
        "subject": "subject-one",
        "email": "same@example.test",
        "label": "Personal",
        "ext_agent_host_id": "urn:uuid:11111111-1111-4111-8111-111111111111",
        "revision": "synthetic-source-revision",
        "nonce": "nonce",
        "access_token": ACCESS,
        "refresh_token": REFRESH,
        "id_token": issuer.identity(),
        "token_type": "Bearer",
        "scopes": SCOPES.split(),
        "saved_at": now - 5,
        "expires_at": now + 3600,
        "earliest_refresh_at": 0,
        **changes,
    }


def test_bytes_import_preserves_host_and_returns_only_safe_metadata(store, issuer, caplog):
    host_id = store.init_host()
    incoming = registration_payload(issuer)
    result = store.import_registration_bytes(json.dumps(incoming).encode())
    saved = _read_private(store._record_path(CLIENT))
    assert saved["ext_agent_host_id"] == host_id != incoming["ext_agent_host_id"]
    assert saved["revision"] != incoming["revision"]
    assert saved["email"] == "same@example.test"
    assert result["connected"] and result["selected"]
    assert result["label"] == "Personal"
    assert store.status()["selected_client_id"] == CLIENT
    assert store._record_path(CLIENT).stat().st_mode & 0o777 == 0o600
    assert store.directory.stat().st_mode & 0o777 == 0o700
    assert {str(request.url) for request in issuer.requests} == {DISCOVERY_URL, JWKS}
    assert all(request.method == "GET" and not request.content for request in issuer.requests)
    for token in (ACCESS, REFRESH, incoming["id_token"]):
        assert token not in json.dumps(result)
        assert token not in json.dumps(store.status())
        assert token not in caplog.text
    assert not list(store.directory.glob(".pending-*"))


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not JSON",
        b"null",
        b"[]",
        b'"synthetic-access-value"',
        b'{"access_token":"synthetic-access-value",}',
        b'{"access_token":"synthetic-access-value","access_token":"second"}',
        b'{"nested":{"a":1,"a":2}}',
        b'{"access_token":1,"access_\\u0074oken":2}',
        b'{"expires_at":NaN}',
        b'{"expires_at":Infinity}',
        b'{"expires_at":-Infinity}',
        b'{"expires_at":1e10000}',
        b'{"value":"\\ud800"}',
        b'{"\\ud800":"value"}',
        b'{"value":"\xff"}',
        b"\xef\xbb\xbf{}",
        "{}".encode("utf-16"),
        b'{"nested":' + b"[" * 17 + b"0" + b"]" * 17 + b"}",
        b"[" * 2000,
        b" " * (1024 * 1024 + 1),
    ],
)
def test_bytes_import_rejects_ambiguous_json_without_files_or_network(
    store, issuer, body, caplog
):
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(body)
    assert error.value.code == "invalid_storage"
    assert ACCESS not in str(error.value)
    assert ACCESS not in caplog.text
    assert not store.directory.exists()
    assert issuer.requests == []


@pytest.mark.parametrize("body", [None, "{}", bytearray(b"{}")])
def test_bytes_import_rejects_non_bytes(store, body):
    with pytest.raises(ChatGPTAuthError, match="invalid"):
        store.import_registration_bytes(body)
    assert not store.directory.exists()


def test_bytes_import_accepts_exact_size_limit(store, issuer):
    body = json.dumps(registration_payload(issuer)).encode()
    body += b" " * (1024 * 1024 - len(body))
    assert store.import_registration_bytes(body)["connected"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 2),
        ("issuer", "https://other.example"),
        ("client_id", "dynamic_agent_client"),
        ("subject", []),
        ("label", {"nested": REFRESH}),
        ("label", "x" * 201),
        ("label", "unsafe\nlabel"),
        ("label", f"Account: {ACCESS}"),
        ("label", f"Account: {REFRESH}"),
        ("email", "x" * 321),
        ("email", "different@example.test"),
        ("email", REFRESH),
        ("nonce", 7),
        ("nonce", "wrong"),
        ("ext_agent_host_id", {"nested": REFRESH}),
        ("revision", [REFRESH]),
        ("access_token", "invalid\ntoken"),
        ("access_token", "x" * (128 * 1024 + 1)),
        ("refresh_token", None),
        ("token_type", "Basic"),
        ("token_type", ["Bearer"]),
        ("scopes", "resource.invoke chatgpt.tokens.use.direct"),
        ("scopes", ["openid"]),
        ("scopes", [*SCOPES.split(), "resource.invoke"]),
        ("scopes", [*SCOPES.split(), ACCESS]),
        ("scopes", [*SCOPES.split(), "a b"]),
        ("scopes", [*SCOPES.split(), "a\\b"]),
        ("scopes", [*SCOPES.split(), 'a"b']),
        ("scopes", [*SCOPES.split(), "x" * 257]),
        ("scopes", SCOPES.split() + [str(i) for i in range(65)]),
        ("scopes", [*SCOPES.split(), []]),
        ("expires_at", True),
        ("expires_at", 0),
        ("expires_at", "3600"),
        ("expires_at", 1e12),
        ("saved_at", False),
        ("saved_at", -1),
        pytest.param("saved_at", time.time() + 86400, id="future-saved-at"),
        ("earliest_refresh_at", "never"),
        ("earliest_refresh_at", -1),
        ("api_key", REFRESH),
        ("client_secret", REFRESH),
        ("unknown_metadata", REFRESH),
    ],
)
def test_bytes_import_rejects_bad_schema_without_damaging_existing_account(
    store, issuer, field, value, caplog
):
    store.import_registration_bytes(json.dumps(registration_payload(issuer)).encode())
    before = {path.name: path.read_bytes() for path in store.directory.glob("*.json")}
    incoming = registration_payload(issuer, **{field: value})
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert ACCESS not in str(error.value) and REFRESH not in str(error.value)
    assert ACCESS not in caplog.text and REFRESH not in caplog.text
    assert {path.name: path.read_bytes() for path in store.directory.glob("*.json")} == before
    assert store.status()["connected"]


@pytest.mark.parametrize("claim", ["iss", "aud", "sub"])
def test_bytes_import_rejects_signed_identity_mismatch(store, issuer, claim):
    overrides = {"iss": "https://other.example", "aud": OTHER, "sub": "subject-two"}
    incoming = registration_payload(
        issuer, id_token=issuer.identity(**{claim: overrides[claim]})
    )
    with pytest.raises(ChatGPTAuthError):
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert not store.directory.exists()


@pytest.mark.parametrize("at_hash", [None, 12, "wrong", "correct"])
def test_bytes_import_validates_optional_signed_access_hash(store, issuer, at_hash):
    expected = base64.urlsafe_b64encode(hashlib.sha256(ACCESS.encode()).digest()[:16])
    identity = issuer.identity(
        at_hash=expected.rstrip(b"=").decode() if at_hash == "correct" else at_hash
    )
    incoming = registration_payload(issuer, id_token=identity)
    if at_hash == "correct":
        assert store.import_registration_bytes(json.dumps(incoming).encode())["connected"]
    else:
        with pytest.raises(ChatGPTAuthError) as error:
            store.import_registration_bytes(json.dumps(incoming).encode())
        assert error.value.code == "invalid_identity"
        assert not store.directory.exists()


def test_bytes_import_allows_expired_hint_and_access_without_refreshing(store, issuer):
    incoming = registration_payload(
        issuer, id_token=issuer.identity(exp=int(time.time()) - 60), expires_at=time.time() - 1
    )
    assert store.import_registration_bytes(json.dumps(incoming).encode())["connected"]
    assert all(request.method == "GET" for request in issuer.requests)


def test_bytes_import_idempotency_preserves_current_tokens_and_metadata(store, issuer):
    incoming = registration_payload(issuer)
    store.import_registration_bytes(json.dumps(incoming).encode())
    before = {path.name: path.read_bytes() for path in store.directory.glob("*.json")}
    incoming.update(saved_at=time.time(), expires_at=time.time() + 7200, label="Replacement")
    result = store.import_registration_bytes(json.dumps(incoming).encode())
    assert result["label"] == "Personal"
    assert {path.name: path.read_bytes() for path in store.directory.glob("*.json")} == before


@pytest.mark.parametrize("newer_timestamp", [False, True])
def test_bytes_import_cannot_replace_working_credentials_even_with_forged_timestamp(
    store, issuer, newer_timestamp
):
    incoming = registration_payload(issuer)
    store.import_registration_bytes(json.dumps(incoming).encode())
    before = {path.name: path.read_bytes() for path in store.directory.glob("*.json")}
    other_access, other_refresh = "other-synthetic-access", "other-synthetic-refresh"
    incoming.update(
        access_token=other_access,
        refresh_token=other_refresh,
        saved_at=time.time() + 60 if newer_timestamp else time.time() - 60,
        expires_at=time.time() + 7200,
    )
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert error.value.code == "account_changed"
    assert {path.name: path.read_bytes() for path in store.directory.glob("*.json")} == before


def test_bytes_import_cannot_change_subject_of_existing_client(store, issuer):
    store.import_registration_bytes(json.dumps(registration_payload(issuer)).encode())
    before = store._record_path(CLIENT).read_bytes()
    incoming = registration_payload(
        issuer, subject="subject-two", id_token=issuer.identity(subject="subject-two")
    )
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert error.value.code == "identity_mismatch"
    assert store._record_path(CLIENT).read_bytes() == before


def test_bytes_import_cannot_roll_back_concurrent_refresh(store, issuer, monkeypatch):
    incoming = registration_payload(issuer)
    store.import_registration_bytes(json.dumps(incoming).encode())
    original_validate = store._validated_import
    refreshed = {}
    replacement_access, replacement_refresh = "refreshed-access", "refreshed-refresh"

    def validate_then_refresh(value):
        verified = original_validate(value)
        issuer.tokens = issuer.response()
        issuer.tokens.update(access_token=replacement_access, refresh_token=replacement_refresh)
        store._access_token(CLIENT, force_refresh=True)
        refreshed.update(_read_private(store._record_path(CLIENT)))
        return verified

    monkeypatch.setattr(store, "_validated_import", validate_then_refresh)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert error.value.code == "account_changed"
    assert _read_private(store._record_path(CLIENT)) == refreshed
    assert refreshed["access_token"] == replacement_access


def test_bytes_import_cancelled_before_validation_has_no_effect(store, issuer):
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(b"{}", cancelled=lambda: True)
    assert error.value.code == "import_cancelled"
    assert not store.directory.exists()
    assert issuer.requests == []


def test_bytes_import_cancelled_during_jwks_has_no_effect(store, issuer):
    cancelled = threading.Event()

    def transport(request):
        response = issuer(request)
        if str(request.url) == JWKS:
            cancelled.set()
        return response

    store._transport = httpx.MockTransport(transport)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(
            json.dumps(registration_payload(issuer)).encode(), cancelled=cancelled.is_set
        )
    assert error.value.code == "import_cancelled"
    assert not store.directory.exists()


def test_bytes_import_cancelled_while_waiting_for_refresh_lock_has_no_effect(
    store, issuer, monkeypatch
):
    store.init_host()
    before = (store.directory / "host.json").read_bytes()
    cancelled, validated = threading.Event(), threading.Event()
    original_validate = store._validated_import

    def validate(value):
        result = original_validate(value)
        validated.set()
        return result

    monkeypatch.setattr(store, "_validated_import", validate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with store._locked():
            future = executor.submit(
                store.import_registration_bytes,
                json.dumps(registration_payload(issuer)).encode(),
                cancelled=cancelled.is_set,
            )
            assert validated.wait(timeout=5)
            cancelled.set()
        with pytest.raises(ChatGPTAuthError) as error:
            future.result(timeout=5)
    assert error.value.code == "import_cancelled"
    assert (store.directory / "host.json").read_bytes() == before
    assert not store._record_path(CLIENT).exists()


def test_cli_import_uses_shared_validation_and_permits_explicit_newer_replacement(
    store, issuer, tmp_path
):
    source = tmp_path / "synthetic-registration.json"
    incoming = registration_payload(issuer)
    _atomic_private(source, incoming)
    assert store.import_registration(source)["connected"]
    newer_access, newer_refresh = "newer-synthetic-access", "newer-synthetic-refresh"
    incoming.update(
        access_token=newer_access,
        refresh_token=newer_refresh,
        saved_at=time.time(),
        expires_at=time.time() + 7200,
    )
    _atomic_private(source, incoming)
    store.import_registration(source)
    assert _read_private(store._record_path(CLIENT))["refresh_token"] == newer_refresh
    incoming["saved_at"] = time.time() - 60
    incoming["refresh_token"] = REFRESH
    _atomic_private(source, incoming)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration(source)
    assert error.value.code == "account_changed"
    assert _read_private(store._record_path(CLIENT))["refresh_token"] == newer_refresh
    incoming["client_secret"] = REFRESH
    _atomic_private(source, incoming)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration(source)
    assert error.value.code == "invalid_storage"


@pytest.mark.parametrize("token", ['synthetic-"quoted"-access', "synthetic-\\-access"])
def test_bytes_import_rejects_json_escaped_token_reflection(store, issuer, token):
    incoming = registration_payload(issuer, access_token=token, label=f"Account: {token}")
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert error.value.code == "invalid_storage"
    assert token not in str(error.value)
    assert not store.directory.exists()


def test_bytes_import_cannot_upgrade_existing_grant_by_changing_unsigned_scopes(store, issuer):
    login(store, issuer, scope="openid profile email offline_access")
    incoming = _read_private(store._record_path(CLIENT))
    incoming["scopes"] = SCOPES.split()
    before = {path.name: path.read_bytes() for path in store.directory.glob("*.json")}
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert error.value.code == "plan_permission_missing"
    assert {path.name: path.read_bytes() for path in store.directory.glob("*.json")} == before
    assert store.status()["selected_client_id"] is None


def test_bytes_import_write_failure_is_sanitized_and_preserves_active_registration(
    store, issuer, monkeypatch
):
    store.import_registration_bytes(json.dumps(registration_payload(issuer)).encode())
    before = {path.name: path.read_bytes() for path in store.directory.glob("*.json")}
    incoming = registration_payload(
        issuer, client_id=OTHER, id_token=issuer.identity(client_id=OTHER)
    )

    def fail(*args):
        raise PermissionError(REFRESH)

    monkeypatch.setattr("agent.chatgpt_auth._atomic_private", fail)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(json.dumps(incoming).encode())
    assert error.value.code == "storage_unavailable"
    assert REFRESH not in str(error.value)
    assert {path.name: path.read_bytes() for path in store.directory.glob("*.json")} == before


def test_bytes_import_cancelled_after_commit_does_not_change_active_selection(
    store, issuer, monkeypatch
):
    store.import_registration_bytes(json.dumps(registration_payload(issuer)).encode())
    before = (store.directory / "host.json").read_bytes()
    cancelled = threading.Event()
    incoming = registration_payload(
        issuer, client_id=OTHER, id_token=issuer.identity(client_id=OTHER)
    )

    def commit_then_cancel(path, value):
        _atomic_private(path, value)
        cancelled.set()

    monkeypatch.setattr("agent.chatgpt_auth._atomic_private", commit_then_cancel)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration_bytes(
            json.dumps(incoming).encode(), cancelled=cancelled.is_set
        )
    assert error.value.code == "import_cancelled"
    assert (store.directory / "host.json").read_bytes() == before
    assert store.status()["selected_client_id"] == CLIENT
    # The completed atomic write cannot be undone by cancelling the worker.
    assert store._record_path(OTHER).exists()
    assert any(account["client_id"] == OTHER for account in store.status()["accounts"])


def test_cli_import_rejects_duplicate_keys_using_shared_parser(store, tmp_path):
    source = tmp_path / "duplicate-synthetic-registration.json"
    source.write_bytes(b'{"access_token":"synthetic-access-value","access_token":"other"}')
    source.chmod(0o600)
    with pytest.raises(ChatGPTAuthError) as error:
        store.import_registration(source)
    assert error.value.code == "invalid_storage"
    assert ACCESS not in str(error.value)
    assert not store.directory.exists()
