"""Official ChatGPT plan-use OAuth for an owner-operated, self-hosted runtime.

Credentials never enter the dashboard or the settings database. The runtime owner
uses the local CLI, then owns this protected directory (0700, files 0600). Unix
file locks serialize rotating refresh tokens across API and agent processes. No
browser, callback, code exchange or grant is initiated by constructing a store.

Reference: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

ISSUER = "https://auth.openai.com"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPES = frozenset({"resource.invoke", "chatgpt.tokens.use.direct"})
_TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)
_TOKEN_FIELDS = (
    "access_token",
    "refresh_token",
    "id_token",
    "expires_at",
    "earliest_refresh_at",
    "nonce",
)
_MAX_FILE_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 1024 * 1024
_CLIENT_ID = re.compile(r"oaiapp_[A-Za-z0-9_-]{1,200}\Z")


class ChatGPTAuthError(RuntimeError):
    """A safe, actionable error; never contains a raw provider response or token."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _client_id(value: Any) -> str:
    if not isinstance(value, str) or not _CLIENT_ID.fullmatch(value):
        raise ChatGPTAuthError("invalid_client", "The issued ChatGPT client ID is invalid.")
    return value


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 * 1024:
        raise ChatGPTAuthError(
            "invalid_response", f"ChatGPT did not return a valid {field_name}."
        )
    return value


def _number(value: Any, default: float = 0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 0 <= value < 1e12:
            return float(value)
    return default


@contextmanager
def _safe_storage() -> Iterator[None]:
    """Keep paths and malformed credential contents out of API/CLI diagnostics."""
    try:
        yield
    except (OSError, UnicodeError):
        raise ChatGPTAuthError(
            "storage_unavailable", "ChatGPT credential storage is unavailable or unsafe."
        ) from None


def _protected_stat(info: os.stat_result, *, directory: bool = False) -> None:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode) or info.st_uid != os.getuid():
        raise ChatGPTAuthError("unsafe_storage", "ChatGPT storage must be owned by this user.")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ChatGPTAuthError(
            "unsafe_storage", "ChatGPT storage needs owner-only permissions (0700/0600)."
        )
    if not directory and info.st_nlink != 1:
        raise ChatGPTAuthError(
            "unsafe_storage", "ChatGPT credential files cannot be hard links."
        )


def _read_private(path: Path) -> dict[str, Any]:
    """Open without following a symlink, check the opened inode, and bound parsing."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        _protected_stat(os.fstat(source.fileno()))
        raw = source.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise ChatGPTAuthError("invalid_storage", "The ChatGPT credential file is too large.")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ChatGPTAuthError(
            "invalid_storage", "The ChatGPT credential file is invalid."
        ) from None
    if not isinstance(value, dict):
        raise ChatGPTAuthError("invalid_storage", "The ChatGPT credential file is invalid.")
    return value


def _atomic_private(path: Path, value: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            json.dump(value, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(repr=False)
class _Attempt:
    """Short-lived, one-use state; repr must never expose login hints or PKCE."""

    client_id: str | None
    host_id: str
    redirect_uri: str
    subject: str | None = None
    label: str | None = None
    state: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(64))
    expires_at: float = field(default_factory=lambda: time.monotonic() + 300)
    consumed: bool = False
    id_token_hint: str | None = None
    email: str | None = None
    revision: str | None = None

    def authorization_url(self, *, consent: bool = False) -> str:
        try:
            parts = urlsplit(self.redirect_uri)
            valid = (
                parts.scheme == "http"
                and parts.hostname == "127.0.0.1"
                and parts.path == "/auth/callback"
                and bool(parts.port)
                and not (parts.username or parts.password or parts.query or parts.fragment)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ChatGPTAuthError("invalid_redirect", "OAuth requires a 127.0.0.1 callback.")
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.verifier.encode()).digest())
        params = {
            "client_id": self.client_id or DYNAMIC_CLIENT_ID,
            "ext_agent_host_id": self.host_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": self.state,
            "nonce": self.nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge.rstrip(b"=").decode(),
        }
        if self.client_id:
            if self.id_token_hint:
                params["id_token_hint"] = self.id_token_hint
            if self.email:
                params["login_hint"] = self.email
        else:
            params["agent_name_hint"] = "Tel-Agent"
        if consent:
            params["prompt"] = "consent"
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    def consume(self, query: str) -> tuple[str, str]:
        if self.consumed or time.monotonic() >= self.expires_at:
            raise ChatGPTAuthError("expired_attempt", "ChatGPT sign-in expired. Start again.")
        self.consumed = True
        try:
            values = parse_qs(query, keep_blank_values=True, max_num_fields=20)
        except ValueError:
            raise ChatGPTAuthError(
                "invalid_callback", "ChatGPT returned an invalid callback."
            ) from None
        if any(len(items) != 1 for items in values.values()):
            raise ChatGPTAuthError("invalid_callback", "ChatGPT returned an invalid callback.")
        if not secrets.compare_digest(
            values.get("state", [""])[0].encode(), self.state.encode()
        ):
            raise ChatGPTAuthError("state_mismatch", "ChatGPT sign-in could not be verified.")
        if "error" in values:
            raise ChatGPTAuthError("access_denied", "ChatGPT sign-in was declined or failed.")
        issued = values.get("client_id", [self.client_id])[0]
        issued = _client_id(issued)
        if self.client_id is not None and issued != self.client_id:
            raise ChatGPTAuthError("identity_mismatch", "ChatGPT returned a different client.")
        return _required_text(values.get("code", [None])[0], "authorization code"), issued


class ChatGPTAuthStore:
    """Isolated account registrations in protected runtime storage, never in a browser.

    `status`, `accounts` and `select` expose metadata only. Call those small disk
    operations through `asyncio.to_thread` from an async API. `access_token` already
    does so; it holds an OS lock throughout refresh and its atomic replacement.
    The optional HTTP transport is a test seam, never an endpoint override.
    """

    def __init__(
        self, directory: Path, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.directory = Path(directory).expanduser().absolute()
        self._transport = transport
        self._discovery_cache: tuple[float, dict[str, Any]] | None = None
        self._jwks_cache: tuple[float, dict[str, Any]] | None = None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if os.name != "posix":
            raise ChatGPTAuthError(
                "unsupported_storage",
                "ChatGPT credential storage requires Linux, macOS or WSL.",
            )
        import fcntl

        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            _protected_stat(self.directory.lstat(), directory=True)
            fd = os.open(
                self.directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "a+") as lock:
                _protected_stat(os.fstat(lock.fileno()))
                deadline = time.monotonic() + 45
                while True:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise ChatGPTAuthError(
                                "storage_busy",
                                "Another process is updating ChatGPT credentials.",
                            ) from None
                        time.sleep(0.025)
                try:
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except (OSError, UnicodeError):
            raise ChatGPTAuthError(
                "storage_unavailable", "ChatGPT credential storage is unavailable or unsafe."
            ) from None

    def _host(self, *, create: bool = True) -> dict[str, Any]:
        path = self.directory / "host.json"
        if not path.exists():
            if not create:
                return {"selected_client_id": None}
            host = {
                "version": 1,
                "ext_agent_host_id": f"urn:uuid:{uuid.uuid4()}",
                "selected_client_id": None,
            }
            _atomic_private(path, host)
            return host
        host = _read_private(path)
        try:
            uuid.UUID(
                _required_text(host.get("ext_agent_host_id"), "host ID").removeprefix(
                    "urn:uuid:"
                )
            )
        except ValueError:
            raise ChatGPTAuthError(
                "invalid_storage", "The saved ChatGPT host ID is invalid."
            ) from None
        if host.get("selected_client_id") is not None:
            _client_id(host["selected_client_id"])
        return host

    def init_host(self) -> str:
        """Create or return this runtime's non-secret, stable host identifier."""
        with self._locked():
            return str(self._host()["ext_agent_host_id"])

    def _record_path(self, client_id: str) -> Path:
        digest = hashlib.sha256(_client_id(client_id).encode()).hexdigest()
        return self.directory / f"registration-{digest}.json"

    def _record(self, client_id: str) -> dict[str, Any]:
        path = self._record_path(client_id)
        if not path.exists():
            raise ChatGPTAuthError(
                "account_missing", "That ChatGPT registration was not found."
            )
        value = _read_private(path)
        if value.get("client_id") != client_id or value.get("issuer") != ISSUER:
            raise ChatGPTAuthError(
                "invalid_storage", "The saved ChatGPT registration is invalid."
            )
        _required_text(value.get("subject"), "account identity")
        scopes = value.get("scopes", [])
        if not isinstance(scopes, list) or not all(isinstance(item, str) for item in scopes):
            raise ChatGPTAuthError("invalid_storage", "The saved ChatGPT grant is invalid.")
        return value

    @staticmethod
    def _metadata(record: dict[str, Any], selected: str | None) -> dict[str, Any]:
        scopes = record.get("scopes", [])
        plan_enabled = isinstance(scopes, list) and PLAN_SCOPES.issubset(scopes)
        connected = (
            bool(record.get("access_token") and record.get("refresh_token")) and plan_enabled
        )
        return {
            "client_id": record["client_id"],
            "label": record.get("label") or record["client_id"],
            "email": record.get("email"),
            "subject": record["subject"],
            "connected": connected,
            "selected": selected == record["client_id"],
            "plan_enabled": plan_enabled,
            "scopes": scopes,
            "expires_at": record.get("expires_at"),
        }

    def _accounts(self, selected: str | None) -> list[dict[str, Any]]:
        accounts = []
        for path in sorted(self.directory.glob("registration-*.json")):
            candidate = _read_private(path)
            client_id = _client_id(candidate.get("client_id"))
            if path != self._record_path(client_id):
                raise ChatGPTAuthError("invalid_storage", "A ChatGPT registration is misnamed.")
            accounts.append(self._metadata(self._record(client_id), selected))
        return accounts

    def accounts(self) -> list[dict[str, Any]]:
        """Return safe, distinct registrations; email is a display hint, never an identity."""
        with _safe_storage():
            if not self._directory_exists():
                return []
            return self._accounts(self._host(create=False).get("selected_client_id"))

    def _directory_exists(self) -> bool:
        try:
            info = self.directory.lstat()
        except FileNotFoundError:
            return False
        _protected_stat(info, directory=True)
        return True

    def status(self) -> dict[str, Any]:
        """Inspect local state only; never authorize, refresh or call an inference endpoint."""
        with _safe_storage():
            if self._directory_exists():
                selected = self._host(create=False).get("selected_client_id")
                accounts = self._accounts(selected)
            else:
                selected, accounts = None, []
            active = next((item for item in accounts if item["selected"]), None)
            connected = bool(active and active["connected"])
            return {
                "connected": connected,
                "selected_client_id": selected,
                "accounts": accounts,
                "reason": None
                if connected
                else (
                    "plan_permission_missing"
                    if active and not active["plan_enabled"]
                    else "sign_in_required"
                ),
            }

    def select(self, client_id: str) -> dict[str, Any]:
        """Explicitly switch an account; fail closed rather than selecting a fallback."""
        with self._locked():
            host = self._host()
            record = self._record(client_id)
            metadata = self._metadata(record, client_id)
            if not metadata["connected"]:
                raise ChatGPTAuthError(
                    "sign_in_required", "Sign in to this ChatGPT account and enable plan usage."
                )
            host["selected_client_id"] = client_id
            _atomic_private(self.directory / "host.json", host)
            return metadata

    def _http(
        self, method: str, url: str, *, data: dict[str, str] | None = None
    ) -> httpx.Response:
        # Tokens go only to the issuer, with no redirects, proxies from the environment,
        # URL credentials or raw response logging. HTTP exceptions are sanitized below.
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme == "https"
                and parsed.hostname == "auth.openai.com"
                and parsed.port in (None, 443)
                and not (parsed.username or parsed.password or parsed.fragment or parsed.query)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ChatGPTAuthError("invalid_endpoint", "ChatGPT advertised an unsafe endpoint.")
        try:
            with httpx.Client(
                timeout=15, follow_redirects=False, trust_env=False, transport=self._transport
            ) as client:
                with client.stream(
                    method, url, data=data, headers={"Accept": "application/json"}
                ) as response:
                    body = bytearray()
                    deadline = time.monotonic() + 30
                    for chunk in response.iter_bytes():
                        if time.monotonic() > deadline:
                            raise ChatGPTAuthError(
                                "network_error",
                                "ChatGPT could not be reached. Try again later.",
                            )
                        if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                            raise ChatGPTAuthError(
                                "invalid_response", "ChatGPT returned an oversized response."
                            )
                        body.extend(chunk)
                    # The stream must be closed before callers parse or retain a response.
                    return httpx.Response(
                        response.status_code,
                        content=bytes(body),
                        request=response.request,
                    )
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError):
            raise ChatGPTAuthError(
                "network_error", "ChatGPT could not be reached. Try again later."
            ) from None

    @staticmethod
    def _json_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code != 200:
            code = "oauth_error"
            try:
                payload = response.json()
                error = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error, dict):
                    error = error.get("code")
                if error in _TERMINAL_REFRESH_ERRORS | {"invalid_client"}:
                    code = error
            except (ValueError, TypeError, RecursionError):
                pass
            if response.status_code == 429 or response.status_code >= 500:
                code = "temporarily_unavailable"
            raise ChatGPTAuthError(
                code, "ChatGPT authorization failed. Check the connection or sign in again."
            )
        try:
            payload = response.json()
        except (ValueError, RecursionError):
            payload = None
        if not isinstance(payload, dict):
            raise ChatGPTAuthError("invalid_response", "ChatGPT returned an invalid response.")
        return payload

    def _discovery(self) -> dict[str, Any]:
        if self._discovery_cache and self._discovery_cache[0] > time.monotonic():
            return self._discovery_cache[1]
        value = self._json_response(self._http("GET", DISCOVERY_URL))
        if (
            value.get("issuer") != ISSUER
            or value.get("authorization_endpoint") != AUTHORIZE_URL
            or value.get("token_endpoint") != TOKEN_URL
        ):
            raise ChatGPTAuthError("invalid_issuer", "ChatGPT discovery could not be verified.")
        self._discovery_cache = (time.monotonic() + 3600, value)
        return value

    def _validate_id_token(
        self,
        token: str,
        client_id: str,
        *,
        nonce: str | None = None,
        allow_expired: bool = False,
    ) -> dict[str, Any]:
        discovery = self._discovery()
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")
            if algorithm not in ("RS256", "ES256") or not isinstance(header.get("kid"), str):
                raise ValueError
            cached = self._jwks_cache and self._jwks_cache[0] > time.monotonic()
            jwks = self._jwks_cache[1] if cached and self._jwks_cache else None
            for attempt in range(2):
                if jwks is None:
                    uri = _required_text(discovery.get("jwks_uri"), "JWKS endpoint")
                    jwks = self._json_response(self._http("GET", uri))
                    self._jwks_cache = (time.monotonic() + 3600, jwks)
                matches = [
                    key
                    for key in jwks.get("keys", [])
                    if isinstance(key, dict)
                    and key.get("kid") == header["kid"]
                    and key.get("use", "sig") == "sig"
                    and key.get("alg", algorithm) == algorithm
                ]
                if len(matches) == 1:
                    break
                if not cached or attempt:
                    raise ValueError
                jwks = None
            signing_key = jwt.PyJWK.from_dict(matches[0], algorithm=algorithm)
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                audience=client_id,
                issuer=ISSUER,
                leeway=5,
                options={
                    "require": ["iss", "sub", "aud", "exp", "iat"],
                    "verify_exp": not allow_expired,
                },
            )
            if any(_number(claims[name], -1) < 0 for name in ("exp", "iat")):
                raise ValueError
            _required_text(claims.get("sub"), "account identity")
            if claims.get("azp") not in (None, client_id):
                raise ValueError
            if (
                isinstance(claims["aud"], list)
                and len(claims["aud"]) > 1
                and claims.get("azp") != client_id
            ):
                raise ValueError
            if nonce is not None and (
                not isinstance(claims.get("nonce"), str)
                or not secrets.compare_digest(claims["nonce"].encode(), nonce.encode())
            ):
                raise ValueError
            return claims
        except (
            jwt.PyJWTError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            OverflowError,
            RecursionError,
        ):
            raise ChatGPTAuthError(
                "invalid_identity", "The ChatGPT identity token could not be verified."
            ) from None

    @staticmethod
    def _apply_tokens(
        record: dict[str, Any], tokens: dict[str, Any], *, initial: bool
    ) -> dict[str, Any]:
        updated = dict(record)
        updated["access_token"] = _required_text(tokens.get("access_token"), "access token")
        if (
            not isinstance(tokens.get("token_type"), str)
            or tokens["token_type"].lower() != "bearer"
        ):
            raise ChatGPTAuthError("invalid_response", "ChatGPT did not return a Bearer token.")
        lifetime = _number(tokens.get("expires_in"))
        if not lifetime:
            raise ChatGPTAuthError("invalid_response", "ChatGPT did not return a token expiry.")
        scope = tokens.get("scope")
        if scope is None and not initial:
            scopes = record.get("scopes", [])  # OAuth omission retains the original grant.
        elif isinstance(scope, str):
            scopes = sorted(set(scope.split()))
        else:
            raise ChatGPTAuthError("invalid_response", "ChatGPT did not return granted scopes.")
        if initial and not PLAN_SCOPES.issubset(scopes) and tokens.get("refresh_token") is None:
            # Identity-only consent may omit offline_access and its refresh token.
            # Keep the verified account hint without enabling inference or selection.
            updated.pop("refresh_token", None)
        else:
            updated["refresh_token"] = _required_text(
                tokens.get("refresh_token"), "replacement refresh token"
            )
        updated.update(
            token_type="Bearer",  # noqa: S106 - OAuth scheme, not a secret
            scopes=scopes,
            expires_at=time.time() + lifetime,
            earliest_refresh_at=_number(tokens.get("earliest_refresh_at")),
            saved_at=time.time(),
            revision=uuid.uuid4().hex,
        )
        if tokens.get("id_token"):
            updated["id_token"] = _required_text(tokens["id_token"], "identity token")
        return updated

    def _clear(self, record: dict[str, Any]) -> None:
        for name in _TOKEN_FIELDS:
            record.pop(name, None)
        record.update(scopes=[], revision=uuid.uuid4().hex)
        _atomic_private(self._record_path(record["client_id"]), record)

    async def access_token(
        self, client_id: str | None = None, *, force_refresh: bool = False
    ) -> str:
        """Get a scoped bearer token; rotating refresh is locked and written atomically."""
        return await asyncio.to_thread(
            self._access_token, client_id, force_refresh=force_refresh
        )

    def _access_token(self, client_id: str | None, *, force_refresh: bool = False) -> str:
        with self._locked():
            client_id = client_id or self._host().get("selected_client_id")
            if not client_id:
                raise ChatGPTAuthError(
                    "sign_in_required", "Continue with ChatGPT using the local setup command."
                )
            record = self._record(client_id)
            if not PLAN_SCOPES.issubset(record.get("scopes", [])):
                raise ChatGPTAuthError(
                    "plan_permission_missing",
                    "Enable ChatGPT plan usage for this registration.",
                )
            if not record.get("access_token") or not record.get("refresh_token"):
                raise ChatGPTAuthError(
                    "sign_in_required", "Sign in to this ChatGPT account again."
                )
            now = time.time()
            expiry = _number(record.get("expires_at"))
            earliest = _number(record.get("earliest_refresh_at"))
            if not force_refresh and expiry > now + 60:
                return str(record["access_token"])
            if earliest > now:
                if not force_refresh and expiry > now:
                    return str(record["access_token"])
                raise ChatGPTAuthError(
                    "refresh_too_early", "ChatGPT token refresh is not available yet."
                )
            try:
                tokens = self._json_response(
                    self._http(
                        "POST",
                        TOKEN_URL,
                        data={
                            "grant_type": "refresh_token",
                            "client_id": client_id,
                            "refresh_token": record["refresh_token"],
                            "resource": RESOURCE,
                        },
                    )
                )
            except ChatGPTAuthError as error:
                if error.code in _TERMINAL_REFRESH_ERRORS or error.code == "invalid_response":
                    self._clear(record)
                raise
            try:
                updated = self._apply_tokens(record, tokens, initial=False)
                if tokens.get("id_token"):
                    identity = self._validate_id_token(tokens["id_token"], client_id)
                    if identity["sub"] != record["subject"]:
                        raise ChatGPTAuthError(
                            "identity_mismatch",
                            "ChatGPT returned a different account. Sign in again.",
                        )
                    if "nonce" in identity and isinstance(record.get("nonce"), str):
                        if not isinstance(identity["nonce"], str) or not secrets.compare_digest(
                            identity["nonce"].encode(), record["nonce"].encode()
                        ):
                            raise ChatGPTAuthError(
                                "invalid_identity",
                                "The ChatGPT identity token could not be verified.",
                            )
            except ChatGPTAuthError:
                # HTTP 200 has rotated the session. Never retry the consumed old token
                # if the replacement response or its identity cannot be validated.
                self._clear(record)
                raise
            _atomic_private(self._record_path(client_id), updated)
            if not PLAN_SCOPES.issubset(updated["scopes"]):
                raise ChatGPTAuthError(
                    "plan_permission_missing", "ChatGPT plan usage is no longer enabled."
                )
            return str(updated["access_token"])

    def _begin(self, redirect_uri: str, client_id: str | None, label: str | None) -> _Attempt:
        with self._locked():
            host = self._host()
            record = self._record(client_id) if client_id else {}
            return _Attempt(
                client_id=client_id,
                host_id=host["ext_agent_host_id"],
                redirect_uri=redirect_uri,
                subject=record.get("subject"),
                label=label or record.get("label"),
                id_token_hint=record.get("id_token"),
                email=record.get("email"),
                revision=record.get("revision"),
            )

    def _complete(self, attempt: _Attempt, query: str) -> dict[str, Any]:
        code, client_id = attempt.consume(query)
        attempt.client_id = client_id
        tokens = self._json_response(
            self._http(
                "POST",
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "code_verifier": attempt.verifier,
                    "redirect_uri": attempt.redirect_uri,
                    "resource": RESOURCE,
                },
            )
        )
        id_token = _required_text(tokens.get("id_token"), "identity token")
        identity = self._validate_id_token(id_token, client_id, nonce=attempt.nonce)
        if attempt.subject is not None and identity["sub"] != attempt.subject:
            raise ChatGPTAuthError("identity_mismatch", "ChatGPT returned a different account.")
        record = self._apply_tokens(
            {
                "version": 1,
                "issuer": ISSUER,
                "subject": identity["sub"],
                "client_id": client_id,
                "email": identity.get("email")
                if isinstance(identity.get("email"), str)
                else None,
                "label": attempt.label or client_id,
                "ext_agent_host_id": attempt.host_id,
                "id_token": id_token,
                "nonce": attempt.nonce,
            },
            tokens,
            initial=True,
        )
        with self._locked():
            host = self._host()
            if host["ext_agent_host_id"] != attempt.host_id:
                raise ChatGPTAuthError(
                    "identity_mismatch", "The ChatGPT runtime host changed during sign-in."
                )
            path = self._record_path(client_id)
            if path.exists():
                old = self._record(client_id)
                if old["subject"] != identity["sub"] or old.get("revision") != attempt.revision:
                    raise ChatGPTAuthError(
                        "account_changed",
                        "This ChatGPT account changed during sign-in. Start again.",
                    )
            _atomic_private(path, record)
            if PLAN_SCOPES.issubset(record["scopes"]):
                host["selected_client_id"] = client_id
                _atomic_private(self.directory / "host.json", host)
            return self._metadata(record, host.get("selected_client_id"))

    def sign_in(
        self,
        client_id: str | None = None,
        *,
        label: str | None = None,
        port: int = 0,
        timeout: int = 300,
        consent: bool = False,
        browser_open: Callable[[str], Any] = webbrowser.open,
    ) -> dict[str, Any]:
        """Local, interactive CLI only. Bind loopback before opening the system browser.

        No URL is printed: returning authorization URLs contain an ID-token hint.
        There is no remote/public callback or manual code-paste fallback.
        """
        if not 1 <= timeout <= 600 or not 0 <= port <= 65535:
            raise ChatGPTAuthError(
                "invalid_setup", "Use a valid port and a timeout of 1-600 seconds."
            )
        query: list[str] = []

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # A request line contains the authorization code; never log it.

            def do_GET(self) -> None:
                parts = urlsplit(self.path)
                expected_host = f"127.0.0.1:{self.server.server_port}"  # type: ignore[attr-defined]
                valid = (
                    parts.path == "/auth/callback"
                    and self.headers.get("Host") == expected_host
                    and not parts.scheme
                    and not parts.netloc
                    and len(self.path) <= 16384
                )
                if valid and not query:
                    query.append(parts.query)
                self.send_response(200 if valid else 404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
                )
                self.end_headers()
                self.wfile.write(
                    b"Return to the Tel-Agent terminal to check the sign-in result."
                )

        class Listener(HTTPServer):
            def get_request(self) -> Any:
                connection, address = super().get_request()
                connection.settimeout(1)
                return connection, address

            def handle_error(self, request: Any, client_address: Any) -> None:
                pass  # Do not dump request data into diagnostics.

        try:
            with Listener(("127.0.0.1", port), Callback) as server:
                server.timeout = 0.25
                redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
                attempt = self._begin(redirect_uri, client_id, label)
                attempt.expires_at = time.monotonic() + timeout
                for retry in range(2):
                    if not browser_open(attempt.authorization_url(consent=consent)):
                        raise ChatGPTAuthError(
                            "browser_unavailable",
                            "Open the setup command on a computer with a system browser.",
                        )
                    while not query and time.monotonic() < attempt.expires_at:
                        server.handle_request()
                    if not query:
                        raise ChatGPTAuthError(
                            "expired_attempt", "ChatGPT sign-in timed out. Start again."
                        )
                    try:
                        return self._complete(attempt, query[0])
                    except ChatGPTAuthError as error:
                        if error.code != "invalid_grant" or retry:
                            raise
                        # Retain the issued ID only in this pending attempt until identity
                        # is validated. Never save an unverified account as a registration.
                        attempt = _Attempt(
                            client_id=attempt.client_id,
                            host_id=attempt.host_id,
                            redirect_uri=attempt.redirect_uri,
                            subject=attempt.subject,
                            label=attempt.label,
                            email=attempt.email,
                            id_token_hint=attempt.id_token_hint,
                            revision=attempt.revision,
                            expires_at=attempt.expires_at,
                        )
                        query.clear()
                raise ChatGPTAuthError(
                    "expired_attempt", "ChatGPT sign-in failed. Start again."
                )

        except OSError:
            raise ChatGPTAuthError(
                "callback_unavailable", "The local ChatGPT callback could not be opened."
            ) from None

    def sign_out(self, client_id: str | None = None) -> dict[str, Any]:
        """Revoke and clear local tokens, retaining the account and host mapping."""
        with self._locked():
            host = self._host()
            selected = client_id or host.get("selected_client_id")
            if not selected:
                return {"signed_out": True, "revocation_confirmed": True}
            record = self._record(selected)
            confirmed = not bool(record.get("refresh_token"))
            if record.get("refresh_token"):
                try:
                    endpoint = _required_text(
                        self._discovery().get("revocation_endpoint"), "revocation endpoint"
                    )
                    for attempt in range(3):
                        try:
                            response = self._http(
                                "POST",
                                endpoint,
                                data={
                                    "token": record["refresh_token"],
                                    "token_type_hint": "refresh_token",
                                    "client_id": selected,
                                },
                            )
                            confirmed = response.status_code == 200
                            if confirmed or response.status_code < 500:
                                break
                        except ChatGPTAuthError as error:
                            if error.code != "network_error":
                                break
                        if attempt < 2:
                            time.sleep(0.25 * 2**attempt)
                except ChatGPTAuthError:
                    pass
            self._clear(record)
            if host.get("selected_client_id") == selected:
                host["selected_client_id"] = None
                _atomic_private(self.directory / "host.json", host)
            return {
                "signed_out": True,
                "revocation_confirmed": confirmed,
                "message": None
                if confirmed
                else (
                    "Signed out locally; remote revocation was not confirmed. "
                    "Disconnect Tel-Agent in ChatGPT Settings."
                ),
            }

    def import_registration(self, source: Path) -> dict[str, Any]:
        """Import one owner-transferred protected file; never overwrite the VM host ID.

        The operator, not an assistant or browser upload, transfers this file over SSH.
        Retained ID tokens may have expired; their signatures/issuer/audience/identity
        are still checked. The transferred session's credentials must have one owner:
        stop using the source runtime before the VM performs rotating refreshes.
        """
        with self._locked():
            host = self._host()  # Must exist before loading a laptop's credential record.
            incoming = _read_private(Path(source))
            client_id = _client_id(incoming.get("client_id"))
            identity = self._validate_id_token(
                _required_text(incoming.get("id_token"), "identity token"),
                client_id,
                allow_expired=True,
            )
            if incoming.get("subject") != identity["sub"] or incoming.get("issuer") != ISSUER:
                raise ChatGPTAuthError(
                    "identity_mismatch", "The imported ChatGPT identity does not match."
                )
            for name in ("access_token", "refresh_token"):
                _required_text(incoming.get(name), name)
            scopes = incoming.get("scopes")
            if (
                not isinstance(scopes, list)
                or not all(isinstance(item, str) for item in scopes)
                or not PLAN_SCOPES.issubset(scopes)
                or not _number(incoming.get("expires_at"))
            ):
                raise ChatGPTAuthError(
                    "invalid_storage", "The imported ChatGPT session is incomplete."
                )
            path = self._record_path(client_id)
            if path.exists() and self._record(client_id)["subject"] != identity["sub"]:
                raise ChatGPTAuthError(
                    "identity_mismatch",
                    "The imported account conflicts with a saved registration.",
                )
            incoming["ext_agent_host_id"] = host["ext_agent_host_id"]
            incoming["revision"] = uuid.uuid4().hex
            _atomic_private(path, incoming)
            host["selected_client_id"] = client_id
            _atomic_private(self.directory / "host.json", host)
            return self._metadata(incoming, client_id)


async def get_access_token(store: ChatGPTAuthStore, client_id: str | None = None) -> str:
    """Integration helper; callers must keep the result server-side."""
    return await store.access_token(client_id)
