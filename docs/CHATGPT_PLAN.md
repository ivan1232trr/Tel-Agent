# ChatGPT plan provider

`chatgpt_plan` is an optional language-model provider using official Sign in with
ChatGPT. It sends eligible streaming Responses API requests with the selected
account's OAuth access token. It does not use an API key or a browser session cookie.

## Eligibility and billing

Sign-in and permission to use a ChatGPT plan are separate. An eligible account
must explicitly grant plan usage, and requests consume that account's existing
allowance. Availability and limits are controlled by OpenAI; the UI lists the
models actually visible to the selected account rather than promising a fixed
model list. A subscription is not an unlimited inference allowance.

This setup follows the open-source, locally hosted and owner-operated self-hosted
VM flow. It does **not** establish eligibility for Tel-Agent Cloud, a paid hosted
service, pooled customer accounts, or a multi-tenant credential broker. Review
OpenAI's current requirements before offering a commercial or remotely hosted
service. A public dashboard URL alone does not make its server a valid OAuth
callback host.

The provider never silently switches to an API key, another account, or another
billing path after an authorization, quota, model or network failure. An explicit
provider change in Settings is a separate choice. ChatGPT plan or credit settings
remain controlled in ChatGPT. Speech recognition, speech synthesis and telephony
remain separate services with their own setup and costs.

Official documentation (checked 2026-10-06):

- [Sign in with ChatGPT and plan usage](https://learn.chatgpt.com/docs/sign-in-with-chatgpt)
- [Open-source integration eligibility](https://developers.openai.com/siwc/token-sharing-open-source)
- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Owner-operated self-hosted VMs](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms)

## Storage and process boundaries

The owner runs the CLI to sign in. An authenticated workspace owner can then
transfer one existing registration through the HTTPS dashboard's file picker.
There is no token paste box or server-side browser login. Never paste credential
files, tokens, authorization callback URLs or browser cookies into chat, tickets
or logs. An assistant must not read or upload the credential file for you.

The CLI requires a POSIX environment: Linux, macOS or WSL. Install the repository's
Python dependencies as described in [CONTRIBUTING.md](../CONTRIBUTING.md), then run
`python scripts/chatgpt_auth.py --help` from its root.

Manual installations use `~/.config/tel-agent/chatgpt` by default.
`LLM_CHATGPT_AUTH_DIR` or the CLI's `--auth-dir` selects another protected directory.
The CLI option comes **before** the subcommand. The API and agent must use the
same directory and OS user. Directories require mode `0700`; files require `0600`
and must be owned by the running user. Symlinks and hard-linked credential files
are rejected. Do not solve a permission error with `chmod 777` or a root runtime.

The credential store contains a stable host identity, separate account
registrations, rotating access/refresh tokens, and retained identity metadata.
It is host-local protected OAuth storage, not a dashboard settings export. Treat
the entire directory as secret material. Keep it out of source control, build
contexts, database backups and broadly accessible backup systems.

In both Compose files:

- `tel-agent-chatgpt-auth` is a dedicated persistent named volume mounted at
  `/var/lib/tel-agent/chatgpt`. The image initializes the directory as the
  non-root `telagent` user (UID 1000), mode `0700`.
- `LLM_CHATGPT_AUTH_DIR` is fixed to that container path. A host path in `.env`
  does not relocate the Compose volume. Custom mounts must preserve the path,
  ownership and permissions explicitly.
- The API and current in-process agent reply loop share this directory. The web,
  database and optional automation containers do not receive it.
- The mount must be writable: refreshes take a shared file lock and atomically
  replace credential files. A read-only secret mount or separate copied directory
  cannot support this. Keep it on one host with reliable local file-lock semantics.
- If running an independently packaged agent process, mount the **same** volume
  read-write at the same path, set `LLM_CHATGPT_AUTH_DIR` identically, and use the
  same UID. This repository's default Compose stack has no standalone agent
  service; do not start a second API just to obtain one.

### Domains behind Coolify

Use the source Compose file and Coolify's existing proxy. Assign the dashboard
domain to `web:38471` and the external API domain to `api:38472`. Set
`TEL_AGENT_PUBLIC_API_URL` and `PUBLIC_BASE_URL` to the HTTPS API origin,
`TEL_AGENT_WEB_ORIGIN` to the HTTPS dashboard origin, and
`TEL_AGENT_TRUSTED_HOSTS` to the API hostname plus `localhost,127.0.0.1`.
The Compose file forwards these settings; rebuild the web image after changing
its API URL. Do not add a second proxy competing for ports 80/443.

Retain the existing Compose project and volume identities when updating an
installation. Confirm a restorable backup and the prior deployment's rollback
before cutover. Coolify's resource clone does not by itself copy stored database
records or files. Do not start duplicate production channel workers against the
same database. Leave ChatGPT unselected until the owner completes sign-in.

## Owner setup on a self-hosted VM

These are instructions for the installation owner. Signing in, approving plan
access and transferring the protected file are deliberate owner actions, not
part of building or testing this provider.

### 1. Prepare the server without connecting ChatGPT

Use the normal [Quick start](../README.md#quick-start). Build from a revision
containing this provider; use the release Compose file only once the chosen
published image is confirmed to contain it. Leave the LLM unconfigured until
the account is imported and a model is chosen.

From the repository directory on the VM, create its stable host identity:

```sh
docker compose exec api python scripts/chatgpt_auth.py init-host
docker compose exec api python scripts/chatgpt_auth.py status
```

Both commands are local-only; they neither log in nor make an inference request.
An unconnected status is expected. Keep the VM's `host.json` and its volume across
restarts. Do not copy a laptop's `host.json` over it.

For published images, replace `docker compose` in every command here with
`docker compose -f docker-compose.release.yml`. Use the same Compose project
name/directory throughout so commands operate on the same containers and volume.

### 2. Sign in on the computer running your browser

On your own computer, use the same Tel-Agent CLI and the intended ChatGPT user
and workspace:

```sh
python scripts/chatgpt_auth.py sign-in --label "My Tel-Agent server"
python scripts/chatgpt_auth.py list
```

The browser opens the official authorization page. Review the account and plan
permission before continuing. The callback listens only on `127.0.0.1` on this
computer. Do not run this flow on the VM and then open its loopback callback in
your laptop's browser; those are different hosts. Do not publish an OAuth callback
port or replace it with the dashboard's address.

The CLI prints account metadata without tokens. Each registration has its own
issued `client_id`; labels and email addresses are display hints, not identity
keys. Choose the intended user and workspace deliberately.

### 3. Transfer one protected registration yourself

The saved file is `registration-<SHA-256 of client_id>.json` in the local auth
directory. To find its name without displaying its contents:

```sh
client_id='REPLACE_WITH_THE_ISSUED_CLIENT_ID_FROM_LIST'
registration=$(python -c 'import hashlib, sys; print("registration-" + hashlib.sha256(sys.argv[1].encode()).hexdigest() + ".json")' "$client_id")
```

Stop using this local session for model discovery or inference before transferring
it. The VM will own future token refreshes. Do **not** sign out the copied local
session as a cleanup step: revocation can invalidate the transferred session too.

#### HTTPS dashboard import (no SSH required)

If sign-in has already succeeded locally, do not repeat it. Sign in to your
Tel-Agent dashboard as a workspace **owner**, then open **Settings → Advanced →
ChatGPT plan → Import an existing registration**. Select the registration JSON
file with the native file picker, review the destination notice, and choose
**Upload and import** yourself. Do not upload `host.json` or a Codex credential file.

Both the dashboard and its configured API must use HTTPS. The API's
`PUBLIC_BASE_URL` must be its canonical HTTPS origin, and `CORS_ORIGINS` must
include the exact HTTPS dashboard origin. Ordinary admin/viewer accounts cannot
import credentials. The upload is limited to 1 MiB, checks the signed identity and
plan-use metadata, and stores the accepted registration only in the existing
protected auth volume. It preserves the VM host ID and never returns token values.

Import does not run inference or select a model/provider. After success, refresh
the model list and explicitly choose the account and model. If the response is
lost, refresh the connection before doing anything else; the browser never retries
an upload automatically. A conflicting upload cannot replace the tokens of an
existing working registration. Do not delete working credentials merely to retry.

#### SSH transfer (optional operator route)

Create a private staging directory on your VM, then transfer only this selected
registration over SSH. Replace `OWNER@VM` with your verified SSH destination;
adjust the local auth path if you configured another one:

```sh
ssh OWNER@VM 'umask 077; mkdir -p "$HOME/.local/share/tel-agent-import"; chmod 0700 "$HOME/.local/share/tel-agent-import"'
scp -p "$HOME/.config/tel-agent/chatgpt/$registration" OWNER@VM:.local/share/tel-agent-import/chatgpt-import.json
```

Do not copy the entire auth directory or the local `host.json`. On the VM, from
the Compose project directory, import the transferred file:

```sh
chmod 0600 "$HOME/.local/share/tel-agent-import/chatgpt-import.json"
docker cp "$HOME/.local/share/tel-agent-import/chatgpt-import.json" "$(docker compose ps -q api):/tmp/chatgpt-import.json"
docker compose exec --user root api sh -c 'chown telagent:telagent /tmp/chatgpt-import.json && chmod 0600 /tmp/chatgpt-import.json'
docker compose exec api python scripts/chatgpt_auth.py import /tmp/chatgpt-import.json
docker compose exec api python scripts/chatgpt_auth.py status
```

The one root command sets ownership of the temporary import file; the application
and import run as `telagent`. Import validates the registration, may fetch official
identity-verification metadata, and preserves the VM's own host ID. It makes no
inference request. After a successful import, remove the temporary copies:

```sh
docker compose exec api rm /tmp/chatgpt-import.json
rm "$HOME/.local/share/tel-agent-import/chatgpt-import.json"
```

Keep any retained local registration protected and inactive. Never distribute
copies of one rotating session to independent servers. OpenAI currently notes
that transferred sessions do not provide host-specific usage attribution or
host-specific revocation; consult the linked VM guide before relying on either.

### 4. Choose the account and model

In Settings, use the ChatGPT plan section to refresh local connection status,
load that account's available models, choose a model, and explicitly select it.
Loading models contacts OpenAI and may refresh a token; it does not generate a
reply. Page load and local status inspection do not initiate OAuth or inference.
Account/model availability errors leave the prior selection intact.

For a standalone environment-configured agent, set `LLM_PROVIDER=chatgpt_plan`
and `LLM_MODEL` to a model slug from that account's catalog. `LLM_API_KEY` is not
used by this provider. The CLI's `select CLIENT_ID` chooses the default registration
for standalone use; dashboard settings can select their own saved registration.
Compose forwards the LLM environment settings, while explicitly saved dashboard
settings take precedence. Restart an environment-configured process after changing
its environment.

Make a deliberate test conversation only when ready to spend the selected
account's allowance. A stored connection or successful model catalog lookup is
not proof that an inference request will succeed or that quota remains.

## Restarts, health and recovery

- Rebuilds and ordinary `docker compose down` preserve the named auth volume.
  `docker compose down -v` removes named volumes, including credentials and the
  stable host identity. Do not use it to upgrade or troubleshoot this setup.
- Changing the Compose project name selects different volumes by default. Keep
  the project identity stable on the same host. A new host needs its own new host
  ID before import; do not clone this volume to create another installation.
- The existing API health check checks service/database health, not ChatGPT
  authorization, quota or model reachability. It never makes paid inference calls.
  `restart: unless-stopped` handles process exits; an unhealthy check alone does
  not trigger a Docker restart. Inspect health and local account status separately.
- The web image checks its local sign-in page without contacting a model provider.
  OAuth connection state is not a deployment readiness requirement.
- If storage is unavailable or unsafe, fix the selected volume's ownership and
  owner-only permissions. A pre-existing/bind-mounted volume is not repaired by
  rebuilding an image. Keep all consumers on the same UID and writable directory.
- If consent or a refresh session expires, sign in locally again with
  `sign-in --client-id ISSUED_CLIENT_ID` (and `--consent` when renewed permission is
  needed), retaining that local registration. Repeat the owner-operated import;
  do not change the server host ID or silently create another billing path.
- `sign-out --client-id ISSUED_CLIENT_ID` revokes and clears a registration's local
  credentials. If the CLI cannot confirm remote revocation, disconnect Tel-Agent
  in ChatGPT Settings as well. This is an explicit owner action, not health-check
  or deployment behavior.
- A failed request or exhausted plan allowance is surfaced to the caller. Wait
  for availability, reauthorize when required, or explicitly choose another
  configured provider. No automatic paid API fallback is enabled.
