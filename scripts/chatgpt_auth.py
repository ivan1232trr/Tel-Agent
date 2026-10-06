"""Owner-operated ChatGPT setup; OAuth runs only on explicit `sign-in`.

Run `python scripts/chatgpt_auth.py --help`. Never paste credential files, token
values or callback URLs into chat, support transcripts, or the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.chatgpt_auth import ChatGPTAuthError, ChatGPTAuthStore
from agent.config import chatgpt_auth_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up ChatGPT plan usage on this runtime.")
    parser.add_argument(
        "--auth-dir",
        type=Path,
        default=chatgpt_auth_directory(),
        help="Protected runtime directory (LLM_CHATGPT_AUTH_DIR).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser(
        "sign-in", help="Continue with ChatGPT in this computer's browser."
    )
    login.add_argument("--client-id", help="Reauthorize this existing registration.")
    login.add_argument("--label", help="A recognizable label; registrations remain separate.")
    login.add_argument(
        "--port", type=int, default=0, help="Loopback port; default selects a free port."
    )
    login.add_argument("--timeout", type=int, default=300)
    login.add_argument(
        "--consent", action="store_true", help="Explicitly request plan-use consent again."
    )
    commands.add_parser("list", help="List account metadata without any tokens.")
    commands.add_parser(
        "status", help="Inspect local connection state without network requests."
    )
    commands.add_parser(
        "init-host", help="Create this VM's stable host ID before importing credentials."
    )
    select = commands.add_parser("select", help="Use an already authorized registration.")
    select.add_argument("client_id")
    logout = commands.add_parser("sign-out", help="Revoke and clear one registration's tokens.")
    logout.add_argument("--client-id")
    importing = commands.add_parser(
        "import", help="Import one owner-transferred 0600 credential file."
    )
    importing.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    store = ChatGPTAuthStore(args.auth_dir)
    try:
        result: object
        if args.command == "sign-in":
            print(
                "Continue with ChatGPT in the system browser. Waiting for the local callback."
            )
            result = store.sign_in(
                args.client_id,
                label=args.label,
                port=args.port,
                timeout=args.timeout,
                consent=args.consent,
            )
        elif args.command == "list":
            result = store.accounts()
        elif args.command == "select":
            result = store.select(args.client_id)
        elif args.command == "init-host":
            result = {"ext_agent_host_id": store.init_host()}
        elif args.command == "sign-out":
            result = store.sign_out(args.client_id)
        elif args.command == "import":
            result = store.import_registration(args.file)
        else:
            result = store.status()
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    except ChatGPTAuthError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ChatGPT setup cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
