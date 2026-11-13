"""Read current authorized CLI access credentials without owning token renewal.

This file is also an isolated stdlib-only command. Its stdout is credential
material for the native provider's captured key_cmd pipe, never a diagnostic.
It does not create children, contact a service, or write an authentication store.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_STORE_BYTES = 65_536
_TOKEN_CHARS = 16_384
_MIN_VALIDITY_SECONDS = 60
_JWT = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


_CATEGORIES = frozenset({
    "credential_invalid", "credential_unavailable", "account_mismatch", "credential_expired",
})
# Any other value is a programming error; it fails closed to this one fixed token.
_UNLISTED = "unlisted"


class AccessCredentialError(RuntimeError):
    """Fixed category only; never retain a credential, path or native exception.

    The category is checked against the closed set when the error is made, so free text
    passed here can never reach the refusal line.
    """

    def __init__(self, category: object) -> None:
        self.category = _UNLISTED
        if type(category) is str and category in _CATEGORIES:
            self.category = category
        super().__init__(self.category)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate_fields")
    return result


def read_access_credential(
    auth_file: Path, account_sha256: str, *, now: float | None = None,
) -> dict[str, Any]:
    """Return an access token for the pinned account, with a short freshness TTL.

    Native Hermes imposes a five-second minimum command cache. Advertising one
    second keeps that bound instead of retaining a copied login until JWT expiry.
    The authorized CLI owner remains responsible for updating its own cache.
    """
    if (type(account_sha256) is not str or _FINGERPRINT.fullmatch(account_sha256) is None):
        raise AccessCredentialError("credential_invalid")
    try:
        if (not auth_file.is_absolute() or auth_file.is_symlink()
                or getattr(auth_file.lstat(), "st_file_attributes", 0) & 0x400
                or not auth_file.is_file()):
            raise AccessCredentialError("credential_unavailable")
        with auth_file.open("rb") as stream:
            raw = stream.read(_STORE_BYTES + 1)
    except AccessCredentialError:
        raise
    except OSError:
        raise AccessCredentialError("credential_unavailable") from None
    try:
        if len(raw) > _STORE_BYTES:
            raise ValueError("store_bound")
        store = json.loads(raw, object_pairs_hook=_object)
        tokens = store["tokens"]
        token = tokens["access_token"]
        if type(token) is not str or len(token) > _TOKEN_CHARS or _JWT.fullmatch(token) is None:
            raise ValueError("token_shape")
        encoded = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)),
                             object_pairs_hook=_object)
        account = payload["https://api.openai.com/auth"]["chatgpt_account_id"]
        expiry = payload["exp"]
        if (type(account) is not str or not account or len(account) > 512
                or type(expiry) not in (int, float) or not math.isfinite(expiry)):
            raise ValueError("claim_shape")
        account_fingerprint = hashlib.sha256(account.encode("utf-8")).hexdigest()
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise AccessCredentialError("credential_invalid") from None
    if account_fingerprint != account_sha256:
        raise AccessCredentialError("account_mismatch")
    timestamp = time.time() if now is None else now
    if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
            or expiry - timestamp <= _MIN_VALIDITY_SECONDS):
        raise AccessCredentialError("credential_expired")
    return {"access_token": token, "expires_in": 1}


def command_argv(
    executable: Path, helper: Path, auth_file: Path, account_sha256: str,
) -> list[str]:
    if (not all(path.is_absolute() for path in (executable, helper, auth_file))
            or _FINGERPRINT.fullmatch(account_sha256) is None):
        raise ValueError("credential_command")
    return [str(executable), "-I", "-B", "-S", str(helper), "--auth-file", str(auth_file),
            "--account-sha256", account_sha256]


def shell_command(argv: list[str], *, windows: bool | None = None) -> str:
    """Encode fixed trusted arguments for Hermes's existing shell-string hook."""
    if (type(argv) is not list or not 1 <= len(argv) <= 16
            or any(type(item) is not str or not 1 <= len(item) <= 2048
                   or "\x00" in item or "\n" in item or "\r" in item for item in argv)):
        raise ValueError("credential_command")
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        if any(re.search(r'["%!^&|<>]', item) for item in argv):
            raise ValueError("credential_command")
        return " ".join('"' + item + '"' for item in argv)
    import shlex

    return shlex.join(argv)


def validate_native_binding(config: dict[str, Any], executable: Path, helper: Path) -> None:
    """Bind the selected native provider and presenter to one reviewed resolver.

    Only the explicit live-Codex profile opts into this contract. Static provider
    profiles keep their existing configuration and never call this validator.
    """
    try:
        model = config["model"]
        selected = model["provider"]
        if type(selected) is not str or not selected.startswith("custom:"):
            raise ValueError("provider_selection")
        name = selected[len("custom:"):]
        if (re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) is None
                or set(config["providers"]) != {name}):
            raise ValueError("provider_selection")
        native = config["providers"][name]
        presentation = config["plugins"]["entries"]["hermes-realtime"]["settings"][
            "presentation"]
        argv = presentation["key_cmd_argv"]
        if (type(argv) is not list or len(argv) != 9
                or any(type(item) is not str for item in argv)
                or argv[1:4] != ["-I", "-B", "-S"] or argv[5] != "--auth-file"
                or argv[7] != "--account-sha256"
                or _FINGERPRINT.fullmatch(argv[8]) is None
                or not Path(argv[6]).is_absolute()):
            raise ValueError("resolver_arguments")
        if (Path(argv[0]).resolve() != executable.resolve()
                or Path(argv[4]).resolve() != helper.resolve()):
            raise ValueError("resolver_origin")
        if ("api_key" in model
                or set(native) != {"base_url", "default_model", "api_mode", "key_cmd",
                                       "extra_headers"}
                or "api_key_env" in presentation):
            raise ValueError("static_override")
        headers = native["extra_headers"]
        if type(headers) is not dict or set(headers) != {"ChatGPT-Account-ID"}:
            raise ValueError("account_header")
        account = headers["ChatGPT-Account-ID"]
        if (type(account) is not str or not 1 <= len(account) <= 512 or not account.isascii()
                or any(ord(character) < 33 or ord(character) > 126 for character in account)
                or hashlib.sha256(account.encode("ascii")).hexdigest() != argv[8]):
            raise ValueError("account_header")
        base_url = native["base_url"]
        if (type(base_url) is not str
                or base_url != "https://chatgpt.com/backend-api/codex"
                or model.get("base_url", base_url) != base_url
                or model.get("api_mode", "codex_responses") != "codex_responses"
                or type(model["default"]) is not str or not model["default"]
                or presentation["base_url"] != base_url
                or presentation["provider"] != "custom"
                or native["api_mode"] != "codex_responses"
                or presentation.get("api_mode") != "codex_responses"
                or native["default_model"] != model["default"]
                or presentation["model"] != model["default"]):
            raise ValueError("resolver_route")
        if native["key_cmd"] != shell_command(argv):
            raise ValueError("resolver_command")
    except (KeyError, TypeError, ValueError, OSError):
        raise ValueError("credential_binding") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a current same-account access token.")
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--account-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        credential = read_access_credential(args.auth_file, args.account_sha256)
    except AccessCredentialError as error:
        print('[codex-access-credential] ' + json.dumps({"version": 1, "category": error.category}),
              file=sys.stderr, flush=True)
        return 1
    print(json.dumps(credential, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
