from __future__ import annotations

import asyncio
from functools import partial

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token

from ..base import AuthUser, TokenValidator


class GoogleIDTokenValidator(TokenValidator):
    def __init__(self, audience: str) -> None:
        self._audience = audience

    async def validate(self, token: str) -> AuthUser | None:
        try:
            loop = asyncio.get_running_loop()
            verify = partial(
                id_token.verify_oauth2_token,
                token,
                GoogleRequest(),
                self._audience,
            )
            idinfo = await loop.run_in_executor(None, verify)
            scopes = idinfo.get("scope", "")
            if isinstance(scopes, str):
                scopes = [s for s in scopes.split() if s]
            elif not isinstance(scopes, list):
                scopes = []

            return AuthUser(
                user_id=idinfo["sub"],
                email=idinfo.get("email"),
                scopes=scopes,
                raw_claims=idinfo,
            )
        except Exception:
            return None
