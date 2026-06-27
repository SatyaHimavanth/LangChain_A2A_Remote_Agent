from __future__ import annotations

from typing import Any

import httpx

from ..base import AuthUser, TokenValidator


class OAuthIntrospectionValidator(TokenValidator):
    """RFC 7662-style token introspection validator."""

    def __init__(
        self,
        introspection_url: str,
        client_id: str,
        client_secret: str,
        *,
        timeout: float = 10.0,
        token_field: str = "token",
        auth_method: str = "basic",
    ) -> None:
        self._introspection_url = introspection_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._token_field = token_field
        self._auth_method = auth_method.lower()

    async def validate(self, token: str) -> AuthUser | None:
        data = {self._token_field: token}

        auth = None
        if self._auth_method == "basic":
            auth = httpx.BasicAuth(self._client_id, self._client_secret)
        elif self._auth_method == "post":
            data["client_id"] = self._client_id
            data["client_secret"] = self._client_secret

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._introspection_url,
                    data=data,
                    auth=auth,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
        except Exception:
            return None

        if not payload.get("active"):
            return None

        scopes = payload.get("scope", [])
        if isinstance(scopes, str):
            scopes = [s for s in scopes.split() if s]
        elif not isinstance(scopes, list):
            scopes = []

        user_id = (
            payload.get("sub")
            or payload.get("username")
            or payload.get("client_id")
            or "oauth-user"
        )

        return AuthUser(
            user_id=str(user_id),
            email=payload.get("email"),
            scopes=[str(s) for s in scopes],
            raw_claims=payload,
        )
