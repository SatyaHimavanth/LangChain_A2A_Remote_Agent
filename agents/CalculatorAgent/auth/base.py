from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class AuthUser:
    user_id: str
    email: str | None = None
    scopes: list[str] = field(default_factory=list)
    raw_claims: dict = field(default_factory=dict)


class TokenValidator(ABC):
    """Swap implementations without touching agent or executor logic."""

    @abstractmethod
    async def validate(self, token: str) -> AuthUser | None:
        """Return AuthUser if valid, None if not. Never raise."""
        raise NotImplementedError
