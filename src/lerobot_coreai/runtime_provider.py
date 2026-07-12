# runtime_provider.py — abstract runtime-provider identity (RFC-0700 §13).
#
# "Add an abstract provider identity now, but defer implementation." The certification
# contracts are already backend-neutral (see stages.RUNTIME_BACKENDS +
# RuntimeProviderStage, v1.3.24a); this formalizes the PROVIDERS themselves with honest
# implementation STATUS, so MLX / PyTorch-reference plug in later without a migration and
# without ever implying they exist today. Common evidence compares every provider
# against ONE LeRobot semantic contract (RFC-0700 §13). Pure Python; no torch/lerobot.

from __future__ import annotations

from dataclasses import dataclass

from .stages import RUNTIME_BACKENDS

# provider status is deliberately honest — see RFC-0700 §12/§13 (no premature MLX port).
#   implemented : real code path exists in this repo today
#   reference   : an oracle the others are compared against (upstream LeRobot/PyTorch)
#   deferred    : an identity reserved now; implementation is future work, not present
_STATUSES = ("implemented", "reference", "deferred")


@dataclass(frozen=True)
class RuntimeProvider:
    backend: str            # one of stages.RUNTIME_BACKENDS
    status: str             # implemented | reference | deferred
    title: str
    description: str

    @property
    def is_available(self) -> bool:
        return self.status == "implemented"

    def to_dict(self) -> dict:
        return {"backend": self.backend, "status": self.status, "title": self.title,
                "description": self.description, "available": self.is_available}


RUNTIME_PROVIDERS = {
    "coreai": RuntimeProvider(
        "coreai", "implemented", "Apple Core AI deployment",
        "Deploy/execute LeRobot policies as Apple Core AI `.aimodel` artifacts through "
        "the CoreAI Runner — the only provider implemented in this repo today."),
    "pytorch_reference": RuntimeProvider(
        "pytorch_reference", "reference", "PyTorch reference oracle",
        "Upstream LeRobot/PyTorch execution used as the numeric-parity oracle every "
        "other provider is compared against; not a deployment target."),
    "mlx": RuntimeProvider(
        "mlx", "deferred", "MLX dynamic/training provider",
        "Reserved identity only (RFC-0700 §13). No implementation and no premature port; "
        "existing community LeRobot-MLX work is conformance-tested before any duplicate."),
}
# every declared backend has exactly one provider identity (no gaps, no extras).
assert set(RUNTIME_PROVIDERS) == set(RUNTIME_BACKENDS)


def get_provider(backend: str) -> RuntimeProvider:
    if backend not in RUNTIME_PROVIDERS:
        raise ValueError(f"unknown runtime provider {backend!r} "
                         f"(known: {sorted(RUNTIME_PROVIDERS)})")
    return RUNTIME_PROVIDERS[backend]


def available_providers() -> tuple:
    """Backends with a real implemented code path today (only ``coreai``)."""
    return tuple(b for b, p in RUNTIME_PROVIDERS.items() if p.is_available)


def require_available(backend: str) -> RuntimeProvider:
    """Fail closed: refuse to route to a reserved/deferred provider as if it were real."""
    p = get_provider(backend)
    if not p.is_available:
        raise NotImplementedError(
            f"runtime provider {backend!r} is {p.status}, not implemented — "
            "it is a reserved identity (RFC-0700 §13), not an available deployment target")
    return p
