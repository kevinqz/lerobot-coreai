# negotiation.py — real Runner observation-encoding negotiation (v1.3.4).
#
# v1.3.3 had a safe default but no negotiation. This selects the encoding as the
# intersection of: the config request, what the plugin supports, and what the
# Runner ANNOUNCES in its capabilities. Fail-closed: an unsupported request or an
# empty intersection raises (unless legacy is explicitly allowed). No egress.

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

from lerobot_coreai.errors import CoreAIPolicyError

from .transport import NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1, VALID_ENCODINGS

# Plugin-supported encodings, safest first (auto prefers this order).
PLUGIN_SUPPORTED = (NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1)
MIN_PROTOCOL = "coreai-runner.v2"

_PROTO_RE = re.compile(r"^(?P<family>[a-zA-Z0-9_.\-]+?)\.v(?P<major>\d+)$")


@dataclass(frozen=True)
class ProtocolIdentifier:
    """A structured runner-protocol identity: ``family`` + integer ``major``.

    v1.3.5 compared only the trailing ``.vN`` suffix, so ``coreai-runner.v3`` and
    ``malicious-protocol.v3`` looked equivalent, and any higher major was accepted
    blindly. v1.3.6 parses family + major so the family must match and a higher
    major is accepted only with an explicit backward-compat declaration.
    """
    family: str
    major: int

    def __str__(self) -> str:
        return f"{self.family}.v{self.major}"


def parse_protocol(version: Any) -> ProtocolIdentifier | None:
    """Parse ``family.vN`` into a ProtocolIdentifier, or None if malformed."""
    if not isinstance(version, str):
        return None
    m = _PROTO_RE.match(version.strip())
    if not m:
        return None
    return ProtocolIdentifier(family=m.group("family"), major=int(m.group("major")))


@dataclass(frozen=True)
class NegotiatedRunnerProtocol:
    """The concrete protocol the plugin will use for one bound runner.

    Produced by ``negotiate_runner_protocol`` from the runner's announced
    capabilities. ``protocol_version`` is the announced value (never a hardcoded
    constant), so the wire request reflects what was actually negotiated.
    """
    protocol_version: str
    observation_encoding: str
    supports_batch: bool = False
    max_batch_size: int = 1
    legacy: bool = False


def _is_backward_compatible(capabilities: Any, minimum: ProtocolIdentifier) -> bool:
    """True if the runner declares backward-compat with ``minimum``."""
    declared = getattr(capabilities, "backward_compatible_with", ()) or ()
    for entry in declared:
        pid = parse_protocol(entry)
        if pid is not None and pid.family == minimum.family and pid.major == minimum.major:
            return True
    return False


def negotiate_runner_protocol(
    *,
    requested_encoding: str,
    capabilities: Any,
    minimum_protocol: str = MIN_PROTOCOL,
    allow_legacy: bool = False,
) -> NegotiatedRunnerProtocol:
    """Negotiate the full runner protocol, fail-closed (v1.3.6).

    ``capabilities`` is a RunnerCapabilities fetched from a live runner (a
    capabilities/transport failure must be raised by the caller, never turned
    into legacy here). Protocol identity is checked structurally:
      - runner announced no protocol_version    -> error, unless allow_legacy
      - different protocol family               -> error
      - major < minimum major                   -> error
      - major > minimum major without a declared
        backward_compatible_with the minimum    -> error
      - malformed / unknown protocol            -> error
      - encoding not announced / no common      -> error (encoding negotiation)
    The negotiated ``protocol_version`` is the announced one, never hardcoded.
    """
    minimum = parse_protocol(minimum_protocol)
    if minimum is None:
        raise CoreAIPolicyError(
            f"invalid minimum_runner_protocol {minimum_protocol!r}.")

    announced_str = getattr(capabilities, "protocol_version", None)

    if announced_str is None:
        if allow_legacy:
            warnings.warn(
                "runner announced no protocol_version; using legacy protocol "
                f"{minimum_protocol!r} + nested_json_v1. Use runtime_binding_mode="
                "'strict' to fail closed.",
                RuntimeWarning, stacklevel=2)
            enc = negotiate_observation_encoding(
                requested_encoding, capabilities, allow_legacy=True)
            return NegotiatedRunnerProtocol(
                protocol_version=minimum_protocol, observation_encoding=enc,
                supports_batch=bool(getattr(capabilities, "supports_batch", False)),
                max_batch_size=int(getattr(capabilities, "max_batch_size", None) or 1),
                legacy=True)
        raise CoreAIPolicyError(
            "runner announced no protocol_version and legacy is not allowed; "
            "refusing to negotiate (use runtime_binding_mode='legacy' to opt in).")

    announced = parse_protocol(announced_str)
    if announced is None:
        raise CoreAIPolicyError(
            f"runner announced a malformed protocol {announced_str!r}; "
            f"expected {minimum.family!r} family, e.g. {minimum_protocol!r}.")
    if announced.family != minimum.family:
        raise CoreAIPolicyError(
            f"runner protocol family {announced.family!r} != required "
            f"{minimum.family!r}; refusing to bind.")
    if announced.major < minimum.major:
        raise CoreAIPolicyError(
            f"runner protocol {announced_str!r} is below the minimum "
            f"{minimum_protocol!r}; refusing to bind.")
    if announced.major > minimum.major and not _is_backward_compatible(
            capabilities, minimum):
        raise CoreAIPolicyError(
            f"runner protocol {announced_str!r} is newer than {minimum_protocol!r} "
            f"and does not declare backward_compatible_with {minimum_protocol!r}; "
            "a newer major may be breaking — refusing to bind.")

    enc = negotiate_observation_encoding(
        requested_encoding, capabilities, allow_legacy=False)
    return NegotiatedRunnerProtocol(
        protocol_version=announced_str, observation_encoding=enc,
        supports_batch=bool(getattr(capabilities, "supports_batch", False)),
        max_batch_size=int(getattr(capabilities, "max_batch_size", None) or 1),
        legacy=False)


def negotiate_observation_encoding(
    requested: str, capabilities: Any, *, allow_legacy: bool = False,
) -> str:
    """Return the negotiated encoding, or fail closed.

    ``capabilities`` is a RunnerCapabilities (or None). Rules:
      - requested (non-auto) not plugin-supported            -> fail
      - runner announces encodings + requested not announced -> fail
      - requested announced                                  -> use it
      - auto                                                 -> first plugin
        encoding the runner announces
      - runner announces nothing + allow_legacy              -> nested_json + warn
      - runner announces nothing + not allow_legacy          -> fail
    """
    if requested != "auto" and requested not in VALID_ENCODINGS:
        raise CoreAIPolicyError(f"unknown observation encoding {requested!r}.")

    announced = tuple(getattr(capabilities, "observation_encodings", ()) or ())

    if not announced:
        if allow_legacy:
            import warnings
            warnings.warn(
                "runner announced no observation_encodings; falling back to "
                "nested_json_v1 (legacy). Set allow_legacy=False to fail closed.",
                RuntimeWarning, stacklevel=2)
            return NESTED_JSON_V1
        raise CoreAIPolicyError(
            "runner announced no observation_encodings and legacy is not allowed; "
            "refusing to guess the wire format.")

    if requested == "auto":
        for enc in PLUGIN_SUPPORTED:
            if enc in announced:
                return enc
        raise CoreAIPolicyError(
            f"no common observation encoding between plugin {PLUGIN_SUPPORTED} "
            f"and runner {announced}.")

    if requested not in announced:
        raise CoreAIPolicyError(
            f"requested encoding {requested!r} not announced by runner {announced}.")
    return requested
