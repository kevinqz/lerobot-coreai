# negotiation_algorithm.py — PURE runner-protocol/encoding negotiation (v1.3.20).
#
# Lives in the BASE package (lerobot-free) so the OFFLINE verifier can re-run the
# EXACT algorithm the plugin runtime used and require the persisted NegotiationRecord
# to equal the recomputed result (P1.2). A record that is self-consistent but
# semantically invalid — e.g. negotiated below the minimum, an undeclared downgrade,
# or an unannounced encoding — is rejected here, not merely hashed.

from __future__ import annotations

import re

# Plugin-supported encodings, safest first (mirrors the plugin's PLUGIN_SUPPORTED).
PLUGIN_SUPPORTED_ENCODINGS = ("nested_json_v1", "typed_array_envelope_v1")
_PROTO_RE = re.compile(r"^(?P<family>[a-zA-Z0-9_.\-]+?)\.v(?P<major>\d+)$")


class NegotiationError(ValueError):
    """Raised when a set of negotiation inputs cannot produce a valid result."""


def parse_protocol(version) -> tuple[str, int] | None:
    """Parse ``family.vN`` into ``(family, major)``, or None if malformed."""
    if not isinstance(version, str):
        return None
    m = _PROTO_RE.match(version.strip())
    if not m:
        return None
    return m.group("family"), int(m.group("major"))


def expected_protocol(minimum_protocol: str, runner_protocol,
                      backward_compatible_with) -> str:
    """Re-derive the protocol the runtime MUST have negotiated, fail-closed.

    Mirrors ``negotiate_runner_protocol``: same family, major >= minimum, a newer
    major only with a declared backward-compat with the minimum. The negotiated
    value is the runner's announced protocol (never hardcoded).
    """
    minimum = parse_protocol(minimum_protocol)
    if minimum is None:
        raise NegotiationError(f"invalid minimum protocol {minimum_protocol!r}.")
    announced = parse_protocol(runner_protocol)
    if announced is None:
        raise NegotiationError(f"malformed runner protocol {runner_protocol!r}.")
    fam, major = announced
    min_fam, min_major = minimum
    if fam != min_fam:
        raise NegotiationError(f"family {fam!r} != required {min_fam!r}.")
    if major < min_major:
        raise NegotiationError(
            f"runner protocol {runner_protocol!r} is below minimum {minimum_protocol!r}.")
    bcw = tuple(backward_compatible_with or ())
    if major > min_major and not any(
            parse_protocol(e) == minimum for e in bcw):
        raise NegotiationError(
            f"runner protocol {runner_protocol!r} is newer than {minimum_protocol!r} "
            "without a declared backward compatibility.")
    return runner_protocol


def expected_encoding(requested_encoding, runner_encodings,
                      plugin_supported=PLUGIN_SUPPORTED_ENCODINGS) -> str:
    """Re-derive the encoding the runtime MUST have negotiated, fail-closed."""
    announced = tuple(runner_encodings or ())
    if not announced:
        raise NegotiationError("runner announced no observation encodings.")
    req = requested_encoding
    if req is not None and req != "auto":
        if req not in plugin_supported:
            raise NegotiationError(f"requested encoding {req!r} not plugin-supported.")
        if req not in announced:
            raise NegotiationError(f"requested encoding {req!r} not announced.")
        return req
    for enc in plugin_supported:            # auto: first supported the runner announces
        if enc in announced:
            return enc
    raise NegotiationError("no common encoding between plugin and runner.")


def expected_negotiation(record: dict) -> tuple[str, str]:
    """Re-run negotiation from a NegotiationRecord's inputs. Returns
    ``(protocol, encoding)`` the runtime must have chosen, or raises."""
    proto = expected_protocol(
        record["minimum_protocol"], record["runner_protocol"],
        record.get("runner_backward_compatible_with"))
    enc = expected_encoding(
        record.get("requested_encoding"), record.get("runner_encodings"))
    return proto, enc
