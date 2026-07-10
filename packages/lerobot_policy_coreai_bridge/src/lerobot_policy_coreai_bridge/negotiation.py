# negotiation.py — real Runner observation-encoding negotiation (v1.3.4).
#
# v1.3.3 had a safe default but no negotiation. This selects the encoding as the
# intersection of: the config request, what the plugin supports, and what the
# Runner ANNOUNCES in its capabilities. Fail-closed: an unsupported request or an
# empty intersection raises (unless legacy is explicitly allowed). No egress.

from __future__ import annotations

from typing import Any

from lerobot_coreai.errors import CoreAIPolicyError

from .transport import NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1, VALID_ENCODINGS

# Plugin-supported encodings, safest first (auto prefers this order).
PLUGIN_SUPPORTED = (NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1)
MIN_PROTOCOL = "coreai-runner.v2"


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
