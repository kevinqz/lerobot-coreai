# test_runtime_provider.py — abstract runtime-provider identity (RFC-0700 §13).

import pytest

from lerobot_coreai.runtime_provider import (
    RUNTIME_PROVIDERS, available_providers, get_provider, require_available,
)
from lerobot_coreai.stages import RUNTIME_BACKENDS


def test_every_backend_has_exactly_one_provider():
    assert set(RUNTIME_PROVIDERS) == set(RUNTIME_BACKENDS)


def test_only_coreai_is_implemented_today():
    assert available_providers() == ("coreai",)
    assert get_provider("coreai").is_available
    assert get_provider("pytorch_reference").status == "reference"
    assert get_provider("mlx").status == "deferred"


def test_deferred_and_reference_providers_fail_closed():
    # a reserved identity must NOT be routable as if it were a real deployment target.
    require_available("coreai")                     # ok
    with pytest.raises(NotImplementedError):
        require_available("mlx")
    with pytest.raises(NotImplementedError):
        require_available("pytorch_reference")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        get_provider("tensorflow")
