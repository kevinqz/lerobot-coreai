# test_metrics.py — tests for action comparison metrics.

import math
import pytest

from lerobot_coreai.metrics import (
    action_to_flat_float_list,
    infer_shape,
    mean_absolute_error,
    max_absolute_error,
    cosine_similarity,
    relative_mae,
)
from lerobot_coreai.errors import ActionParityError


class FakeTensor:
    def __init__(self, data):
        self._data = data
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self._data


class TestFlatten:
    def test_nested_list(self):
        assert action_to_flat_float_list([[1.0, 2.0], [3.0]]) == [1.0, 2.0, 3.0]

    def test_flat_list(self):
        assert action_to_flat_float_list([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_scalar(self):
        assert action_to_flat_float_list(42.0) == [42.0]

    def test_fake_tensor(self):
        assert action_to_flat_float_list(FakeTensor([1.0, 2.0])) == [1.0, 2.0]


class TestShape:
    def test_nested_shape(self):
        assert infer_shape([[0.0] * 7] * 16) == [16, 7]

    def test_ragged_returns_none(self):
        assert infer_shape([[0.0] * 7, [0.0] * 3]) is None

    def test_scalar_returns_none(self):
        assert infer_shape(42.0) is None


class TestMetrics:
    def test_cosine_identical(self):
        a = [[0.1, 0.2, 0.3]]
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-10)

    def test_mae_zero_identical(self):
        a = [[0.1, 0.2, 0.3]]
        assert mean_absolute_error(a, a) == pytest.approx(0.0)

    def test_max_mae(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.5]
        assert max_absolute_error(a, b) == pytest.approx(0.5)

    def test_relative_mae(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0]
        assert relative_mae(a, b) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ActionParityError, match="shapes differ"):
            cosine_similarity([[0.0] * 7], [[0.0] * 6])

    def test_different_sizes_raises(self):
        with pytest.raises(ActionParityError):
            mean_absolute_error([1.0, 2.0], [1.0])

    def test_cosine_near_zero_actions(self):
        """Both actions near zero → cosine should be 1.0 (not ill-conditioned)."""
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_fake_tensor_metrics(self):
        """Metrics should work with fake tensors."""
        a = FakeTensor([1.0, 2.0, 3.0])
        b = FakeTensor([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)
        assert mean_absolute_error(a, b) == pytest.approx(0.0)
