# test_failure_taxonomy.py — tests for the failure taxonomy builder (v0.8.2).

import pytest

from lerobot_coreai.failure_taxonomy import build_failure_taxonomy, classify_failure


class TestClassifyFailure:
    def test_runner(self):
        assert classify_failure({"stage": "action.generate", "type": "RunnerTimeoutError"}) == "runner"

    def test_environment(self):
        assert classify_failure({"stage": "simulator.step", "type": "CoreAIPolicyError"}) == "environment"

    def test_validation(self):
        assert classify_failure({"stage": "observation.adapt", "type": "ObservationValidationError"}) == "validation"

    def test_unknown(self):
        assert classify_failure({"stage": "loop", "type": "RuntimeError"}) == "unknown"


class TestBuildFailureTaxonomy:
    def test_empty_errors(self):
        t = build_failure_taxonomy([])
        assert t["total_failures"] == 0
        assert t["by_stage"] == {}
        assert t["first_failure"] is None

    def test_by_stage_and_type(self):
        errors = [
            {"stage": "action.generate", "type": "RunnerTimeoutError", "message": "a", "episode": 0, "step": 5},
            {"stage": "action.generate", "type": "RunnerTimeoutError", "message": "b", "episode": 0, "step": 9},
            {"stage": "simulator.step", "type": "CoreAIPolicyError", "message": "c", "episode": 1, "step": 2},
        ]
        t = build_failure_taxonomy(errors)
        assert t["total_failures"] == 3
        assert t["by_stage"] == {"action.generate": 2, "simulator.step": 1}
        assert t["by_type"] == {"RunnerTimeoutError": 2, "CoreAIPolicyError": 1}

    def test_classified(self):
        errors = [
            {"stage": "action.generate", "type": "RunnerTimeoutError"},
            {"stage": "simulator.step", "type": "CoreAIPolicyError"},
            {"stage": "observation.adapt", "type": "ObservationValidationError"},
        ]
        t = build_failure_taxonomy(errors)
        assert t["classified"]["runner"] == 1
        assert t["classified"]["environment"] == 1
        assert t["classified"]["validation"] == 1

    def test_first_failure(self):
        errors = [
            {"stage": "action.generate", "type": "RunnerTimeoutError", "message": "first", "episode": 0, "step": 12},
            {"stage": "simulator.step", "type": "CoreAIPolicyError", "message": "second", "episode": 1, "step": 3},
        ]
        t = build_failure_taxonomy(errors)
        assert t["first_failure"]["episode"] == 0
        assert t["first_failure"]["step"] == 12
        assert t["first_failure"]["stage"] == "action.generate"
        assert t["first_failure"]["message"] == "first"
