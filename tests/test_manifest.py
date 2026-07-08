# test_manifest.py — tests for lerobot-coreai.json parsing and validation.

import json
import copy

import pytest

from lerobot_coreai.manifest import LeRobotCoreAIManifest, load_manifest
from lerobot_coreai.errors import ManifestError


class TestManifestParsing:
    def test_valid_manifest_parses(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert m.schema_version == "lerobot-coreai.v0"
        assert m.runtime == "coreai"
        assert m.framework_name == "lerobot"
        assert m.framework_version == "0.6.0"
        assert m.policy_repo_id == "kevinqz/EVO1-SO100-CoreAI"
        assert m.policy_source_repo_id == "lerobot/evo1_so100"
        assert m.policy_type == "evo1"
        assert m.robot_type == "so100"
        assert m.robot_fps == 30

    def test_observation_features_parsed(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert "observation.images.wrist" in m.observation_features
        assert m.observation_features["observation.images.wrist"].dtype == "image"
        assert m.observation_features["observation.images.wrist"].shape == [3, 224, 224]
        assert m.observation_features["observation.images.wrist"].required is True

        assert "observation.state" in m.observation_features
        assert m.observation_features["observation.state"].dtype == "float32"
        assert m.observation_features["observation.state"].shape == [7]

        assert "task" in m.observation_features
        assert m.observation_features["task"].required is False

    def test_action_features_parsed(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert "action" in m.action_features
        assert m.action_features["action"].dtype == "float32"
        assert m.action_features["action"].shape == [16, 7]

    def test_graphs_parsed(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert len(m.graphs) == 1
        assert m.graphs[0].name == "action_denoise_step"
        assert m.graphs[0].role == "denoise_step"

    def test_host_loop_parsed(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert m.host_loop_required is True
        assert m.host_loop_type == "flow_matching"
        assert m.host_loop_solver == "euler"
        assert m.host_loop_num_steps == 10

    def test_evaluation_parsed(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert m.evaluation_metric == "action_parity"
        assert m.evaluation_status == "passed"
        assert m.evaluation_n_obs == 8
        assert m.evaluation_min_chunk_cosine is not None
        assert m.parity_passed is True
        assert m.proves_numeric_fidelity is True
        assert m.proves_task_success is False
        assert m.proves_robot_safety is False

    def test_safety_parsed(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

        assert m.default_mode == "dry_run"
        assert m.real_actuation_requires_confirmation is True

    def test_raw_preserved(self, valid_manifest_dict):
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        assert m.raw == valid_manifest_dict


class TestManifestValidation:
    def test_missing_required_field_raises(self, valid_manifest_dict):
        del valid_manifest_dict["runtime"]
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

    def test_wrong_runtime_raises(self, valid_manifest_dict):
        valid_manifest_dict["runtime"] = "pytorch"
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

    def test_wrong_framework_raises(self, valid_manifest_dict):
        valid_manifest_dict["framework"]["name"] = "pytorch"
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

    def test_missing_observation_features_raises(self, valid_manifest_dict):
        del valid_manifest_dict["features"]["observation"]
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

    def test_additional_property_rejected(self, valid_manifest_dict):
        valid_manifest_dict["unknown_field"] = "x"
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

    def test_bad_schema_version_raises(self, valid_manifest_dict):
        valid_manifest_dict["schema_version"] = "wrong-format"
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)

    def test_bad_safety_mode_raises(self, valid_manifest_dict):
        valid_manifest_dict["safety"]["default_mode"] = "dangerous"
        with pytest.raises(ManifestError, match="schema validation"):
            LeRobotCoreAIManifest.from_dict(valid_manifest_dict)


class TestManifestOptional:
    def test_no_host_loop(self, valid_manifest_dict):
        """A manifest without host_loop should still parse."""
        del valid_manifest_dict["coreai"]["host_loop"]
        valid_manifest_dict["coreai"]["host_loop_required"] = False
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        assert m.host_loop_required is False
        assert m.host_loop_type is None

    def test_no_graphs(self, valid_manifest_dict):
        """A manifest without graphs should still parse."""
        del valid_manifest_dict["coreai"]["graphs"]
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        assert m.graphs == []

    def test_parity_not_run(self, valid_manifest_dict):
        """A manifest with evaluation status 'not_run' should parse."""
        valid_manifest_dict["evaluation"]["status"] = "not_run"
        m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        assert m.parity_passed is False
