# test_from_pretrained.py — tests for CoreAIPolicy.from_pretrained with network mocked.

import pytest

from lerobot_coreai.policy import CoreAIPolicy
from lerobot_coreai.manifest import LeRobotCoreAIManifest


class TestFromPretrained:
    def test_from_pretrained_builds_policy(self, monkeypatch, valid_manifest_dict):
        """from_pretrained should download the manifest and build a CoreAIPolicy."""
        monkeypatch.setattr(
            "lerobot_coreai.policy.load_manifest",
            lambda repo_id, revision="main": LeRobotCoreAIManifest.from_dict(valid_manifest_dict),
        )
        policy = CoreAIPolicy.from_pretrained("kevinqz/EVO1-SO100-CoreAI")

        assert policy.repo_id == "kevinqz/EVO1-SO100-CoreAI"
        assert policy.policy_type == "evo1"
        assert policy.robot_type == "so100"
        assert policy.parity_passed is True

    def test_from_pretrained_custom_runner_url(self, monkeypatch, valid_manifest_dict):
        """from_pretrained should accept a custom runner_url."""
        monkeypatch.setattr(
            "lerobot_coreai.policy.load_manifest",
            lambda repo_id, revision="main": LeRobotCoreAIManifest.from_dict(valid_manifest_dict),
        )
        policy = CoreAIPolicy.from_pretrained(
            "kevinqz/EVO1-SO100-CoreAI",
            runner_url="http://mac-studio.local:8710",
        )
        assert policy.config.runtime.runner_url == "http://mac-studio.local:8710"

    def test_from_pretrained_select_action_raises(self, monkeypatch, valid_manifest_dict):
        """select_action should raise NotImplementedError in v0.1."""
        monkeypatch.setattr(
            "lerobot_coreai.policy.load_manifest",
            lambda repo_id, revision="main": LeRobotCoreAIManifest.from_dict(valid_manifest_dict),
        )
        policy = CoreAIPolicy.from_pretrained("kevinqz/EVO1-SO100-CoreAI")
        with pytest.raises(NotImplementedError, match="v0.2"):
            policy.select_action({})

    def test_from_pretrained_manifest_accessor(self, monkeypatch, valid_manifest_dict):
        """The .manifest property should expose the parsed manifest."""
        monkeypatch.setattr(
            "lerobot_coreai.policy.load_manifest",
            lambda repo_id, revision="main": LeRobotCoreAIManifest.from_dict(valid_manifest_dict),
        )
        policy = CoreAIPolicy.from_pretrained("kevinqz/EVO1-SO100-CoreAI")
        assert policy.manifest.policy_type == "evo1"
        assert policy.manifest.framework_version == "0.6.0"
