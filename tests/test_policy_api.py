# test_policy_api.py — tests for the CoreAIPolicy class.

import pytest

from lerobot_coreai.policy import CoreAIPolicy
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.config import CoreAIRuntimeConfig
from lerobot_coreai.errors import ManifestError


class TestCoreAIPolicyFromManifest:
    def _make_policy(self, valid_manifest_dict) -> CoreAIPolicy:
        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        return CoreAIPolicy(manifest)

    def test_policy_metadata(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)

        assert p.repo_id == "kevinqz/EVO1-SO100-CoreAI"
        assert p.policy_type == "evo1"
        assert p.robot_type == "so100"
        assert p.parity_passed is True

    def test_policy_repr(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        r = repr(p)
        assert "CoreAIPolicy" in r
        assert "EVO1-SO100-CoreAI" in r
        assert "evo1" in r
        assert "so100" in r

    def test_policy_config(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        cfg = p.config

        assert cfg.type == "coreai"
        assert cfg.path == "kevinqz/EVO1-SO100-CoreAI"
        assert cfg.policy_type == "evo1"
        assert cfg.robot_type == "so100"
        assert "observation.images.wrist" in cfg.observation_features
        assert "action" in cfg.action_features

    def test_policy_eval(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        assert p.eval() is p

    def test_policy_train_raises(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        with pytest.raises(NotImplementedError, match="inference"):
            p.train()

    def test_policy_to_coreai(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        assert p.to("coreai") is p
        assert p.to("auto") is p

    def test_policy_to_invalid_device(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        with pytest.raises(ValueError, match="device"):
            p.to("cuda")

    def test_policy_reset(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        p.reset()  # should not raise

    def test_policy_select_action_no_runner(self, valid_manifest_dict):
        """v0.2: select_action raises RunnerNotReachableError when no runner is configured."""
        from lerobot_coreai.errors import RunnerNotReachableError
        p = self._make_policy(valid_manifest_dict)
        with pytest.raises(RunnerNotReachableError):
            p.select_action({})

    def test_policy_manifest_accessor(self, valid_manifest_dict):
        p = self._make_policy(valid_manifest_dict)
        assert p.manifest.policy_type == "evo1"
        assert p.manifest.robot_type == "so100"


class TestCoreAIPolicyRuntimeConfig:
    def test_default_runtime_config(self, valid_manifest_dict):
        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        p = CoreAIPolicy(manifest)

        assert p.config.runtime.type == "coreai"
        assert p.config.runtime.mode == "dry_run"
        assert p.config.runtime.runner_url == "unix:///tmp/coreai-runner.sock"

    def test_custom_runtime_config(self, valid_manifest_dict):
        manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
        runtime = CoreAIRuntimeConfig(
            runner_url="http://mac-studio.local:8710",
            mode="shadow",
        )
        p = CoreAIPolicy(manifest, runtime=runtime)

        assert p.config.runtime.runner_url == "http://mac-studio.local:8710"
        assert p.config.runtime.mode == "shadow"
