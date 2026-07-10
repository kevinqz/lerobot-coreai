# test_lerobot_config_bridge.py — LeRobot-shaped bridge config (v1.1.0).

from lerobot_coreai.lerobot_config import BRIDGE_POLICY_TYPE, CoreAIBridgeConfig
from lerobot_coreai.manifest import LeRobotCoreAIManifest


def test_config_from_manifest(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    cfg = CoreAIBridgeConfig.from_manifest(m)
    assert cfg.policy_type == BRIDGE_POLICY_TYPE == "coreai_bridge"
    assert cfg.robot_type == m.robot_type
    assert cfg.device == "coreai"
    assert cfg.input_features  # observation features mapped
    assert cfg.output_features  # action features mapped


def test_config_never_claims_training_or_registry(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    cfg = CoreAIBridgeConfig.from_manifest(m)
    assert cfg.training_supported is False
    assert cfg.native_registry is False
    d = cfg.to_dict()
    assert d["training_supported"] is False
    assert d["native_registry"] is False
    assert d["policy_type"] == "coreai_bridge"
