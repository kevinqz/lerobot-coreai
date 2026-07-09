# test_sim_package_integration.py — end-to-end sim --package-run (v0.8.4).

import json
from unittest.mock import MagicMock, patch

from lerobot_coreai.sim import SimConfig, run_sim_mode


def _make_mock_policy(manifest_dict):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    mock = MagicMock()
    mock.predict_action.return_value = {
        "action": [[0.0] * 7] * 16, "metadata": {"timing": {"total_ms": 12.3}},
    }
    mock.manifest = LeRobotCoreAIManifest.from_dict(manifest_dict)
    mock.policy_type = "evo1"
    mock.robot_type = "so100"
    mock.parity_passed = True
    mock.policy_repo_id = "test/policy"
    return mock


class TestSimPackageIntegration:
    def test_package_run_creates_bundle(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True, package_output_dir=bundle_dir,
            )
            result = run_sim_mode(config)
        assert result.ok
        assert (bundle_dir / "bundle_manifest.json").is_file()
        assert (bundle_dir / "checksums.json").is_file()
        assert (bundle_dir / "source_run" / "sim_report.json").is_file()

    def test_report_includes_bundle_section(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True, package_output_dir=bundle_dir,
            )
            result = run_sim_mode(config)
        # The on-disk report must record the bundle.
        report = json.loads((output_dir / "sim_report.json").read_text())
        assert report["bundle"]["created"] is True
        assert report["bundle"]["output_dir"] == str(bundle_dir)

    def test_default_bundle_dir_is_output_dir_bundle(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True,  # no explicit package_output_dir
            )
            run_sim_mode(config)
        assert (output_dir / "bundle" / "bundle_manifest.json").is_file()

    def test_no_package_run_skips_bundle(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
            )
            result = run_sim_mode(config)
        assert "bundle" not in result.report
        assert not (output_dir / "bundle").exists()

    def test_observations_dir_excluded_by_default(self, tmp_path, valid_manifest_dict):
        output_dir = tmp_path / "run"
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True, package_output_dir=bundle_dir,
            )
            run_sim_mode(config)
        assert not (bundle_dir / "source_run" / "observations").exists()

    def test_packaged_trace_contains_sim_completed(self, tmp_path, valid_manifest_dict):
        # The bundled trace must represent the finalized run (sim.completed
        # written and trace closed before packaging copies it).
        output_dir = tmp_path / "run"
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True, package_output_dir=bundle_dir,
            )
            run_sim_mode(config)
        bundled_trace = bundle_dir / "source_run" / "sim_trace.jsonl"
        assert bundled_trace.is_file()
        assert "sim.completed" in bundled_trace.read_text()

    def test_packaged_bundle_verifies(self, tmp_path, valid_manifest_dict):
        # End-to-end: a bundle produced by sim --package-run passes verification.
        from lerobot_coreai.sim_bundle import verify_sim_bundle
        output_dir = tmp_path / "run"
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True, package_output_dir=bundle_dir,
            )
            run_sim_mode(config)
        result = verify_sim_bundle(bundle_dir)
        assert result.ok, result.invariant_failures + result.checksum_failures

    def test_package_failure_does_not_fail_sim(self, tmp_path, valid_manifest_dict):
        # If packaging raises, the sim result stays ok and a warning is recorded.
        output_dir = tmp_path / "run"
        bundle_dir = tmp_path / "bundle"
        mock_policy = _make_mock_policy(valid_manifest_dict)
        with patch("lerobot_coreai.sim.CoreAIPolicy.from_pretrained", return_value=mock_policy), \
             patch("lerobot_coreai.sim.package_sim_run", side_effect=RuntimeError("boom")):
            config = SimConfig(
                policy_path="test/p", output_dir=output_dir, env_type="fake",
                confirm_sim_egress=True, episodes=1, max_steps_per_episode=2, fps=0,
                package_run=True, package_output_dir=bundle_dir,
            )
            result = run_sim_mode(config)
        assert result.ok  # sim itself succeeded
        report = json.loads((output_dir / "sim_report.json").read_text())
        assert report["bundle"]["created"] is False
        assert any("bundle packaging failed" in w for w in report.get("warnings", []))
