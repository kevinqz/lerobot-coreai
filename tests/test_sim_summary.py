# test_sim_summary.py — tests for the markdown summary builder (v0.8.2).

from lerobot_coreai.sim_summary import build_sim_summary_markdown


def _report(**overrides):
    base = {
        "policy": {"path": "test/policy", "runtime": "coreai"},
        "environment": {"type": "fake"},
        "mode": "sim",
        "lerobot_coreai_version": "0.8.2",
        "metrics": {"episodes_completed": 3, "episodes_requested": 3, "steps_completed": 30,
                    "actions_generated": 30, "actions_sent_to_simulator": 30},
        "episode_metrics": {"success_rate": 0.8, "mean_reward": 42.0, "median_reward": 40.5},
        "latency_metrics": {"runner_p50_ms": 12.0, "runner_p95_ms": 15.2,
                            "env_step_p95_ms": 8.1, "loop_p95_ms": 25.0},
        "action_metrics": {"nan_action_steps": 0, "inf_action_steps": 0, "shape_changes": 0},
        "failure_metrics": {"total_errors": 0, "runner_errors": 0,
                            "env_errors": 0, "validation_errors": 0},
        "safety": {"actions_sent_to_robot": 0, "action_egress": "simulator_only",
                   "simulator_egress_enabled": True, "robot_egress_enabled": False,
                   "physical_actuation_possible": False, "motor_commands_available": False,
                   "robot_connected": False},
        "claims": {"proves_sim_task_success": True, "proves_real_task_success": False,
                   "proves_robot_safety": False, "proves_real_world_safety": False},
        "files": {"report": "sim_report.json", "summary": "sim_summary.md"},
    }
    base.update(overrides)
    return base


class TestSimSummary:
    def test_contains_safety_block(self):
        md = build_sim_summary_markdown(_report())
        assert "## Safety" in md
        assert "Robot egress enabled: False" in md

    def test_real_claims_false(self):
        md = build_sim_summary_markdown(_report())
        assert "Proves real task success: False" in md
        assert "Proves robot safety: False" in md
        assert "Proves real-world safety: False" in md

    def test_sim_success_shown(self):
        md = build_sim_summary_markdown(_report())
        assert "Proves sim task success: True" in md

    def test_null_metrics_render_na(self):
        report = _report(episode_metrics={"success_rate": None, "mean_reward": None, "median_reward": None})
        md = build_sim_summary_markdown(report)
        assert "Success rate: n/a" in md
        assert "Mean reward: n/a" in md

    def test_contains_files_section(self):
        md = build_sim_summary_markdown(_report())
        assert "## Files" in md
        assert "- sim_report.json" in md

    def test_never_claims_real_world_success(self):
        md = build_sim_summary_markdown(_report())
        # The summary must never assert real-world success.
        assert "real-world success: True" not in md.lower()
