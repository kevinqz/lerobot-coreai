# test_official_cli_rollout.py — the WS1 policy half (v1.3.27.2): the OFFICIAL
# `lerobot-eval` CLI drives the coreai_bridge policy against the registered
# coreai_cert_env to a completed rollout, through a real subprocess, with the bridge
# policy doing genuine inference against a stub coreai-runner.v2 server.
#
# Runs only where the lerobot[dataset] extra is present (lerobot-eval imports
# lerobot.datasets at module load) — i.e. the rollout CI jobs.

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("torch")

from lerobot_policy_coreai_bridge.artifact import build_plugin_artifact  # noqa: E402

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None
_A, _HOR = 6, 1     # coreai_cert_env action/state dim + horizon


def _manifest():
    # features MUST match coreai_cert_env: observation.state (6,), action (HOR, 6),
    # NO cameras.
    return {
        "schema_version": "lerobot-coreai.v0", "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0", "commit": None},
        "policy": {"repo_id": "k/cert", "source_repo_id": "k/cert", "type": "evo1",
                   "class": None, "config_class": None},
        "robot": {"type": "so100", "action_representation": "joint_position_delta",
                  "fps": 10},
        "features": {"observation": {
            "observation.state": {"dtype": "float32", "shape": [_A], "required": True},
            "task": {"dtype": "string", "required": False}},
            "action": {"action": {"dtype": "float32", "shape": [_HOR, _A]}}},
        "normalization": {"format": "lerobot", "path": "n.json", "sha256": None},
        "coreai": {"artifact_format": "aimodel", "runner": "coreai-runner",
                   "graphs": [{"name": "g", "role": "denoise_step"}],
                   "host_loop_required": False},
        "evaluation": {"metric": "action_parity", "status": "passed", "n_obs": 8,
                       "min_chunk_cosine": 0.9999, "max_action_mae": None,
                       "max_relative_action_mae": None, "proves_numeric_fidelity": True,
                       "proves_task_success": False, "proves_robot_safety": False},
        "safety": {"default_mode": "dry_run",
                   "real_actuation_requires_confirmation": True},
        "contracts": {
            "processor": {"observation_input": {"owner": "coreai_runner",
                                                "expects": "raw_lerobot_observation"},
                          "action_output": {"owner": "coreai_runner",
                                            "returns": "postprocessed_environment_action"}},
            "batch": {"schema_version": "coreai-batch-contract.v3",
                      "native_batch": {"supported": True, "max_batch_size": 4,
                                       "required_slot_isolation": "independent"},
                      "client_split": {"supported": True, "max_batch_size": 4,
                                       "allowed_state_scopes": ["stateless"]},
                      "fallback": "split_and_stack",
                      "queue": {"layout": "time_major_batched",
                                "commit_semantics": "atomic_queue_commit"},
                      "observation_stage": "lerobot_policy_preprocessor_output.v1"}},
    }


def _handler():
    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _j(self, code, o):
            b = json.dumps(o).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/v1/health":
                return self._j(200, {"status": "ok"})
            if self.path == "/v1/capabilities":
                return self._j(200, {
                    "runtime": "coreai-runner", "supports": {"action": True},
                    "protocol_version": "coreai-runner.v2",
                    "observation_encodings": ["nested_json_v1"],
                    "action_batching": {"supported": True, "max_batch_size": 4,
                                        "semantics": "native",
                                        "slot_isolation": "independent"},
                    "inference_state": {"scope": "stateless",
                                        "supports_session_ids": False,
                                        "reset_scope": "none"}})
            return self._j(404, {})

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            opts = body.get("options", {})
            if "batch_size" in opts:
                b = int(opts["batch_size"])
                action = [[[0.0] * _A] for _ in range(b)]     # [B, HOR, A]
            else:
                action = [[0.0] * _A]                          # [HOR, A]
            return self._j(200, {"action": action})
    return Hd


def _build_cert_artifact(tmp_path):
    src = tmp_path / "coreai"
    src.mkdir()
    (src / "lerobot-coreai.json").write_text(json.dumps(_manifest()))
    art = tmp_path / "plugin"
    build_plugin_artifact(str(src), str(art), runner_url_env="COREAI_RUNNER_URL")
    return str(art)


@pytest.mark.skipif(not _HAS_DATASETS,
                    reason="lerobot-eval needs the lerobot[dataset] extra (rollout jobs)")
def test_official_cli_drives_bridge_policy_to_completed_rollout(tmp_path):
    art = _build_cert_artifact(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler())
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        env = dict(os.environ)
        env["COREAI_RUNNER_URL"] = f"http://127.0.0.1:{port}"
        with tempfile.TemporaryDirectory() as out:
            argv = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
                    "--env.type=coreai_cert_env", f"--policy.path={art}",
                    "--eval.n_episodes=1", "--eval.batch_size=1",
                    f"--output_dir={out}/eval"]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=600,
                                  env=env)
        combined = proc.stdout + proc.stderr
        # the REAL official CLI completed a rollout of the bridge policy on our env.
        assert proc.returncode == 0, combined[-2000:]
        assert "End of eval" in combined
        assert "pc_success" in combined
    finally:
        srv.shutdown()
