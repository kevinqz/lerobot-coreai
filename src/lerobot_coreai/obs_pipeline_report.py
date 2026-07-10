# obs_pipeline_report.py — report writers for the observation bridge (v1.1.5).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_obs_bridge_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "obs_bridge_report.json", "w") as f:
        json.dump(report, f, indent=2)
    (output_dir / "obs_bridge_report.md").write_text(build_obs_bridge_markdown(report))


def build_obs_bridge_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Observation Pipeline Bridge Check",
        "",
        f"- OK: {report.get('ok')}",
        f"- Policy: {report.get('policy_path')}",
        f"- Input source: {report.get('input_source')}",
        f"- Frame index: {report.get('frame_index')}",
        "",
        "## Checks",
    ]
    for c in report.get("checks", []):
        mark = "✅" if c["passed"] else "❌"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"- {mark} `{c['name']}` ({c['severity']}){detail}")
    if report.get("dropped_keys"):
        lines += ["", f"Dropped keys: {report['dropped_keys']}"]
    if report.get("warnings"):
        lines += ["", "## Warnings"] + [f"- {w}" for w in report["warnings"]]
    lines += [
        "",
        "Proves the observation mapping is valid for this sample only — not task "
        "success, not physical safety.",
        "",
    ]
    return "\n".join(lines)
