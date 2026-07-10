# policy_card_reports.py — report writers for the policy card (v1.2.3).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_policy_card_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Policy Card Report",
        "",
        f"- OK: {report.get('ok')}",
        f"- Source mode: {report.get('source_mode')}",
        f"- Source verified: {report.get('source_verified')}",
        f"- Policy: {report.get('policy_path')}",
        f"- Artifact id: {report.get('artifact_id')}",
        "",
        "## Sections written",
    ]
    for s in report.get("sections_written", []):
        lines.append(f"- {s}")
    lines += [
        "",
        "Proves the card was generated from verified evidence — not physical "
        "safety, task success, training, or actuation.",
        "",
    ]
    return "\n".join(lines)


def write_policy_card_report(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    path.with_suffix(".md").write_text(build_policy_card_report_markdown(report))
