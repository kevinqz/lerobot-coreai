# artifact_index_reports.py — human-readable output for the artifact index (v1.2.2).

from __future__ import annotations

from typing import Any


def format_entries_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "(no artifacts indexed)"
    lines = []
    for e in entries:
        sig = "signed✓" if e.get("signature_verified") else "unsigned"
        rc = e.get("release_check_passed")
        rc_s = "release✓" if rc else ("release✗" if rc is False else "release?")
        lines.append(
            f"- {e.get('artifact_id')}\n"
            f"    type={e.get('artifact_type')} channel={e.get('release_channel')} "
            f"{sig} {rc_s}\n"
            f"    policy={e.get('policy_path')} dataset={e.get('dataset_repo_id')}")
    return "\n".join(lines)


def format_verify(checks: list[dict[str, Any]], ok: bool) -> str:
    lines = ["lerobot-coreai artifact-index verify", "=" * 50]
    for c in checks:
        mark = "✓" if c["passed"] else "✗"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"{mark} {c['name']}{detail}")
    lines.append("=" * 50)
    lines.append("Index verified." if ok else "Index verification FAILED.")
    return "\n".join(lines)
