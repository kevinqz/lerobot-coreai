# dataset_metadata_hash.py — canonical LeRobot metadata-tree hash (v1.3.25).
#
# A dataset's identity is its metadata TREE (info.json, stats.json, tasks.parquet,
# episode metadata parquet), not a mutable branch name. This computes a deterministic,
# mtime/permission-INDEPENDENT root over the metadata file CONTENT: allowlist → POSIX
# relative path → per-file content sha256 → ordered (path, digest) list → canonical
# JSON hash. Pure Python; no lerobot.

from __future__ import annotations

import hashlib
from pathlib import Path

from .rollout_evidence_schema import canonical_json_sha256

METADATA_TREE_HASH_ALGORITHM = "lerobot-metadata-tree-sha256.v1"

# the metadata files that DEFINE dataset identity (content-hashed; data/videos are
# NOT part of the metadata tree — content verification is a separate claim).
_META_DIR = "meta"
_ALLOWED_TOP = ("info.json", "stats.json", "tasks.parquet")
_EPISODES_SUBDIR = "episodes"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _safe_rel(rel: str) -> bool:
    parts = rel.split("/")
    return ".." not in parts and "" not in parts and not rel.startswith("/")


def collect_metadata_files(root: str | Path) -> dict[str, str]:
    """Return {posix_relpath: content_sha256} over the allowlisted metadata tree.

    Only regular files under ``meta/`` matching the allowlist (+ episode parquet)
    are included. Symlinks and path-escapes are rejected."""
    root = Path(root)
    meta = root / _META_DIR
    files: dict[str, str] = {}
    if not meta.is_dir():
        raise FileNotFoundError(f"no metadata directory at {meta}")
    candidates: list[Path] = []
    for name in _ALLOWED_TOP:
        p = meta / name
        if p.exists():
            candidates.append(p)
    ep_dir = meta / _EPISODES_SUBDIR
    if ep_dir.is_dir():
        candidates += [p for p in sorted(ep_dir.rglob("*.parquet")) if p.is_file()]
    for p in candidates:
        if p.is_symlink():
            raise ValueError(f"symlink not allowed in metadata tree: {p}")
        rel = p.relative_to(root).as_posix()
        if not _safe_rel(rel):
            raise ValueError(f"unsafe metadata path {rel}")
        files[rel] = _sha256_file(p)
    if f"{_META_DIR}/info.json" not in files:
        raise FileNotFoundError("metadata tree missing meta/info.json")
    return files


def metadata_tree_sha256(files: dict[str, str]) -> str:
    """Canonical root over the ordered (path, digest) list (mtime-independent)."""
    return canonical_json_sha256({"algorithm": METADATA_TREE_HASH_ALGORITHM,
                                  "files": sorted(files.items())})


def compute_metadata_tree(root: str | Path) -> tuple[dict[str, str], str]:
    files = collect_metadata_files(root)
    return files, metadata_tree_sha256(files)
