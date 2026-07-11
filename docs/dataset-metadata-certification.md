# Real LeRobotDatasetMetadata evidence (v1.3.25)

> Bind policy semantics to the exact LeRobot dataset metadata tree that defines robot
> type, fps, features, names, shapes, tasks and statistics.

v1.3.25 replaces synthetic metadata stand-ins (`SimpleNamespace`) in the certified
path with a **real, on-disk `LeRobotDatasetMetadata`** and a deterministic, mtime-
independent **metadata-tree root**, bound to the v1 FeatureContract.

## Deterministic metadata-tree hash

`dataset_metadata_hash` computes a content-only root over the allowlisted metadata
tree — `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`,
`meta/episodes/**/*.parquet`:

1. allowlist metadata files (data/videos are NOT part of identity);
2. POSIX relative path, symlinks + path-escapes rejected;
3. per-file **content** sha256 (never `stat`/mtime);
4. ordered `(path, digest)` list;
5. canonical-JSON hash, algorithm id `lerobot-metadata-tree-sha256.v1`.

So mtime/permission changes don't move the root, but any content change to
info/stats/tasks/episodes does.

## DatasetMetadataEvidence v1

`capture_dataset_metadata_evidence(meta, root, repo_id, revision, root_kind)` reads a
duck-typed metadata object (the **real** `LeRobotDatasetMetadata`, or any object with
the same properties) — `robot_type`, `fps`, `features`, `camera_keys`, `names`,
`shapes`, `total_tasks`, `total_episodes` — plus the tree root. Claims:
`dataset_metadata_verified=true` (scoped to the exact root/revision) but
`dataset_content_verified=false` (content is a separate, later proof).
`verify_dataset_metadata_evidence` is fully offline: schema-valid + the root
recomputes from disk; a `hub_snapshot` without a `resolved_commit` is **not**
certificate-grade (a mutable branch name is not an identity).

## Binding to the FeatureContract

`bind_metadata_to_feature_contract` enforces (fail-closed): every required
contract observation/action feature exists in the metadata; dtype/component-dim/
names+order agree; camera keys map to an image-family modality; a normalization
`stats_ref` must exist in the metadata; robot_type compatibility; positive fps.

## How the fixture is real (not synthetic)

The lerobot-gated test builds a **video-free** dataset (state + action, 2 episodes)
with lerobot's own writer (`LeRobotDataset.create` → `add_frame` → `save_episode` →
`finalize`), then loads it back through the real `LeRobotDatasetMetadata(root=…)`
with **no network**. This runs on both the `lerobot-stable` (0.6.0) and `lerobot-dev`
(pinned) CI jobs. Generating at runtime (rather than committing version-specific
parquet) keeps it loadable across both lerobot versions.

## CLI

```bash
lerobot-coreai dataset-metadata inspect --repo-id local/fixture --root DS --output ev.json
lerobot-coreai dataset-metadata verify  --metadata-evidence ev.json --root DS
lerobot-coreai dataset-metadata bind-feature-contract \
  --metadata-evidence ev.json --feature-contract fc.json --fail-on-mismatch
```

## v1.3.26.3 — multimodal + static cross-version + certificate grade

Closes the external review's remaining v1.3.25 gaps:

- **Multimodal fixture** — a committed, static, byte-exact metadata tree at
  `tests/fixtures/lerobot_dataset_v3_multimodal/` exercising `observation.state`,
  **`observation.images.front` + `observation.images.wrist`** (two cameras), `action`
  and `task` — so `camera_keys`, image modality, shapes and per-feature names are all
  covered.
- **Static cross-version** — both the `lerobot-stable` (0.6.0) and `lerobot-dev`
  rollout jobs read the **same committed bytes** through the real
  `LeRobotDatasetMetadata` and must reproduce the **pinned** `metadata_tree_sha256`
  (`sha256:07cefa4a…`) + identical semantic evidence — proving cross-version
  interpretation of *one* artifact (the runtime `create→…→finalize` test still covers
  intra-version round-trip).
- **Certificate grade** — `capture_dataset_metadata_evidence(..., evidence_grade=
  "certificate")` records the real **loader identity** (module/class/lerobot version)
  and **refuses a duck-typed stand-in**; the verifier requires the official
  `lerobot.…LeRobotDatasetMetadata` loader for certificate-grade evidence.

## Not yet

- **Full episode/stats/content verification** (`dataset_content_verified`) — still
  metadata *identity* only.
- A **video-mode** fixture variant (ffmpeg-encoded) alongside the image-mode one —
  deferred (ffmpeg dependency).
- Official-eval CLI (v1.3.27), signed evidence (v1.3.28), Apple runtime (v1.4.0) —
  each promoted only by its own proof.
