# fabric_adapter.py — isolated adapter for coreai-fabric export (v0.6).
#
# All coreai_fabric imports are isolated here.

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CoreAIPolicyError


def require_coreai_fabric() -> None:
    """Ensure coreai-fabric is installed."""
    try:
        import coreai_fabric  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        raise CoreAIPolicyError(
            "Export requires coreai-fabric. Install with:\n"
            '  pip install "lerobot-coreai[fabric]"\n'
            "No robot commands were sent."
        ) from None


@dataclass
class FabricExportConfig:
    torch_policy_path: str
    output_dir: Path
    policy_type: str | None = None
    robot_type: str | None = None
    model_id: str | None = None
    output_repo_id: str | None = None
    fabric_config: Path | None = None
    fabric_profile: str | None = None
    fabric_target: str = "coreai"
    artifact_name: str | None = None


@dataclass
class FabricExportResult:
    ok: bool
    artifact_path: Path | None = None
    manifest_path: Path | None = None
    model_id: str | None = None
    output_dir: Path = Path(".")
    metadata: dict[str, Any] = field(default_factory=dict)


def run_fabric_export(config: FabricExportConfig) -> FabricExportResult:
    """Run coreai-fabric export to produce a .aimodel artifact.

    Strategy: CLI first (most stable), then Python API, then error.
    """
    # Strategy 1: Try coreai-fabric CLI first.
    fabric_bin = shutil.which("coreai-fabric")
    if fabric_bin:
        return _run_fabric_cli(config, fabric_bin)

    # Strategy 2: Try Python API.
    try:
        import coreai_fabric  # type: ignore[import-not-found]  # noqa: F401
        from coreai_fabric.recipes import find_recipe, load_recipe  # type: ignore[import-not-found]
        # If fabric has a programmatic export API, use it here.
        pass
    except ImportError:
        pass

    raise CoreAIPolicyError(
        "coreai-fabric is not available. Install with:\n"
        '  pip install "lerobot-coreai[fabric]"\n'
        "Or use --skip-fabric with --existing-artifact.\n"
        "No robot commands were sent."
    )


def _run_fabric_cli(config: FabricExportConfig, fabric_bin: str) -> FabricExportResult:
    """Run coreai-fabric new → convert → verify → publish via CLI."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe_id = config.model_id or config.torch_policy_path.split("/")[-1].lower()
    artifact_name = config.artifact_name or f"{recipe_id}.aimodel"

    # Step 1: Scaffold recipe.
    cmd_new = [
        fabric_bin, "new", config.torch_policy_path,
        "--output-dir", str(output_dir),
    ]
    if config.policy_type:
        cmd_new += ["--policy-type", config.policy_type]

    result = subprocess.run(cmd_new, capture_output=True, text=True)
    if result.returncode != 0:
        raise CoreAIPolicyError(
            f"coreai-fabric new failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    # Step 2: Convert.
    cmd_convert = [fabric_bin, "convert", recipe_id]
    result = subprocess.run(cmd_convert, capture_output=True, text=True, cwd=str(output_dir))
    if result.returncode != 0:
        raise CoreAIPolicyError(
            f"coreai-fabric convert failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    # Step 3: Verify.
    cmd_verify = [fabric_bin, "verify", recipe_id]
    result = subprocess.run(cmd_verify, capture_output=True, text=True, cwd=str(output_dir))
    if result.returncode != 0:
        raise CoreAIPolicyError(
            f"coreai-fabric verify failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    # Locate artifact.
    artifact_path = output_dir / "build" / recipe_id / artifact_name
    if not artifact_path.exists():
        # Try common alternative locations.
        for candidate in output_dir.rglob("*.aimodel"):
            artifact_path = candidate
            break
        else:
            raise CoreAIPolicyError(
                f"Export completed but .aimodel artifact not found in {output_dir}"
            )

    # Generate lerobot-coreai.json if we have a lerobot block in the recipe.
    manifest_path = output_dir / recipe_id / "lerobot-coreai.json"
    try:
        from .lerobot_manifest_gen import generate_manifest_from_recipe
        recipe_path = output_dir / "recipes" / f"{recipe_id}.yaml"
        if recipe_path.exists():
            manifest = generate_manifest_from_recipe(recipe_path, artifact_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(__import__("json").dumps(manifest, indent=2) + "\n")
    except Exception:
        manifest_path = None  # Best-effort.

    return FabricExportResult(
        ok=True,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        model_id=recipe_id,
        output_dir=output_dir,
        metadata={"fabric_cli": fabric_bin, "recipe_id": recipe_id},
    )
