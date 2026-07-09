# Observation Adapters

Observation adapters provide declarative transformation of raw observations from sources
into the shape the policy expects. Added in v0.7.2.

## What adapters do

- Inject `task` text into observations
- Inject `observation.state` from a vector or JSON file
- Map image key aliases (`camera_front` → `observation.images.front`)
- Check required keys (explicit or manifest-driven)
- Drop unknown keys not in the manifest

Adapters return warnings for non-fatal issues and raise for fatal ones (missing required
key, invalid state JSON, non-numeric state vector).

## Usage via CLI

```bash
lerobot-coreai shadow \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --observation-source camera \
  --adapter.require-task \
  --adapter.require-state \
  --adapter.required-keys "observation.images.wrist,observation.state" \
  --task "pick up the cube" \
  --state-vector "0,0,0,0,0,0,0" \
  --output-dir runs/shadow-adapted
```

### Image key mapping

For sources that produce non-standard image key names:

```bash
--adapter.image-map "camera_front=observation.images.front,wrist=observation.images.wrist"
```

### Drop unknown keys

When `--adapter.drop-unknown-keys` is set, only keys present in the manifest's
`observation_features` (plus `task`) are kept. Non-manifest keys are dropped with a warning.

## Usage via Python

```python
from lerobot_coreai.observation_adapters import ObservationAdapterConfig, adapt_observation

config = ObservationAdapterConfig(
    image_key="observation.images.wrist",
    task="pick up the cube",
    state_vector=[0.0] * 7,
    require_task=True,
)

result = adapt_observation(raw_observation, config)
# result.observation — the adapted dict
# result.keys_present — sorted list of keys
# result.warnings — non-fatal warnings
```

## Important

Adapters do **not** duplicate manifest validation. Observation/action validation against
the manifest is handled by `CoreAIPolicy.predict_action()`. The adapter's job is shaping
and mapping, not deep validation.
