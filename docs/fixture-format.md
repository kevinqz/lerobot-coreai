# Observation Fixture Format

Observation fixtures are JSON files that provide a static observation batch for dry-run rollout.

## Flat format (simple)

```json
{
  "observation.images.wrist": "assets/wrist.png",
  "observation.state": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
  "task": "pick up the cube"
}
```

- Image paths (`observation.images.*` keys with string values) are resolved relative to the fixture file's directory
- State values are passed as-is (lists of floats)
- The `task` key is a plain string

## Typed format (explicit kinds)

```json
{
  "observation": {
    "observation.images.wrist": {
      "kind": "image",
      "path": "assets/wrist.png"
    },
    "observation.state": {
      "kind": "tensor",
      "dtype": "float32",
      "shape": [7],
      "value": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
    },
    "task": {
      "kind": "text",
      "value": "pick up the cube"
    }
  }
}
```

- `kind: "image"` — resolves `path` relative to fixture directory
- `kind: "tensor"` — uses `value` directly
- `kind: "text"` — uses `value` directly

## Rules

- File must be `.json`
- Both formats return a flat `dict[str, Any]` batch
- Image existence is NOT validated (allows portable/mocked fixtures)
