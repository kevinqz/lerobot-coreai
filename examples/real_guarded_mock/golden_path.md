# Golden path: guarded real mode with the mock adapter (no hardware)

This walks the full safety chain end-to-end using the **mock** adapter, so it
touches no hardware and sends nothing to a real robot. It demonstrates the
sequence, artifacts, and gates — not physical safety.

> The mock adapter is safe by construction. Running against a real controller is
> the operator's responsibility, behind every gate, and proves nothing about
> physical safety on its own.

## 1. Produce sim evidence, gate it, bundle it

```bash
lerobot-coreai sim --policy.path kevinqz/EVO1-SO100-CoreAI --env pusht \
  --episodes 5 --safety.profile so100-sim-default --output-dir runs/evo1-sim
lerobot-coreai safety-gate --run-dir runs/evo1-sim --output-dir gates/evo1
lerobot-coreai package-sim-run --run-dir runs/evo1-sim --output-dir publish/evo1-bundle
lerobot-coreai verify-sim-bundle --bundle-dir publish/evo1-bundle
```

## 2. Operator approval + release readiness

```bash
lerobot-coreai approve-bundle --bundle-dir publish/evo1-bundle \
  --operator "You" --output-dir approvals/evo1
lerobot-coreai release-readiness --bundle-dir publish/evo1-bundle \
  --approval approvals/evo1/approval_manifest.json --output-dir readiness/evo1
```

## 3. Preflight, then a bounded guarded session (mock)

```bash
lerobot-coreai real --mode preflight \
  --policy.path kevinqz/EVO1-SO100-CoreAI --runner.url http://127.0.0.1:8710 \
  --robot.adapter mock --robot.type so100 \
  --safety.profile so100-sim-default \
  --readiness-report readiness/evo1/release_readiness_report.json \
  --approval approvals/evo1/approval_manifest.json \
  --bundle-dir publish/evo1-bundle --output-dir runs/evo1-real-preflight

lerobot-coreai real --mode guarded \
  --policy.path kevinqz/EVO1-SO100-CoreAI --runner.url http://127.0.0.1:8710 \
  --robot.adapter mock --robot.type so100 \
  --safety.profile so100-sim-default \
  --readiness-report readiness/evo1/release_readiness_report.json \
  --approval approvals/evo1/approval_manifest.json \
  --bundle-dir publish/evo1-bundle --operator "You" \
  --max-steps 10 --fps 10 \
  --i-understand-this-may-move-real-hardware \
  --i-have-physical-emergency-stop-ready \
  --i-confirm-robot-workspace-is-clear \
  --abort-file /tmp/ABORT \
  --output-dir runs/evo1-real-guarded
```

Touch `/tmp/ABORT` (or press Ctrl-C) at any point to e-stop the session
(`operator_abort`).

## 4. Audit the session offline

```bash
lerobot-coreai verify-real-session --run-dir runs/evo1-real-guarded \
  --bundle-dir publish/evo1-bundle \
  --approval approvals/evo1/approval_manifest.json \
  --readiness-report readiness/evo1/release_readiness_report.json \
  --output-dir verification/evo1-real
```

Artifacts produced along the way include `real_arming_manifest.json` (the armed
envelope), `real_metrics.json/csv/md` (loop timing), and `real_report.json`.
None of them claim physical safety.
