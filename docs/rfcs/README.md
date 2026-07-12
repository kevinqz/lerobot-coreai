# Core AI Ecosystem RFCs (governing this repo)

Vendored from the ecosystem RFC pack (2026-07-12) for traceability (RFC-0800 §13).
These describe the **target** architecture; they do not claim unfinished capabilities
already exist.

- [RFC-0000 — Ecosystem architecture & boundaries](RFC-0000-ECOSYSTEM-ARCHITECTURE.md)
- [RFC-0700 — `lerobot-coreai` as the Apple deployment provider](RFC-0700-LEROBOT-COREAI.md)
- [RFC-0800 — Cross-repo delivery plan & release train](RFC-0800-CROSS-REPO-DELIVERY-PLAN.md)
- [RFC-0100 — coreai-interop](RFC-0100-COREAI-INTEROP.md) · [RFC-0400 — Runner/RuntimeCore](RFC-0400-COREAI-RUNNER.md)
- [RFC-0900 — lerobot-coreai-apple (robot-brain app)](RFC-0900-LEROBOT-COREAI-APPLE-APP.md)

> Vendored at the **mobile-robot-brain-amendment-1** edition (2026-07-12). It adds a second
> new repo (`lerobot-coreai-apple`), the Gemma-planner→policy→gateway→robot architecture,
> the Runner RuntimeCore/service split, and this repo's **gateway-reference** role.

## This repo's roadmap (RFC-0700 §19) and where we are

`LR1` contract convergence · `LR2` real Fabric bundle · `LR3` real Swift Runner ·
`LR4` L4 matrix · `LR5` protected signing · `LR6` Diffusion · `LR7` VLA · `LR8` async/RTC ·
`LR9` remote Server · **`LR10` Robot Gateway reference (done — `robot_gateway.py`,
[docs/robot-gateway.md](../robot-gateway.md))** · `LR11` mobile-recording bridge ·
`LR12` Apple-app conformance.

**Current: L3 (Official Eval), `test_only`** — see
[conformance-levels-l0-l6.md](../conformance-levels-l0-l6.md) and
`lerobot-coreai conformance-level`. Phase-0 truth repair (RFC-0800 §3) is done: docs
distinguish L3 test-only from L4 real Core AI, and a regression test enforces it.

**Blocked on external artifacts** (not on this repo): `coreai-interop` generated client
(LR1), a real Fabric `.aimodel` (LR2), and a real Swift CoreAI Runner (LR3) — all
upstream in the ecosystem RFCs.
