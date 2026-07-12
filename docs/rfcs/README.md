# Core AI Ecosystem RFCs (governing this repo)

Vendored from the ecosystem RFC pack (2026-07-12) for traceability (RFC-0800 §13).
These describe the **target** architecture; they do not claim unfinished capabilities
already exist.

- [RFC-0000 — Ecosystem architecture & boundaries](RFC-0000-ECOSYSTEM-ARCHITECTURE.md)
- [RFC-0700 — `lerobot-coreai` as the Apple deployment provider](RFC-0700-LEROBOT-COREAI.md)
- [RFC-0800 — Cross-repo delivery plan & release train](RFC-0800-CROSS-REPO-DELIVERY-PLAN.md)

## This repo's roadmap (RFC-0700 §19) and where we are

`LR1` contract convergence · `LR2` real Fabric bundle · `LR3` real Swift Runner ·
`LR4` L4 matrix · `LR5` protected signing · `LR6` Diffusion · `LR7` VLA · `LR8` async/RTC ·
`LR9` remote Server.

**Current: L3 (Official Eval), `test_only`** — see
[conformance-levels-l0-l6.md](../conformance-levels-l0-l6.md) and
`lerobot-coreai conformance-level`. Phase-0 truth repair (RFC-0800 §3) is done: docs
distinguish L3 test-only from L4 real Core AI, and a regression test enforces it.

**Blocked on external artifacts** (not on this repo): `coreai-interop` generated client
(LR1), a real Fabric `.aimodel` (LR2), and a real Swift CoreAI Runner (LR3) — all
upstream in the ecosystem RFCs.
