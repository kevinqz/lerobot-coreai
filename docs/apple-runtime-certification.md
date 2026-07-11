# Apple / CoreAI Runtime Certification — diagnostic harness (v1.4.0 machinery)

> Run the exact signed artifact on a real Apple/CoreAI runtime and prove the hardware
> execution matches the certified contract.

This ships the **machinery** — `AppleRuntimeIdentity v1`, `AppleRuntimeCertificate v1`,
capture, the hard gate, the offline verifier, and a CLI. The actual **certification
run requires real Apple Silicon** (a real CoreAI Runner + a real `.aimodel`, no fake
runner), which the maintainer runs on their own hardware. On Linux CI (and until a
real run happens) `apple_runtime_certified` is held **false** — by construction, not
by omission.

## The gate (why CI can't fake it)

`apple_runtime_certified` is derived, never asserted. It is true **only** when:

- `identity.hardware.cpu_arch == "arm64"` **and** `identity.execution.process_arch ==
  "arm64"` (no Rosetta/x86 process),
- `identity.software.macos_version` is present (a real macOS host),
- every real-execution check passes: `real_runner_used`, `real_aimodel_loaded`,
  `all_required_cases_passed`, `numeric_parity_passed`, `official_eval_chain_bound`,
  `signed_evidence_verified`,
- and (in the verifier) a **signed official-eval certificate** is bound.

A GitHub-hosted Linux runner fails the arch + macOS conditions, so the claim is false
there regardless of the check flags. The verifier recomputes the gate and rejects a
forged `true`, an identity tamper, or a certified claim with no signed chain.

## Promotion authority — no self-certification (v1.3.26.8)

The 3rd external review's finding was that the *contract* was near-SotA but the
*authority* still trusted the producer: a caller could hand `build_apple_runtime_certificate`
a dict of `True` booleans. That path is closed.

- The public builder is now **`build_diagnostic_apple_runtime_report`** and carries
  `evidence_grade: "diagnostic"` — its `apple_runtime_certified` is **always false**,
  no matter what checks the caller passes. The verifier rejects a diagnostic report
  whose claim was forged to `true`.
- A **true** certificate (`evidence_grade: "certificate"`) can be produced only by
  **`promote_apple_runtime_certificate`**, which accepts *only* unforgeable
  `Verified*` receipts — `VerifiedCoreAIRuntimeReceipt`,
  `VerifiedSignedOfficialEvalCertificate`, `VerifiedModelConversionEvidence`,
  `VerifiedTrustPolicy` (see `authority.py`). A plain dict or bool in any slot raises
  `TypeError`; the checks are **derived from the receipt substance**, never supplied.
- Each `Verified*` is minted **only** by its verifier (constructor sealed by a
  module-private token), and each verifier re-derives its substance from bytes: the
  runtime receipt must prove a real runner (no `fake_runner`), the `.aimodel` opened
  with its root == the certified artifact, the full case matrix, and numeric parity;
  the signed official-eval must verify its Ed25519 signature + trust policy; the
  conversion must re-pass numeric parity from raw arrays.

So no public function accepts booleans to promote, and **no manually-built JSON can
obtain a true `apple_runtime_certified`** — the gate still also requires the arm64
Apple identity, which Linux CI can never satisfy.

## Provenance authority (v1.3.26.11)

An earlier review correctly noted that the `Verified*` types had merely **moved**
self-attestation into the receipts, and that a Python object sealed by a module-private
token is **type discipline, not a security boundary**. This release closes the
provenance gap for everything that is deterministic offline:

- **Trust anchor is pinned (P0.6).** A high claim promotes only under
  `as_verified_official_trust_policy`, which requires the policy to match the pinned
  official anchor identity (`policy_id` + issuers ⊆ the official set), require
  certificate grade, and carry **no dev key**. A producer that mints its own key + its
  own policy is rejected — "verified signature" no longer means "signed by a key the
  caller chose to trust."
- **Signatures bind to real certificate bytes (P0.7/WS4).**
  `as_verified_signed_official_eval` now requires the underlying `OfficialEvalCertificate`,
  **re-runs its verifier** (must be certificate grade + certified), and cross-binds the
  signed root to a fresh `certificate_root_sha256` of those bytes. A signature over a
  bare, unverified root is refused.
- **Checks are derived from receipt reports (P0.3).** The official-eval receipt carries
  the executor's own `schema_report` / `replay_report`; the promoter derives its checks
  from them instead of hardcoding `True`.
- **Full evidence-graph root closure (P0.4/WS5).** A certified `OfficialEvalCertificate`
  must bind all eleven roots (artifact, feature, dataset, processor-parity, policy-execution,
  model-conversion, processor-stage, runtime-support, negotiation, runner-capabilities,
  rollout-matrix) — **non-null**. The verifier rejects a certified certificate with any
  null root.
- **Artifact cross-binding (P0.5).** Apple promotion refuses unless the `.aimodel` the
  runner executed is the *same* artifact bound by the conversion evidence, the runtime
  identity, and the certified official-eval (`runtime.aimodel == conversion.artifact ==
  identity.model == official_eval.artifact_root`). "Certified the wrong artifact" is a
  hard error.

**Honest limits.** The `Verified*` seal remains a type guard, not authenticity;
authenticity comes from verified bytes + a trusted executor identity + a signed receipt
+ the pinned trust policy. The two things CI cannot fabricate — a real `lerobot-eval`
subprocess (WS1) and a real CoreAI Runner + `.aimodel` handshake (WS2) — stay deferred
to v1.3.27 / v1.4.0; until an executor-owned, signed receipt exists, those receipts are
built from declared fields and the high claims remain **false** in CI.

## Reproducible procedure (maintainer, on Apple Silicon)

1. Start the **real** CoreAI Runner on loopback; note its binary sha256 + version.
2. Capture the identity:
   `lerobot-coreai apple-runtime probe --coreai-runner-version <v> --aimodel-sha256 <h> --output identity.json`
   (on Apple Silicon this records `arm64` + the macOS version).
3. Drive the certified B=1 / native / split multimodal matrix through the **real**
   runner (front+wrist+state+task), compare outputs to the approved reference
   (numeric parity), and collect the check flags + performance summary.
4. Mint the verified receipts (`verify_coreai_runtime_receipt`,
   `as_verified_signed_official_eval`, `as_verified_model_conversion`,
   `as_verified_trust_policy`) from the real run's artifacts, then call
   `promote_apple_runtime_certificate(identity=…, runtime_receipt=…, official_eval=…,
   conversion=…, trust_policy=…)` — the gate promotes `apple_runtime_certified` only if
   the arm64-Apple identity holds *and* every receipt verifies. (A diagnostic-only
   snapshot uses `build_diagnostic_apple_runtime_report`, which never certifies.)
5. Verify offline: `lerobot-coreai apple-runtime verify --certificate cert.json
   --identity identity.json`.
6. Sign it with the **protected release key** via the signed-evidence primitive
   (`certificate_type="apple_runtime"`).

The claim's scope is always the **exact** recorded hardware/OS/SDK/runner/`.aimodel`
tuple — never "Apple Silicon in general", never physical safety or real-robot task
success.

## Not yet (needs the real run / hardware)

- The actual signed `AppleRuntimeCertificate` with `apple_runtime_certified=true` —
  requires the maintainer's Apple-Silicon run + the protected release key.
- A self-hosted macOS arm64 CI runner for the release gate.
- The functional/reliability/performance capture wired to the real runner (the schema
  fields + gate are in; the real measurements come from the hardware run).
