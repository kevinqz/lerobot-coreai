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

## Reproducible procedure (maintainer, on Apple Silicon)

1. Start the **real** CoreAI Runner on loopback; note its binary sha256 + version.
2. Capture the identity:
   `lerobot-coreai apple-runtime probe --coreai-runner-version <v> --aimodel-sha256 <h> --output identity.json`
   (on Apple Silicon this records `arm64` + the macOS version).
3. Drive the certified B=1 / native / split multimodal matrix through the **real**
   runner (front+wrist+state+task), compare outputs to the approved reference
   (numeric parity), and collect the check flags + performance summary.
4. Build the certificate with `build_apple_runtime_certificate(identity=…, checks=…,
   signed_official_eval_certificate_sha256=…, …)` — the gate promotes
   `apple_runtime_certified` only if all conditions hold.
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
