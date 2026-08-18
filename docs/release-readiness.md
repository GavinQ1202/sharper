# v0.2 release readiness

This is a contract-facing readiness checklist for maintainers and users. It
is not a release announcement and does not grant permission to publish a
package.

## Current status

| Item | Current fact | Final target or gate |
| --- | --- | --- |
| Package version | `0.2.0` source and metadata | Wheel/sdist/runtime parity PASS |
| v0.1 compatibility | Existing workflow, report, CLI, result fields, defaults, and exports remain the compatibility baseline | Compatibility gates PASS |
| Task20 workflow | Typed opt-in orchestration and I6 integration are implemented; targeted repair and bounded closure are complete | `A–X = 24/24 PASS`; terminal state is `Release Ready — Not Released` |
| Task20 JSON | `task20.policy.v1` and `task20.warning.v1` adapters are implemented | Closed-schema parity and clean-install evidence PASS |
| Task20 CLI | `sharper v02-run` is implemented | CLI smoke and distribution evidence PASS |
| Task20 root exports | Nine exact symbols are active in the `0.2.0` root surface | Permanent v0.1 compatibility preserved |
| Examples | All five approved v0.2 example files exist and run with synthetic inputs | Final matrix/distribution evidence recorded |
| Distribution/CI | `0.2.0` wheel/sdist build, additive CI gates, and source-free offline smoke are implemented | Local/structural CI readiness evidence is recorded; no remote runner claim |
| Release action | No tag, push, upload, publish, or release has occurred | Remains prohibited until separately authorized |

The final implementation status is:

- Full Implementation Review: `NO-GO — CONSUMED` (quota `0`)
- Frozen findings: `T20-IR-01..04` = `CLOSED` (4 CLOSED / 0 OPEN)
- Targeted repair: `COMPLETE`; repair checkpoint: `82aadc06a1912f4a2e132170d53a61719d8a75cc`
- Bounded Implementation Closure: `PASS`
- A–X: `24/24 PASS`
- Implementation: `REPAIRED AND CLOSURE-VALIDATED`

The terminal wording is `Release Ready — Not Released`: the readiness gates
have been validated while no release action has been performed. CI readiness is
based on local/structural evidence; this document makes no claim of a remote
GitHub Actions run.

## Required readiness gates

### Public surface and version

- The current ordered `sharper.__all__` preserves the v0.1 compatibility
  manifest and appends exactly the approved Task20 symbols:
  `V02ScoreValidationRequest`, `V02AuditRequest`, `V02PreLoanRequest`,
  `V02PostLoanRequest`, `V02GovernanceRequest`, `V02WorkflowRequest`,
  `V02WorkflowResult`, `run_v02_workflow`, and `generate_v02_report`.
- Root imports, signatures, field order, defaults, and docs agree with the
  approved public surface.
- Version metadata, artifact names, and runtime `__version__` agree on the
  authorized target `0.2.0`; the Full Implementation Review is
  `NO-GO — CONSUMED` and its quota is `0`.

### Compatibility and behavior

- The full v0.1 test suite remains intact and verifies v0.1 signatures,
  dataclass fields, defaults, reports, CLI behavior, and exports.
- Task20 remains opt-in. The default `run_analysis`,
  `generate_analysis_report`, and `sharper analyze` paths do not acquire v0.2
  sections or change their output bundle.
- Python and CLI runs use the same fixed owner call order and result semantics;
  enabled owner functions are called exactly once.
- Reports consume frozen results and Figures without fitting, predicting,
  evaluating, executing rules, reconstructing alerts, or recomputing metrics.

### Verification and tooling

After `bash scripts/verify-uv-env.sh`, the project `.venv` must pass:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build --no-isolation
```

The final gate also requires controlled wheel/sdist clean-install checks,
source-free import checks, distribution smoke checks, examples, CLI smoke,
and the supported CI/Python matrix. A result produced by a system Python or a
different virtual environment is not valid release evidence.

### Distribution boundary

The target distribution distinction is:

- wheel: runtime-focused package contents and the verified public runtime
  surface;
- sdist: source distribution containing the approved source, documentation,
  and v0.2 examples.

The final gate must verify build metadata, version parity, wheel and sdist
contents, clean installation without the source checkout, import behavior,
and the absence of unapproved mandatory dependencies. The controlled
source-free offline distribution smoke and the four direct final acceptance
tests passed at this checkpoint.

### Approved examples

The final v0.2 example set is exactly:

```text
examples/v02_score_validation.py
examples/v02_preloan.py
examples/v02_postloan.py
examples/v02_combined_report.py
examples/v02_cli_json.py
```

Examples must use only the approved public API, use synthetic or illustrative
inputs, disclose that policy/alert outputs are offline evidence, and run
deterministically. Their creation and smoke validation are implementation
evidence; the final matrix and implementation review evidence are recorded in
the completed Task20 checkpoints.

### CI and offline evidence

CI must preserve the v0.1 compatibility layer while checking the current
release surface, full tests, Ruff, build, clean installation, examples, CLI
smoke, and supported Python versions. The readiness evidence must be
reproducible without network access after dependencies are available locally;
benchmark downloads, raw datasets, generated reports, caches, lock-file
changes, credentials, and publishing permissions are not release artifacts.

## Release boundary

Task20 reaches release readiness after the gates above, the approved contract
checklist, and the final implementation review closure recorded here.
Readiness does not perform any external action. The following remain outside
this task's release-readiness operation:

```text
git push
git tag
PyPI upload
GitHub release
publishing
deployment
```

The source and metadata target is `0.2.0`; this document makes no claim of a
published, deployed, production-ready, or externally released package.
