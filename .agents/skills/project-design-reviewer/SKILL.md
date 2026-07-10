---
name: project-design-reviewer
description: Review a Python data-analysis or machine-learning package design before implementation and issue a Go, Conditional Go, or No-Go decision. Use when evaluating SPEC.md, AGENTS.md, README.md, package architecture, public API, testing strategy, dependency choices, or an implementation roadmap for unclear scope, oversized v0.1 plans, overengineering, API complexity, weak tests, ML data leakage, unnecessary dependencies, or poor implementation sequencing.
---

# Project Design Reviewer

Review the design as an implementation gate. Be direct, evidence-based, and biased toward a smaller v0.1. Do not write implementation code.

## Review Workflow

1. Inspect `SPEC.md`, `AGENTS.md`, `README.md`, `pyproject.toml`, proposed or existing package structure, API sketches, test plans, examples, and roadmap. Note missing artifacts; do not invent their contents.
2. Reconstruct the claimed v0.1 in one short paragraph: target user, problem, primary workflow, deliverables, and explicit exclusions.
3. Trace each primary workflow through the proposed modules, public API, tests, and roadmap. Flag gaps and contradictions across documents.
4. Review every area below. Report material findings first instead of narrating the review process.
5. Recommend the smallest correction for each finding. Prefer deleting, deferring, merging, or simplifying before adding abstractions.
6. Issue exactly one final verdict: **Go**, **Conditional Go**, or **No-Go**.

## Review Areas

### Scope and v0.1

- Require a specific target user, concrete problem, two or three primary workflows, measurable acceptance criteria, and explicit non-goals.
- Flag features without a demonstrated v0.1 workflow.
- Flag broad platform language such as “all data,” “any model,” “extensible framework,” or “production-ready” when the design does not bound it.
- Recommend a coherent thin slice that can be released and learned from.

### Architecture and Overdesign

- Require clear module ownership and simple dependency direction.
- Flag layers, adapters, registries, plugin systems, abstract factories, dependency injection, generic workflow engines, deep inheritance, or persistence frameworks without an immediate v0.1 need.
- Flag duplicate concepts, circular responsibilities, catch-all modules, and architecture that optimizes for hypothetical future integrations.
- Prefer ordinary functions, focused classes, scikit-learn composition, and explicit data flow.

### Public API

- Trace the shortest user path from import to useful result.
- Require a small, typed, documented, testable public surface with deliberate exports.
- Flag overlapping entry points, boolean-heavy signatures, generic configuration dictionaries, ambiguous return types, hidden global state, and public exposure of internal orchestration.
- Require task-specific behavior where classification and regression differ.

### Testing Strategy

- Require a mapping from important contract or risk to test type, fixture, and expected evidence.
- Flag plans that say only “add unit/integration tests,” rely mainly on coverage percentage, or omit negative and edge cases.
- Require public API tests, install/import smoke tests, deterministic examples, invalid-input behavior, and release-critical checks.
- For ML workflows, require small synthetic fixtures that can prove split isolation, preprocessing behavior, metric correctness, and reproducibility.

### ML Leakage and Evaluation

- Treat preprocessing, imputation, encoding, scaling, feature selection, resampling, or model fitting before train/test splitting as a blocker.
- Require learned transformations to be fitted through a pipeline on training data only and independently inside each cross-validation fold.
- Flag target leakage, duplicate-row leakage, group leakage, time leakage, post-outcome features, test-set model selection, threshold tuning on test data, and full-dataset feature engineering.
- Require explicit `random_state` propagation to supported randomized operations.
- Require classification and regression evaluation paths and metrics to remain distinct.

### Dependencies

- Require every runtime dependency to support a named v0.1 capability.
- Flag dependencies used only for trivial helpers, optional workflows, speculative integrations, development tooling, or functionality already supplied by the standard library or a required core dependency.
- Keep development, documentation, and optional dependencies out of the minimum runtime install.

### Implementation Roadmap

- Require thin, verifiable milestones with an observable outcome and acceptance evidence.
- Put contracts, package skeleton, safe data split, minimal preprocessing pipeline, and one simple baseline workflow before secondary models or infrastructure.
- Flag roadmaps that build frameworks before a vertical slice, postpone leakage tests, mix unrelated features, or lack dependency ordering.
- Require documentation and executable examples to evolve with the public API rather than arrive only at the end.

### Document Consistency

- Check that README promises fit the SPEC scope.
- Check that AGENTS.md enforces, rather than contradicts, architecture and verification rules.
- Check that the directory tree supports the stated module boundaries.
- Check that API examples, tests, and roadmap use the same names and concepts.

## Findings Format

List findings in descending severity:

1. **Blocker** — implementation would be unsafe, invalid, or aimed at an undefined product.
2. **Major** — likely to cause substantial rework, scope failure, API instability, or untrustworthy ML results.
3. **Minor** — worthwhile design correction that does not block the first implementation slice.

For every finding include:

- concise problem statement;
- exact file and section, or mark the required artifact as missing;
- concrete consequence;
- smallest recommended correction;
- whether it blocks the verdict.

Do not soften a material problem with vague phrasing. Do not report mere style preferences as findings.

## Verdict Rules

### Go

Issue **Go** only when:

- v0.1 scope and non-goals are clear and appropriately small;
- primary workflows, architecture, API, tests, and roadmap agree;
- no unresolved leakage or evaluation-validity risk exists;
- no blocker or verdict-blocking major finding remains;
- the first implementation slice has objective acceptance evidence.

### Conditional Go

Issue **Conditional Go** when the direction is sound but a short, explicit set of design corrections must be completed before or during the first bounded implementation slice. List each condition as a binary, verifiable requirement. Do not use this verdict to disguise a blocker.

### No-Go

Issue **No-Go** when scope is undefined or materially oversized, architecture or API requires major redesign, leakage controls are unsafe or absent, testing cannot establish correctness, or the roadmap would lock in the wrong foundation. State what must change before another review.

## Required Output

Produce:

1. **Verdict** — Go, Conditional Go, or No-Go, followed by a one-sentence rationale.
2. **Scope summary** — reconstructed v0.1 and the smallest recommended v0.1 if different.
3. **Findings** — severity-ordered and evidence-linked.
4. **Conditions or required changes** — binary and prioritized.
5. **Recommended implementation order** — concise, dependency-aware, and verification-driven.
6. **Review coverage** — identify each requested artifact as reviewed, missing, or not applicable.

Do not create or modify implementation files. Do not implement fixes. Modify design documents only when the user explicitly asks for a separate revision after the review.
