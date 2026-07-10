---
name: python-package-architect
description: Design and review small, maintainable Python package architectures without implementing them. Use when a user asks to design a Python package, review package architecture, choose a src layout or pyproject.toml structure, define module boundaries or a public API, plan tests, README, examples, packaging, or release readiness, generate SPEC.md or AGENTS.md for a Python package, or plan the implementation order.
---

# Python Package Architect

Design the smallest coherent package that satisfies the stated use cases. Produce decisions and reviewable artifacts, not implementation code, unless the user explicitly requests implementation.

## Workflow

1. Inspect the repository before proposing changes. Read existing `pyproject.toml`, package tree, tests, documentation, and contributor instructions when present.
2. State the package goal, primary users, core use cases, constraints, and non-goals. Make conservative assumptions when details are missing and label assumptions that materially affect the design.
3. Define an MVP boundary. Defer optional integrations, plugin systems, abstraction layers, and speculative extensibility until a concrete use case requires them.
4. Design a `src/` layout. Give every package and module one clear responsibility. Keep dependency direction simple and avoid circular or cross-layer knowledge.
5. Sketch `pyproject.toml` at the configuration-section level: build backend, package metadata, supported Python versions, runtime dependencies, optional dependency groups, entry points if needed, and tool configuration. Do not invent precise versions without evidence.
6. Design the public API from user workflows. Keep it small, typed, documented, testable, and stable. Distinguish public exports from internal modules. Prefer plain functions and focused classes over frameworks or deep inheritance.
7. Design tests around observable contracts. Cover public behavior, error cases, type expectations, package installation/imports, examples, and release-critical metadata. Match test layout to the source layout where useful without mechanically duplicating every file.
8. Plan README, examples, API reference, contributor guidance, and release preparation. Ensure examples exercise only the proposed public API.
9. Order implementation as thin, verifiable vertical slices. Put foundations and public contracts before optional features; attach a verification outcome to every phase.
10. Review the result for unclear ownership, unnecessary layers, premature abstractions, oversized API surface, hidden coupling, and missing acceptance criteria.

## Required Deliverable

Always include:

- proposed directory tree;
- module responsibilities table;
- public API draft;
- test strategy;
- ordered implementation plan.

Also include `pyproject.toml`, documentation, examples, and release-readiness decisions when relevant. Read [references/design-output.md](references/design-output.md) and follow its structure for a full design, architecture review, `SPEC.md`, or `AGENTS.md`.

## Design Rules

- Use `src/<import_name>/` and keep tests outside `src/`.
- Keep the MVP small and explicit. Add a module only when it owns a distinct concept or change boundary.
- Make dependencies flow toward stable domain concepts. Keep I/O, optional integrations, and presentation concerns at the edges.
- Define each module by what it owns, what it may depend on, and what it must not do.
- Expose public names deliberately through package entry points such as `__init__.py`; treat unexported modules as internal.
- Specify types, inputs, outputs, errors, side effects, and minimal examples for public API items.
- Prefer compatibility-preserving API evolution. Flag decisions that would require deprecation or a major version change.
- Avoid catch-all `utils`, `common`, `base`, or `manager` modules unless their contents have one precise responsibility.
- Avoid plugin systems, registries, dependency injection frameworks, deep class hierarchies, and configuration machinery without demonstrated MVP need.
- Separate design recommendations from observed repository facts.

## Artifact Rules

- Generate `SPEC.md` as a product and architecture contract: scope, decisions, structure, API, tests, milestones, and acceptance criteria.
- Generate `AGENTS.md` as concise instructions for future coding agents: architectural invariants, public/internal boundaries, verification commands, and change rules. Do not duplicate the full specification.
- Do not create implementation files, placeholder modules, or production code during design-only work.
- If implementation is explicitly requested, preserve the approved design as the source of truth and implement only the requested scope.
