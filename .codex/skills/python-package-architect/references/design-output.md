# Design Output

Use the sections below for a complete design. Adapt headings to the requested artifact, but retain the five required deliverables.

## 1. Goal and MVP boundary

- Problem and target users
- Primary workflows
- In scope
- Explicitly deferred
- Assumptions and constraints

## 2. Architecture decisions

Summarize each consequential decision, its rationale, and the simpler alternative rejected. Cover `src` layout, dependency direction, packaging, supported Python versions, and optional dependencies where relevant.

## 3. Proposed directory tree

Show only justified files and directories. Annotate non-obvious entries briefly.

## 4. Module responsibilities

Use a table with:

| Module | Owns | May depend on | Must not do |
|---|---|---|---|

Every proposed module must appear exactly once.

## 5. Public API draft

For each public symbol, specify:

- import path and typed signature;
- purpose and minimal usage example;
- input and output contracts;
- documented exceptions or failure result;
- side effects and state, if any;
- testable acceptance behavior.

List internal-only modules separately. Keep the public surface smaller than the internal implementation surface.

## 6. Packaging and tooling

Plan relevant `pyproject.toml` sections without writing implementation:

- build system and package discovery;
- project metadata and Python support policy;
- runtime versus optional and development dependencies;
- type-checking, linting, testing, coverage, and build configuration;
- package data and command entry points, only when required.

## 7. Test strategy

Map each important contract or risk to a test level:

| Contract or risk | Test type | Location | Evidence of success |
|---|---|---|---|

Include public API behavior, invalid inputs, type checking, installation/import smoke tests, example verification, and distribution metadata. Add numerical, property-based, compatibility, or performance tests only when the package domain requires them.

## 8. Documentation and examples

Plan:

- README promise, installation, quick start, and project status;
- one minimal example per primary workflow;
- API reference generated from or checked against public signatures;
- contributor and architecture guidance;
- versioning, changelog, migration, and release notes when needed.

## 9. Implementation order

Break work into thin phases. For each phase state:

1. outcome;
2. files or architectural area;
3. dependency on earlier phases;
4. verification and acceptance criterion.

Do not use vague phases such as “build core” or “add tests.”

## 10. Risks and open decisions

Separate blockers from deferrable questions. Recommend the smallest reversible choice for unresolved decisions.

## Review Mode

When reviewing an existing package, add an answer-first findings section. Rank findings by impact, cite exact repository paths, explain the violated invariant or user consequence, and recommend the smallest design correction. Do not rewrite the architecture merely to match personal preference.

## AGENTS.md Mode

Keep `AGENTS.md` operational and concise:

- package purpose and architecture map;
- module ownership and dependency rules;
- public API stability rules;
- required test and documentation updates;
- authoritative verification commands discovered in the repository;
- prohibited shortcuts and generated-file rules.

Do not invent commands that have not been configured.
