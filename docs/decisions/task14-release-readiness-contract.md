# Task 14 文档、示例与 v0.1 发布验证公共契约

## 状态、身份与范围

**状态：Implemented — Go。** 本记录冻结 Task 14 的发布准备范围；Task 14 已完成。
任何改变本记录冻结的 public-surface audit、分发验证、文件 allowlist 或 scope 前，
必须同步评审本文件、`SPEC.md`、`IMPLEMENTATION_PLAN.md` 和 `AGENTS.md`。

**正式名称：** Task 14 — 文档、示例与 v0.1 发布验证。

**前置依赖：** Tasks 01--13 已完成，特别是 Task 13 已提供完整 workflow、Markdown/HTML assets bundle 和 CLI。

Task 14 将既有 v0.1 能力整理为可发布、可安装、可执行的文档与验证证据。它不实现新的分析、特征、图、模型、workflow、reporting、CLI 或文件事务行为，也不改变任何 Tasks 03--13 的 public API、稳定错误、Figure ownership、bundle transaction 或依赖方向。

release readiness 只表示本地和 CI 发布门禁已经具备证据；它不等于上传 PyPI、创建 tag、创建 GitHub release、修改远程仓库或提交 commit。核心验证不得把可联网下载依赖作为通过条件。

## 目标与非目标

目标：

- 文档、quickstart、analysis guide、leakage guide、API reference、两个 examples 和 changelog 只承诺已实现的 v0.1 能力；
- 对既有 `sharper.__all__`、public signatures、dataclass fields、type hints、docstrings 与 CLI help 建立发布门槛；
- 在离线条件下验证 sdist/wheel、clean install、`import sharper`、CLI、最小 CSV workflow、examples 与可选 Excel extra 的边界；
- 固定发布前生成物、cache、lock file 与示例输出不得进入仓库。

非目标：

- 不新增、删除或改名 public symbol；不修改已有 exports、signatures、frozen dataclass fields、稳定错误、CLI 参数、exit code 或 stdout/stderr 语义；
- 不修改 `src/sharper/`、`workflow.py`、`reporting.py`、`cli.py`，也不重做 Task 13 验证；
- 不新增或修改核心运行时依赖、lock file、dashboard、server、notebook、网络服务、interactive HTML、release/upload tooling；
- 不发布到 package index，不创建 release tag、GitHub release、自动 commit 或 push；
- 不实现 v0.2/v0.3 功能、性能测试矩阵、跨平台 binary packaging、新的 CI abstraction 或 Task 13 风格 fault matrix。

## Public API、版本与 metadata

Task 14 **不新增 public API**。`sharper.__all__`、既有 public signatures、frozen dataclass field order/type hints、`sharper = "sharper.cli:app"` entry point 和 Task 13 CLI behavior 只审计，不改动。文档和 examples 只能从 `sharper` 顶层 import 已存在的 public names；不得 import private helpers、直接依赖 `src` path、访问 result internals 来重算数据，或把 optional Excel dependency 当成核心安装要求。

Task 14 的冻结版本为 `0.1.0`，且必须一致于 `sharper.__version__`、dynamic version metadata、wheel metadata、sdist metadata、两个分发文件名、目标临时 venv console 的 version output、目标临时 venv Python 的 module-CLI version output、`CHANGELOG.md` 的 `0.1.0` 章节以及 README 中出现的版本（若有）。Task 14 不得 bump version、修改 version source、创建 tag/release 或上传 package index。若文档或 metadata 偏离 `0.1.0`，可在 allowlist 内修正；若 `src` version source 偏离，必须报告 blocker，不能在 Task 14 修改 `src`。

metadata 审计必须验证 package name `sharper`、version `0.1.0`、MIT license、现有 Python 支持范围、README/readme content type、wheel/sdist metadata 一致、现有 console entry point、`excel` extra 与其现有 `openpyxl` 声明，且没有未冻结 extra 或 publish-only requirement。README content type 必须为 `text/markdown`，或 `pyproject.toml` 已声明的等价正确值。

## 文档、examples 与对象 ownership

`docs/quickstart.md` 说明无 target 与可选 model 两条端到端流程。`docs/analysis-guide.md` 仅解释已实现结果和探索性限制。`docs/leakage.md` 保持 split-first、Pipeline、holdout-only、time/group/entity 风险边界。`docs/api.md` 仅列顶层 public API；Task 14 implementation checklist 必须将其“Tasks 11/12 尚未接入 workflow、report generation 或 CLI”的过期表述改为与已完成 Task 13 一致，但不得借机全面重写 API 文档。`CHANGELOG.md` 记录 `0.1.0` 已实现能力与已知限制，不声称 v0.2/v0.3 能力或已发布状态。

两个 examples 都只使用 public API，不读网络、环境隐式数据或当前日期，不打开 GUI，不保存 caller-owned Figure；报告成功后遵守 Task 13 reporting 的 Figure ownership。它们只在调用方给定的临时输出目录内读写，并明确 Markdown/HTML 都是 report + PNG assets bundle。

### Basic example

`examples/basic_analysis.py` 是确定性的最小 public-API workflow。唯一接口为：

~~~
python examples/basic_analysis.py --output-dir <TEMP_DIR>
~~~

脚本在内部创建小型确定性 tabular DataFrame 或临时 CSV，使用固定 random seed、显式 `target` 与固定 `task="classification"`，调用 `run_analysis(..., include_model=False)`，再生成固定 Markdown report + sibling PNG assets directory。它必须保留 `target_analysis`，且 `training`、`evaluation` 均为 `None`；不需要 Excel extra 或 interactive UI。成功 exit code 为 0，stdout 仅为简短成功摘要或为空；report、assets 与临时输入均位于 `<TEMP_DIR>`，由调用它的测试清理。

### Baseline/full example

`examples/baseline_modeling.py` 是完整模型 workflow 与静态 report bundle。唯一接口为：

~~~
python examples/baseline_modeling.py --output-dir <TEMP_DIR>
~~~

脚本在内部创建小型确定性分类数据或临时 CSV，使用固定 random seed、显式 `target`、固定 `task="classification"` 与 `include_model=True`，并生成固定 HTML report + sibling PNG assets directory。成功 exit code 为 0；它必须产生 report、assets、至少一个 PNG，且无 staging/backup 残留。它不依赖网络、Excel extra、private API 或源码 checkout，不打开 GUI，也不污染仓库。

examples 必须被包含在 sdist 并从提取的 sdist 中运行。它们无需作为 wheel package data；wheel clean-install 不得声称从 wheel 内运行 examples，而是运行等价的最小 installed-public-API smoke。examples 的内部变量、helper、临时目录随机名称、HTML/CSS 和 stdout 文案不属于冻结接口。

## Packaging、metadata 与文件系统行为

Task 14 对 `pyproject.toml` 的允许修改仅限于现有字段缺失或与已冻结事实不一致时的以下项目：`[project].name`、`dynamic`（仅维持既有动态版本策略）、`description`、`readme`、`requires-python`、`license`、`authors`、已有或明确需要的 `maintainers`、`classifiers`、`keywords`、`urls`、`[project.optional-dependencies].excel`、`[project.scripts]` 中既有 `sharper` entry point、build package-data/LICENSE inclusion 配置，以及仅为 distribution-test 收集所需的 pytest marker/configuration。

明确禁止修改核心 runtime dependency 的包名或版本范围，新增/删除核心依赖，build backend，version strategy，package discovery，`src` layout，新 console script 或 CLI command 名，lock file，twine/publishing dependencies，pre-commit 或无关工具配置。若发现可证明 metadata 错误（直接 import 的核心依赖完全缺失、包名明显错误、或现有范围使已声明并通过的 Python 环境无法安装），报告 blocker；Task 14 默认不直接修改核心依赖范围，除非合同修订并重新 review。

测试可在 pytest `tmp_path`、系统临时目录或 build tooling 临时目录生成 wheel、sdist、临时 venv、CSV/XLSX、report 和 PNG assets。执行前必须记录工作区基线；只能清理由本任务当前执行创建的 `dist/`、`build/`、临时 venv/wheelhouse、example 临时输入/report/assets、当前运行创建的 `.pytest_cache` 和未跟踪 `__pycache__`/`.pyc`。不得清理执行前已有的 cache、无法确认来源的文件、HEAD 跟踪 `.pyc`、用户文件、既有未跟踪 contracts、Tasks 10--13 内容或 `docs/.DS_Store`。

## Distribution、clean install 与离线验证

使用本地项目构建：

~~~
.venv/bin/python -m build
~~~

构建证据至少包含 `dist/sharper-0.1.0-py3-none-any.whl` 与 `dist/sharper-0.1.0.tar.gz`；若后端使用标准等价 sdist 名称，以实际标准名称为准，但 version 必须为 `0.1.0`。

wheel 与 sdist 必须分别 build、分别安装、分别验证。每个 artifact 使用当前测试解释器的 `venv` 在系统临时目录创建独立临时环境；不得复用开发 `.venv`、不得在同一临时 venv 安装 wheel 与 sdist、不得 editable install，并在任务结束时删除这些环境。

核心验证不得访问外网。pip 安装使用 `--no-index`，依赖只能来自本地 wheelhouse、当前已安装环境或本地 cache；build frontend/build dependencies 同样必须来自当前环境或本地来源。若无法离线提供全部运行时 dependencies，可在临时 venv 预装本地可用 dependencies 后以 `pip install --no-index --no-deps <artifact>` 安装 artifact；这只证明 Sharper distribution 本身可安装，dependencies metadata 必须另行检查。联网下载依赖绝不是 Go 条件。

每个安装后命令在仓库根目录外的系统临时目录执行，清除或覆盖 `PYTHONPATH`，不得将仓库 `src/` 加入 `PATH` 或 `sys.path`，也不得激活开发 `.venv` 或使用 editable install。`PATH` 可以以目标临时 venv 的 bin 目录开头并保留最小继承路径，但不得依赖任何其他 `sharper` executable。验证必须断言 `sharper.__file__` 位于相应临时 venv 的 `site-packages`，不位于仓库 `src/` 或当前 checkout。

每个 artifact 的 console CLI 必须直接调用其目标临时 venv 内的 executable：POSIX 为 `<venv>/bin/sharper`，其他平台按 `venv` 规则解析等价路径。测试必须断言预期 console path 存在、位于目标 wheel 或 sdist venv 内，且不指向开发 `.venv`、系统全局安装或另一临时环境；subprocess command 的第一项必须等于该预期 path。不得以通过全局 `PATH` 解析的 `sharper` 或单独的 `shutil.which("sharper")` 作为来源；若使用 `which`，必须先限定 `PATH` 并断言其结果等于预期 path。

每个 artifact 的 module CLI 必须直接使用同一目标临时 venv 的 Python：POSIX 为 `<venv>/bin/python -m sharper.cli`，其他平台按 `venv` 规则解析等价路径。测试必须断言预期 Python path 存在、位于目标 venv 内，且 wheel smoke 使用 wheel venv Python、sdist smoke 使用 sdist venv Python；subprocess command 的第一项必须等于该预期 Python path。不得通过当前 shell 或开发环境解析 `python -m sharper.cli`。

wheel 与 sdist 的独立 installed-artifact smoke 都必须覆盖：`import sharper`、`sharper.__version__ == "0.1.0"`、public exports、目标 venv console 的 `--help`、`--version`、`analyze --help` 和最小 CSV CLI analysis，以及目标 venv Python 的 `-m sharper.cli --help`、`-m sharper.cli --version`。help/version 与分析均 exit 0；两个 version 输出均为 `sharper 0.1.0`、stderr 为空，最小分析无 traceback。每个 artifact 必须同时证明 package import、实际 console executable 和实际 module Python 分别来自同一目标 venv；wheel 与 sdist 不得共享 venv、executable 或 Python。提取 sdist 的环境还必须运行两个 examples；wheel 环境运行与它们等价的最小 installed-public-API smoke，且不得宣称 examples 属于 wheel。

wheel 与 sdist 均必须验证 LICENSE inclusion、metadata license 与现有 MIT LICENSE 一致、README 被作为 project readme 使用，且 README content type 正确。Task 14 不改写或更换 LICENSE。

## Excel extra 离线边界

extra 名称冻结为 `excel`，现有 dependency 为 `openpyxl`。wheel/sdist metadata 必须证明该 extra 只包含既有 `openpyxl` 声明、`openpyxl` 不进入核心 dependencies，且 core install 后 CSV 仍可用。无 extra 时不必卸载开发环境中的 `openpyxl`；若验证 Excel failure，临时 venv 必须确实未安装它，并使用既有稳定错误，不新增 public error。

只有本地 wheelhouse 具有 `openpyxl` 及必要本地 dependencies 时，才可离线执行 `pip install --no-index --find-links <LOCAL_WHEELHOUSE> "sharper[excel]"` 与单 sheet Excel smoke。若 CI 没有本地 wheelhouse，只验证 extra metadata，并在已有 `openpyxl` 的主测试环境执行 Excel smoke；在线安装 extra 不是核心 Go 条件。

## 验证合同

测试至少证明：

| 风险/合同 | 位置 | 证据 |
|---|---|---|
| public-surface drift | `tests/test_public_api.py` | `__all__`、signatures、frozen result fields、type hints、docstrings 与 documented import paths 一致 |
| docs/examples use only public API | `tests/test_public_api.py` | 文档/examples 不导入 private symbols，声明能力不超出 v0.1 |
| basic/baseline examples | `tests/test_distribution.py` 或 CI | sdist 环境中确定运行、临时 report + assets、无仓库污染；wheel 使用等价 API smoke |
| build/install | `tests/test_distribution.py` 或 CI | wheel/sdist 分别 build、离线 isolated install、site-packages import，以及目标 venv console/module/CSV smoke |
| Excel boundary | `tests/test_distribution.py` 或 CI | offline metadata、core CSV、可行时 local-wheelhouse extra smoke |
| metadata/docs | tests 或 CI | version、license/README inclusion、Python declaration、changelog、entry point 与 metadata 一致 |

Task 14 不创建 Task 13 风格的 fault matrix。对可选 dependency、subprocess 或 clean-install 的失败，测试断言可操作的 existing error/installation evidence；不新增公共稳定异常。

## CI 范围与安全边界

Task 14 仅允许新建或修改 `.github/workflows/ci.yml`；当前不存在该文件是允许新建它的理由，禁止新增任何其他 workflow 文件。CI 使用 `ubuntu-latest`，Python matrix 为 `3.10`、`3.11`、`3.12`、`3.13`，与现有 `requires-python` 和 classifiers 一致。一个或少量 jobs 必须覆盖 checkout、Python setup、package/test dependency install、pytest、Ruff check、Ruff format check、build、distribution tests 和 examples smoke；可在一个主 Python 版本运行 build/distribution/examples，以避免重复。job/step 名及内部 helper 不冻结。

workflow 必须声明：

~~~yaml
permissions:
  contents: read
~~~

允许的触发只有常规验证 `push`、`pull_request`、`workflow_dispatch`。CI 不得授予 `id-token: write`、`contents: write`、`packages: write`、`actions: write` 或 release 权限；不得使用 PyPI/GitHub release token 或 publishing secrets，运行 `twine upload`、trusted publishing、tag/release/automatic commit/push、deployment environment 或 package-index upload。测试 artifact upload 可选，且不是门禁。

## LICENSE、CHANGELOG 与确定性

LICENSE 已存在且为 MIT；Task 14 只核验，不改写。仅当 LICENSE 缺失或与 metadata 明确不一致时报告 blocker，不擅自替换文本。

允许更新 `CHANGELOG.md`。`0.1.0` 条目至少概括 data loading/schema/quality/summary、feature workflow、classification/regression、evaluation/plots、完整 workflow、Markdown/HTML reports、CLI 与 release validation。不得虚构或使用未来发布日期、复制完整 commit history、记录内部审查轮次/P1/P2/Codex/模型/私有过程、声称发布 PyPI 或写入未实现能力；可使用无日期 `## 0.1.0` 标题。

示例 fixture、output path、random state 和 expected CLI text 必须确定。Task 14 文档/tests 可调用 public package API、subprocess CLI 和 packaging tools；不得令低层模块依赖 tests、docs、workflow/reporting/CLI，或让 examples 成为 runtime package dependencies。

## 文件 allowlist

Task 14 实现仅可修改或创建以下文件：

~~~text
README.md
CHANGELOG.md
LICENSE (verification only; do not modify by default)
docs/quickstart.md
docs/analysis-guide.md
docs/leakage.md
docs/api.md
docs/decisions/task14-release-readiness-contract.md
examples/basic_analysis.py
examples/baseline_modeling.py
tests/test_public_api.py
tests/test_distribution.py
.github/workflows/ci.yml
pyproject.toml
SPEC.md
IMPLEMENTATION_PLAN.md
AGENTS.md
~~~

`src/sharper/` is not allowed to change: Task 14 audits, but does not alter, exports or behavior. No other `src/`, `tests/`, example, docs, or workflow file is allowed. Do not modify Tasks 03--13 decision contracts, dependency lock files, generated artifacts/cache files or `docs/.DS_Store`; do not add “other necessary files”, “related docs”, “other examples” or “necessary CI files” as an exception.

## Acceptance gates 与后续边界

Before Task 14 is complete, the project-local interpreter must pass:

~~~
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
git diff --check
git diff --cached --check
~~~

The approved CI matrix must additionally provide the frozen clean-install, distribution, examples and CLI smoke evidence. A release may proceed only after documentation, examples, distribution metadata and all release gates agree. Publishing, tagging and post-`0.1.0` API evolution are outside this task.
