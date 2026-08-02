# Sharper Implementation Plan

## 1. 计划边界

Tasks 01--14 记录 `SPEC.md` 已批准并完成的 v0.1；v0.2 roadmap 已通过 review，
Task 15 implementation 已完成且 bounded closure review 为 `Go`；Task 16 contract 已批准为
`Approved — Go`，implementation 已完成且 review 为 `Go`；Task 17 contract 已批准为
`Approved — Go`，implementation 尚未开始。Tasks 18--20 contracts 和 implementation 均未开始；
v0.2 整体尚未完成或发布，当前 package version 仍为 `0.1.0`。
v0.1 保持以下完整轻量闭环：

```text
CSV/Excel
  -> schema 与数据画像
  -> 数据质量
  -> 分析挖掘与特征建议
  -> 任务型可视化
  -> 可选分类/回归基线
  -> Markdown/HTML 报告
  -> Python workflow 与 CLI
```

Tasks 01--14 不实现 v0.2/v0.3 能力。v0.2 只按
`docs/decisions/v02-roadmap-contract.md` 和本计划的 Tasks 15--20 推进；新增 tree
model family、数据驱动分箱 transform、group aggregate transformer、target
encoding、WOE 和监督分箱已经延期，不得重新塞入 v0.2 Task。

`SPEC.md` 定义产品定位、模块边界、公共原则和已批准版本能力；本计划是任务拆分和
实现顺序的执行依据。若治理文件出现阶段划分、ownership 或交付顺序冲突，应先同步
并 review 文档，再开始实现。

v0.1 的分发契约冻结为：distribution name `sharper`、import name `sharper`、MIT license、初始版本 `0.1.0`。

专项 skill 按 Task 边界使用：

- `analytics-workflow-builder` 最早用于 Task 03 或 Task 04，不用于 Task 01 或 Task 02。
- `feature-engineering-builder` 不用于 Tasks 01–04；仅从 Task 09 特征工程工作开始使用。
- `visualization-system-builder` 不用于 Task 01 或任何尚未进入可视化工作的 Task；仅从 Task 10 可视化工作开始使用。

## 2. 每个任务的共同完成规则

每个任务必须：

1. 作为单一、可独立 review 的提交完成，不夹带后续任务或无关重构。
2. 同时提交实现、对应测试和该任务影响到的最小文档更新。
3. 为新增 public API 提供完整 type hints、docstring、异常和副作用说明。
4. 不静默修改调用者的 DataFrame。
5. 在运行任何 Python 验证命令前执行 `bash scripts/verify-uv-env.sh`，并只使用
   `.venv/bin/python` 作为控制解释器（临时 distribution clean-install venv 除外；
   详见 `AGENTS.md`）。
6. 运行并通过：

   ```bash
   .venv/bin/python -m pytest
   .venv/bin/python -m ruff check .
   .venv/bin/python -m ruff format --check .
   ```

7. 涉及打包、CLI、public exports 或发布时，额外运行：

   ```bash
   .venv/bin/python -m build --no-isolation
   ```

8. 若任务依赖的前置任务尚未合并，不得通过临时重复实现绕开依赖。
9. Tasks 15--20 必须分别遵循 contract -> contract review -> implementation -> diff
   review -> final Go；roadmap 和本计划不代替单 Task 的精确决策记录，也不允许跨
   Task 提前实现后续能力。

下面的“文件”是该任务允许创建或修改的主要范围。`README.md`、`docs/api.md` 或 docstring 只在 public behavior 改变时做局部同步。

## 3. 有序任务

### Task 01 — 包骨架、工具配置与最小公共导入契约

**目标**

建立可安装、可导入、可 lint、可测试的最小 `src` layout；实现版本与 `__all__` 契约，并冻结分发名、import 名、license、Python 支持范围和依赖分组。Task 01 不冻结任何领域结果类型。

**依赖**

无。

**创建/修改文件**

- `pyproject.toml`
- `src/sharper/__init__.py`
- `tests/test_public_api.py`
- `tests/test_distribution.py`
- `README.md`
- `CHANGELOG.md`
- `LICENSE`（MIT）

**Public API**

- 包可通过 `import sharper` 导入。
- `sharper.__version__` 存在且初始值为 `"0.1.0"`，并作为项目版本的单一来源。
- 本任务不新增领域函数。
- 仅建立明确的最小 `__all__` 机制；不得提前导出尚未实现的符号或后续 Task 的结果类型。

**明确不属于本任务**

- `SchemaReport`、`DataFrameSummary`、`QualityReport` 或其他领域结果类型。
- 自定义异常体系或 `exceptions.py`；v0.1 默认使用 `OSError` 和 `ValueError`。
- `cli.py`、CLI script entry point、CLI help。
- CSV I/O、schema inference、`summarize_dataframe`、`check_data_quality`。
- analytics、feature engineering、visualization、modeling、workflow 或 report generation。

**测试文件**

- `tests/test_public_api.py`
- `tests/test_distribution.py`

**pytest 覆盖点**

- 从已安装包而非仓库根目录导入 `sharper`。
- `__version__ == "0.1.0"`，且分发 metadata 版本一致。
- `__all__` 不包含内部 `_types` 或未实现名称。
- 项目 metadata（包括 `sharper` 名称与 MIT license）、Python 3.10+ 声明、核心/optional/dev 依赖分组正确。
- sdist 和 wheel 均能构建并在干净环境导入。
- 核心安装不强制安装 Excel optional dependency。

**验收标准**

- `python -m build` 成功生成 sdist 与 wheel。
- clean install 后 `import sharper` 成功。
- pytest 与 Ruff 三条质量命令通过。
- 未创建 `cli.py`、`exceptions.py`、业务实现占位函数、领域结果类型、lock file 或无用途抽象。
- 本任务不运行或验收 `sharper --help`；CLI 入口与 help 在 Task 05 验收。

---

### Task 02 — CSV 读取边界

**目标**

提供可靠的本地 CSV 输入，建立所有后续流程统一的 pandas DataFrame 边界。

**依赖**

Task 01。

**创建/修改文件**

- `src/sharper/io.py`
- `src/sharper/__init__.py`
- `tests/test_io.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
load_csv(path: str | Path, **read_options: Any) -> pd.DataFrame
```

只承诺文档列出的常用 pandas 读取选项；不将 pandas 的全部未来参数视为 Sharper 稳定契约。

**测试文件**

- `tests/test_io.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 正常 UTF-8 CSV、显式分隔符和 dtype 选项。
- 空文件、缺失文件、目录路径、坏格式和不支持选项。
- 列名和数据值不被隐式清洗或改写。
- 返回值类型为 DataFrame。
- public export、签名、type hints 和 docstring。

**验收标准**

- 代表性 CSV 能以确定结果读取。
- 错误保留底层因果链并给出可操作消息。
- 除读取文件外无副作用。

---

### Task 03 — Schema 推断与 DataFrame 摘要

**目标**

实现物理 dtype 与逻辑角色分离，并产出稳定的数据集级和列级画像。

**依赖**

Task 01；测试可使用 Task 02 的读取结果，但领域实现不得依赖 `io`。

**冻结合同**

实现前必须遵守已接受的
`docs/decisions/task03-schema-summary-contract.md`。该记录冻结
`ColumnSchema`、`TargetCandidate`、`SchemaReport`、`DataFrameSummary`、
逻辑类型、`column_summary` 明细 schema、推断优先级、rate 分母、空
DataFrame 和 target candidate 行为。若实现需要改变字段或规则，必须先
更新并评审决策记录与 `SPEC.md`，不得在代码中自行扩展。

**创建/修改文件**

- `src/sharper/_types.py`（仅在 schema 契约确有跨模块小型类型别名或枚举需要时创建）
- `src/sharper/schema.py`
- `src/sharper/summary.py`
- `src/sharper/__init__.py`
- `tests/fixtures/` 中的共享小型数据夹具
- `tests/test_schema.py`
- `tests/test_summary.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
infer_schema(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    id_threshold: float = 0.98,
) -> SchemaReport

summarize_dataframe(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
) -> DataFrameSummary
```

实现已冻结的 `ColumnSchema`、`TargetCandidate`、`SchemaReport` 和
`DataFrameSummary` 最小公开字段。

这些结果类型首次在本任务实现，不得增加决策记录之外的字段，也不得在
Task 01 预先定义公共占位类型。

**测试文件**

- `tests/test_schema.py`
- `tests/test_summary.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- numeric、categorical、datetime、boolean、text-like、ID-like、unknown。
- pandas nullable dtype、混合 object、日期字符串、全缺失列和常量列。
- ID 阈值边界、显式 target、target candidate 只提示不确认。
- 重复列名、0 行有列、0 行 0 列和非法阈值。
- 非字符串列名稳定报错，消息包含
  `DataFrame column names must all be strings`，且不发生列名转换。
- shape、memory、missing、unique、分位数与 pandas 基准一致。
- `column_summary` 的 17 个冻结列名、顺序、数值适用范围和 pandas
  缺失语义，以及普通、0 行有列和 0 行 0 列结果的固定 dtype。
- rate 的非缺失分母、零分母行为、全缺失列非 constant 语义。
- 传入 schema 与 DataFrame 不匹配时的可操作 `ValueError`。
- pandas bool 与 object/string/category 纯布尔值使用各自冻结的
  confidence 和 reason code。
- object/string/category/StringDtype 中仅由大小写不敏感、忽略前后空格的
  `"true"`/`"false"` token 组成时判为 boolean；字符串 `"0"`/`"1"`、
  数字 `0`/`1` 和全缺失列均不得因此判为 boolean。
- 全唯一 mixed object 仍以 `mixed_object_unknown` 判为 unknown；即使
  列名包含 id/uuid/key 或唯一率达到 identifier 阈值，也不得判为
  identifier。
- explicit target、精确/包含名称信号、identifier 排除和各 task type
  使用冻结的 confidence 映射与 reason codes；缺失显式 target 的错误消息
  包含 `target column not found`。
- 相同输入产生稳定角色、固定 confidence、封闭 reason codes 和排序。
- 输入 DataFrame 不变。

**验收标准**

- 混合类型夹具得到可解释且确定的 schema/summary。
- target candidate 不触发任何 target-aware 行为。
- 0 行或 0 列输入返回字段和明细 schema 完整的确定结果，不因空输入崩溃。
- 公开 dataclass 字段、逻辑类型集合和明细表 schema 与 Task 03 决策记录
  完全一致。
- 实现不得产生决策记录之外的 confidence 值、reason code 或弱 target
  信号。
- Task 03 不包含 duplicate rows、`QualityReport`、outlier、correlation
  或 target relationship。
- 结果类型可被后续 quality、workflow 和 reporting 消费，无 renderer 逻辑。

---

### Task 04 — 固定数据质量规则

**目标**

实现 v0.1 全部固定质量规则，只报告问题和建议，不自动修复数据。

**依赖**

Task 03。

**冻结合同**

实现前必须遵守已接受的
`docs/decisions/task04-quality-contract.md`。该记录冻结
`QualityIssue`、`QualityReport`、severity、issue code、函数签名、规则
阈值与分母、空输入、互斥和稳定排序。若实现需要改变字段、文本或规则，
必须先更新并评审决策记录与 `SPEC.md`，不得在代码中自行扩展。

**创建/修改文件**

- `src/sharper/quality.py`
- `src/sharper/__init__.py`
- `tests/test_quality.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
check_data_quality(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    missing_threshold: float = 0.40,
) -> QualityReport
```

实现决策记录冻结的 `QualityIssue` 和 `QualityReport` 字段，不得增加
schema、summary、时间戳、路径或随机 ID。

`QualityIssue` 与 `QualityReport` 已由 Task 04 决策记录冻结，并首次在
本任务实现；不得在更早 Task 定义公共占位类型。

**测试文件**

- `tests/test_quality.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 重复行、全缺失、高缺失、常量、近常量、高基数、疑似 ID。
- 数值 inf、混合 Python 类型、datetime 解析失败提示。
- 阈值正好命中、略高/略低及非法阈值。
- 0x0、0 行有列、全缺失、重复列名、非字符串列名和无问题数据。
- `missing_threshold` 的 `(0, 1]` 边界和冻结错误消息。
- 传入 schema 的 shape、列名和顺序匹配验证。
- duplicate rows 使用 `duplicated(keep=False)` 的计数、比例和 NaN 语义，
  且不检查重复 index 或业务实体。
- all-missing/high-missing、all-missing/constant、
  constant/near-constant、identifier/high-cardinality 的互斥。
- near-constant 固定 `0.95` 边界；categorical 高基数同时满足
  `unique_count > 50` 与 `unique_rate > 0.50`。
- issue code、严重度、count/ratio/threshold、稳定排序和冻结建议文本。
- severity counts 始终按 `info`、`warning`、`error` 包含全部 key。
- 不静默删除、填充、转换或修改输入。

**验收标准**

- `QualityIssue` 和 `QualityReport` 的 dataclass 字段、顺序与类型完全符合
  Task 04 决策记录。
- 每条冻结质量规则均有独立参数化测试，且只产生批准的 11 个 issue code
  和 3 个 severity。
- 相同数据产生稳定 issue codes、证据、文本及 table/code/原列顺序排序。
- 0x0 和 0 行有列只产生对应的 `empty_dataframe`；无问题结果具有空
  `issues`、零 `issue_count` 和三个零值 severity counts。
- 未传入 schema 时可复用 `infer_schema`；不调用
  `summarize_dataframe`，不在 `QualityReport` 中嵌入 schema 或 summary。
- 无 outlier、correlation、target relationship、特征、绘图、建模、
  评估、报告、CLI、自动清洗、重复 index/实体、leakage 或业务规则检查。
- 无自动修复或隐藏 schema 重写，输入 DataFrame 不变。

---

### Task 05 — Minimal workflow, Markdown report, and CLI vertical slice

**目标**

形成第一个可运行闭环：CSV → schema/summary/quality → `AnalysisRun` → Markdown。CLI 与 Python 必须使用同一个 workflow。

**依赖**

Tasks 02–04。

**冻结合同**

实现前必须遵守已接受的
`docs/decisions/task05-workflow-report-cli-contract.md`。该记录冻结
`AnalysisRun`、`ReportArtifact`、两个 public function、Markdown 章节与
格式、CLI 参数、stdout/stderr、exit code、help 和错误行为。若实现需要
改变字段、文本或行为，必须先更新并评审决策记录与 `SPEC.md`，不得在
代码中自行扩展。

**创建/修改文件**

- `src/sharper/workflow.py`
- `src/sharper/reporting.py`
- `src/sharper/cli.py`
- `src/sharper/__init__.py`
- `pyproject.toml`
- `tests/test_workflow.py`
- `tests/test_reporting.py`
- `tests/test_cli.py`
- `tests/test_public_api.py`
- `README.md`

**Public API**

```python
run_analysis(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    task: str | None = None,
    include_model: bool = False,
    id_columns: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    random_state: int = 42,
) -> AnalysisRun

generate_analysis_report(
    run: AnalysisRun,
    output_path: str | Path,
    *,
    title: str = "Sharper Analysis Report",
    format: str = "markdown",
    overwrite: bool = True,
) -> ReportArtifact
```

本任务只实现 `format="markdown"`；请求 HTML 必须使用决策记录冻结的
`ValueError`，不得生成不完整文件。`AnalysisRun` 只包含决策记录冻结的
schema、summary、quality、调用参数、skipped 和 warnings 字段，不提前
保留后续分析、图表、特征、模型或评估结果字段。

CLI：

```text
sharper analyze INPUT --output report.md
```

使用 Typer。`src/sharper/cli.py`、`[project.scripts]` 入口和 CLI help
首次在本任务实现；Task 01 不创建或验收这些内容。Task 05 参数、默认值、
输出通道和 exit code 以决策记录为准，不提供 model、HTML、plot、feature
或 debug 选项。

**测试文件**

- `tests/test_workflow.py`
- `tests/test_reporting.py`
- `tests/test_cli.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- `AnalysisRun`、`ReportArtifact` 的 frozen dataclass 字段、顺序和类型。
- 无 target 的 `AnalysisRun` 包含 schema、summary、quality 和固定 skipped。
- target/task、`include_model=True`、random state、id/exclude columns、
  缺失列、重复参数和重叠参数的冻结验证与错误消息。
- target candidate 不自动启用 target-aware 分析。
- Markdown 标题清理、固定章节/表格顺序、值格式、warnings/skipped、
  UTF-8 和单个末尾换行。
- 无 quality issue 文本和有 issue table。
- Markdown-only、输出目录创建、目录拒绝、覆盖策略和 I/O 失败。
- CLI 两级 help、正常退出、非法路径/format、no-overwrite，以及
  stdout/stderr 和 0/1/2 exit code。
- CLI 与 Python 对同一 CSV 产生相同章节集合。
- 输入 DataFrame 不变，Tasks 01–04 回归测试继续通过。

**验收标准**

- Python API 和 CLI 都能生成可阅读 Markdown 质量报告。
- CLI 不复制 schema、quality 或报告算法，只调用 workflow/reporting。
- `sharper --help` 和 `sharper analyze --help` 可用。
- Task 05 不包含 analysis、feature engineering、visualization、modeling、
  evaluation、HTML、dashboard、registry、plugin 或 batch processing。

---

### Task 06 — Excel single-sheet I/O

**目标**

补齐 v0.1 第二种输入格式，保持 Excel 引擎为 optional dependency。
Task 06 只提供 Python API 读取本地 `.xlsx` 单 sheet，不修改 CLI、
workflow 或 reporting。

**API 决策记录**

必须遵守 `docs/decisions/task06-excel-io-contract.md`。该记录冻结
`load_excel` 签名、`.xlsx` 单 sheet 范围、`read_options` 白名单、
optional `excel` extra、错误类型和稳定消息、public export 以及测试
合同。若实现需要改变这些行为，必须先同步评审该记录、`SPEC.md` 和
`IMPLEMENTATION_PLAN.md`。

**依赖**

Tasks 01、02。

**创建/修改文件**

- `src/sharper/io.py`
- `src/sharper/__init__.py`
- `pyproject.toml`
- `tests/test_io.py`
- `tests/test_public_api.py`
- `README.md`
- `docs/api.md`

**Public API**

```python
load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    **read_options: Any,
) -> pd.DataFrame
```

**测试文件**

- `tests/test_io.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 默认首个 sheet、按名称和索引选择 sheet。
- `sheet_name=None` 以及 list、tuple、set 必须拒绝。
- 非 `.xlsx` 后缀、缺失 sheet、坏文件、缺失文件、目录和非法路径。
- 未安装 Excel extra 时抛出 `ImportError`，消息包含
  `Install sharper[excel] to read Excel files`。
- 不支持多 sheet 返回 dict；该用法必须拒绝。
- `read_options` 仅允许 `header`、`names`、`usecols`、`dtype`、
  `na_values`、`keep_default_na`、`skiprows`、`nrows`。
- `sheet_name` 是显式 keyword-only 参数，不属于 `read_options` 合同；
  Python duplicate-keyword `TypeError` 不是 Sharper public error
  contract，也不要求测试。
- `read_options` 中的 `engine` 必须以稳定消息拒绝。
- 不调用 `infer_schema`、`summarize_dataframe` 或 `check_data_quality`。
- CSV-only 核心安装仍可导入和运行。

**验收标准**

- 安装 `excel` extra 后能读取一个本地 `.xlsx` 单 sheet 并返回
  `pd.DataFrame`，保留 pandas 原始列名和值。
- 未安装 extra 不影响包导入、CSV 或其他功能。
- 不增加多表联合分析能力。
- 不修改 `src/sharper/cli.py`、`src/sharper/workflow.py` 或
  `src/sharper/reporting.py`；不实现 Excel CLI、workflow/reporting
  集成、schema、summary、quality、清洗、写入或 HTML。

---

### Task 07 — Non-target feature analysis

**目标**

实现 v0.1 核心 non-target feature analysis，超越简单 `describe()`，并遵守
固定计算预算。Task 07 只包含 numeric feature analysis、categorical
feature analysis、numeric pairwise correlations 和 numeric outlier
detection；不修改 CLI、workflow 或 reporting。

**API 决策记录**

必须遵守 `docs/decisions/task07-analysis-contract.md`。该记录冻结
Task 07 的四个 public function 签名、四个 result dataclass 字段、
输出表 schema、skipped reason vocabulary 与 precedence、错误行为、
deterministic ordering、public export 和测试合同。若实现需要改变这些行为，
必须先同步评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。

**依赖**

Task 03。Task 07 基于 pandas dtype 自动选择列，不调用 `infer_schema`、
`summarize_dataframe` 或 `check_data_quality`。

**创建/修改文件**

- `src/sharper/analysis.py`
- `src/sharper/__init__.py`
- `tests/test_analysis.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
analyze_numeric_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> NumericAnalysis

analyze_categorical_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    top_n: int = 10,
) -> CategoricalAnalysis

compute_correlations(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    max_columns: int = 50,
    min_periods: int = 2,
) -> CorrelationAnalysis

detect_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> OutlierAnalysis
```

签名和 `NumericAnalysis`、`CategoricalAnalysis`、`CorrelationAnalysis`
与 `OutlierAnalysis` 的字段必须与 Task 07 决策记录完全一致。

**测试文件**

- `tests/test_analysis.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- public API exports、签名、result dataclass frozen behavior 和字段顺序。
- shared input validation：non-DataFrame、非字符串 DataFrame column names、
  重复 DataFrame column names、missing requested column、重复 requested
  column、非字符串 requested column 和输入不变性。
- Numeric analysis：自动选择 numeric non-boolean columns、显式 columns
  顺序、跳过 non-numeric/all-missing、固定 summary columns/dtypes、
  `zero_count`/`zero_rate` 和空结果 schema。
- Categorical analysis：自动选择 object/string/category/bool columns、显式
  columns 顺序、跳过 numeric/all-missing、非法 `top_n`、固定
  summary/top_categories columns/dtypes、`top_n` budget 和 first-appearance
  tie break。
- Correlation：自动选择 numeric non-boolean columns、显式 columns 顺序、
  非法 method/`max_columns`/`min_periods`、跳过
  non-numeric/all-missing/constant/insufficient columns、`max_columns`
  truncation、`exceeds_max_columns`、long-form pair order、`n_pairs`、
  `min_periods`、无 diagonal rows 和空结果 schema。
- Outlier detection：只支持 `iqr`、非法 `threshold`、跳过
  non-numeric/all-missing/constant/insufficient/non-finite columns、IQR
  lower/upper bound、summary/outliers schema、原始 `row_index` label、
  deterministic ordering 和无 outlier 空结果 schema。
- Task 01-06 回归测试仍通过，且无 workflow/reporting/CLI changes。

**验收标准**

- 所有结果记录有效样本量、缺失处理、截断和 skipped reasons，且只使用
  决策记录冻结的 skipped reason codes 与 precedence。
- 所有输出表的 columns、dtypes 和 deterministic ordering 与决策记录一致。
- 不删除异常值，不修改输入。
- 不实现 target relationship analysis、grouped analysis、feature
  engineering、visualization、modeling、evaluation、report generation、
  workflow integration、CLI integration、automatic cleaning、data mutation 或
  custom exceptions。
- workflow 可在后续任务接入结果而无需改变 public result contract。

---

### Task 08 — 分组比较与 Target Relationship Analysis

**目标**

补齐普通分组比较，以及分类/回归两条明确分离的 target relationship 分析路径。

**API 决策记录**

必须遵守 `docs/decisions/task08-group-target-analysis-contract.md`。该记录冻结
Task 08 的两个 public function 签名、两个 frozen result dataclass 字段、
全部 output DataFrame schemas/dtypes、四条统计路径、effect sizes、预算、
skipped reason vocabulary/precedence、missing/constant/infinity/small-sample
行为、stable errors、deterministic ordering、public exports 和测试合同。
若实现需要改变这些行为，必须先同步评审该记录、`SPEC.md` 和本计划。

**依赖**

Task 07。

**创建/修改文件**

- `src/sharper/analysis.py`
- `src/sharper/__init__.py`
- `tests/test_analysis.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
compare_groups(
    df: pd.DataFrame,
    group_by: str,
    *,
    values: Sequence[str] | None = None,
    max_groups: int = 20,
) -> GroupComparison

analyze_target_relationships(
    df: pd.DataFrame,
    target: str,
    *,
    task: Literal["classification", "regression"],
    features: Sequence[str] | None = None,
) -> TargetAnalysis
```

**测试文件**

- `tests/test_analysis.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- public exports、签名、frozen dataclass 字段顺序和完整 fixed table schemas/
  dtypes，包括 empty results。
- 单 categorical group key 下的 count、missing count、mean、median、分位数、
  missing group 披露、20-group budget、频数排序和 first-appearance tie break。
- classification × numeric Kruskal-Wallis/epsilon squared；classification ×
  categorical Chi-square/Cramér's V 和通用 `category_details` rate contract。
- regression × numeric Pearson/absolute Pearson r；regression × categorical
  target summary、Kruskal-Wallis/epsilon squared。
- target/category/feature budgets、全部 skipped reasons 和 precedence。
- missing、constant、infinity、insufficient sample/group、invalid task/column/
  budget 和全部 stable messages。
- 固定 `TASK08_MIN_GROUP_SIZE=2`、两条 Kruskal retained-group 规则、
  complete-case category budget 和 classification zero-count Cartesian cells。
- SciPy/effect-size 非有限结果的 `statistical_test_not_applicable`、封闭
  limitations vocabulary/顺序及精确 row limitation 文本。
- deterministic ordering、输入不变性、不调用 Task 07 public analysis
  functions 和 Tasks 01-07 回归。
- Task 08 real-numeric predicate、complex group value/target/feature behavior，
  以及 bool group、20/21 category boundary、前置 skip 后 50/51 eligible budget、
  duplicate index、mixed hashable categories 和 non-2×2 Chi-square。

**验收标准**

- 无 target 的普通分组分析可独立调用。
- 分类与回归行为不通过猜测 target dtype 选择；必须使用显式 task。
- 不训练模型、不生成因果结论、不自动进行多重模型式搜索或多重比较校正。
- Task 08 不修改 workflow、reporting、CLI、I/O 或 Task 07 合同；完整集成留给
  Task 13。

---

### Task 09 — Feature suggestions and safe stateless derivation

**目标**

按照已接受的 `docs/decisions/task09-feature-engineering-contract.md` 实现确定性、
有预算、可解释的候选建议，以及 v0.1 白名单内的 arithmetic、datetime
component 和显式 reference-date days-since 无状态物化。Fitted、target-aware、
aggregate candidates 只建议，不物化。

**依赖**

Task 03 schema contracts 和 `infer_schema`。Task 07 只是 sequencing
prerequisite；Task 09 不 import/call Task 07 或 Task 08 public analysis API，
不实现 correlation heuristic。

**创建/修改文件**

- `src/sharper/features.py`
- `src/sharper/__init__.py`
- `tests/test_features.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
suggest_feature_derivations(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    target: str | None = None,
    exclude_columns: Sequence[str] = (),
    reference_date: str | date | datetime | pd.Timestamp | None = None,
    max_suggestions: int = 50,
) -> FeatureSuggestionReport

derive_features(
    df: pd.DataFrame,
    suggestions: Sequence[FeatureSuggestion],
    *,
    copy: bool = True,
) -> FeatureDerivationResult
```

公开并严格遵守决策记录冻结字段的 `FeatureSuggestion`、
`FeatureSuggestionReport` 和 `FeatureDerivationResult`；三个类型均为
`dataclass(frozen=True)`。不得新增其他 feature public API。

**测试文件**

- `tests/test_features.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- exact signatures、type hints/docstrings、exports、三个 frozen dataclass 的字段/
  顺序/类型。
- shared DataFrame、schema、target、exclusions、reference date、budget validation
  及稳定错误和 precedence。
- real numeric/timezone-naive datetime/categorical eligibility；target、explicit exclusions、
  identifier、all-missing、constant、unsupported dtype 和 exact duplicate-content
  exclusion precedence；三种 column states 构成完整 partition，duplicate comparison
  只在包含 unsupported dtype 的六步前置过滤后的候选域中执行，并断言
  `unsupported_dtype` 优先于 `duplicate_content`。
- datetime、ratio、difference、product、binning/group aggregate/target encoding
  candidates 的封闭 vocabulary、risk/reason/requires-fit 映射；六种 datetime
  types 均固定 priority 1，稳定生成顺序不变。
- 固定 per-type budgets、global 49/50/51 boundaries、pair direction、命名、冲突
  过滤、identity 去重、priority 和 stable ordering。
- 显式 reference date 的 Timestamp/datetime/date dispatch、规范化、timezone-aware
  拒绝、无 eligible datetime column metadata 及无 current-date dependency；
  timezone-aware datetime source 拒绝、Monday=0/Sunday=6 与 weekend 定义。
- requires-fit、unsupported type、missing source、每种 materializable/suggestion-only
  canonical fields、output collision、Sequence container 和 copy 的 fail-fast
  validation；source count、existence、dtype compatibility 固定置于 canonical
  fields 前，所有 validation 在临时计算前完成。
- arithmetic source 先转 `float64`、large-integer overflow、zero/missing/non-finite
  行为；datetime extension dtypes、missing 与正负 days-since。
- index/原列/dtype 保留、新列追加顺序、`copy=True`/`copy=False`/empty
  suggestions、unexpected computation failure transaction-like atomicity，以及
  pandas `deep=True` 不递归复制 object cells；正常结果 skipped fields 为空。
- target values 和 held-out-only category/extreme 不影响 materialized formulas；
  不调用 Task 07/08 analysis。

**验收标准**

- v0.1 不出现 learned/fixed binning、group aggregate、target encoding、WOE、
  监督分箱、target-aware materialization、fitted transformer 或 correlation-based
  search。
- 候选数量遵守冻结的 per-type/global hard budgets，相同输入顺序稳定并披露
  budget 结果。
- 每个物化特征可追溯到 canonical formula、source columns 和 parameters。
- 不修改 workflow、reporting、CLI、I/O、analysis、pyproject dependency groups
  或 Tasks 01–08 contracts。
- 项目本地 pytest、Ruff lint、Ruff format check 和 build 全部通过。

---

### Task 10 — 数据分析任务型可视化

**目标**

为非模型分析生成 report-ready Figure；统计分析型图表优先使用 seaborn，
matplotlib 作为底层 backend、Figure/Axes 契约和低级 fallback。绘图消费数据
或既有分析结果，不隐藏重算统计。

**API 决策记录**

必须遵守 `docs/decisions/task10-visualization-contract.md`。该记录冻结六个
public function 签名、`PlotResult`/`PlotCollection` 字段、每类图的数据来源和
算法、预算、metadata、排序、错误、空结果、Figure 生命周期、全局状态与测试
合同。Task 09 仅是 sequencing prerequisite，不是 Task 10 实际 API 依赖；不得
绘制 feature suggestions。

**依赖**

Tasks 07–09。

**创建/修改文件**

- `src/sharper/visualization.py`
- `src/sharper/__init__.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`
- `docs/api.md`
- `README.md`
- `docs/decisions/task12-regression-baseline-evaluation-visualization-contract.md`
- `SPEC.md`
- `IMPLEMENTATION_PLAN.md`
- `AGENTS.md`
- `README.md`

**Public API**

```python
def plot_distributions(
    df: pd.DataFrame,
    *,
    max_plots: int = 20,
    sample_size: int = 10_000,
) -> PlotCollection: ...

def plot_missingness(
    df: pd.DataFrame,
    *,
    max_columns: int = 50,
) -> PlotResult: ...

def plot_correlations(result: CorrelationAnalysis) -> PlotResult: ...

def plot_outliers(
    result: OutlierAnalysis,
    *,
    max_plots: int = 20,
) -> PlotCollection: ...

def plot_group_comparison(result: GroupComparison) -> PlotCollection: ...
def plot_target_relationships(result: TargetAnalysis) -> PlotCollection: ...
```

本任务按决策记录冻结 `PlotResult` 和 `PlotCollection` 的全部字段。模型评估图留给 Tasks 11–12。

**测试文件**

- `tests/test_visualization.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 数值直方、类别 top-N、缺失率、相关热图、异常值、分组和 target 图。
- 返回对象包含 Figure、chart type、source、item 和冻结的采样/截断 metadata；
  collection 空结果不创建 skipped plot。
- 最多 20 张图、最多 10,000 行绘图样本、50 列预算。
- 空列、全缺失、常量、高基数、不适用 target 图。
- 不调用 `show()`、不默认写文件、不依赖全局状态。
- 不在函数中随意修改全局 matplotlib 或 seaborn style。
- seaborn 是统计分析型图表的首选实现；matplotlib 用于底层对象契约和
  seaborn 不适用时的低级 fallback。
- 使用 Agg backend；测试结束关闭所有 Figure。
- monkeypatch/spy 证明消费结果的函数不重新调用分析计算。
- 验证 Figure ownership、rcParams/backend/global seaborn state、稳定 metadata、
  参数/结果 schema errors、无 bare Axes、确定性颜色和输入不变性。

**验收标准**

- 每种批准任务至少能生成一个可保存 Figure，或按决策记录返回空 collection / 正常空 Figure。
- 所有抽样和截断在返回元数据中披露。
- 不建立多可视化后端系统；无 Plotly、Altair、Bokeh、交互式 dashboard
  或主题系统。
- 不修改 workflow、reporting、CLI、I/O、analysis、features、`pyproject.toml`、
  dependency groups 或 Tasks 01–09 contracts；不保存文件。

---

### Task 11 — 分类基线、分类评估与评估图

**目标**

实现严格 split-first 的分类基线及独立评估，建立 leakage 回归测试。

**依赖**

Tasks 03、09、10。

**创建/修改文件**

- `src/sharper/modeling.py`
- `src/sharper/evaluation.py`
- `src/sharper/visualization.py`
- `src/sharper/__init__.py`
- `tests/test_modeling.py`
- `tests/test_evaluation.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`
- `docs/leakage.md`
- `docs/api.md`
- `README.md`

**Public API**

```python
train_classifier(
    df: pd.DataFrame,
    target: str,
    *,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
    time_column: str | None = None,
    estimator: ClassifierMixin | None = None,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> TrainingResult: ...
evaluate_classifier(result: TrainingResult) -> ClassificationEvaluation
evaluate_model(result: TrainingResult) -> ClassificationEvaluation | RegressionEvaluation
plot_classification_evaluation(
    result: ClassificationEvaluation,
) -> PlotCollection
```

`evaluate_model` 在本任务只需正确分派 classification；regression 分支在 Task 12 完成。

**测试文件**

- `tests/test_modeling.py`
- `tests/test_evaluation.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- split 后才 fit imputer、encoder、scaler 和 estimator。
- Task 03 schema、ID-like、dtype 与最终 feature eligibility 只从 `X_train` 决定；holdout 不影响 feature order。
- ColumnTransformer 数值/类别路径、未知测试类别和缺失值。
- 默认 Logistic Regression、显式 feature 列和自定义 classifier。
- `exclude_columns` 排除 posterior/future/entity 风险列；`time_column`、datetime 或 timedelta 输入稳定拒绝。
- 单类、小样本、缺 target、非法 split、非 classifier estimator。
- 测试独有类别/极值不进入 fitted state。
- target、后验字段、ID 和 exclude columns 不进入特征。
- 重复行/索引警告；实体重复或时间风险拒绝/警告符合 SPEC。
- holdout 不参与特征、阈值或模型选择。
- 相同 random_state 得到相同 split、列和指标；自定义 estimator 随机性被披露。
- estimator clone/ownership、错误 predict/probability/classes 输出与 result consistency。
- accuracy、balanced accuracy、macro F1、二分类 ROC AUC 和混淆矩阵与 sklearn 一致。
- 多分类和无 `predict_proba` estimator 的适用指标。
- 混淆矩阵/ROC Figure 或明确 skipped reason。

**验收标准**

- leakage 测试能证明 preprocessing state 只来自训练集。
- 评估只消费 holdout，分类指标与回归指标不混用。
- fitted Pipeline、split metadata、schema snapshot、random_state 和 warnings 可审查。

---

### Task 12 — 回归基线、回归评估与评估图

**目标**

在不复制分类流程的前提下完成独立回归路径和任务匹配指标。Task 11 的
`TrainingResult` 保持分类专用；Task 12 冻结独立的 `RegressionTrainingResult`，
并只扩展 `evaluate_model` 的回归分派能力。

**API 决策记录**

必须遵守 `docs/decisions/task12-regression-baseline-evaluation-visualization-contract.md`。
该记录冻结回归训练/评估/绘图签名、keyword-only 参数、
`RegressionTrainingResult`/`RegressionEvaluation` 字段、validation precedence、
holdout-only 数据流、leakage 边界、指标/图/metadata、determinism、stable errors
和测试合同。它不改变 Task 11 的分类结果或分类分支行为。

**依赖**

Task 11。

**创建/修改文件**

- `src/sharper/modeling.py`
- `src/sharper/evaluation.py`
- `src/sharper/visualization.py`
- `src/sharper/__init__.py`
- `tests/test_modeling.py`
- `tests/test_evaluation.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
train_regressor(
    df: pd.DataFrame,
    target: str,
    *,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
    time_column: str | None = None,
    estimator: RegressorMixin | None = None,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> RegressionTrainingResult
evaluate_regressor(result: RegressionTrainingResult) -> RegressionEvaluation
evaluate_model(
    result: TrainingResult | RegressionTrainingResult,
) -> ClassificationEvaluation | RegressionEvaluation
plot_regression_evaluation(
    result: RegressionEvaluation,
) -> PlotCollection
```

**测试文件**

- `tests/test_modeling.py`
- `tests/test_evaluation.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 默认 Ridge 与自定义 regressor。
- 与分类相同的 split-first、未知类别、缺失和 leakage 保护。
- 非数值/缺失/非有限/常量 target、小样本、非法 split 和 classifier 误传。
- MAE、RMSE、R² 与 sklearn metrics 一致。
- 预测/残差表索引和 holdout 标签对齐。
- `evaluate_classifier` 拒绝 regression，`evaluate_regressor` 拒绝 classification。
- `evaluate_model` 严格依据批准 training result 的 frozen `task` 分派。
- 残差图和预测对比图；常量 target 稳定拒绝，valid result 恒返回两张图。
- random_state 下结果可复现。

**验收标准**

- 分类与回归共享内部安全预处理构造，但保持独立训练验证、结果类型和评估契约。
- 不新增树模型、模型搜索、交叉验证或 AutoML。
- 所有回归指标和图只使用 holdout。

---

### Task 13 — 完整 Workflow、静态 HTML 与 CLI 收口

**状态：已完成。** 最终交付完整 analysis workflow、classification/regression
orchestration、无模型 target/task analysis、frozen `AnalysisRun`、Markdown/HTML
report + PNG assets bundle、deterministic staging/backup/commit 与
rollback/compensation、Figure ownership、CSV/XLSX CLI、eager root `--version` 和
no-recomputation。最终验证基线：579 passed；Ruff check passed；Ruff format check
passed；build passed；`git diff --check` passed；`git diff --cached --check` passed。

**目标**

将所有已实现领域结果接入唯一 workflow，完成 Markdown/HTML 报告和完整 CLI 参数。

**依赖**

Tasks 06–12。

**创建/修改文件**

- `src/sharper/workflow.py`
- `src/sharper/reporting.py`
- `src/sharper/cli.py`
- `tests/test_workflow.py`
- `tests/test_reporting.py`
- `tests/test_cli.py`
- `tests/test_public_api.py`
- `README.md`
- `docs/quickstart.md`
- `docs/analysis-guide.md`
- `docs/api.md`
- `docs/decisions/task13-full-workflow-static-html-cli-contract.md`
- `SPEC.md`
- `IMPLEMENTATION_PLAN.md`
- `AGENTS.md`

**API 决策记录与 Public API**

必须遵守已接受的
`docs/decisions/task13-full-workflow-static-html-cli-contract.md`。该记录冻结
Task 13 的正式身份、扩展后的 `AnalysisRun` 字段、`run_analysis` 签名、完整
workflow 数据流、result-only reporting、Markdown/HTML section/asset/file 行为、
CLI、validation、stable errors、determinism、Figure lifecycle、测试合同和本任务
allowlist。Task 13 扩展 Task 05 的 `AnalysisRun`，接入既有 analysis、feature
suggestion、plot、training/evaluation 和嵌套预算结果；不新增 public workflow
class 或配置系统，`ReportArtifact` 字段保持不变。

完整 CLI：

```text
sharper analyze INPUT
  --output PATH
  --format markdown|html
  --target TARGET
  --task classification|regression
  --id-column COLUMN
  --exclude-column COLUMN
  --feature COLUMN
  --time-column COLUMN
  --group-by COLUMN
  --reference-date YYYY-MM-DD
  --max-suggestions INTEGER
  --model / --no-model
  --test-size FLOAT
  --random-state INTEGER
  --overwrite / --no-overwrite
  --debug / --no-debug
```

**测试文件**

- `tests/test_workflow.py`
- `tests/test_reporting.py`
- `tests/test_cli.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 无 target 完整分析：画像、质量、非 target 分析、特征建议和图。
- 显式 `--group-by` 的 group comparison 及其图；未指定时记录固定 skipped。
- 显式 target 但不建模：target relationship 存在、training/evaluation 为空。
- 显式分类/回归建模：对应 training、evaluation 和模型图存在。
- target candidate 永不自动触发 target-aware 分析。
- id/exclude columns、random_state、非法参数组合和缺失列。
- 默认预算以及每项 requested/actual/reason 披露。
- Markdown/HTML 章节一致，HTML 转义和内部静态模板正确。
- 图像资产保存、相对链接、输出目录、覆盖策略、deterministic staging/backup
  rollback 和每个写失败点的最终文件状态；普通 pre-acquisition failure 与
  residual staging/backup conflict 分别测试。前者断言无新文件系统修改、PNG、
  staging 或 backup、Python Figures 保持 open 和精确错误；后者分别覆盖四个
  temporary path 及多个同时残留，断言所有预存路径原样保留、无新路径/PNG、
  final 不变、Python 不关闭 Figure、精确 `FileExistsError`，以及 CLI `finally`
  最终关闭 Figures。
- `AnalysisRun`/nested result/plot/Figure preflight tampering 在任何 I/O 前以
  稳定错误拒绝，且不重算上游结果。
- wrong `run` type（`None`、dict、任意 object 与 exact-type 要求下的 subclass）
  精确抛出 `TypeError("run must be an AnalysisRun")`，不创建输出或关闭 Python
  caller Figures；至少一个 tampered exact `AnalysisRun` 精确抛出
  `ValueError("analysis run has invalid schema")`。
- CLI 与直接 Python workflow 的章节、warnings、skipped 和预算元数据一致。
- Excel CLI、带/不带模型、root eager `--version` 与 subcommand `--version`
  parser error、debug/普通错误输出和退出码。
- 每个 upstream public API 的一次调用、固定顺序、无 workflow/reporting
  recomputation、raw DataFrame 不保留及 Figure 成功/失败后关闭。
- reporting ownership 取得前的失败保持 Python caller Figure ownership；CLI 在
  此阶段失败时负责关闭其 workflow run 的 Figures，取得后不重复关闭。

**验收标准**

- 一条 CLI 命令能对 CSV/Excel 生成可审阅 Markdown 或静态 HTML 报告。
- HTML 不增加第三方 renderer 或交互式依赖。
- CLI 不包含领域算法；reporting 不重算分析；workflow 不读取或写文件。
- 完整流程仍遵守所有 leakage 和预算规则。
- 本任务允许文件精确以 Task 13 决策记录的 allowlist 为准；不得修改
  `src/sharper/__init__.py`、`pyproject.toml`、依赖、上游 contracts 或生成文件。

---

### Task 14 — 文档、示例与 v0.1 发布验证

**状态：已完成（Go）。** 已同步 README、CHANGELOG 和 API 文档，交付 basic 与
baseline examples、public-surface drift tests、wheel/sdist artifact validation、独立
clean-install、package/console/module 三重来源验证、uv offline dependency
installation、CLI/CSV/API/examples smoke、LICENSE/README/Excel extra metadata
验证和 CI workflow；本任务未实际发布。

最终验证基线：580 tests collected；full pytest passed；Ruff check passed；Ruff
format check passed；wheel/sdist build passed；wheel clean-install passed；sdist
clean-install passed；examples passed；CLI smoke passed；`git diff --check` passed；
`git diff --cached --check` passed。

**目标**

让文档只承诺实际实现的 v0.1，并验证源码分发、wheel、示例和 CLI 的发布可用性。

**依赖**

Tasks 01–13。

**创建/修改文件**

- `README.md`
- `CHANGELOG.md`
- `LICENSE`（仅核验；默认不得修改）
- `docs/quickstart.md`
- `docs/analysis-guide.md`
- `docs/leakage.md`
- `docs/api.md`
- `docs/decisions/task14-release-readiness-contract.md`
- `examples/basic_analysis.py`
- `examples/baseline_modeling.py`
- `tests/test_public_api.py`
- `tests/test_distribution.py`
- `.github/workflows/ci.yml`
- `pyproject.toml`（仅 Task 14 contract 冻结的 metadata、package-data/license inclusion 或 distribution-test 配置）
- `SPEC.md`
- `IMPLEMENTATION_PLAN.md`
- `AGENTS.md`

**Public API**

- 不新增 API。
- 审核并冻结 `sharper.__all__`、全部 public signatures、结果 dataclass 字段和 CLI help。

**测试文件**

- `tests/test_public_api.py`
- `tests/test_distribution.py`
- 示例执行可放入 `tests/test_distribution.py` 或 CI，不新增重复领域测试。

**pytest 覆盖点**

- 每个 public symbol 可从文档声明的 import path 导入。
- 每个 public function/class/dataclass 有 type hints 和完整 docstring。
- README 与 examples 只使用 public API。
- 两个示例在小型夹具上确定执行。
- wheel/sdist 分别离线 clean install、site-packages import，以及各自临时 venv 的 console executable/module Python help/version 和最小 CSV 分析。
- sdist 中两个 example 确定执行；wheel 运行等价 installed-public-API smoke，不宣称 examples 被 wheel 包含。
- 缺 Excel extra 时核心流程仍可用；仅在本地依赖可用时离线安装 extra 并执行 Excel smoke。
- distribution metadata、LICENSE/README inclusion、README content type、Python 支持声明和 entry point。

**验收标准**

- Python 3.10--3.13 的 Ubuntu CI matrix 通过 pytest、Ruff；build、distribution、examples 和 CLI smoke 至少在一个主版本提供证据。
- README quickstart、CLI 和 examples 均可执行。
- CHANGELOG 准确列出 v0.1 能力与已知限制。
- 不包含 v0.2/v0.3 API、lock file、构建产物、生成报告、发布/upload 行为或额外 workflow 文件。

---

### Task 15 — Binary Risk Validation and Business Metrics

**状态：Implementation complete — review Go。**

**合同：** `docs/decisions/task15-binary-risk-validation-contract.md`（Approved — Go）。

v0.2 roadmap 与 Task 15 contract 均已批准；Task 15 implementation 已完成且 bounded
closure review 为 `Go`。Task 16 contract 已批准为 `Approved — Go`，implementation 已完成
且 review 为 `Go`；Task 17 contract 已批准为 `Approved — Go`，implementation 尚未开始。
Tasks 18--20 contracts 和 implementation 均未开始。不得跨 Task 提前实现。

**目标**

在保持 v0.1 分类 API 与随机 holdout 默认行为不变的前提下，建立风险型二分类的
validation、OOF、ranking/probability metrics 和基础业务指标语义。

**依赖**

v0.1 Tasks 01--14 稳定基线。Task 15 与 Task 16 是可独立 contract/review 的基础
能力。

**范围**

- 显式 positive label、风险方向和 score provenance；
- 严格区分任意有限 `ranking_score` 与 `[0, 1]` 内、对应显式正类的
  `event_probability`；
- stratified/group/time validation、frozen fold membership 和 OOF predictions；
- time fold 同时执行 `observation_time < fold_cutoff` 与
  `label_available_time <= fold_cutoff`，记录 horizon、reporting delay、成熟、排除和
  purged 行；
- ROC-AUC、准确命名的 PR 指标、normalized Gini、KS、gains、lift 和 capture；
- 仅对 event probability 计算 calibration、Brier/log loss 和 expected-loss primitives；
- 仅在 train/validation/OOF 上比较 caller 预声明的有界 threshold/band 候选并报告
  analytical operating evidence；
- 基础 exposure、成熟 observed loss 与 event-probability expected loss 汇总。

**边界**

不执行贷前规则、贷后预警，不分配业务动作，不消费业务 constraint，不计算
action-dependent profit/payoff，不训练新模型族，不实现 WOE/target encoding 或策略
optimizer，不修改 workflow/reporting/CLI。精确 API、结果、错误和测试由 Task 15
独立合同冻结。

---

### Task 16 — Data Quality and Leakage Audit

**状态：Contract Approved — Go；Implementation complete — review Go。**

**合同：** `docs/decisions/task16-data-quality-leakage-contract.md`（Approved — Go）。

Task 15 implementation 已完成且 review 为 `Go`；Task 16 contract 已批准，implementation
已完成且 review 为 `Go`。Task 17 contract 已批准为 `Approved — Go`，implementation 尚未开始；
Tasks 18--20 contracts 和 implementation 均未开始；v0.2 整体尚未完成或发布，当前 package
version 仍为 `0.1.0`。

**目标**

提供不改变 v0.1 `check_data_quality` 的 opt-in 质量与 leakage audit，并成为 Tasks
17/18 shared condition truth 的唯一 owner。

**依赖**

v0.1 Tasks 03、04、14 稳定合同；与 Task 15 可独立推进。提供 target/score/folds 时
只消费 Task 15 已冻结语义，不重算其 metrics 或 folds。

**范围**

- 特殊值、range、allowed-values、cross-column、时间和结构质量检查；
- missingness profiling/drift：reference/current rates、绝对/相对变化、全缺失/恢复
  列以及 schema/missing-pattern differences；
- suspected ID、near-copy、target proxy 和 post-outcome evidence；
- entity/group/time/point-in-time leakage、availability、outcome support 和 duplicate
  entity-time audit；
- private closed condition kernel：封闭 operators、三值逻辑、Boolean composition、
  missing/effective/expiration semantics、预算、确定性和输入不可变性。

**边界**

只报告 evidence，不自动删列、填值、截尾、重采样、修复或替用户作业务决策。
condition kernel 不导出为 public DSL，不接受 `eval`、任意 Python、callable、函数、
脚本、插件或动态 operator。精确 API 和 private module placement 由 Task 16 独立合同
冻结。

---

### Task 17 — Pre-loan Eligibility Rules and Decision Strategy Simulation

**状态：Contract Approved — Go；Implementation not started。**

**合同：** `docs/decisions/task17-preloan-eligibility-strategy-contract.md`（Approved — Go）。

唯一一次full contract review为`No-Go`；targeted contract fixes已完成；bounded contract closure
为`Go`，P0/P1/P2均为`0`。Task 17合同阶段已完成，不得再次进行开放式Task 17 full contract
review。Task 17 implementation尚未开始；Tasks 18--20尚未开始；v0.2整体尚未完成或发布，当前
package version仍为`0.1.0`。

**目标**

在独立聚焦的 pre-loan policy 模块中执行 caller-defined 准入规则和冻结策略的离线
回放、约束评估与比较，不执行真实审批。

**依赖**

硬依赖 Task 16；使用 score、outcome 或 business metrics 时消费 Task 15 frozen
results。不得依赖 Task 18。

**范围**

- hard/soft/refer、数据完整性、资格、产品、信用政策、偿付能力和敞口等通用规则；
- 规则组合、priority、stop-on-hit、冲突、缺失、effective/expiration 和 version；
- caller-frozen score bands/cutoffs、open/closed bounds、ties 和 fallback；
- approve/decline/refer/request-information 等仅作为 caller symbolic action-name 示例的
  deterministic offline action simulation；
- 显式 `action_name`/`action_role` mapping；无 mapping 时只输出 action distribution；
- manual-review capacity、风险、敞口、budget/rate constraints 的可行性与 gap；
- 仅以 event probability 计算 expected loss/basic payoff，以成熟 observed outcomes
  计算 observed replay；
- rule hits、reasons、base/final action、override、provenance 和相同支持集上的
  champion/challenger 离线比较。

**边界**

只执行 caller 冻结的规则、bands 和 policies；constraints 不生成或改写动作。不建立
任意代码执行规则引擎、通用 DSL、solver 或 optimizer，不自动选择 winner/cutoff，
不声称真实策略收益，不执行审批。只消费 Task 16 kernel，不复制 condition evaluator。

---

### Task 18 — Post-loan Early Warning and Lifecycle Monitoring

**状态：规划已批准，合同未开始。**

**目标**

在独立聚焦的 post-loan monitoring 模块中，对 `customer/account × observation date`
执行 point-in-time 预警、alert backtest 和生命周期分析，不执行账户或催收动作。

**依赖**

硬依赖 Task 16；使用模型分数时可选消费 Task 15。不得依赖 Task 17 或其 action
result。

**范围**

- explicit entity、observation/available/event time、prior-only windows、horizon、
  maturity/censoring 和边界语义；
- level/change/trend/persistence/combination/state-transition/peer/history rules；
- first/repeated alerts、persistence、episodes、cooldown、resolution 和 reopen；
- alert/rule-hit rate、event capture/recall、precision、false-alert share、false-positive
  rate、lead time、burden 和 unresolved/duplicate metrics；
- vintage、cohort age、MOB、roll-rate、roll-forward/back、cure、cohort 和 maturity；
- no-alert、single-threshold、current/challenger rules、model score 和 model+rules 的
  同支持集离线比较。

**边界**

future events 只用于 matured-horizon backtest，不进入当时 signal/rule/alert state。
不自动推荐或执行催收动作，不优化渠道，不建立实时监控服务，不实现反欺诈，也不
把历史关联描述为 alert 的因果效果。只消费 Task 16 kernel，不复制 condition
evaluator；alert/history result 与 Task 17 action result 分离。

---

### Task 19 — Explainability, Champion/Challenger and Governance

**状态：规划已批准，合同未开始。**

**目标**

在独立聚焦的 explanation/comparison/governance owner 中汇总模型、规则、策略和预警
的解释、比较与审计 evidence。

**依赖**

Tasks 15、16、17、18 全部完成并冻结结果。

**范围**

- coefficient、native/permutation importance 和 source-feature provenance；
- 同一 frozen folds/rows 上的 model champion/challenger comparison；
- 消费 Task 17 policy comparison 和 Task 18 warning comparison，形成 comparison
  inventory；
- model/policy/alert reason provenance、reason codes、mapping coverage、rule path、
  versioning、override audit 和 fallback；
- prediction drift、model performance-by-time/group、稳定性与 bounded reproducibility
  audit metadata；不建立 v0.3 的完整 run manifest；
- governance purpose、owner、assumptions、limitations、monitoring evidence 和 issue
  status。

**边界**

只消费 Tasks 15--18 frozen results，不重新计算其 metrics、missingness drift、input
profiles、conditions、rules、actions、alerts、backtests、policy comparisons 或 lifecycle
tables。不生成 adverse-action notice、合规认证、因果结论或自动业务授权。

---

### Task 20 — v0.2 Integration and Release Readiness

**状态：规划已批准，合同未开始。**

**目标**

把 Tasks 15--19 frozen public results 接入独立 opt-in v0.2 workflow、静态报告和 CLI，
完成文档、示例、兼容性、distribution 和 CI readiness，但不实际发布。

**依赖**

Tasks 15--19 全部完成并通过各自合同。

**范围**

- 新的 opt-in workflow/result，不扩充 v0.1 `AnalysisRun`；
- score validation、pre-loan eligibility、post-loan warning 三条独立路径及可组合报告；
- result-only 静态 Markdown/HTML 与既有 Figure/asset ownership；
- 只解析 versioned、closed、pure-data JSON policy/warning spec 的 CLI adapter；未知
  schema version、field/operator、非 JSON、duplicate key 或超预算嵌套明确失败；
- examples、API/guide/leakage 文档和发行准备说明；
- 永久 `v0.1 compatibility invariants` 与 `current release surface` tests；
- pytest/Ruff、wheel/sdist、独立 clean-install、CLI/examples smoke 和 CI readiness。

**边界**

workflow 只编排 public APIs，reporting 不重算，CLI 不含领域算法。CLI 不接受
YAML/TOML、Python、函数、脚本、模板、环境变量或路径展开，不建立通用规则 DSL。
Task 20 不 tag、push、upload、创建 release 或实际发布；版本和 exports 的精确迁移只
能由 Task 20 独立合同授权。

## 4. 依赖顺序总览

```text
01 Packaging
├── 02 CSV I/O
├── 03 Schema + Summary
│   ├── 04 Quality
│   │   └── 05 Minimal Workflow + Markdown + CLI
│   └── 07 Core Analysis
│       ├── 08 Group + Target Analysis
│       └── 09 Feature Discovery
│           └── 10 Analytical Visualization
│               └── 11 Classification
│                   └── 12 Regression
├── 06 Excel I/O
└──────────────────────────────┐
05–12 ────────────────────────> 13 Full Workflow + HTML + CLI
13 ───────────────────────────> 14 Release Readiness
```

Task 06 可在 Task 05 之后与 Tasks 07–09 独立排期，但必须在 Task 13 前完成。其余任务按编号顺序执行最容易保持 public contract 稳定。

v0.2 在稳定 v0.1 基线上使用以下 DAG：

```text
Task 15 ─┐
         ├─> Task 17 ─┐
Task 16 ─┤             ├─> Task 19 ─> Task 20
         └─> Task 18 ─┘
```

Task 15、16 是基础能力。Task 17 硬依赖 Task 16，并在使用 score/outcome/business
metrics 时消费 Task 15；Task 18 硬依赖 Task 16，使用模型分数时可选消费 Task 15。
Tasks 17/18 并列且不相互依赖。Task 19 等待 Tasks 15--18 frozen results，Task 20
等待 Task 19。每个 Task 必须先建立并通过独立 contract review，禁止通过跨 Task
复制逻辑或提前实现绕开依赖。

## 5. v0.1 最终发布门槛

只有同时满足以下条件才视为 v0.1 完成：

- CSV/Excel 均能进入统一 workflow。
- Python API 与 CLI 对同一输入产生一致章节、warnings、skipped 和预算披露。
- 数据画像、质量、分析挖掘、特征建议、任务型图表、可选分类/回归和报告全部存在。
- split-first、Pipeline、ColumnTransformer、holdout-only evaluation 和 leakage 专项测试通过。
- 分类与回归评估严格分开。
- v0.1 只物化批准的无状态特征。
- 所有 public API typed、documented、tested。
- pytest、Ruff、build、clean install、示例和 CLI smoke tests 全部通过。
- 没有实现或承诺 SPEC 中推迟到 v0.2/v0.3 的功能。
