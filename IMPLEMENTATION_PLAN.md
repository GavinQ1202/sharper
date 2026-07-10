# Sharper v0.1 Implementation Plan

## 1. 计划边界

本计划只实现 `SPEC.md` 已批准的 v0.1。它保持以下完整轻量闭环：

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

不实现 v0.2/v0.3 能力，包括数据驱动分箱 transform、group aggregate transformer、target encoding、WOE、监督分箱、交叉验证、group/time-aware split、树模型、模型比较、交互式图表和数据漂移。

`SPEC.md` 定义产品定位、模块边界、公共原则和最终 v0.1 能力；本计划是任务拆分和实现顺序的执行依据。若两者出现阶段划分或交付顺序冲突，应先修订 `SPEC.md` 与本计划对齐，再开始实现。

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
5. 运行并通过：

   ```bash
   python -m pytest
   python -m ruff check .
   python -m ruff format --check .
   ```

6. 涉及打包、CLI、public exports 或发布时，额外运行：

   ```bash
   python -m build
   ```

7. 若任务依赖的前置任务尚未合并，不得通过临时重复实现绕开依赖。

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

### Task 06 — Excel 单表读取

**目标**

补齐 v0.1 第二种输入格式，保持 Excel 引擎为 optional dependency。

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
- 缺失 sheet、坏文件、缺失文件和非法路径。
- 未安装 Excel extra 时的明确安装提示。
- 不支持多 sheet 返回 dict；该用法必须拒绝。
- CSV-only 核心安装仍可导入和运行。

**验收标准**

- 安装 `excel` extra 后能读取一个 `.xlsx` sheet。
- 未安装 extra 不影响包导入、CSV 或其他功能。
- 不增加多表联合分析能力。

---

### Task 07 — 单变量分析、相关性与异常值

**目标**

实现 v0.1 核心非 target 分析，超越简单 `describe()`，并遵守固定计算预算。

**依赖**

Task 03。

**创建/修改文件**

- `src/sharper/analysis.py`
- `src/sharper/__init__.py`
- `tests/test_analysis.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
analyze_numeric_features(...) -> NumericAnalysis
analyze_categorical_features(...) -> CategoricalAnalysis
compute_correlations(...) -> CorrelationAnalysis
detect_outliers(...) -> OutlierAnalysis
```

签名必须与 `SPEC.md` 完全一致。

**测试文件**

- `tests/test_analysis.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 数值分位数、偏度、零值率和有效样本量。
- 类别频数、比例、稀有水平、top-20 与截断披露。
- Pearson/Spearman、`min_periods`、50 列预算和稳定列顺序。
- IQR/MAD 异常值、边界值、NaN、inf、常量和小样本。
- columns 缺失、类型不适用、空选择和零列。
- 与手算、pandas、SciPy 基准在明确容差内一致。
- 无可分析列时返回 skipped reasons，不伪造统计量。

**验收标准**

- 所有结果记录有效样本量、缺失处理、截断和 skipped reasons。
- 不删除异常值，不修改输入。
- workflow 可接入结果而无需改变 public result contract。

---

### Task 08 — 分组比较与 Target Relationship Analysis

**目标**

补齐普通分组比较，以及分类/回归两条明确分离的 target relationship 分析路径。

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

- 单类别 group key 下的 count、missing count、mean、median 和分位数。
- 缺失 group、超过 20 组的频数截断及稳定排序。
- 多 group key、非法 group、非数值 values 和非法预算被拒绝。
- 分类 target：数值分组摘要、类别交叉表/比例、有限卡方结果。
- 回归 target：数值相关、类别分组 target 摘要。
- 缺 target、单类 target、任务不匹配、缺失 feature 和小样本。
- 检验结果包含样本量、统计量、p 值和探索性限制。

**验收标准**

- 无 target 的普通分组分析可独立调用。
- 分类与回归行为不通过猜测 target dtype 选择；必须使用显式 task。
- 不训练模型、不生成因果结论、不自动进行多重模型式搜索。

---

### Task 09 — 特征建议与安全无状态物化

**目标**

实现有预算、可解释的候选发现，以及 v0.1 允许的四类安全 transform。

**依赖**

Tasks 03、07。

**创建/修改文件**

- `src/sharper/features.py`
- `src/sharper/__init__.py`
- `tests/test_features.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
suggest_feature_derivations(...) -> FeatureSuggestionReport
derive_features(...) -> FeatureDerivationResult
```

公开 `FeatureSuggestion`、`FeatureSuggestionReport` 和 `FeatureDerivationResult` 的已批准最小字段。

**测试文件**

- `tests/test_features.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- ratio、difference、product 和日期候选。
- 分箱、group aggregate、target-aware 候选只返回 `requires_fit=True`/不可物化标记。
- target、ID、常量和明显重复列被排除。
- 总建议最多 50、每类预算、去重和稳定优先级。
- 零分母转缺失并记录 warning。
- 显式 reference date、缺失日期、非法日期和不可复现当前日期禁用。
- 列名冲突、输入列缺失、重复 suggestion 和 `copy` 行为。
- `requires_fit=True` suggestion 被 `derive_features` 拒绝。
- 返回 applied、skipped、warnings；输入不变。

**验收标准**

- v0.1 不出现 fitted transformer、target encoding、监督分箱或组聚合物化。
- 候选数量有硬预算且相同输入顺序稳定。
- 每个物化特征可追溯到公式和来源列。

---

### Task 10 — 数据分析任务型可视化

**目标**

为非模型分析生成 report-ready Figure；统计分析型图表优先使用 seaborn，
matplotlib 作为底层 backend、Figure/Axes 契约和低级 fallback。绘图消费数据
或既有分析结果，不隐藏重算统计。

**依赖**

Tasks 07–09。

**创建/修改文件**

- `src/sharper/visualization.py`
- `src/sharper/__init__.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`
- `docs/api.md`

**Public API**

```python
plot_distributions(...) -> PlotCollection
plot_missingness(...) -> PlotResult
plot_correlations(result: CorrelationAnalysis) -> PlotResult
plot_outliers(...) -> PlotCollection
plot_group_comparison(...) -> PlotCollection
plot_target_relationships(...) -> PlotCollection
```

本任务冻结 `PlotResult` 和 `PlotCollection` 的最小字段。模型评估图留给 Tasks 11–12。

**测试文件**

- `tests/test_visualization.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 数值直方/箱线、类别 top-N、缺失率、相关热图、异常值、分组和 target 图。
- 返回对象包含 Figure、task、采样/截断元数据和 skipped reason。
- 最多 20 张图、最多 10,000 行绘图样本、50 列预算。
- 空列、全缺失、常量、高基数、不适用 target 图。
- 不调用 `show()`、不默认写文件、不依赖全局状态。
- 不在函数中随意修改全局 matplotlib 或 seaborn style。
- seaborn 是统计分析型图表的首选实现；matplotlib 用于底层对象契约和
  seaborn 不适用时的低级 fallback。
- 使用 Agg backend；测试结束关闭所有 Figure。
- monkeypatch/spy 证明消费结果的函数不重新调用分析计算。

**验收标准**

- 每种批准任务至少能生成一个可保存 Figure 或明确 skipped result。
- 所有抽样和截断在返回元数据中披露。
- 不建立多可视化后端系统；无 Plotly、Altair、Bokeh、交互式 dashboard
  或主题系统。

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

**Public API**

```python
train_classifier(...) -> TrainingResult
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
- ColumnTransformer 数值/类别路径、未知测试类别和缺失值。
- 默认 Logistic Regression、显式 feature 列和自定义 classifier。
- 单类、小样本、缺 target、非法 split、非 classifier estimator。
- 测试独有类别/极值不进入 fitted state。
- target、后验字段、ID 和 exclude columns 不进入特征。
- 重复行/索引警告；实体重复或时间风险拒绝/警告符合 SPEC。
- holdout 不参与特征、阈值或模型选择。
- 相同 random_state 得到相同 split、列和指标；自定义 estimator 随机性被披露。
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

在不复制分类流程的前提下完成独立回归路径和任务匹配指标。

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
train_regressor(...) -> TrainingResult
evaluate_regressor(result: TrainingResult) -> RegressionEvaluation
evaluate_model(result: TrainingResult) -> ClassificationEvaluation | RegressionEvaluation
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
- 非数值/缺失 target、小样本、非法 split 和 classifier 误传。
- MAE、RMSE、R² 与 sklearn metrics 一致。
- 预测/残差表索引和 holdout 标签对齐。
- `evaluate_classifier` 拒绝 regression，`evaluate_regressor` 拒绝 classification。
- `evaluate_model` 严格依据 `TrainingResult.task` 分派。
- 残差图和预测对比图；常量 target 等边界明确 skipped/报错。
- random_state 下结果可复现。

**验收标准**

- 分类与回归共享内部安全预处理构造，但保持独立训练验证和评估契约。
- 不新增树模型、模型搜索、交叉验证或 AutoML。
- 所有回归指标和图只使用 holdout。

---

### Task 13 — 完整 Workflow、静态 HTML 与 CLI 收口

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

**Public API**

- 在单独评审并同步 Task 13 决策后扩展 Task 05 的 `AnalysisRun`，接入已
  实现的 analysis、feature、plot、training/evaluation 和预算结果；不得
  在 Task 05 提前预留这些字段。
- 完成 `generate_analysis_report(..., format="html")`。
- 不增加新的 public workflow class 或配置系统。

完整 CLI：

```text
sharper analyze INPUT
  --output PATH
  --format markdown|html
  --target TARGET
  --task classification|regression
  --id-column COLUMN
  --exclude-column COLUMN
  --model / --no-model
  --random-state INTEGER
```

**测试文件**

- `tests/test_workflow.py`
- `tests/test_reporting.py`
- `tests/test_cli.py`
- `tests/test_public_api.py`

**pytest 覆盖点**

- 无 target 完整分析：画像、质量、非 target 分析、特征建议和图。
- 显式 target 但不建模：target relationship 存在、training/evaluation 为空。
- 显式分类/回归建模：对应 training、evaluation 和模型图存在。
- target candidate 永不自动触发 target-aware 分析。
- id/exclude columns、random_state、非法参数组合和缺失列。
- 默认预算以及每项 requested/actual/reason 披露。
- Markdown/HTML 章节一致，HTML 转义和内部静态模板正确。
- 图像资产保存、相对链接、输出目录、覆盖策略和写失败。
- CLI 与直接 Python workflow 的章节、warnings、skipped 和预算元数据一致。
- Excel CLI、带/不带模型、debug/普通错误输出和退出码。

**验收标准**

- 一条 CLI 命令能对 CSV/Excel 生成可审阅 Markdown 或静态 HTML 报告。
- HTML 不增加第三方 renderer 或交互式依赖。
- CLI 不包含领域算法；reporting 不重算分析；workflow 不读取或写文件。
- 完整流程仍遵守所有 leakage 和预算规则。

---

### Task 14 — 文档、示例与 v0.1 发布验证

**目标**

让文档只承诺实际实现的 v0.1，并验证源码分发、wheel、示例和 CLI 的发布可用性。

**依赖**

Tasks 01–13。

**创建/修改文件**

- `README.md`
- `CHANGELOG.md`
- `docs/quickstart.md`
- `docs/analysis-guide.md`
- `docs/leakage.md`
- `docs/api.md`
- `examples/basic_analysis.py`
- `examples/baseline_modeling.py`
- `tests/test_public_api.py`
- `tests/test_distribution.py`
- 必要的 CI 配置文件
- `pyproject.toml`（仅发布 metadata、版本下界验证结果和测试配置）

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
- sdist/wheel clean install、`import sharper`、`sharper --help` 和最小分析命令。
- 缺 Excel extra 时核心流程仍可用；安装 extra 后 Excel smoke test 通过。
- distribution metadata、license、README 渲染和 Python 支持声明。

**验收标准**

- Python 3.10+ 支持矩阵中的所有配置通过 pytest、Ruff、build 和 clean-install smoke tests。
- README quickstart、CLI 和 examples 均可执行。
- CHANGELOG 准确列出 v0.1 能力与已知限制。
- 不包含 v0.2/v0.3 API、lock file、构建产物或生成报告。

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
