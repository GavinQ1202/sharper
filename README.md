# Sharper

Sharper 是一个面向结构化表格数据的 Python 分析工具包。它计划把数据读取、质量检查、画像、关系挖掘、候选特征发现、分析型可视化、可选基线建模和报告导出组织为一条轻量、可复现的工作流。

它不是单纯的 EDA 工具，也不是 sklearn 的薄封装。建模只是完整分析闭环的一部分；变量关系、特征发现和可解释的分析输出同样重要。

核心工作流分为四个同等重要的部分：数据画像与质量、关系和分组挖掘、特征发现与任务型可视化、可选的可靠基线建模。

## 目标用户

- 希望快速理解 CSV/Excel 的数据分析师
- 需要发现特征并建立可靠基线的数据科学家与建模人员
- 需要可复现统计分析和报告的研究人员
- 需要快速产出表格数据分析报告的用户

## v0.1 计划能力

- 读取本地 CSV 和 Excel 单表
- 推断数值、类别、日期、ID 等列角色，并提示疑似 target
- DataFrame 摘要与数据质量报告
- 数值/类别变量分析、缺失与异常值分析
- 相关性、分组对比和分类/回归目标关系分析
- ratio、difference、product、日期、分箱和组聚合候选建议；v0.1 只物化不需要拟合状态的安全特征
- 围绕分布、缺失、相关、异常值、分组、目标关系和模型评估的静态分析图表
- 基于 sklearn Pipeline/ColumnTransformer 的分类与回归基线
- 严格分离的分类/回归评估
- Markdown/HTML 分析报告
- `sharper analyze` 完整流程 CLI

v0.1 会刻意限制算法、候选数量和图表类型，以优先保证正确性、可解释性和 data leakage 防护。数据驱动分箱和 group aggregate 在 v0.1 只作为建议，不提供 fit/transform。AutoML、深度学习、Web dashboard、数据库、分布式计算、feature store、MLflow 和 SHAP 不在 v0.1 范围。

## 当前可用能力

Tasks 01–13 已完成：包骨架、CSV/Excel Python API 读取、schema/summary、
minimal data quality API、最薄 Markdown CLI，以及独立的 non-target numeric、
categorical、correlation、outlier、group comparison 和 classification/regression
target relationship、feature suggestion 与 safe stateless derivation Python APIs
已可用。Task 13 将这些结果接入统一 workflow；Markdown 与 HTML 都写出确定性的
报告与 PNG assets bundle，CLI 同时支持 CSV 与单 sheet XLSX：

```bash
sharper analyze data.csv --output report.md
```

完整 CLI 不提供 dashboard、server 或交互式 HTML，也不重算既有分析结果。

Task 09 只物化不需要拟合状态的 arithmetic 和 timezone-naive datetime 特征；
binning、group aggregate 与 target encoding 仍只生成不可物化的结构化建议。完整
workflow 会接入既有 group/target analysis、feature suggestion、visualization 和
classification/regression baseline results，但不会重算它们。

Python API `load_excel` 可通过 optional `excel` extra 读取本地 `.xlsx`
单 sheet；CLI 也支持本地 `.xlsx` 单 sheet 输入。

## 完整工作流

```python
from sharper import (
    analyze_target_relationships,
    generate_analysis_report,
    infer_schema,
    load_csv,
    load_excel,
    run_analysis,
    suggest_feature_derivations,
)

df = load_csv("data.csv")
excel_df = load_excel("data.xlsx", sheet_name=0)
schema = infer_schema(df)
suggestions = suggest_feature_derivations(df, schema=schema)

# target 必须由用户确认；schema 候选不会自动触发 target-aware 分析。
target_analysis = analyze_target_relationships(
    df,
    target="outcome",
    task="classification",
)

run = run_analysis(
    df,
    target="outcome",
    task="classification",
    include_model=False,
)
generate_analysis_report(run, "report.html", format="html")
```

完整 CLI：

```bash
sharper analyze data.csv --output report.html
```

建模必须显式指定目标、任务并启用模型：

```bash
sharper analyze data.csv \
  --target outcome \
  --task classification \
  --model \
  --output report.html
```

Python 与 CLI 共用同一个 workflow。未显式确认 target 和 task 时，不会执行 target-aware 分析或建模。

## 当前状态

**状态：v0.1 已实现，Task 14 发布准备已完成。** Tasks 01–14 已完成；本地验证和
CI 门禁已就绪，尚未发布到 PyPI。

v0.2 [roadmap contract](docs/decisions/v02-roadmap-contract.md) 已批准。Task 15 的独立、
opt-in 二分类风险验证 API 与 Task 16 opt-in data audit 已实现并通过 review；Task 17
opt-in decision-strategy simulation 已实现并通过 review；Task 18 opt-in post-loan early
warning/lifecycle monitoring implementation 已完成，唯一一次 full implementation review 与
bounded implementation closure 已通过，final validation 已完成。
Tasks 19–20 尚未开始，v0.2 整体也尚未发布。当前 package version 仍为 `0.1.0`。

## Opt-in 二分类风险验证（Task 15）

Task 15 提供显式 estimator 或 external prediction source 的 leakage-aware validation、
OOF evidence、ranking/probability metrics、calibration、gains/lift、预声明 threshold 分析
与基础 exposure/loss 汇总。它不接入现有 workflow、report 或 CLI：

```python
from sklearn.linear_model import LogisticRegression

from sharper import (
    BinaryRiskValidationConfig,
    plot_binary_risk_validation,
    validate_binary_risk,
)

config = BinaryRiskValidationConfig(
    validation_mode="stratified_kfold",
    n_splits=5,
    thresholds=(0.20, 0.50, 0.80),
    threshold_kind="event_probability",
)
result = validate_binary_risk(
    df,
    "outcome",
    config=config,
    estimator=LogisticRegression(max_iter=1000),
    exclude_columns=("future_value",),
)
figure = plot_binary_risk_validation(result, kind="gains")
figure.savefig("gains.png")
```

Figure 由 caller 持有并负责保存或关闭。Task 15 只表示该 opt-in 能力已经实现，不代表
Tasks 18–20、v0.2 workflow/CLI integration 或 v0.2 release 已完成。

## Opt-in 数据质量与泄漏审计（Task 16）

Task 16 新增 `DataAuditRoles`、`ColumnAuditRule`、`DataAuditConfig`、
`DataAuditResult` 和 `audit_data_quality`。它只返回确定性的质量、missingness、drift、
leakage 与 point-in-time evidence，不修改输入，也不自动修复数据：

```python
from sharper import DataAuditRoles, audit_data_quality

result = audit_data_quality(
    current,
    reference=reference,
    roles=DataAuditRoles(
        target="outcome",
        features=("income", "balance"),
        observation_time="observed_at",
        feature_available_time_map=(("balance", "balance_available_at"),),
    ),
)
```

shared condition kernel 是 private implementation，不是 public DSL。Task 16 未接入现有
workflow、report 或 CLI，也不表示 Tasks 18–20 或 v0.2 release 已完成。

## Opt-in 贷前决策策略模拟（Task 17）

Task 17 新增 `StrategyCondition`、`DecisionRule`、`DecisionConstraint`、
`DecisionStrategyConfig`、`DecisionStrategyResult` 和 `simulate_decision_strategy`。它执行
caller 冻结的两阶段 eligibility/decision rules，返回离线 simulated actions、rule evidence、
约束测量、业务 evidence、历史动作 transitions 和匿名 segment/time stability summaries：

```python
from datetime import datetime

import pandas as pd

from sharper import (
    DecisionRule,
    DecisionStrategyConfig,
    StrategyCondition,
    simulate_decision_strategy,
)

data = pd.DataFrame({"income": [40_000, 80_000]})
condition = StrategyCondition(
    "atomic", "lt", "column", "income", "literal", 50_000
)
rule = DecisionRule("income_review", "eligibility", 10, condition, "manual_review")
config = DecisionStrategyConfig(
    strategy_key="preloan_policy",
    strategy_version="v1",
    effective_from=datetime(2025, 1, 1),
    expires_at=None,
    evaluation_time=datetime(2025, 1, 2),
    rules=(rule,),
    default_action_name="selected",
    unknown_action_name="manual_review",
    action_role_mapping=(
        ("selected", "selected"),
        ("manual_review", "review"),
    ),
)
result = simulate_decision_strategy(data, config)
```

Conditions 使用封闭 operators 并委托 Task 16 private kernel；不存在 expression、callable 或
任意代码执行入口。Ranking score 不会升级为 probability；expected loss/payoff 只消费已对齐的
Task 15 event probability。Constraints 只测量 frozen actions，不搜索 cutoff、改写 actions 或
选择 winner。该 API 不执行真实审批，不接入现有 workflow/report/CLI，也不表示 v0.2 已发布。

## Opt-in 贷后预警与生命周期监测（Task 18）

Task 18 新增 `MonitoringCondition`、`EarlyWarningRule`、`WarningScenario`、`LifecycleState`、
`LifecycleMonitoringConfig`、`LifecycleMonitoringResult` 和 `monitor_lifecycle`。该入口只执行
caller 冻结的 point-in-time、离线预警与生命周期 evidence，不发送通知、不执行账户或催收动作，
也不接入现有 workflow/report/CLI：

```python
from sharper import monitor_lifecycle

result = monitor_lifecycle(data, config)
```

`LifecycleMonitoringResult` 固定包含 11 张 typed tables：`observation_history`、
`rule_evaluations`、`notifications`、`alert_episodes`、`event_matches`、`state_history`、
`state_transitions`、`monitoring_summary`、`scenario_comparison`、`lifecycle_summary` 和
`provenance`。`scenario_comparison` 固定 16 列，并只覆盖九个 scenario-bearing scopes：
`scenario`、`scenario_rule`、`scenario_alert_level`、`scenario_segment`、`scenario_time`、
`scenario_cohort`、`scenario_vintage`、`scenario_state`、`scenario_transition`；`overall`、
`segment_time`、`cohort_time` 和 `vintage_state` 不参与比较。`scenario`/`scenario_rule` 的
source-local scenario ordinal 不进入 equality identity，其余七个 scope 的 normalized subordinate
ordinal 必须对齐。Summary resource gates 按 `monitoring_summary_rows` → `lifecycle_summary_rows`
→ `scenario_comparison_rows` 执行，comparison 上限为 200,000 行。

Task 18 implementation 状态为 `Implementation complete — review Go`；final validation 已完成。
Task 18 contract 与 AM-04 均为 `Approved — Go`，package version 仍为 `0.1.0`，v0.2 尚未发布。

## Opt-in 模型治理（Task 19）

Task 19 提供 `evaluate_governance` 与 `plot_model_governance`，用于离线消费 Tasks 15--18
的冻结结果以及调用者预先计算的结构化 attribution、prediction profile、performance evidence
和 governance metadata。它输出解释、漂移、稳定性、candidate comparison、模拟 recommendation、
metadata 与 provenance 十张有类型的表，并提供五种只读取结果表的 matplotlib 图。

该能力不读取或执行模型，不自动 promotion、审批或部署，也不提供因果推断、法律公平认证或
adverse-action notice。结构化 attribution 仅支持 coefficient direction、native importance 和
预先计算的 holdout/OOF permutation importance。Task 19 implementation 已在 targeted remediation、bounded closure 与
post-review final validation 后完成，当前为 `Implemented — Post-Review Closure Complete`；package version 仍为
`0.1.0`，v0.2 尚未发布。

### Development environment

This repository uses a uv-managed virtual environment at `.venv`.

Create or update tools explicitly in that environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python <development-tools>
```

Verify the environment before running project commands:

```bash
bash scripts/verify-uv-env.sh
```

Run Python tooling with the project interpreter:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m build --no-isolation
```

Do not use the system Python for project validation.

从源码安装开发环境：

```bash
python -m pip install -e ".[dev]"
```

从构建产物安装而不依赖源码 checkout：

```bash
python -m pip install dist/sharper-0.1.0-py3-none-any.whl
```

Sharper 支持 Python 3.10+，使用 `src` layout。运行时依赖为 pandas、
NumPy、SciPy、scikit-learn、matplotlib、seaborn 和 Typer；Excel Python API 支持
使用可选依赖：

```bash
pip install "sharper[excel]"
# 或源码 editable 安装：
python -m pip install -e ".[excel]"
```

依赖版本下界将在对应功能实现并通过支持矩阵验证后确定。

`run_analysis(..., include_model=False)` 可在不训练模型时执行显式 target/task
relationship analysis；设置 `include_model=True` 后，`task="classification"` 与
`task="regression"` 分别执行对应的 split-first baseline。Markdown 和 HTML 都是
静态报告与 PNG assets bundle，不包含 interactive dashboard 或 server。
