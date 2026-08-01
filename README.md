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
opt-in 二分类风险验证 API 已实现并等待 implementation diff review；Tasks 16–20 尚未
实现，v0.2 整体也尚未发布。当前 package version 仍为 `0.1.0`。

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
Tasks 16–20、v0.2 workflow/CLI integration 或 v0.2 release 已完成。

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
