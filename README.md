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

Tasks 01–08 已完成：包骨架、CSV/Excel Python API 读取、schema/summary、
minimal data quality API、最薄 Markdown CLI，以及独立的 non-target numeric、
categorical、correlation、outlier、group comparison 和 classification/regression
target relationship Python APIs 已可用：

```bash
sharper analyze data.csv --output report.md
```

Task 05 CLI 仍只运行 CSV → schema → summary → quality → Markdown。Task 07
non-target analysis 尚未接入 workflow、reporting 或 CLI。

Task 08 的 group comparison 与 classification/regression target relationship
仅作为独立 Python API 提供，尚未接入 workflow、reporting 或 CLI。完整集成
留给 Task 13。HTML、特征、绘图和建模仍是后续任务。

Python API `load_excel` 可通过 optional `excel` extra 读取本地 `.xlsx`
单 sheet。Excel 输入的 `sharper analyze input.xlsx` CLI 支持仍是后续完整
workflow/CLI 任务。

## 未来完整工作流（占位）

> 以下完整分析、HTML 和建模 API 尚未实现。

未来 Python 工作流：

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

未来完整 CLI：

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

**状态：v0.1 实现中。** 当前已完成 Tasks 01–08。本文“未来完整工作流”中的
HTML、特征、绘图和建模示例暂不可运行，Tasks 07–08 analysis 也尚未接入完整
workflow/CLI。

从源码安装开发环境：

```bash
python -m pip install -e ".[dev]"
```

Sharper 支持 Python 3.10+，使用 `src` layout。运行时依赖为 pandas、
NumPy、SciPy、scikit-learn、matplotlib 和 Typer；Excel Python API 支持
使用可选依赖：

```bash
pip install "sharper[excel]"
# 或源码 editable 安装：
python -m pip install -e ".[excel]"
```

依赖版本下界将在对应功能实现并通过支持矩阵验证后确定。
