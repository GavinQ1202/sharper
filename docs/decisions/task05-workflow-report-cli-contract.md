# Task 05 Minimal Workflow、Markdown Report 与 CLI 公共契约

## 状态

已接受。本文是 Task 05 实现前的 API / CLI 决策记录。Task 05 的实现、
测试和文档必须遵守本文；修改本文冻结的 public API、结果字段、Markdown
格式或 CLI 行为需要先同步评审 `SPEC.md` 和
`IMPLEMENTATION_PLAN.md`。

## 范围

Task 名称冻结为：
**Task 05 — Minimal workflow, Markdown report, and CLI vertical slice**。

Task 05 只实现最薄闭环：

1. 读取 CSV；
2. 推断 schema；
3. 生成 DataFrame summary；
4. 检查 minimal data quality；
5. 组装 `AnalysisRun`；
6. 生成 Markdown report；
7. 通过 `sharper analyze INPUT --output report.md` 运行上述流程。

Task 05 CLI 只执行 CSV → schema → summary → quality → Markdown，不执行
SPEC 中未来完整 CLI 的后续分析能力。Task 05 不实现 outlier detection、
correlation analysis、target relationship analysis、feature engineering、
visualization、modeling、evaluation、HTML report、dashboard、automatic
data cleaning、workflow registry、plugin system、batch directory
processing、notebook generation 或 cloud/database connectors。

## 公开结果类型

两个结果类型均使用 `dataclass(frozen=True)`。

### `AnalysisRun`

字段按以下顺序冻结：

| 字段 | 类型 | Task 05 语义 |
|---|---|---|
| `schema` | `SchemaReport` | `infer_schema` 的结果 |
| `summary` | `DataFrameSummary` | `summarize_dataframe` 的结果 |
| `quality` | `QualityReport` | `check_data_quality` 的结果 |
| `target` | `str \| None` | 用户提供并验证的 target |
| `task` | `str \| None` | `"classification"`、`"regression"` 或 `None` |
| `include_model` | `bool` | Task 05 中只能为 `False` |
| `id_columns` | `tuple[str, ...]` | 已验证但尚未应用的列 |
| `exclude_columns` | `tuple[str, ...]` | 已验证但尚未应用的列 |
| `random_state` | `int` | 只保存，不参与随机计算 |
| `skipped` | `tuple[str, ...]` | 固定顺序的未执行能力 |
| `warnings` | `tuple[str, ...]` | 固定顺序的阶段性提示 |

`AnalysisRun` 不包含 generated time、input/output path、file name、
duration、random ID、visualization、model、evaluation 或 feature
engineering results。`skipped` 和 `warnings` 必须确定且不可使用无结构
dict 替代。

### `ReportArtifact`

字段按以下顺序冻结：

| 字段 | 类型 | Task 05 语义 |
|---|---|---|
| `path` | `Path` | 用户传入并解析为 `Path` 的 output path |
| `format` | `str` | 固定为 `"markdown"` |
| `title` | `str` | 清理后的报告标题 |

结果不包含 report content、generated time、file size、checksum、asset
list、绝对/相对路径转换元数据或其他非确定字段。

## `run_analysis`

签名冻结为：

```python
def run_analysis(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    task: str | None = None,
    include_model: bool = False,
    id_columns: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    random_state: int = 42,
) -> AnalysisRun: ...
```

规则：

- `df` 必须是 pandas DataFrame，且函数不修改它。
- 非字符串列名沿用 Task 03/04 `ValueError`，消息包含
  `DataFrame column names must all be strings`。
- 非空 `target` 必须存在，否则 `ValueError` 消息包含
  `target column not found`。
- 非空 `task` 只允许 `"classification"` 或 `"regression"`；否则
  `ValueError` 消息包含 `task must be classification or regression`。
- `task` 非空但 `target is None` 时，`ValueError` 消息包含
  `task requires target`。
- `target` 非空而 `task is None` 时允许；target 只传给 schema inference，
  不运行 target analysis。
- `include_model=True` 时，`ValueError` 消息包含
  `modeling is not available in Task 05`。
- `id_columns` 和 `exclude_columns` 只验证并保存，不实际排除列。任一列
  不存在时，`ValueError` 消息包含 `column not found`。
- 两组列不得重叠；重叠时 `ValueError` 消息包含
  `id_columns and exclude_columns must not overlap`。
- 任一组内部出现重复列名时，`ValueError` 消息包含
  `duplicate column parameter`；不得静默去重。
- `random_state` 必须是 `int`；否则 `ValueError` 消息包含
  `random_state must be an integer`。

Workflow 必须依次组合批准的 public API：

```python
schema = infer_schema(df, target=target)
summary = summarize_dataframe(df)
quality = check_data_quality(df, schema=schema)
```

Workflow 不复制 schema、summary 或 quality 算法，不调用 CLI，不渲染
Markdown，也不读写文件。

### `skipped`

Task 05 固定为：

```python
("modeling", "visualization", "feature_engineering")
```

### `warnings`

只允许以下文本，并按适用条件以固定顺序输出：

1. target 非空：
   `target recorded but target analysis is not available in Task 05`
2. task 非空：
   `task recorded but modeling is not available in Task 05`
3. `id_columns` 非空：
   `id_columns recorded but not applied in Task 05`
4. `exclude_columns` 非空：
   `exclude_columns recorded but not applied in Task 05`

无适用 warning 时返回空 tuple，不生成其他 warning 文本。

## `generate_analysis_report`

签名冻结为：

```python
def generate_analysis_report(
    run: AnalysisRun,
    output_path: str | Path,
    *,
    title: str = "Sharper Analysis Report",
    format: str = "markdown",
    overwrite: bool = True,
) -> ReportArtifact: ...
```

规则：

- Task 05 只支持 `format="markdown"`；否则抛出 `ValueError`，消息包含
  `only markdown reports are supported in Task 05`。
- `output_path` 接受 `str` 或 `Path`，返回时使用 `Path(output_path)`。
- 已存在目录不能作为 output path；抛出 `ValueError`，消息包含
  `output_path must be a file path`。
- 父目录不存在时自动创建。
- 文件存在且 `overwrite=True` 时覆盖；`overwrite=False` 时抛出
  `FileExistsError`，消息包含 `output file already exists`。
- 使用 UTF-8 写入，文件末尾恰有一个换行符。
- 写入失败保留原生 `OSError`，不包装为自定义异常。
- 不实现 HTML、模板系统、外部 Markdown renderer 或非确定字段。

## Markdown 合同

报告章节顺序固定为：

1. `# {escaped_title}`
2. `## Overview`
3. `## Schema`
4. `## DataFrame Summary`
5. `## Data Quality`
6. `## Skipped Capabilities`
7. `## Warnings`

### 标题

将换行替换为空格并去除首尾空白；不做复杂 Markdown escaping。清理后
为空时使用 `"Sharper Analysis Report"`。`ReportArtifact.title` 保存
清理后的标题。

### Overview

按以下顺序包含 bullet：

- `Rows: {n_rows}`
- `Columns: {n_columns}`
- `Quality issues: {issue_count}`

### Schema

使用 pipe table，每个 schema column 按原始列顺序一行，列顺序为：

1. `Column`
2. `Logical type`
3. `Pandas dtype`
4. `Missing rate`
5. `Unique count`
6. `ID-like`

### DataFrame Summary

使用 `DataFrameSummary.column_summary` 的原始行顺序和以下列：

1. `column`
2. `logical_type`
3. `missing_rate`
4. `unique_count`
5. `is_constant`
6. `is_id_like`

### Data Quality

无 issue 时输出 `No data quality issues detected.`。有 issue 时按
`QualityReport.issues` 既有顺序输出 pipe table，列顺序为：

1. `Code`
2. `Severity`
3. `Scope`
4. `Column`
5. `Count`
6. `Ratio`
7. `Suggestion`

### Skipped Capabilities 与 Warnings

`run.skipped` 为空时输出 `No capabilities were skipped.`，否则按 tuple
顺序输出 bullet。`run.warnings` 为空时输出 `No warnings.`，否则按 tuple
顺序输出 bullet。

### 固定格式

- Markdown tables 使用 pipe table，不使用外部依赖。
- `None` 渲染为空字符串。
- float rate 固定为四位小数，例如 `0.2500`。
- bool 渲染为 `true` 或 `false`。
- 文件末尾恰有一个换行符。
- 不包含时间戳、路径、随机 ID 或 Python object repr。

## CLI

使用 Typer，并在 `pyproject.toml` 注册：

```toml
[project.scripts]
sharper = "sharper.cli:app"
```

Task 05 只实现 `analyze`，并支持：

```bash
sharper --help
sharper analyze --help
sharper analyze INPUT --output report.md
```

### `analyze` 参数

| 参数 | 形式 | 默认值 |
|---|---|---|
| `INPUT` | 必填 CSV 路径 argument | 无 |
| `--output`, `-o` | 必填 Markdown 输出路径 option | 无 |
| `--target` | 可选字符串 | `None` |
| `--task` | 可选字符串，仅 classification/regression | `None` |
| `--id-column` | 可重复 option | 空 tuple |
| `--exclude-column` | 可重复 option | 空 tuple |
| `--random-state` | integer option | `42` |
| `--format` | option | `markdown` |
| `--overwrite / --no-overwrite` | boolean option | overwrite `True` |

Task 05 不提供 `--model`、`--include-model`、`--html`、`--plot`、
`--visualize`、`--feature` 或 `--debug`。

CLI 依次调用 `load_csv(input_path)`、`run_analysis(...)` 和
`generate_analysis_report(...)`。CLI 不直接调用 schema、summary 或
quality API，不写 Markdown，也不实现参数处理和错误展示之外的业务逻辑。

### stdout、stderr 与 exit code

成功时 stdout 恰有一行：

```text
Report written to: {str(Path(output_path))}
```

不向 stdout 打印 Markdown。Typer usage error 的 exit code 为 2。预期
runtime `ValueError`、`OSError` 和 `FileExistsError` 由 CLI 捕获，只将
消息打印到 stderr，不输出 traceback，exit code 为 1。成功 exit code
为 0。

预期 runtime error 包括 input file 缺失或为目录、CSV parse failure、
非法 target/task/format、output path 为目录、no-overwrite 时文件已存在
和 report write `OSError`。

### Help

`sharper --help` 至少包含 `analyze`。

`sharper analyze --help` 至少包含 `INPUT`、`--output`、`--target`、
`--task`、`--id-column`、`--exclude-column`、`--random-state`、
`--format`、`--overwrite` 和 `--no-overwrite`。不冻结完整 Typer
排版。

## 测试合同

Task 05 必须测试：

1. `AnalysisRun` 和 `ReportArtifact` frozen dataclass 字段；
2. `run_analysis` 正常路径、target/task、model rejection、列参数验证、
   random state 和输入不变性；
3. Markdown 写入、固定章节顺序、无 issue 文本、issue table、skipped、
   warnings、标题清理、UTF-8、单个末尾换行；
4. HTML rejection、目录 output、覆盖和 I/O failure；
5. 两级 CLI help、成功、非法 input/format 和 no-overwrite；
6. CLI 与 Python workflow 对同一 CSV 产生相同章节；
7. public API/`__all__` 以及 Tasks 01–04 回归测试。

所有实现和审查使用：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```
