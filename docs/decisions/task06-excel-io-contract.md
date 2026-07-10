# Task 06 Excel Single-Sheet I/O 公共契约

## 状态

已接受。本文是 Task 06 实现前的 API 决策记录。Task 06 的实现、测试和
文档必须遵守本文；修改本文冻结的 public API、文件格式、参数边界、错误
行为或测试合同需要先同步评审 `SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。

## 范围

Task 名称冻结为：
**Task 06 — Excel single-sheet I/O**。

Task 06 只新增一个 public API：

```python
def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    **read_options: Any,
) -> pd.DataFrame: ...
```

Task 06 只做：

1. 从本地 `.xlsx` 文件读取单个 sheet；
2. 返回 `pandas.DataFrame`；
3. 保留 pandas 解析得到的原始列名和值；
4. 使用 optional `excel` extra 提供 Excel engine；
5. 更新 public API、文档和测试合同。

Task 06 不做：

1. 多 sheet 读取；
2. sheet 合并；
3. Excel 写入；
4. Excel CLI；
5. workflow、reporting 或 CLI 集成；
6. schema inference；
7. summary；
8. quality checks；
9. cleaning；
10. type coercion；
11. column normalization；
12. modeling；
13. visualization；
14. HTML 或 dashboard。

Task 06 不修改 Task 05 CLI。`sharper analyze` 仍只支持 Task 05 已冻结
的 CSV → Markdown 闭环。

## 文件格式

Task 06 v0.1 只承诺支持本地 `.xlsx` 文件。

不承诺支持：

- `.xls`
- `.xlsm`
- `.xlsb`
- `.ods`
- remote URLs
- file-like objects

如果 `path` 后缀不是 `.xlsx`，抛出 `ValueError`，消息必须包含：

```text
only .xlsx files are supported in Task 06
```

## `path` 参数

`path` 只接受 `str` 或 `pathlib.Path`。其他类型抛出 `ValueError`，消息
必须包含：

```text
path must be a string or Path
```

文件不存在时抛出 `OSError`，消息必须包含：

```text
Excel file not found
```

如果 `path` 是目录，抛出 `OSError`，消息必须包含：

```text
Excel path is a directory
```

Task 06 只读文件，不创建目录，不写文件。

## `sheet_name` 参数

`sheet_name` 只接受 `str` 或 `int`，默认值为：

```python
sheet_name=0
```

必须拒绝：

- `None`
- `list`
- `tuple`
- `set`
- 任何可能触发 pandas 返回多 sheet `dict` 的值

非法 `sheet_name` 抛出 `ValueError`，消息必须包含：

```text
sheet_name must be a string or integer
```

如果 sheet 不存在，抛出 `ValueError`，消息必须包含：

```text
sheet not found
```

`load_excel` 必须保证只返回 `pd.DataFrame`，绝不返回
`dict[str, pd.DataFrame]`。

## `read_options` 白名单

Task 06 不允许任意透传 pandas `read_excel` 参数。

只允许以下 `read_options`：

- `header`
- `names`
- `usecols`
- `dtype`
- `na_values`
- `keep_default_na`
- `skiprows`
- `nrows`

禁止以下参数：

- `engine`
- `engine_kwargs`
- `storage_options`
- `parse_dates`
- `date_format`
- `converters`
- `thousands`
- `decimal`
- `comment`
- `index_col`
- 任何未列入白名单的参数

如果用户传入未允许的 read option，抛出 `ValueError`，消息必须包含：

```text
unsupported Excel read option
```

`sheet_name` 是 `load_excel` 的显式 keyword-only 参数，不属于
`read_options` 白名单或黑名单。用户必须通过显式 `sheet_name` 参数选择
sheet。由于 Python 会在函数调用绑定阶段处理显式关键字参数，重复传入
`sheet_name` 可能由 Python 抛出 `TypeError`，且函数体不会执行；这不是
Sharper 的 public error contract。Task 06 不要求测试
`sheet_name inside read_options rejected`。

如果用户传入 `engine`，抛出 `ValueError`，消息必须包含：

```text
engine cannot be overridden in Task 06
```

## Excel engine 与 optional dependency

Task 06 使用：

```python
engine="openpyxl"
```

用户不能覆盖 engine。

`openpyxl` 必须放在 optional dependency group：

```toml
[project.optional-dependencies]
excel = ["openpyxl>=..."]
```

不得把 `openpyxl` 放入 core runtime dependencies。

如果缺少 `openpyxl`，`load_excel` 必须抛出 `ImportError`，消息必须包含：

```text
Install sharper[excel] to read Excel files
```

README 应说明：

```bash
pip install "sharper[excel]"
```

或 editable dev 场景下的等价安装方式。

## pandas 行为边界

Task 06 允许 pandas 负责实际解析，但不能把 pandas 的所有
`read_excel` 参数暴露为 public contract。

规则：

1. 返回值必须是 `pd.DataFrame`；
2. 不修改列名；
3. 不清洗数据；
4. 不推断 schema；
5. 不调用 `infer_schema`；
6. 不调用 `summarize_dataframe`；
7. 不调用 `check_data_quality`；
8. 不修改输入文件；
9. 不写任何输出文件；
10. pandas 解析失败时抛出 `ValueError`；
11. pandas 解析失败的错误信息必须包含：

```text
failed to read Excel file
```

## public API 与 `__all__`

Task 06 实现后：

- `load_excel` 应从 `sharper.io` 导出；
- `load_excel` 应加入 `src/sharper/__init__.py`；
- `load_excel` 应加入 `__all__`。

Task 06 不修改已有 public API 的行为：

- `load_csv`
- `infer_schema`
- `summarize_dataframe`
- `check_data_quality`
- `run_analysis`
- `generate_analysis_report`

## CLI、workflow 与 reporting

Task 06 不修改：

- `src/sharper/cli.py`
- `src/sharper/workflow.py`
- `src/sharper/reporting.py`

Task 06 不允许：

- `sharper analyze input.xlsx`
- Excel workflow integration
- Excel report generation through CLI

README 可以说明：

- Python API 支持 `load_excel`；
- CLI Excel 支持是未来任务，不是 Task 06 当前能力。

## 测试合同

Task 06 tests must cover:

1. `load_excel` reads a simple `.xlsx` file；
2. `path` as `str`；
3. `path` as `Path`；
4. `sheet_name` by index；
5. `sheet_name` by name；
6. `sheet_name=None` rejected；
7. `sheet_name` list、tuple 和 set rejected；
8. non-`.xlsx` suffix rejected；
9. missing file raises `OSError`；
10. directory path raises `OSError`；
11. unsupported read option rejected；
12. `engine` inside `read_options` rejected；
13. missing sheet raises `ValueError`；
14. bad Excel file raises `ValueError`；
15. return type is `pd.DataFrame`；
16. no schema、summary 或 quality calls；
17. public API / `__all__` includes `load_excel`；
18. CSV-only Task 02 tests still pass；
19. Task 01–05 tests still pass。

Optional dependency tests:

- If `openpyxl` is installed in the test environment, actual `.xlsx` tests
  should run.
- Missing `openpyxl` behavior may be tested with monkeypatch / import mocking.
- Do not make the entire suite fail merely because optional Excel extra is
  unavailable; skip actual Excel read tests with a clear reason if `openpyxl`
  is unavailable.
- In the current project `.venv`, if `openpyxl` is installed through dev/test
  extras, actual Excel tests should run.

## 验证命令

Task 06 implementation and review must use:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```

Do not use system `python3`.
Do not rely on `python` command.

## 明确推迟

以下能力明确推迟到后续任务或 v0.2+：

- 多 sheet 读取或合并；
- Excel CLI；
- Excel workflow/reporting integration；
- Excel 写入；
- schema、summary、quality 自动集成；
- cleaning、type coercion 或 column normalization；
- HTML、visualization、feature engineering、modeling、evaluation、
  dashboard 或 automatic data cleaning。
