# Task 10 数据分析任务型可视化公共契约

## 状态与范围

已接受。本文是 Task 10 实现前的 API 决策记录。实现、测试和 API 文档必须遵守本文；改变签名、结果字段、图型、数据映射、预算、metadata、错误、排序、Figure 生命周期或全局状态行为前，必须先同步评审本文、`SPEC.md` 与 `IMPLEMENTATION_PLAN.md`。

Task 名称冻结为：**Task 10 — 数据分析任务型可视化**。

Task 10 提供 report-ready、静态、确定性的 Python Figure API。它只使用 seaborn 和 matplotlib；不接入 workflow、reporting 或 CLI，不保存图片/文件，不生成 HTML、dashboard 或交互组件，不支持 Plotly、Altair、Bokeh，不绘制 Task 09 feature suggestions，也不新增依赖或 custom exception。Task 09 只是 sequencing prerequisite，不是实际 API 依赖。

Task 10 不修改 Tasks 01–09 的 public API、result schema、errors、ordering 或 reason codes。尤其不修改 `AnalysisRun`，也不改变 Task 05 固定的 `"visualization"` skipped capability；完整 workflow/reporting/CLI 集成留给 Task 13。

## Public API

以下签名冻结；不得增加 `ax`、`fig`、`show`、`save`、`palette`、`style`、`**kwargs` 或其他 public plotting API：

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

Task 11/12 的模型评估图不属于本文。

## Public result types

两个类型均必须使用 `@dataclass(frozen=True)`；字段名、顺序和注解冻结如下。容器及 Figure 不承诺 deep immutability，但库函数不得修改输入 result、其 DataFrame 或其 metadata。`PlotResult` 与 `PlotCollection` 是供后续已批准 visualization task 复用的通用容器；后续 task 必须在自己的已接受合同中冻结其新图型、source、metadata 和 collection 语义，且不得改变下述 Task 10 六个 API 的行为。

```python
@dataclass(frozen=True)
class PlotResult:
    figure: matplotlib.figure.Figure
    chart_type: str
    title: str
    source: str
    item: str | None
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PlotCollection:
    requested_count: int
    available_count: int
    actual_count: int
    truncated: bool
    truncation_reason: str | None
    plots: tuple[PlotResult, ...]
```

`figure` 始终是存在的 `matplotlib.figure.Figure`，不使用 `None`；不保存 Axes、path、timestamp、duration、random ID 或 file artifact。空 collection 的 `plots == ()`。没有 skipped `PlotResult`，也没有 Task 10 skipped-reason vocabulary：不适用的 collection 图以空 collection 表示，单 `PlotResult` API 以正常但无 data artists 的 Figure 表示。

`metadata` 是按本文每图型指定顺序的不可变 `(key, value)` tuple；key/value 均为字符串。整数使用十进制字符串，浮点使用 `format(value, ".12g")`，布尔使用 `"true"`/`"false"`，无值使用 `"none"`。列表使用紧凑 JSON array：`json.dumps(values, ensure_ascii=False, separators=(",", ":"))`；其中 column name 保留原字符串，类别/group 值先逐项 `str(value)`。不得添加、删除、重排或以 dict 替代。

对本合同的六个 Task 10 API，`chart_type` 只允许：`"distribution_histogram"`、`"distribution_categories"`、`"missingness_rate"`、`"correlation_heatmap"`、`"outlier_rate"`、`"group_median"`、`"target_classification_numeric"`、`"target_classification_categorical"`、`"target_regression_categorical"`。该封闭词汇不限制后续 task 在其独立合同中使用同一通用容器的 chart type。

对本合同的六个 Task 10 API，`source` 只允许 `"dataframe"`、`"correlation_analysis"`、`"outlier_analysis"`、`"group_comparison"`、`"target_analysis"`。该封闭词汇不限制后续 task 在其独立合同中使用同一通用容器的 source。`item` 是 column/value/feature 名，或没有单一项目时为 `None`。

所有 `max_plots` 必须是非 bool 的 `int` 且 `1 <= max_plots <= 20`；所有 `max_columns` 必须是非 bool 的 `int` 且 `1 <= max_columns <= 50`；所有 `sample_size` 必须是非 bool 的 `int` 且 `1 <= sample_size <= 10_000`。对本合同的六个 Task 10 API，`requested_count` 是公开 `max_plots` 参数，或没有该参数时固定的内部 Figure budget `20`；`available_count` 是预算前可生成 Figure 数；`actual_count == len(plots)`；`truncated` 当且仅当 `available_count > requested_count`；`truncation_reason` 仅为 `None` 或 `"max_plots"`。截断始终保留稳定顺序的前项。后续 task 的 collection 计数语义必须各自在独立合同中冻结，且不改变本段的 Task 10 语义。

### 完整 metadata schema

下表是唯一允许的 metadata schema。每行的 keys 必须按列出的顺序出现；未列出的 key 一律禁止。

| chart_type | Ordered keys and frozen values |
|---|---|
| `distribution_histogram` | `column`: source column name；`dtype`: `str(df[column].dtype)`；`finite_count`: sampling 前 finite non-missing value count；`missing_count`: `df[column].isna().sum()`；`non_finite_count`: non-missing positive/negative infinity count；`sample_size_requested`: `sample_size`；`sample_size_actual`: plotted finite values count；`bins`: frozen bin count。 |
| `distribution_categories` | `column`: source column name；`dtype`: `str(df[column].dtype)`；`non_missing_count`: non-missing value count；`missing_count`: `df[column].isna().sum()`；`available_categories`: distinct non-missing category count；`displayed_categories`: retained category labels in plot order的 JSON list；`truncated_categories`: `available_categories > 20`；`category_limit`: `20`。 |
| `missingness_rate` | `n_rows`: `len(df)`；`requested_columns`: `max_columns`；`available_columns`: `len(df.columns)`；`analyzed_columns`: retained source column names in DataFrame order的 JSON list；`truncated_columns`: `len(df.columns) > max_columns`；`truncation_reason`: `"max_columns"` when truncated, otherwise `"none"`。 |
| `correlation_heatmap` | `method`: `result.method`；`analyzed_columns`: `result.analyzed_columns` in order的 JSON list；`pair_rows`: valid long-form pair row count；`missing_pairs`: analyzed-column unordered pairs absent from `correlations`的 JSON list of two-string arrays in matrix traversal order；`input_max_columns`: `result.max_columns`；`input_truncated`: `result.truncated`；`annotation_format`: `".2f"`。 |
| `outlier_rate` | `displayed_features`: collection retained feature names in order的 JSON list；`truncated_features`: `available_count > requested_count`；`outlier_count`: current summary row `outlier_count`；`outlier_rate`: current summary row `outlier_rate`；`lower_bound`: current summary row `lower_bound`；`upper_bound`: current summary row `upper_bound`；`threshold`: `result.threshold`。 |
| `group_median` | `value`: current value name；`displayed_groups`: current value summary rows的 group labels in order的 JSON list；`finite_medians`: current value finite `median` count；`metric`: `"median"`；`error_bars`: `"q25_q75"`。 |
| `target_classification_numeric` | `feature`: current feature name；`analysis_type`: `"classification_numeric"`；`target_categories`: current feature `target_category` labels in row order的 JSON list；`metric`: `"median"`；`error_bars`: `"q25_q75"`。 |
| `target_classification_categorical` | `feature`: current feature name；`analysis_type`: `"classification_categorical"`；`feature_categories`: current feature category labels in order的 JSON list；`target_categories`: target-category labels in the first feature-category block order的 JSON list；`metric`: `"count"`。 |
| `target_regression_categorical` | `feature`: current feature name；`analysis_type`: `"regression_categorical"`；`feature_categories`: current feature category labels in order的 JSON list；`target_categories`: `[]` encoded as a JSON list；`metric`: `"target_median"`。 |

For `outlier_rate`, the collection-wide `displayed_features` and `truncated_features` values are repeated unchanged in every PlotResult in that collection. All other metadata is per-Figure exactly as its table row defines. Empty collections contain no PlotResult and express their budget state only through PlotCollection fields.

## 共享 Figure 与全局状态合同

每张图必须创建独立的新 Figure；函数不接收外部 Axes，返回也不暴露 bare Axes。库代码不得调用 `plt.show()` 或 `plt.close()`，Figure 所有权和关闭责任属于调用者。库代码不得切换 matplotlib backend、修改 `matplotlib.rcParams`、调用 `plt.style.use`、`sns.set_theme`、`sns.set_style` 或 `sns.set_palette`。颜色和样式只能作为单次绘图调用的显式参数；禁止随机 jitter、随机采样及任何 hash-order 迭代。

除相关热图外，单 series 使用 `#4C78A8`。分类×类别图按 legend 顺序循环使用：`#4C78A8`、`#F58518`、`#54A24B`、`#E45756`、`#72B7B2`、`#B279A2`、`#FF9DA6`、`#9D755D`、`#BAB0AC`。直方图和 bar charts 只有 y-axis 的 major dashed grid（alpha `0.3`）；heatmap 无 grid。除相关热图的 cell annotation 外，Task 10 不添加数值 annotation、reference line、p-value、statistic、effect size、significance、importance 或 causal wording。

## 输入、验证与依赖边界

`plot_distributions` 与 `plot_missingness` 是唯一接收 raw DataFrame 的函数。它们只允许本文定义的绘图必要本地计算，且不得调用 `infer_schema`、`summarize_dataframe`、`check_data_quality` 或任何 Task 07/08/09 public function。

其余四个函数是 result-only：只能读取下列冻结 result 中的 DataFrame 和 metadata，且不得重新计算 correlation、outlier、group comparison 或 target statistics，也不得调用任何 Task 07/08/09 public function。

| Function | 允许读取 |
|---|---|
| `plot_correlations` | `CorrelationAnalysis.analyzed_columns`、`method`、`max_columns`、`min_periods`、`truncated`、`correlations` |
| `plot_outliers` | `OutlierAnalysis.analyzed_columns`、`threshold`、`summary`；`outliers` 只可用于 schema/integrity validation，不能用于绘图 |
| `plot_group_comparison` | `GroupComparison.group_by`、`analyzed_values`、`summary` |
| `plot_target_relationships` | `TargetAnalysis.target`、`task`、`analyzed_features`、`numeric_details`、`category_details`、`statistical_tests` |

Result-only functions must not read raw input data, including any data that a caller separately retains. Task 09 `FeatureSuggestionReport` and `FeatureDerivationResult` are not accepted or consumed.

### Stable errors and validation precedence

Only built-in `ValueError` is public. A raw-data function validates, in order: DataFrame type, string column names, duplicate names, then its numeric keyword parameters. A result-only function validates result type, its numeric keyword parameters, then the required frozen result schema. `bool` is never an integer.

Stable message substrings are:

| Condition | Message substring |
|---|---|
| non-DataFrame | `df must be a pandas DataFrame` |
| non-string DataFrame columns | `DataFrame column names must all be strings` |
| duplicate DataFrame columns | `duplicate DataFrame column names are not supported` |
| invalid `max_plots` | `max_plots must be an integer from 1 to 20` |
| invalid `max_columns` | `max_columns must be an integer from 1 to 50` |
| invalid `sample_size` | `sample_size must be an integer from 1 to 10000` |
| wrong correlation result | `result must be a CorrelationAnalysis` |
| wrong outlier result | `result must be an OutlierAnalysis` |
| wrong group result | `result must be a GroupComparison` |
| wrong target result | `result must be a TargetAnalysis` |
| malformed correlation result | `correlation result has invalid schema` |
| malformed outlier result | `outlier result has invalid schema` |
| malformed group result | `group result has invalid schema` |
| malformed target result | `target result has invalid schema` |
| unhashable raw categorical value | `categorical column contains unhashable values` |

Malformed schema means a required table does not have exactly the corresponding Task 07/08 frozen column order and dtypes, or required result metadata is outside its frozen vocabulary. This validation does not coerce, sort, fill, or mutate the result. In addition, the following relational validation is mandatory.

#### `CorrelationAnalysis`

All of the following raise `ValueError("correlation result has invalid schema")`: duplicate ordered pair; unordered duplicate pair; a pair endpoint outside `analyzed_columns`; a diagonal row; a row whose `method` differs from `result.method`; a missing required column; an invalid dtype; a non-finite stored correlation; or any duplicate matrix mapping conflict. Each unordered pair may occur at most once, in the Task 07 stored column order. Duplicate rows are invalid; an implementation must never use a last-row-wins rule.

#### `OutlierAnalysis`

All of the following raise `ValueError("outlier result has invalid schema")`: a `summary` or `outliers` table schema mismatch; a summary feature outside `analyzed_columns`; a missing or duplicate summary row for an analyzed feature; an outlier feature outside `analyzed_columns`; or disagreement between summary `method`/`threshold` and result metadata. `outliers` is validated only, never used as plotting data.

#### `GroupComparison`

All of the following raise `ValueError("group result has invalid schema")`: a summary schema missing `value` or `group`; a summary value outside `analyzed_values`; a missing value block for an analyzed value when summary is non-empty; duplicate `(value, group)` rows; group/value row order that cannot be reconciled with the stored Task 08 summary order; or result metadata inconsistent with the summary's value blocks. The function must not sort, deduplicate, or reconstruct a replacement group order.

#### `TargetAnalysis`

All of the following raise `ValueError("target result has invalid schema")`: a numeric-details, category-details, or statistical-tests schema mismatch; a detail/test feature outside `analyzed_features`; duplicate numeric `(feature, target_category)` rows; duplicate classification categorical `(feature, feature_category, target_category)` rows; duplicate regression categorical `(feature, feature_category)` rows; duplicate statistical-test feature rows; a detail table incompatible with `result.task`; or target-category blocks inconsistent with the applicable frozen Task 08 detail-table schema. The function must not deduplicate, fill missing cells, or infer target categories from other features.

## Chart contracts

### `plot_distributions`

Candidates are traversed in original DataFrame column order. Supported numeric columns are real pandas numeric, non-boolean, non-complex dtypes. Supported categorical columns are object, string, category and boolean dtypes. Datetime, timedelta, complex and all other dtypes are excluded. All-missing columns are excluded. Numeric columns drop missing and positive/negative infinity; if no finite value remains they are excluded. Numeric constants are supported. Categorical missing values are excluded from frequency counts; a categorical column with only missing values is excluded. An unhashable non-missing category is a function-level error.

Each supported candidate creates one Figure, subject to `max_plots`. Numeric charts are histograms with no KDE. They use the first `sample_size` finite values in original row order. Bins use Freedman--Diaconis on that sample: `ceil((max - min) / (2 * IQR / n**(1/3)))`; if `n < 2`, IQR is zero, or the formula is non-finite, use `ceil(sqrt(n))`; clamp to 1--50. A constant uses one bin. Categorical charts are frequency bars for at most 20 categories, ranked by count descending with first non-missing appearance as the tie-break. Missing is not a category and is not separately drawn.

Histogram title is `"{column} distribution"`, x label is `{column}`, y label is `"Count"`, and there is no legend. Category title is `"{column} category frequency"`, x label is `{column}`, y label is `"Count"`, and there is no legend. The categorical tick labels are `str(category)` in the retained order. Histogram and category results use, respectively, `chart_type="distribution_histogram"` and `"distribution_categories"`, always use `source="dataframe"`, and use the column name as `item`.

Histogram and category metadata must use exactly the corresponding `distribution_*` row in the complete metadata schema. `PlotCollection.available_count` counts supported columns before `max_plots`; category truncation is recorded only in that Figure's metadata, not collection truncation. No candidate returns an empty collection.

### `plot_missingness`

This function uses the first `max_columns` DataFrame columns in original order. For each, `missing_rate = missing_count / len(df)`; when there are zero rows the rate is `0.0`. It creates one vertical bar Figure even for a 0-column frame. The title is `"Missingness by column"`, x label is `"Column"`, y label is `"Missing rate"`, y limits are `[0.0, 1.0]`, there is no legend or value annotation, and bars remain in DataFrame order. Its sole PlotResult has `chart_type="missingness_rate"`, `source="dataframe"`, `item=None`, and metadata exactly as the `missingness_rate` row in the complete metadata schema.

### `plot_correlations`

The function only reads `CorrelationAnalysis.correlations`. It creates one `"correlation_heatmap"` Figure with `source="correlation_analysis"` and `item=None`. Matrix columns are `analyzed_columns` in their stored order. Initialize the diagonal to `1.0`; for each long-form pair, set both symmetric cells to its finite `correlation`. Omitted pairs remain `NaN` and are masked. A non-finite stored correlation is malformed schema. There is no sorting and no recomputation. If there are zero analyzed columns, draw a normal empty axes without annotations or colorbar; a single analyzed column displays only its diagonal.

The title is `"Correlation heatmap ({method})"`; x and y labels are `"Feature"`; the colorbar label is `"Correlation"`; use seaborn `vlag`, `vmin=-1.0`, `vmax=1.0`, `center=0.0`, and annotations formatted `.2f` for unmasked cells. Metadata exactly follows the `correlation_heatmap` row in the complete metadata schema.

### `plot_outliers`

The only approved outlier chart is a per-feature outlier-rate bar chart using `OutlierAnalysis.summary`; `outliers` is only schema-validated and is not used for plotting. One summary row in `analyzed_columns` order produces one Figure even when `outlier_count == 0`. `max_plots` retains the first features. Every result has `chart_type="outlier_rate"`, `source="outlier_analysis"`, and its column as `item`. The title is `"{column} outlier rate (IQR)"`, x label is `{column}`, y label is `"Outlier rate"`, y limits are `[0.0, 1.0]`, and there is no legend. Metadata exactly follows the `outlier_rate` row in the complete metadata schema. An empty valid summary returns an empty collection.

### `plot_group_comparison`

The function only reads `GroupComparison.summary`. For each `analyzed_values` item with rows, it creates one Figure in that value order; groups follow their first appearance within that value's summary rows. Every result has `chart_type="group_median"`, `source="group_comparison"`, and its value name as `item`. x is group, y is median; finite q25/q75 values create asymmetric q25--q75 error bars. A missing median has no bar/error bar but retains its x tick in the stored order. The title is `"{value} by {group_by}"`, x label is `{group_by}`, y label is `"{value} median"`, and there is no legend. Metadata exactly follows the `group_median` row in the complete metadata schema. An empty valid summary returns an empty collection.

### `plot_target_relationships`

The function creates only the following result-derived figures, in `analyzed_features` order and then the stored row/category order:

| Path | Source table | Figure |
|---|---|---|
| classification × numeric | `numeric_details` | one Figure per feature; x=`target_category`, y=feature median, q25--q75 error bars |
| classification × categorical | `category_details` | one Figure per feature; grouped count bars, x=`feature_category`, series=`target_category` |
| regression × categorical | `category_details` | one Figure per feature; x=`feature_category`, y=`target_median` |
| regression × numeric | none | no Figure; `statistical_tests` is not plotted |

All target results use `source="target_analysis"` and the feature name as `item`. Classification-numeric uses `chart_type="target_classification_numeric"`; its title is `"{feature} by {target}"`, x label is `{target}`, y label is `"{feature} median"`, and it has no legend. Classification-categorical uses `chart_type="target_classification_categorical"`; its title is `"{feature} by {target}"`, x label is `{feature}`, y label is `"Count"`, and has a legend titled `{target}` in stored target-category order; every source zero-count cell produces a zero-height bar rather than being omitted. Regression-categorical uses `chart_type="target_regression_categorical"`; its title is `"{feature} by {target}"`, x label is `{feature}`, y label is `"{target} median"`, and has no legend. All target metadata exactly follows the corresponding row in the complete metadata schema.

The collection has the common fixed Figure budget 20. Detail rows with missing numeric summary values retain their x categories but no bar/error bar. Empty valid detail tables return an empty collection. The function does not plot `statistical_tests`, p-values, statistics, effect sizes, significance, importance or causal language.

### Label conversion and duplicate display labels

For group labels, target categories, and feature categories, data positions are the consecutive integer positions of source rows in their frozen source order. Tick and legend text is exactly `str(value)`. The implementation must not pass these values through matplotlib's implicit categorical mapping, merge categories, deduplicate labels, or sort labels. Thus values `1` and `"1"` keep two distinct positions in source order even though both visible labels are `"1"`. This rule also applies to duplicate group/category labels and to the JSON-list metadata representation; the source position, not the rendered string, determines identity.

## Determinism, input immutability, and test contract

All item, category, series and subplot order follows the source order stated above; no alphabetic, p-value, effect-size, hash, or random ordering is allowed. All raw DataFrames and structured result inputs (including their DataFrames, index, columns, dtypes, values and metadata) remain unchanged.

Task 10 implementation tests must cover exact signatures/exports; frozen dataclass fields; Figure type; independent Figures; no bare Axes; no show/close; unchanged backend/rcParams/global seaborn state; raw/result input immutability; result-only spies proving no Task 07/08/09 public function calls; every exact chart/title/label/series order; empty input/result; NaN/infinity/constant/all-missing/mixed dtype; duplicate index/category labels; parameter validation and schema errors; budgets and metadata; deterministic non-random sampling; headless backend and Figure cleanup; Tasks 01–09 regression; and a diff audit proving no unrelated modules, dependencies or lock files changed. Tests must use Agg (or another already configured headless backend) before importing plotting code; library code must not select it.

## Allowed implementation files and deferred work

Task 10 implementation may modify only:

- `src/sharper/visualization.py`
- `src/sharper/__init__.py`
- `tests/test_visualization.py`
- `tests/test_public_api.py`
- `docs/api.md`
- `README.md`

It must not modify analysis, features, workflow, reporting, CLI, I/O, `pyproject.toml`, dependency groups, lock files or Tasks 01–09 contracts. File saving/export, report embedding, HTML, workflow composition and CLI integration remain Task 13 work.
