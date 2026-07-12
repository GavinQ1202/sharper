# Task 13 完整 Workflow、静态 HTML 与 CLI 收口公共契约

## 状态、身份与范围

**状态：已接受。** 本文冻结 Task 13 实现前的唯一集成合同；实现、测试和文档不得偏离本文。改变本文件冻结的字段、签名、章节、文件布局、错误、顺序或 allowlist 前，必须先同步更新并评审 `SPEC.md`、`IMPLEMENTATION_PLAN.md` 和本文。

**正式名称：** Task 13 — 完整 Workflow、静态 HTML 与 CLI 收口。

**用户目标：** 对一个内存中的单表 DataFrame，或经 CLI 读取的一个本地 CSV / `.xlsx` 单 sheet，得到同一份可审阅、确定的完整分析结果与 Markdown/静态 HTML 报告；显式确认 target/task 后可选地运行一条严格 holdout-only 的分类或回归基线。

Task 13 是 Task 14 发布准备之前的最后一个实现任务。上游为 Tasks 03--12：schema/summary、quality、CSV/Excel I/O、non-target analysis、group/target analysis、feature suggestions、Figures、classification 和 regression training/evaluation。它不改变任何上游 public API 或结果合同。

Task 13 只做 orchestration、结果渲染、Figure asset export 和 CLI 参数映射。它不做新统计、自动 target/task 推断、特征物化、fit/transform、CV/tuning、threshold search、新模型/metric/plot、dashboard、server、notebook UI、network、telemetry、background execution、plugin、依赖或 lock file。

## Public API

不新增 workflow class 或 configuration dataclass；保留既有顶层 exports `AnalysisRun`、`run_analysis`、`ReportArtifact` 和 `generate_analysis_report`，只按下文扩展已有前两者的合同。`ReportArtifact` 字段不变：`path: Path`、`format: str`、`title: str`，顺序不变。

```python
def run_analysis(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    task: Literal["classification", "regression"] | None = None,
    include_model: bool = False,
    id_columns: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    features: Sequence[str] | None = None,
    time_column: str | None = None,
    group_by: str | None = None,
    reference_date: str | date | datetime | pd.Timestamp | None = None,
    max_suggestions: int = 50,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> AnalysisRun: ...

def generate_analysis_report(
    run: AnalysisRun,
    output_path: str | Path,
    *,
    title: str = "Sharper Analysis Report",
    format: Literal["markdown", "html"] = "markdown",
    overwrite: bool = True,
) -> ReportArtifact: ...
```

All arguments after `df` / `output_path` are keyword-only. `features` and non-`None` `time_column` require `include_model=True`; when allowed, `features=None` preserves the frozen training API's default selection. `group_by=None` means no group comparison. `reference_date` and `max_suggestions` are passed only to Task 09 suggestion generation; no suggestions are materialized. `test_size` is passed only to selected training; it is nevertheless validated on every call. `random_state=None` is allowed and is forwarded unchanged.

`AnalysisRun` remains `@dataclass(frozen=True)`. Its exact field order and annotations are:

```python
schema: SchemaReport
summary: DataFrameSummary
quality: QualityReport
target: str | None
task: Literal["classification", "regression"] | None
include_model: bool
id_columns: tuple[str, ...]
exclude_columns: tuple[str, ...]
features: tuple[str, ...] | None
time_column: str | None
group_by: str | None
reference_date: str | None
max_suggestions: int
test_size: float
random_state: int | None
numeric_analysis: NumericAnalysis
categorical_analysis: CategoricalAnalysis
correlation_analysis: CorrelationAnalysis
outlier_analysis: OutlierAnalysis
group_comparison: GroupComparison | None
target_analysis: TargetAnalysis | None
feature_suggestions: FeatureSuggestionReport
training: TrainingResult | RegressionTrainingResult | None
evaluation: ClassificationEvaluation | RegressionEvaluation | None
distribution_plots: PlotCollection
missingness_plot: PlotResult
correlation_plot: PlotResult
outlier_plots: PlotCollection
group_plots: PlotCollection | None
target_plots: PlotCollection | None
evaluation_plots: PlotCollection | None
skipped: tuple[str, ...]
warnings: tuple[str, ...]
limitations: tuple[str, ...]
```

`features` is `None` exactly when the caller supplied `None`, otherwise the caller-order tuple. `reference_date` is the normalized `FeatureSuggestionReport.reference_date`. The four non-target analysis fields and `feature_suggestions` are always present, including their upstream fixed empty results. `group_comparison` is present iff `group_by is not None`; `target_analysis` is present iff both `target` and `task` are non-`None`; `training` and `evaluation` are both present iff `include_model=True`. When present, their task literals must agree with `AnalysisRun.task`; a classification training/evaluation can never occupy a regression run, and vice versa.

The four required plot fields are the direct return values from their existing Task 10 APIs. The optional collection fields are present exactly with their source results: group iff `group_comparison` is present; target iff `target_analysis` is present; evaluation iff `evaluation` is present. Reporting flattens collections only while rendering, in this exact order: distribution plots; missingness; correlation; outliers; group; target; evaluation. No `PlotResult` is duplicated in another field. Retaining the existing `PlotCollection` objects preserves every frozen requested/available/actual/truncation budget for both nonempty and empty sources; no parallel mutable budget dictionary is added.

`skipped` is the ordered subset of `("group_comparison_not_requested", "target_analysis_not_requested", "modeling_not_requested", "evaluation_not_requested")` selected by: absent `group_by`; absent target/task pair; `include_model=False`; and absent model evaluation, respectively. `warnings` is the stable de-duplicated concatenation of the selected training result's `warnings`, or `()` without a model. `limitations` is the stable de-duplicated concatenation of `target_analysis.limitations` (if any), then `training.limitations`, then `evaluation.limitations`; it does not restate QualityReport issues.

All dataclasses are shallowly frozen. Workflow does not copy or retain the input DataFrame, its index, or its raw values. It retains only upstream result objects; training results retain their already-frozen holdout snapshot/pipeline/estimator under Tasks 11--12 ownership rules. Callers own mutable nested DataFrames, estimators and Figures until the reporting figure-ownership acquisition point frozen below. Reporting never mutates a non-Figure result field.

## Workflow data flow and no recomputation

Workflow is the only layer allowed to access the raw DataFrame after its own validation. It does not read or write files, render a report, save a Figure, call CLI, call private helpers from upstream modules, or retain the DataFrame.

It validates once, then calls each listed public API at most once and in this exact order:

1. `infer_schema(df, target=target)`.
2. `summarize_dataframe(df, schema=schema)`.
3. `check_data_quality(df, schema=schema)`.
4. `analyze_numeric_features(df, columns=analysis_columns)`.
5. `analyze_categorical_features(df, columns=analysis_columns)`.
6. `compute_correlations(df, columns=analysis_columns)`.
7. `detect_outliers(df, columns=analysis_columns)`.
8. `compare_groups(group_frame, group_by, values=None)` only when `group_by` is set; `group_frame` is the raw DataFrame restricted to `analysis_columns`, which necessarily contains `group_by`, in original order.
9. `analyze_target_relationships(df, target, task=task, features=analysis_columns)` only when both target and task are set.
10. `suggest_feature_derivations(df, schema=schema, target=target, exclude_columns=effective_exclusions, reference_date=reference_date, max_suggestions=max_suggestions)`.
11. When `include_model=True`, exactly one of `train_classifier` or `train_regressor`, followed by exactly one `evaluate_model(training)`.
12. `plot_distributions(df)`, `plot_missingness(df)`, `plot_correlations(correlation_analysis)`, and `plot_outliers(outlier_analysis)`; then conditional `plot_group_comparison`, `plot_target_relationships`, and exactly one appropriate evaluation-plot API.

`analysis_columns` is input-column order excluding target (when set), every `id_columns` member, and every `exclude_columns` member. `effective_exclusions` is the de-duplicated input-order concatenation of `id_columns` then `exclude_columns`; it is used only for Task 09 suggestions and model training. `group_by` must not be target, an ID column, or an excluded column; it may be present in `analysis_columns` and `compare_groups` itself selects its numeric value columns. Training receives the caller's `features`, `effective_exclusions`, `time_column`, `test_size` and `random_state` unchanged. No derived feature is passed to training.

Reporting may only read `AnalysisRun`, its frozen nested result fields and the stored `PlotResult.figure` objects. It may flatten the already-stored `PlotCollection` fields in their frozen order, format values, select the already-stored report columns, escape text, and preserve frozen order. It must not call any `infer_*`, `summarize_*`, `check_*`, `analyze_*`, `compare_groups`, `suggest_*`, `derive_*`, `train_*`, `evaluate_*`, `predict`, sklearn metric, or `plot_*` API. In particular it never retrains, reevaluates, predicts, regenerates statistics, or replaces a stored Figure. CLI only parses arguments, chooses the frozen input reader, calls workflow exactly once, calls reporting exactly once, and presents the result.

The allowed dependency graph is:

```text
cli -> io + workflow + reporting
workflow -> schema + summary + quality + analysis + features + modeling + evaluation + visualization
reporting -> AnalysisRun + frozen result types + matplotlib Figure export
```

No edge may reverse: lower-level modules do not import workflow/reporting/CLI; modeling/evaluation do not import reporting; visualization does not import reporting; reporting does not import estimator execution logic.

## Workflow validation and stable errors

Workflow validation precedence is: DataFrame type; column-name validity; scalar options (`include_model`, `task`, `test_size`, `random_state`, `max_suggestions`); target/task relationship; id/exclude/features sequences; column existence/overlap; time/group columns; then delegated step execution. `bool` is not an integer, real, or random state.

| Condition | Exception and exact message |
|---|---|
| non-DataFrame | `ValueError("df must be a pandas DataFrame")` |
| non-unique or non-string names | `ValueError("DataFrame column names must be unique strings")` |
| non-bool `include_model` | `ValueError("include_model must be a boolean")` |
| invalid task | `ValueError("task must be classification or regression")` |
| task without target | `ValueError("task requires target")` |
| model without target/task | `ValueError("modeling requires target and task")` |
| absent target | `ValueError(f"target column not found: {target!r}")` |
| invalid id/exclude sequence | `ValueError("id_columns and exclude_columns must be sequences of unique column names")` |
| missing id/exclude member | `ValueError(f"column not found: {column!r}")` |
| id/exclude overlap / target in either set | `ValueError("id_columns and exclude_columns must not overlap")` / `ValueError("target must not appear in id_columns or exclude_columns")` |
| invalid features | `ValueError("features must be a non-empty sequence of unique column names")` |
| feature absent / target / effective exclusion | `ValueError(f"feature column not found: {feature!r}")`, `ValueError("target must not appear in features")`, or `ValueError("features and exclude_columns must not overlap")` |
| feature or time option without a model | `ValueError("features require include_model=True")` / `ValueError("time_column requires include_model=True")` |
| invalid time column / missing time column | `ValueError("time_column must be a column name string or None")` / `ValueError(f"time column not found: {time_column!r}")` |
| invalid group column / missing group column / forbidden group column | `ValueError("group_by must be a column name string or None")` / `ValueError(f"group column not found: {group_by!r}")` / `ValueError("group_by must not be target, id, or excluded")` |
| invalid test size | `ValueError("test_size must be strictly between 0 and 1")` |
| invalid random state | `ValueError("random_state must be a non-negative integer or None")` |
| invalid maximum suggestions | `ValueError("max_suggestions must be a positive integer")` |

The `reference_date` validation and all remaining domain validation retain their Task 03--12 frozen messages. Any selected public API failure is re-raised as `ValueError(f"workflow step failed: {step}: {error}")` with the original `Exception` as `__cause__`; `step` is exactly its function name from the numbered data-flow list. `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` are never caught. Workflow does not attempt recovery, fallback task dispatch, rerun, coercion, deduplication, or partial result return.

## Reporting preflight validation

Reporting performs an exact-type, non-recomputing preflight before creating a
parent directory, staging path, backup path, or PNG. `type(run) is
AnalysisRun` is required; every other input raises `TypeError("run must be an
AnalysisRun")`. All other preflight failures raise
`ValueError("analysis run has invalid schema")` with the original `Exception`
as `__cause__`. `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` are
never caught.

Preflight only compares observable frozen fields. It never calls an analysis,
feature-materialization, train, predict, evaluate, metric, or plot API, and
never creates a result or Figure. It may use an upstream shared result-schema
validator only when that validator has the same non-recomputing boundary.

The following invariants are mandatory:

1. `AnalysisRun` is a frozen dataclass with exactly the fields, field order,
   annotations, literals, and tuple/`None` container forms frozen above. It has
   no reporting-relevant dynamic attribute. `warnings` and `limitations` are
   tuples of non-empty strings; `skipped` is the frozen ordered vocabulary;
   no value is a timestamp, duration, random ID, raw DataFrame, or source path.
2. `schema`, `summary`, `quality`, all four non-target analyses, and
   `feature_suggestions` have their exact Task 03/04/07/09 result type and
   observable table/tuple/dict schema: required columns and order, required
   dtypes, no duplicate columns, upstream row order, and the upstream fixed
   vocabulary/metadata. Reporting does not calculate a value to validate it.
3. Optional values use `None`, never an empty tuple/dict as a sentinel.
   `group_comparison`/`group_plots` are both present exactly when `group_by` is
   non-`None`; `target_analysis`/`target_plots` are both present exactly when
   `target` and `task` are present; absent steps match the corresponding
   `skipped` code.
4. With `task is None`, training, evaluation, and `evaluation_plots` are all
   `None`. With `task="classification"` and `include_model=True`, `training`
   is `TrainingResult`, `evaluation` is `ClassificationEvaluation`, and
   `evaluation_plots` is the Task 11 collection. With `task="regression"`,
   they are respectively `RegressionTrainingResult`, `RegressionEvaluation`,
   and the Task 12 collection. `training` and `evaluation` are both `None`
   iff `include_model=False`; no cross-task pair is valid.
5. Every present training/evaluation pair has the same task and target as the
   run; training feature order, effective exclusions, time column, test size,
   random state, and holdout positions agree with recorded run configuration;
   evaluation holdout positions agree with training. Classification classes and
   regression prediction-table fields remain exactly those frozen by Tasks 11
   and 12. The preflight only compares these fields; it never recomputes a
   prediction or metric.
6. Each plot field has its exact `PlotResult` or `PlotCollection` type. Every
   collection has a tuple `plots`, `actual_count == len(plots)`, frozen
   requested/available/truncation consistency, and the source-specific plot
   order/type/source/metadata defined by Tasks 10--12. Every `PlotResult` has
   a non-empty string title/chart type/source, its permitted item/metadata,
   and a `matplotlib.figure.Figure` whose number exists in
   `matplotlib.pyplot.get_fignums()` and whose canvas exists. The same Figure
   identity may occur only once across all seven plot fields. Preflight does
   not call `savefig`, `show`, `close`, or redraw a plot; an export failure is
   handled as a staging failure below.
7. The report tables are the exact upstream tables already named by their
   owning result: `DataFrameSummary.column_summary`, all Task 07/08 detail
   tables, Task 09 suggestions, and Task 11/12 evaluation tables/metric
   tuples. They are consumed only after the preceding structural checks. No
   original input DataFrame, replacement summary, reconstructed quality issue,
   or copied metric representation is permitted.

The **figure-ownership acquisition point** is immediately after all seven
pre-I/O stages succeed: arguments/path validation, overwrite validation,
residual staging/backup validation, exact `AnalysisRun` type, complete
preflight, pure in-memory rendering, and non-writing Figure checks. Before
that point reporting only borrows `run`: every failure leaves all Figures open,
does not mutate the run, and does not create a directory, file, PNG, staging,
or backup path. Thus invalid format/path, overwrite conflict, residual-path
conflict, wrong run type, malformed run, and invalid/closed/duplicate Figure
all leave Figure ownership with the Python caller.

Only when reporting is about to create assets staging does it acquire cleanup
ownership of the unique live Figures enumerated in the frozen plot-field
order. From that point, an outer `finally` closes each acquired Figure once in
that order after success, staging failure, backup failure, commit failure,
cleanup failure, or compensation failure. It never calls `plt.close("all")`
or closes a Figure outside the run. A non-`AnalysisRun`, a non-enumerable
malformed plot field, or an already closed Figure is never acquired or closed.
`run_analysis` callers retain open Figures until they either close them or
cross this acquisition point through `generate_analysis_report`.

## Reporting and file contract

`format` accepts exactly `"markdown"` and `"html"`. Both render the same 17 ordered semantic sections, always present even when their result is empty:

1. `Overview`; 2. `Schema`; 3. `DataFrame Summary`; 4. `Data Quality`; 5. `Numeric Feature Analysis`; 6. `Categorical Feature Analysis`; 7. `Correlations`; 8. `Outliers`; 9. `Feature Suggestions`; 10. `Group Comparison`; 11. `Target Relationships`; 12. `Model Training`; 13. `Model Evaluation`; 14. `Visualizations`; 15. `Skipped Capabilities`; 16. `Warnings`; 17. `Limitations`.

Markdown begins with `# {cleaned_title}`; HTML is the standard-library, deterministic static rendering of the same Markdown semantic structure, with escaped title/body values, one `<h1>`, then matching `<h2>` headings in that order. Title cleaning remains Task 05: replace line breaks by spaces, trim, and use `Sharper Analysis Report` if empty. Markdown has exactly one terminal `\n`; HTML has exactly one terminal `\n`. No timestamps, duration, absolute path, random identifier, object repr, dynamic template value, or network resource is emitted.

Tables preserve the order and values already fixed by their result contracts. Section sources are uniquely fixed: Overview reads `summary`, `quality`, and recorded options; Schema / DataFrame Summary / Data Quality read their identically named fields; the next five analytical sections read their identically named result fields; Model Training and Model Evaluation read only their corresponding optional fields; Visualizations reads only stored plot fields; and the final three read their identically named tuples. Empty optional results render the literal `Not requested.`; empty executed tables render `No applicable results.`; an empty Figure tuple renders `No visualizations were generated.`. Metrics keep their stored tuple order. `skipped`, `warnings`, and `limitations` render in their stored tuple order. Markdown uses pipe tables and HTML uses `<table>`; neither recalculates a value or changes result ordering.

For every Figure in the frozen plot-field flattening order, reporting writes one PNG asset with deterministic name `plot-001.png`, `plot-002.png`, and so on, in the sibling directory `{output_path.stem}_assets`. Markdown references it as `![{plot.title}]({output_path.stem}_assets/plot-NNN.png)` and HTML as an escaped `<img>` with the same slash-separated relative URL and `alt=plot.title`. PNG is the only asset format. Thus both frozen formats are a report-and-assets bundle; Markdown is not a single-file format in Task 13. `ReportArtifact.path` is exactly `Path(output_path)`, `format` is the requested literal, and `title` is the cleaned title; asset paths are intentionally not a second mutable result representation.

For final report path `P`, final asset directory `A = P.parent / f"{P.stem}_assets"`, the six transaction paths are exactly:

```text
report final:           P
assets final:           A
report staging:         P.parent / f".{P.name}.sharper-staging"
assets staging:         P.parent / f".{A.name}.sharper-staging"
report backup:          P.parent / f".{P.name}.sharper-backup"
assets backup:          P.parent / f".{A.name}.sharper-backup"
```

All six paths are siblings in `P.parent`; no system temporary directory,
timestamp, PID, UUID, random suffix, or cross-filesystem move is used. Staging
and backup paths are never final paths. An assets directory is created even
when a valid run has zero Figures, so final report and assets always have the
same two-target overwrite semantics.

Validation order before any filesystem mutation is: report arguments/path;
final-target overwrite policy; pre-existing staging/backup conflict; complete
preflight above; pure in-memory rendering; non-writing Figure checks. With
`overwrite=False`, an existing `P` or `A` raises
`FileExistsError("output file or asset directory already exists")`; neither
staging/backup path is created and no Figure is closed until normal final
cleanup. This final-target overwrite conflict therefore takes precedence when
`overwrite=False` and a final and a residual path both exist. Independently of
the `overwrite` value, any pre-existing one of the four staging/backup paths
raises `FileExistsError("staging or backup path already exists")` after the
final-target policy permits processing; Sharper never guesses, deletes, or
restores a prior transaction.

After successful validation, the unique protocol is the same for both overwrite
values. With `overwrite=False`, the already-checked finals are absent, so the
backup phase is skipped; staging, commit, rollback, cleanup, and Figure
ownership below are otherwise identical. With `overwrite=True`, the protocol is:

1. Create `P.parent` only after preflight; create assets staging; export every
   PNG to assets staging; write, flush, and close report staging with links to
   final `A`. A staging failure deletes both staging paths, leaves both finals
   unchanged, creates no backup, and raises the stable write error below with
   cause.
2. Backup in this order: atomically replace `A` with assets backup when it
   exists, then atomically replace `P` with report backup when it exists. If
   the second backup fails, restore any completed assets backup to `A`, delete
   both staging paths, retain no backup, and raise the stable write error.
3. Commit in this order: atomically replace assets staging with `A`, then
   atomically replace report staging with `P`. If either commit fails, delete
   any newly committed final asset directory, restore report backup then assets
   backup to their finals when they existed, delete both staging paths and any
   remaining backup, and raise the stable write error. The report is never
   committed before the assets it references exist.
4. Once report staging has replaced `P`, the bundle is committed. Cleanup first
   deletes report backup, then assets backup. If report-backup deletion fails,
   stop cleanup immediately: retain both backups, retain the new finals, return
   no artifact, and raise the stable write error with that deletion exception
   as `__cause__`. Do not attempt assets-backup deletion. If report-backup
   deletion succeeds but assets-backup deletion fails, retain only assets
   backup, retain the new finals, return no artifact, and raise the same stable
   write error with that deletion exception as `__cause__`. Neither cleanup
   failure rolls back a committed bundle. A later invocation fails before I/O
   under the residual-path conflict rule; users must remove the stated backup.
   Only when both deletions succeed does reporting return `ReportArtifact`.
   File handles are closed before every replace.

If a compensation operation itself fails, reporting raises the stable write
error with that failure as cause and preserves every remaining final, staging,
and backup path rather than attempting another destructive guess. This is the
only permitted exceptional rollback state. Individual `replace` operations are
atomic only to their path; the two-path bundle uses the stated compensation and
does not claim filesystem-transaction atomicity across interruption.

The resulting failure matrix is frozen:

| Failure point | Final report/assets | Staging | Backup | Figures | Result |
|---|---|---|---|---|---|
| argument/path validation, overwrite conflict, wrong run type, malformed `AnalysisRun`, or pure in-memory rendering/Figure preflight | unchanged | report/assets staging: none | report/assets backup: none (the four temporary paths were absent before this call) | acquisition point not crossed; open/caller-owned; Python reporting does not close; CLI `finally` closes its run's unique live Figures | respective stable validation or preflight error |
| pre-existing report/assets staging or report/assets backup (one or more, regardless of `overwrite`) | unchanged | pre-existing staging: unchanged; new staging: none | pre-existing backup: unchanged; new backup: none | acquisition point not crossed; open/caller-owned; Python reporting does not close; CLI `finally` closes its run's unique live Figures | `FileExistsError("staging or backup path already exists")` |
| any PNG or report-staging write | unchanged | both removed | none | closed/report-consumed | stable write error with cause |
| assets backup | unchanged | both removed | none | closed/report-consumed | stable write error with cause |
| report backup | old bundle restored | both removed | none | closed/report-consumed | stable write error with cause |
| assets commit | old bundle restored | both removed | none | closed/report-consumed | stable write error with cause |
| report commit | old bundle restored | both removed | none | closed/report-consumed | stable write error with cause |
| report-backup cleanup | new bundle remains | none | both backups remain | closed/report-consumed | stable write error with cause |
| assets-backup cleanup | new bundle remains | none | assets backup remains | closed/report-consumed | stable write error with cause |
| compensation failure | preserved recoverable paths at the failed step | preserved | preserved | closed/report-consumed | stable write error with cause |

| Condition | Exception and exact message |
|---|---|
| wrong run type | `TypeError("run must be an AnalysisRun")` |
| malformed exact `AnalysisRun` | `ValueError("analysis run has invalid schema")` |
| invalid output type / directory | `ValueError("output_path must be a string or Path")` / `ValueError("output_path must be a file path")` |
| invalid title / format / overwrite | `ValueError("title must be a string")` / `ValueError("format must be markdown or html")` / `ValueError("overwrite must be a boolean")` |
| occupied final `P` or `A` with `overwrite=False` (this takes precedence even when a residual temporary path also exists) | `FileExistsError("output file or asset directory already exists")` |
| pre-existing report/assets staging or backup path (regardless of `overwrite`) | `FileExistsError("staging or backup path already exists")` |
| mkdir, Figure export, staging, backup cleanup, or final write failure | `OSError("failed to write report output")`, retaining the triggering `OSError` or Figure-export `Exception` as `__cause__` |

For both pre-acquisition rows, reporting creates no parent directory, final,
staging, backup, other output, or PNG; final report/assets retain their exact
call-entry state. The residual-path row additionally preserves every existing
temporary path byte-for-byte and does not restore, delete, or overwrite it.
Run validation completes before a directory is created or a Figure is
exported. Reporting catches no `BaseException` and closes Figures only after
the figure-ownership acquisition point above.

## CLI contract

The existing entry point remains `sharper = "sharper.cli:app"`; command name remains `sharper analyze INPUT`. Root `--version` uses eager precedence: before any root validation, scan the argv segment before the first `analyze` subcommand (or all argv if it is absent) for `--version`. If present, write exactly `sharper {__version__}` plus one newline to stdout, write nothing to stderr, exit 0, and do not read an input, call workflow, or call reporting. Therefore `sharper --version`, `sharper --version --help`, `sharper --help --version`, `sharper --version --unknown`, `sharper --unknown --version`, and `sharper --version analyze ...` all return only the version success result. Once `analyze` is encountered, `--version` is not a subcommand option: `sharper analyze --version` and `sharper analyze INPUT --version` are normal argument-parsing errors with stdout empty, Typer stderr usage/error output, exit 2, and zero I/O/workflow/reporting calls. `sharper --help` and `sharper analyze --help` retain their existing help behavior when root eager version is absent. It accepts a local `.csv` or `.xlsx` input by case-insensitive suffix. CSV has no exposed pandas read options; it calls `load_csv(input_path)` with no options. Excel calls `load_excel(input_path)` with the frozen default single sheet and no read options. No other input suffix, URL, directory, file-like object, multiple sheet, encoding, delimiter, engine, or pandas parsing option is supported by this CLI.

```text
sharper analyze INPUT --output PATH
  [--format markdown|html]
  [--target TARGET --task classification|regression]
  [--id-column COLUMN]... [--exclude-column COLUMN]...
  [--feature COLUMN]... [--time-column COLUMN] [--group-by COLUMN]
  [--reference-date YYYY-MM-DD] [--max-suggestions INTEGER]
  [--model/--no-model] [--test-size FLOAT] [--random-state INTEGER]
  [--overwrite/--no-overwrite] [--debug/--no-debug]
```

`INPUT` and `--output/-o` are required. Defaults are: format `markdown`; target/task `None`; repeated lists empty; time/group/reference date `None`; max suggestions `50`; model `False`; test size `0.20`; random state `42`; overwrite `True`; debug `False`. Repeated `--feature` maps to `features=None` when absent, otherwise its tuple; all other repeatables map to tuples. The CLI performs no task inference: `--target` without `--task` is analysis-only; `--task` without target and `--model` without both target/task fail through workflow validation.

Validation precedence is: Typer argument parsing (exit 2); input path/suffix check; `load_csv`/`load_excel`; workflow configuration validation; workflow execution; reporting/output validation and write; final status. A non-CSV/non-XLSX suffix raises `ValueError("only .csv and .xlsx inputs are supported by analyze")`. CLI passes all accepted options once to `run_analysis`, then calls `generate_analysis_report` once; it never calls any other domain API or writes a report itself.

On analyze success stdout is exactly one line, `Report written to: {str(Path(output_path))}`, and stderr is empty; exit status is 0. Typer parse/usage errors use its normal stderr and exit 2. With `--no-debug`, expected `ValueError`, `OSError`, and `FileExistsError` print exactly `str(error)` once to stderr, no traceback or native pandas/sklearn/Typer traceback, and exit 1. If reporting fails before the figure-ownership acquisition point, CLI closes the run's unique live Figures once in frozen plot-field order in a `finally` before either normal error presentation or debug re-raise; if reporting acquired ownership, CLI performs no Figure cleanup. With `--debug`, those exceptions are not caught after that `finally`, so their standard traceback is preserved and the command has a non-zero exit status. Help includes every option above plus `sharper --help` listing `analyze`; its exact Typer layout is not frozen.

## Determinism, tests, and allowlist

With the same input and non-`None` random state, result field order, selections, warning/limitation order, result tables, Figure order/metadata, section order, asset names, relative links, Markdown/HTML bytes and CLI success message are deterministic. No timestamp, duration, random asset name, random sampling beyond existing frozen plot behavior, or random ID is introduced.

Task 13 tests must lock public signatures, dataclass field order/frozen state and exports; both analysis-only and target/model branches; group selection; exact once-per-step call counts/order; no raw-DataFrame retention; no duplicate computation; classification/regression dispatch and wrong-type rejection; every preflight invariant and tampered nested result/collection/Figure; budget propagation; every frozen failure-matrix row through filesystem monkeypatching; report section/order/escaping/empty states/asset links/overwrite/residual conflict/Figure ownership/closure; CLI help/version, CSV/Excel, all options, normal/debug errors, 0/1/2 exits, and exactly-once workflow/report calls. Wrong-run-type tests must cover `None`, `dict`, an arbitrary object, and an `AnalysisRun` subclass, each asserting `pytest.raises(TypeError, match=r"^run must be an AnalysisRun$")`, no filesystem modification/PNG/staging/backup, and no reporting Figure close; a CLI-owned workflow run still closes through its `finally`. At least one tampered exact `AnalysisRun` test must instead assert `ValueError("analysis run has invalid schema")`. Ordinary pre-acquisition failure tests must assert no filesystem modification or PNG, no new staging/backup path, open caller-owned Figures, and the exact error; their CLI counterparts must assert the `finally` closure. Residual-path conflict tests must separately cover pre-existing report staging, assets staging, report backup, assets backup, and multiple simultaneous paths; each asserts every pre-existing temporary path is unchanged, no new path or PNG, unchanged finals, open caller-owned Figures for Python, CLI `finally` closure, and exact `FileExistsError`. Tests must separately assert first cleanup failure stops the second deletion with both backups retained, and second cleanup failure occurs only after report-backup deletion. Root version tests cover every eager precedence example above and prove zero I/O/workflow/reporting; subcommand version tests cover both frozen parse errors. Tasks 01--12 must remain unchanged and pass their full regression suite.

The preflight matrix must independently cover: wrong run type; wrong task;
missing/crossed classification-regression branch; target/features/holdout-position
mismatch; malformed warnings/limitations; every named upstream result table;
tampered prediction/metric data; malformed collection counts/order/type/source/
metadata; duplicate Figure identity; closed or non-Figure value; and each
optional-step/`skipped` inconsistency. Every case asserts the exact schema
error, no created path or PNG, and the frozen Figure-cleanup outcome.

The complete and exclusive Task 13 implementation allowlist is:

```text
src/sharper/workflow.py
src/sharper/reporting.py
src/sharper/cli.py
tests/test_workflow.py
tests/test_reporting.py
tests/test_cli.py
tests/test_public_api.py
README.md
docs/quickstart.md
docs/analysis-guide.md
docs/api.md
docs/decisions/task13-full-workflow-static-html-cli-contract.md
SPEC.md
IMPLEMENTATION_PLAN.md
AGENTS.md
```

No `src/sharper/__init__.py`, `pyproject.toml`, dependency group, upstream contract, lock/cache/build/generated report, `docs/.DS_Store`, or file outside this list may change. This contract-creation turn changes only documentation; it authorizes no implementation.
