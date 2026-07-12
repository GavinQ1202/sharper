# Task 11 分类基线、分类评估与评估图公共契约

## 1. 身份、前置任务与范围

**正式名称：** Task 11 — 分类基线、分类评估与评估图。

**目标：** 对结构化 DataFrame 提供严格 split-first 的分类基线、仅 holdout 的独立评估，以及确定性的静态分类评估图和 leakage 回归测试。

**前置任务：** Tasks 03、09、10。Task 03 提供 schema/ID-like 契约；Task 09 是计划中的 sequencing prerequisite；Task 10 冻结通用 `PlotResult`、`PlotCollection` 和 Figure 生命周期。Task 12 的回归路径在本任务之后；Task 13 才接入 workflow、reporting 和 CLI。

Task 11 实现分类训练、分类评估、混淆矩阵图和适用时的二分类 ROC 图。只有训练入口接受 raw DataFrame；评估和绘图只消费本文冻结的 result objects。

本任务不实现：回归、workflow、reporting、CLI、`AnalysisRun` 变更、文件/图片输出、HTML、持久化、交互 UI、网络访问、cross-validation、模型比较、调参、阈值优化、概率校准、自动特征工程、自动清洗、Task 07--10 统计或图的重算、新依赖、lock file 或公共自定义异常。

## 2. 架构、依赖与副作用边界

依赖方向固定为 `modeling -> schema`、`evaluation -> modeling`、`visualization -> evaluation`。`modeling.py` 不依赖 evaluation/visualization；`evaluation.py` 不依赖 visualization；`visualization.py` 不依赖 workflow/reporting/CLI。Task 11 不调用 Tasks 07--10 public APIs。

`train_classifier` 依次验证输入、选择排除列、拒绝 time-risk、分离 `X/y`、确定 train/holdout membership、仅从训练分区确定列和 fit Pipeline，再返回 `TrainingResult`。`evaluate_classifier` 只消费 `TrainingResult` 的 holdout；`plot_classification_evaluation` 只读取已通过完整 validation 的 `ClassificationEvaluation`，不得 fit、predict 或重新评估。所有 validation 在相应的 split、fit、prediction、Figure 或文件副作用之前完成。所有 public function 不修改输入，也不产生文件或部分外部输出。

## 3. Public API 和 exports

Task 11 只新增以下 `sharper` 顶层 exports；不得增加别名、裸 Axes API、`ax`、`fig`、`show`、`save`、`style`、`figsize`、`palette` 或 `**kwargs`：

```python
def train_classifier(
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

def evaluate_classifier(result: TrainingResult) -> ClassificationEvaluation: ...

def evaluate_model(
    result: TrainingResult,
) -> ClassificationEvaluation | RegressionEvaluation: ...

def plot_classification_evaluation(
    result: ClassificationEvaluation,
) -> PlotCollection: ...
```

`RegressionEvaluation` remains a forward public type reference owned by Task 12; Task 11 neither creates nor exports it. Until Task 12 is implemented, `evaluate_model` is a thin classification-only convenience dispatcher. Task 12's accepted contract extends it to accept the new `RegressionTrainingResult` and invoke `evaluate_regressor` once; this does not alter the classification input, its classification validation, or its exactly-once `evaluate_classifier` behavior. `evaluate_classifier` performs the unique complete TrainingResult schema validation. Neither Task 11 nor its Task 12 extension makes `evaluate_model` train, split, fit, clone, infer schema, select features or preprocess.

## 4. Frozen result types and ownership

The exact frozen dataclasses are:

```python
@dataclass(frozen=True)
class TrainingResult:
    task: Literal["classification"]
    target: str
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    time_column: str | None
    schema: SchemaReport
    pipeline: Pipeline
    estimator: ClassifierMixin
    classes: tuple[str | int | bool, ...]
    train_row_positions: tuple[int, ...]
    test_row_positions: tuple[int, ...]
    X_test: pd.DataFrame
    y_test: tuple[str | int | bool, ...]
    test_size: float
    random_state: int | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

@dataclass(frozen=True)
class ClassificationEvaluation:
    task: Literal["classification"]
    target: str
    holdout_positions: tuple[int, ...]
    classes: tuple[str | int | bool, ...]
    y_true: tuple[str | int | bool, ...]
    y_pred: tuple[str | int | bool, ...]
    score_kind: Literal["predict_proba", "decision_function"] | None
    positive_label: str | int | bool | None
    scores: tuple[float, ...] | None
    roc_curve: tuple[tuple[float, float, float], ...]
    metrics: tuple[tuple[str, float], ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    roc_auc: float | None
    limitations: tuple[str, ...]
```

Fields are in exactly this order. `schema` is a Task 03 `SchemaReport` made from train rows only. Train/test positions are zero-based original row positions, unique and disjoint, and together cover the input exactly once. `X_test` contains only holdout columns in `feature_columns` order; `y_test` aligns with `test_row_positions`. `estimator` is exactly `pipeline.named_steps["estimator"]`, the fitted clone. `warnings` is an ordered subset of `("duplicate_index", "duplicate_rows", "custom_estimator_random_state_not_managed")`; `limitations` is an ordered subset of `("random_state_none", "custom_estimator_determinism_not_guaranteed")`.

The dataclasses are shallowly frozen. `Pipeline`, estimator and `X_test` are caller-owned mutable objects; callers may inspect or mutate them, but a subsequently malformed result is rejected. Public functions never mutate them. Validation checks observable task, names, positions, table shape/order, pipeline/estimator identity, fitted `classes_`, and metadata consistency; it does not attempt to prove an estimator's opaque learned state has not been externally changed.

`classes` use first appearance order in the original target, never Python cross-type sorting. Labels must be scalar `str`, integral non-bool integer, or `bool`; no missing label and no mixture of those three kinds is allowed. `classes` are unique by value and type. `y_true`/`y_pred` contain only `classes`; all holdout-related sequences have length `n_test == len(holdout_positions)`. For binary evaluation, `positive_label` is `classes[1]`; for multiclass it is `None`.

`metrics` is exactly, in order, `(("accuracy", value), ("balanced_accuracy", value), ("f1_macro", value))`, with finite float values in `[0,1]`. Macro F1 is calculated with `zero_division=0`; precision, recall, PR-AUC, log loss and weighted/micro metrics are not Task 11 outputs. `confusion_matrix` is a square tuple of non-negative built-in integers in `classes` order and sums to `n_test`.

For binary score-capable evaluation, `score_kind` is `"predict_proba"` or `"decision_function"`, `scores` has one finite score per holdout row, `roc_curve` is non-empty `(false_positive_rate, true_positive_rate, threshold)` triples, and `roc_auc` is finite in `[0,1]`. A probability score is the estimator `classes_` column mapped to `positive_label`. A decision-function score is used unchanged when `positive_label == estimator.classes_[1]`, otherwise negated; any other binary class mapping is invalid output. ROC detail is exactly `sklearn.metrics.roc_curve(y_true, scores, pos_label=positive_label, drop_intermediate=False)` and AUC is `sklearn.metrics.roc_auc_score(tuple(value == positive_label for value in y_true), scores)`. ROC rates are finite in `[0,1]`; thresholds are finite except that the first may be positive infinity, matching the frozen sklearn ROC representation. For multiclass or unavailable scores, all four are respectively `None`, `None`, `()`, `None`; `limitations` then contains exactly one applicable reason, `"multiclass_roc_unavailable"` or `"score_unavailable"`. No raw full DataFrame, timestamp, duration or random artifact ID is stored.

## 5. Training, exclusions and time risk

`df` must be a pandas DataFrame with unique string column names. `target` is a present string column. `exclude_columns` is a non-string sequence of unique present string names; it may not contain `target`. `time_column` is `None` or a present string name distinct from `target`. If explicit `features` is supplied, it is a non-string non-empty sequence of unique present string names, excludes `target`, and has no overlap with `exclude_columns`.

Exclusions and target removal occur before schema inference. After the split positions are fixed, `infer_schema(X_train)` is called only on the training partition after that removal; `infer_schema` is never called on the complete DataFrame to decide model features. Task 03 ID-like detection, dtype eligibility and every data-dependent final feature eligibility decision read training rows only. This includes unique-rate/identifier evidence and any inspected constant status; Task 11 has no near-constant eligibility rule. Final retained features preserve original DataFrame column order among the training-validated candidates. Default feature selection removes Task 03 ID-like columns; explicit ID-like feature columns are rejected. Posterior, future and entity-risk semantics cannot be reliably inferred from dtype/schema; the caller must supply every known such column in `exclude_columns`. Once supplied it cannot enter the feature set, split-time schema work, preprocessing or estimator input. `exclude_columns` order never affects feature order.

Task 11 supports random holdout only. `time_column is not None` always rejects before split. Any datetime or timedelta DataFrame column also rejects before split, including a column listed in `exclude_columns`; pure numeric time encodings cannot be reliably detected and must be declared through `time_column`. Complex feature dtypes reject. Eligible paths are real non-boolean numeric and categorical/object/string/category/boolean. At least one eligible feature is required.

Target labels must be complete, valid under the label rule above, have at least two classes, and have at least two rows per class. `test_size` is a non-bool finite real strictly between zero and one. `random_state` is a non-bool integer `>= 0` or `None`. Splitting is exactly `train_test_split(..., stratify=y, random_state=random_state)`; infeasible stratification rejects. `random_state=42` is the default.

The default is a newly constructed `LogisticRegression(max_iter=1000, random_state=random_state)`. A custom estimator must expose `_estimator_type == "classifier"`, `fit` and `predict`, then be passed through `sklearn.base.clone`; clone failure rejects. The caller's estimator is never fitted or modified. A custom clone preserves its own parameters; Sharper never writes its `random_state`. Thus a custom estimator with its own randomness guarantees only deterministic splitting, records `"custom_estimator_random_state_not_managed"` and `"custom_estimator_determinism_not_guaranteed"`, and does not promise identical fitted state. After interface validation, an `Exception` from `fit` is raised as `ValueError("classifier estimator fit failed")` with the original exception as its cause; `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` are never caught.

After membership is fixed, a Pipeline fits a training-only `ColumnTransformer`: numeric `SimpleImputer(strategy="median")` then `StandardScaler`; categorical `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")`; then the estimator clone. No imputer statistic, category vocabulary, scaler state, class weight, threshold, selection decision or fitted estimator state may observe holdout rows. Input DataFrame/index/columns/dtypes/values remain unchanged.

## 6. Evaluation and estimator-output validation

`evaluate_classifier` accepts only a fully valid classification `TrainingResult`. It calls `pipeline.predict(X_test)` exactly once and never fits, splits or scores training data. After interface validation, an `Exception` from `predict` is raised as `ValueError("classifier estimator prediction failed")` with the original exception as its cause; `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` are never caught. A successful prediction must be one-dimensional, length `n_test`, finite/non-missing as applicable, and contain only fitted classes.

The fitted estimator must expose non-empty, unique `classes_` with the same type-aware label set as `TrainingResult.classes`; its order may differ and is mapped explicitly. For binary results evaluation first uses `predict_proba` when callable; otherwise it uses `decision_function` when callable; otherwise ordinary metrics and confusion matrix remain valid but score/ROC fields are unavailable. An `Exception` from the selected `predict_proba` call is raised as `ValueError("classifier estimator probability prediction failed")`; an `Exception` from the selected `decision_function` call is raised as `ValueError("classifier estimator score prediction failed")`; both preserve their causes and neither catches `KeyboardInterrupt`, `SystemExit`, or `GeneratorExit`. `predict_proba` must be two-dimensional with shape `(n_test, len(estimator.classes_))`, finite values in `[0,1]`, and each row sum within `1e-12` of one. Binary `decision_function` must be a finite one-dimensional length-`n_test` score mapped to the frozen positive label. No score interface is invoked for multiclass results.

Metrics are exactly `sklearn.metrics.accuracy_score(y_true, y_pred)`, `sklearn.metrics.balanced_accuracy_score(y_true, y_pred)`, and `sklearn.metrics.f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)`, in the frozen order. Even when holdout lacks a training class, macro F1 includes every `classes` item through `labels=classes`; balanced accuracy uses sklearn's standard classes actually present in `y_true` and does not add an absent class. All three metrics must be finite and ordinary sklearn metric warnings are not exposed to callers. The shared complete `ClassificationEvaluation` validator may deterministically reconstruct these metrics, the confusion matrix and applicable ROC detail from frozen fields with absolute tolerance `1e-12`; it never calls an estimator. `plot_classification_evaluation` itself does not call sklearn metric functions or rebuild metrics: it calls the shared validator, then only reads the validated fields. Figure creation occurs after that validation completes.

## 7. Classification evaluation plots

Task 10's nine chart types, five sources, metadata schemas and collection semantics remain unchanged for its six APIs. Task 11 independently uses the common containers with only these new values:

| chart type | source / item | exact title and axes | exact metadata |
|---|---|---|---|
| `classification_confusion_matrix` | `"classification_evaluation"` / `None` | title `"Classification confusion matrix"`; x `"Predicted label"`; y `"True label"` | `(("target", target), ("classes", compact_json), ("n_test", decimal_n_test), ("metric", "count"))` |
| `classification_roc_curve` | `"classification_evaluation"` / `None` | title `"ROC curve ({target})"`; x `"False positive rate"`; y `"True positive rate"`; legend `"ROC"` | `(("target", target), ("classes", compact_json), ("positive_label", str(positive_label)), ("n_test", decimal_n_test), ("score_kind", score_kind), ("roc_auc", format(roc_auc, ".12g")), ("metric", "roc_auc"))` |

`compact_json` is `json.dumps([str(label) for label in classes], ensure_ascii=False, separators=(",", ":"))`; `decimal_n_test` is `str(len(holdout_positions))`; counts use decimal strings. Metadata is an exact ordered tuple of string pairs, with no extra key. Figures use consecutive integer tick locations and `str(label)` in frozen class order; equal visible text remains distinct positions. Confusion cells are integer-count annotations. No p-value, effect size, importance, significance or causal wording appears.

The collection has no caller-controlled budget: `requested_count=2` means the two approved chart slots attempted by this API, not a user request. `available_count=2` and `actual_count=2` for valid binary score results; both are `1` for multiclass or score-unavailable results. `actual_count == len(plots)`, `truncated is False`, and `truncation_reason is None` in every valid result. Plot order is confusion matrix then ROC if available. A valid evaluation never returns an empty collection.

Every figure is new and caller-owned. The function accepts no Figure/Axes, does not save, show or close, switch backend, mutate `rcParams`, or call a global seaborn style/theme/palette API. Invalid evaluation results fail before Figure creation. The ROC figure reads stored `roc_curve`; it never derives coordinates, predicts or recomputes a metric. This contract freezes plot semantics, data source, title, axes, order, metadata and Figure lifecycle only; unless expressly listed, colormap, line color, grid and reference-line styling are deterministic implementation details rather than public vocabulary.

## 8. Stable errors and precedence

Only built-in `ValueError` is public. `bool` is never an integer or real. Exact messages are:

| condition | message |
|---|---|
| wrong DataFrame | `df must be a pandas DataFrame` |
| invalid DataFrame names | `DataFrame column names must be unique strings` |
| invalid target/time argument | `target must be a column name string` / `time_column must be a column name string or None` |
| absent target/time/feature/exclusion | `target column not found: {target!r}` / `time column not found: {time_column!r}` / `feature column not found: {feature!r}` / `excluded column not found: {column!r}` |
| invalid features/exclusions | `features must be a non-empty sequence of unique column names` / `exclude_columns must be a sequence of unique column names` |
| target in exclusions/features or overlap | `target must not appear in exclude_columns` / `target must not appear in features` / `features and exclude_columns must not overlap` |
| time risk | `time-ordered classification is not supported` |
| no eligible feature / bad feature dtype | `no eligible model features` / `unsupported model feature dtype: {feature!r}` |
| invalid target labels/classes | `classification target labels must be complete homogeneous scalar values` / `classification target must contain at least two classes with two rows each` |
| invalid split/seed | `test_size must permit a stratified split strictly between 0 and 1` / `random_state must be a non-negative integer or None` |
| invalid estimator / clone | `estimator must be a classifier` / `classifier estimator could not be cloned` |
| estimator interface/output/execution | `classifier estimator does not support required prediction interface` / `classifier estimator has invalid output` / `classifier estimator fit failed` / `classifier estimator prediction failed` / `classifier estimator probability prediction failed` / `classifier estimator score prediction failed` |
| wrong/malformed training result | `result must be a TrainingResult` / `training result has invalid schema` |
| Task 12 dispatch before Task 12 implementation | `evaluate_model only supports classification in Task 11` |
| wrong/malformed evaluation result | `result must be a ClassificationEvaluation` / `classification evaluation result has invalid schema` |

`train_classifier` precedence is DataFrame type, scalar keyword types/ranges, DataFrame names, target, exclusions, time risk, features, target labels/classes, feature eligibility, estimator/clone/interface, split, fit execution, then fitted-output validation. `evaluate_classifier` precedence is result type, TrainingResult schema, task, observable fitted-estimator interface, prediction execution, prediction output, score execution, score output, then evaluation construction. Before Task 12 implementation, `evaluate_model` precedence is result type, unsupported task, then delegated `evaluate_classifier` validation/error. Task 12's accepted contract supersedes only that dispatcher rule: it first accepts the `TrainingResult | RegressionTrainingResult` union, then dispatches by task to its exactly-once delegate; classification-result validation and errors remain unchanged. Plot precedence is result type, shared ClassificationEvaluation schema/detail/metric consistency validation, availability, then Figure creation. Duplicate/missing/unknown/non-finite/wrong-dtype result members use the corresponding frozen malformed-schema error; no coercion, filling, sorting, recovery or native numpy/sklearn exception leakage is permitted.

## 9. Determinism, testing and allowed files

With default estimator and a non-`None` identical random state, split positions, feature order, schema snapshot, pipeline column order, classes, evaluation detail, metrics, plot order and metadata are deterministic. A custom estimator's internal randomness is its caller's responsibility as frozen above. No timestamp, generated ID, random jitter, hash-order iteration, network input or filesystem-derived name is used.

Tests must cover exact signatures/exports/frozen field order; exclusions and posterior/future/entity exclusion; declared time, datetime and timedelta rejection; split-first train-only schema/ID-like/eligibility decisions with holdout-only categories/extremes; clone/ownership and custom randomness; estimator fit/predict/probability/score execution wrapping and malformed output; first-appearance label order, mixed-label rejection, positive label and zero division; binary/multiclass/score-unavailable evaluation; exact metric reconstruction; `evaluate_model` one-time delegation with no fit/split; plot no-predict/no-metric calls, Figure lifecycle/global state and no Figure on validation failure; input/result immutability; Tasks 01--10 regressions; and dependency/file-boundary checks.

Task 11 implementation may modify only the Task 11 plan allowlist: `src/sharper/modeling.py`, `src/sharper/evaluation.py`, `src/sharper/visualization.py`, `src/sharper/__init__.py`, `tests/test_modeling.py`, `tests/test_evaluation.py`, `tests/test_visualization.py`, `tests/test_public_api.py`, `docs/leakage.md`, `docs/api.md`, `README.md`, and this contract. It must not modify workflow, reporting, CLI, I/O, schema, summary, quality, analysis, features, Tasks 01--10 contracts, dependencies, lock files, caches, build outputs or `docs/.DS_Store`. The narrowly scoped Task 10 clarification made before Task 11 implementation is not implementation authorization to alter Task 10 again. README's separate Task 10 completion-state inconsistency is a P3 documentation-cleanup item, does not block Task 11, and is handled with Task 11 completion documentation. Task 12+ remains unimplemented. On completion docs/api, docs/leakage and README must describe Task 11 as an independent API; Task 13 integration remains deferred.
