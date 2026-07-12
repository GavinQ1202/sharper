# Task 12 回归基线、回归评估与评估图公共契约

## 1. 身份、目标与范围

**正式名称：** Task 12 — 回归基线、回归评估与评估图。

**目标：** 对结构化 `pandas.DataFrame` 提供严格 split-first 的回归基线、仅 holdout 的独立评估，以及确定性的静态回归评估图。回归路径共享已批准的安全预处理原则，但不复用分类专用的标签、分层切分、ROC 或混淆矩阵语义。

**上游依赖：** Task 03 的 `SchemaReport`/`infer_schema` 契约；Task 10 的 `PlotResult`、`PlotCollection` 和 Figure 生命周期；Task 11 已完成的分类路径及 `evaluate_model` 的分类分派。Task 09 只是计划中的 sequencing prerequisite，不是 Task 12 的运行时 API 依赖。

**下游边界：** Task 13 才能把训练、评估或 Figure 接入 `workflow.py`、`reporting.py`、HTML、图片资产或 CLI。Task 12 不修改 `AnalysisRun`，不改变 Task 05 已冻结的 `"modeling"`/`"visualization"` skipped capability。

Task 12 只实现回归训练、回归评估、预测值对实际值图和残差图。它不实现：分类行为变更、workflow、reporting、CLI、文件/图片导出、HTML、持久化、交互 UI、网络访问、cross-validation、模型比较、调参、阈值优化、概率校准、自动特征工程、自动清洗、树模型、AutoML、Task 07--10 统计/图重算、新依赖、lock file 或公共自定义异常。

`TrainingResult` 是 Task 11 冻结的分类专用类型，Task 12 不改变其字段、注解、分类路径或分类错误行为。为避免将 `classes` 等分类语义伪装为回归数据，Task 12 新增独立的 `RegressionTrainingResult`；Task 12 对 `evaluate_model` 的唯一兼容性扩展是接受该新类型并分派给 `evaluate_regressor`。

## 2. 架构、数据流与副作用边界

依赖方向固定为 `modeling -> schema`、`evaluation -> modeling`、`visualization -> evaluation`。`modeling.py` 不依赖 evaluation/visualization；`evaluation.py` 不依赖 visualization；`visualization.py` 不依赖 workflow/reporting/CLI。

| public function | 允许输入/读取 | 禁止读取或调用 |
|---|---|---|
| `train_regressor` | caller 的 raw DataFrame、Task 03 `infer_schema(X_train)` | Task 07--11 public APIs；完整 DataFrame 的 data-dependent schema/ID-like/feature decision |
| `evaluate_regressor` | 已验证 `RegressionTrainingResult` 的 fitted pipeline、`X_test`、`y_test`、holdout positions | raw DataFrame、fit、split、schema inference、feature selection、Task 07--11 public APIs |
| `evaluate_model` | `TrainingResult` 或 `RegressionTrainingResult` 的运行时类型与 `task` | fit、split、schema inference、任何绘图或指标重算 |
| `plot_regression_evaluation` | 已验证 `RegressionEvaluation.predictions`、`metrics`、metadata | raw DataFrame、estimator、pipeline、predict、fit、split、sklearn metric functions、Tasks 07--11 public APIs |

允许直接导入 Task 03 的 `SchemaReport`/`infer_schema`、Task 10 的容器类型和 Task 11 的 `TrainingResult`/`ClassificationEvaluation` 类型，以维持这些已批准的接口；这不是调用对应 Task 的 public function。回归 `train_regressor`、`evaluate_regressor` 和 `plot_regression_evaluation` 路径不调用 Task 07--11 public APIs。唯一例外是 `evaluate_model` 的 classification branch：它精确调用 Task 11 `evaluate_classifier` 一次；regression branch 只精确调用 `evaluate_regressor` 一次。该 dispatcher 扩展不得引入任何新的 Task 11 classification signature、validation、return semantics 或其他行为变化。所有 public functions 不修改输入、result、其 DataFrame、pipeline、estimator、Figure 或 metadata，不写文件，也不产生部分外部输出。

训练路径必须先完成输入验证、target/exclusion/time-risk 验证和 `X/y` 分离，再固定 train/holdout membership；只在 `X_train` 上调用 `infer_schema`、决定 ID-like/最终可用 feature，并 fit `ColumnTransformer`/`Pipeline`。已知 target direct derivative、future、posterior 或 entity-risk 列必须由调用者通过 `exclude_columns` 提供；它们在 split、schema、feature selection、fit 前移除。随机 holdout 不是 time-safe；`time_column`、任何 datetime/timedelta 列均在 split 前拒绝。holdout 只用于一次最终预测、指标和图；不得参与 fitted state、特征决定、阈值、模型选择或重试。

## 3. Public API 与 exports

Task 12 只新增以下顶层 `sharper` exports，且只按本节签名实现。除已有 `evaluate_model` 的回归分派外，不增加别名、裸 Axes API、`ax`、`fig`、`show`、`save`、`style`、`figsize`、`palette` 或 `**kwargs`：

```python
def train_regressor(
    df: pd.DataFrame,
    target: str,
    *,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
    time_column: str | None = None,
    estimator: RegressorMixin | None = None,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> RegressionTrainingResult: ...

def evaluate_regressor(
    result: RegressionTrainingResult,
) -> RegressionEvaluation: ...

def evaluate_model(
    result: TrainingResult | RegressionTrainingResult,
) -> ClassificationEvaluation | RegressionEvaluation: ...

def plot_regression_evaluation(
    result: RegressionEvaluation,
) -> PlotCollection: ...
```

`features`、`exclude_columns`、`time_column`、`estimator`、`test_size` 和 `random_state` 全部是 keyword-only。`evaluate_model` 是唯一便利分派函数：对 `TrainingResult(task="classification")` 精确调用 Task 11 `evaluate_classifier` 一次，对 `RegressionTrainingResult(task="regression")` 精确调用 `evaluate_regressor` 一次，并返回各 delegate 产生的同一对象；它不调用完整 result validator、训练、预测、split、fit、clone、schema inference 或绘图。classification branch 的 signature、validation 和 return semantics 保持 Task 11 冻结行为不变。

## 4. 冻结结果类型、字段与所有权

以下都是 `@dataclass(frozen=True)`；字段名、顺序和类型注解完全冻结：

```python
@dataclass(frozen=True)
class RegressionTrainingResult:
    task: Literal["regression"]
    target: str
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    time_column: str | None
    schema: SchemaReport
    pipeline: Pipeline
    estimator: RegressorMixin
    train_row_positions: tuple[int, ...]
    test_row_positions: tuple[int, ...]
    X_test: pd.DataFrame
    y_test: tuple[float, ...]
    test_size: float
    random_state: int | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

@dataclass(frozen=True)
class RegressionEvaluation:
    task: Literal["regression"]
    target: str
    holdout_positions: tuple[int, ...]
    predictions: pd.DataFrame
    metrics: tuple[tuple[str, float], ...]
    limitations: tuple[str, ...]
```

`RegressionTrainingResult.schema` 必须是只用 train rows 构造的 Task 03 `SchemaReport`。train/test positions 是零基原始行位置、各自唯一且互不相交，并恰好覆盖输入的全部位置。`X_test` 仅含 `feature_columns`（该顺序）；`y_test` 是与 `test_row_positions` 等长的 built-in `float` tuple。`estimator is pipeline.named_steps["estimator"]`，且是调用者 estimator 的 fitted clone。`warnings` 是按以下词汇顺序出现的无重复子集：`("duplicate_index", "duplicate_rows", "custom_estimator_random_state_not_managed")`；`limitations` 是按以下词汇顺序出现的无重复子集：`("random_state_none", "custom_estimator_determinism_not_guaranteed")`。

`predictions` 的 schema 严格为列顺序 `("row_position", "actual", "predicted", "residual")`，dtypes 严格为 `("int64", "float64", "float64", "float64")`，行顺序与 `holdout_positions` 相同；`row_position` 恰等于该 tuple，`actual` 恰等于训练 result 的 `y_test`，`residual == actual - predicted`。所有 `actual`、`predicted`、`residual` 均为有限值。`metrics` 精确且依序为 `(("mae", value), ("rmse", value), ("r2", value))`，各 value 是 finite built-in `float`；MAE/RMSE 非负，R² 可为任意有限实数。`RegressionEvaluation.limitations == ()`；v0.1 不保留未声明的评估降级路径。

dataclass 仅浅层 frozen：pipeline、estimator、`X_test` 与 `predictions` 是调用者持有的可变对象。库函数不修改它们；外部修改致使 observable schema 或一致性失效时，后续调用必须稳定拒绝。结果中不得存储完整 raw DataFrame、原始 index 标签、timestamp、duration、random ID、文件 path 或 artifact。

## 5. 训练、评估、图与 validation 合同

`df` 必须为列名唯一且全为 string 的 DataFrame。`target` 是存在的 string 列。`features` 与 `exclude_columns` 是非 string sequence 的唯一 present string names；`features` 非空；二者均不得包含 target 且不得重叠。`time_column` 是 `None` 或存在且不同于 target 的 string 列。

回归 target 必须是 real、non-bool、non-complex numeric dtype，所有值非缺失且 finite，并至少包含两个不同有限值；无隐式 dtype coercion。`test_size` 是 non-bool finite real 且严格在 `(0, 1)`；`random_state` 是 non-bool non-negative integer 或 `None`。随机切分精确为 `train_test_split(..., random_state=random_state)`，不 stratify；切分后 train 与 holdout 均至少有两行。默认 estimator 是新建 `Ridge(random_state=random_state)`；custom estimator 必须有 `_estimator_type == "regressor"`、`fit` 和 `predict`，并通过 `sklearn.base.clone`。调用者 estimator 不被 fit 或修改；自定义 clone 的内部随机性由调用者负责并记录相应 warning/limitation。

训练 feature eligibility 与 Task 11 一致但不读取其 public API：只接受 real non-bool non-complex numeric，或 object/string/category/bool；datetime、timedelta、complex 均拒绝。default selection 排除 Task 03 ID-like 列；explicit ID-like feature 拒绝；至少保留一个 eligible feature。拟合的 Pipeline 只能在 train data 上构造和 fit：numeric 是 `SimpleImputer(strategy="median")` 再 `StandardScaler`，categorical 是 `SimpleImputer(strategy="most_frequent")` 再 `OneHotEncoder(handle_unknown="ignore")`，然后 fitted regressor。输入 DataFrame/index/columns/dtypes/values 不得修改。

`evaluate_regressor` 先完整验证 `RegressionTrainingResult`，再只调用 `pipeline.predict(X_test)` 一次。预测输出只接受 numeric `numpy.ndarray`、`pandas.Series` 或由 `numpy.asarray` 得到 non-object、real、non-bool numeric dtype 的 array-like；它必须一维、长度等于 holdout rows，且全部 finite。bool dtype、object dtype、string numeric、array-valued elements、mixed Python objects、NaN、positive/negative infinity、错误 shape 或错误长度均稳定以 `ValueError("regressor estimator has invalid output")` 拒绝；不得依赖或暴露 numpy/sklearn 原生异常，也不得做 dtype coercion。不得 fit、split 或读取训练 data。返回表使用 `actual - predicted` residual；metrics 精确为 `sklearn.metrics.mean_absolute_error`、`sqrt(sklearn.metrics.mean_squared_error)` 和 `sklearn.metrics.r2_score(y_true, y_pred, force_finite=True)` 的相应结果。constant holdout target 不报错，唯一采用 sklearn `force_finite=True` 语义；三项 metric 始终 finite、`RegressionEvaluation.limitations` 仍为 `()`，且不得静默以本库逻辑将 NaN 替换为零或其他值。shared `RegressionEvaluation` validator 可由 frozen table 重建这些三项并以绝对容差 `1e-12` 比较，但绝不调用 estimator。`plot_regression_evaluation` 只调用该 shared validator，之后只读取 frozen fields；不调用 sklearn metrics 或计算新统计。

Task 10 的九个 chart type、五个 source、metadata schema 和六个 API 行为不变。Task 12 仅为已有容器新增下列封闭值：

| chart type | source / item | title、axes 与固定元素 | exact ordered metadata |
|---|---|---|---|
| `regression_predicted_vs_actual` | `"regression_evaluation"` / `None` | title `"Predicted vs actual ({target})"`；x `"Actual value"`；y `"Predicted value"`；一条实际值范围的 `y=x` reference line，legend `"Ideal"` | `(("target", target), ("n_test", decimal_n_test), ("metric", "prediction"))` |
| `regression_residuals` | `"regression_evaluation"` / `None` | title `"Residuals ({target})"`；x `"Predicted value"`；y `"Residual (actual - predicted)"`；一条 `y=0` reference line，legend `"Zero residual"` | `(("target", target), ("n_test", decimal_n_test), ("metric", "residual"))` |

`decimal_n_test` 是 `str(len(holdout_positions))`。不采样、不 jitter、不排序、不截断；点严格依照 `predictions` 行顺序绘制。collection 恒为 `requested_count=2`、`available_count=2`、`actual_count=2`、`truncated=False`、`truncation_reason=None`，plot 顺序严格为 predicted-vs-actual 后 residuals；有效 result 不返回空 collection。每张图都是新建、caller-owned `Figure`；不接收或返回 bare Axes，不 show/save/close，不切 backend，不修改 `rcParams`，不调用 global seaborn style/theme/palette API。除上述 reference lines 外，不显示 p-value、effect size、importance、significance 或因果措辞。

## 6. Stable errors、malformed result 与 precedence

公共异常只有 built-in `ValueError`；`bool` 从不视为 integer 或 real。消息精确如下：

| condition | message |
|---|---|
| wrong DataFrame | `df must be a pandas DataFrame` |
| invalid DataFrame names | `DataFrame column names must be unique strings` |
| invalid target/time argument | `target must be a column name string` / `time_column must be a column name string or None` |
| absent target/time/feature/exclusion | `target column not found: {target!r}` / `time column not found: {time_column!r}` / `feature column not found: {feature!r}` / `excluded column not found: {column!r}` |
| invalid features/exclusions | `features must be a non-empty sequence of unique column names` / `exclude_columns must be a sequence of unique column names` |
| target in exclusions/features or overlap | `target must not appear in exclude_columns` / `target must not appear in features` / `features and exclude_columns must not overlap` |
| time risk | `time-ordered regression is not supported` |
| invalid target | `regression target must be complete finite real numeric values` / `regression target must contain at least two distinct values` |
| no eligible feature / bad dtype | `no eligible model features` / `unsupported model feature dtype: {feature!r}` |
| invalid split/seed | `test_size must be strictly between 0 and 1` / `test_size must produce at least two train and holdout rows` / `random_state must be a non-negative integer or None` |
| invalid estimator / clone | `estimator must be a regressor` / `regressor estimator could not be cloned` |
| estimator interface/output/execution | `regressor estimator does not support required prediction interface` / `regressor estimator has invalid output` / `regressor estimator fit failed` / `regressor estimator prediction failed` |
| wrong/malformed training result | `result must be a RegressionTrainingResult` / `regression training result has invalid schema` |
| wrong/malformed evaluation result | `result must be a RegressionEvaluation` / `regression evaluation result has invalid schema` |
| unsupported dispatcher input/task | `result must be a TrainingResult or RegressionTrainingResult` / `evaluate_model supports only classification or regression` |

`train_regressor` precedence is DataFrame type, scalar keyword types/ranges, DataFrame names, target, exclusions, time risk, features, target values, feature eligibility, estimator/clone/interface, split, fit execution, then fitted-output validation. `evaluate_regressor` precedence is result type, full training-result schema, observable fitted-estimator interface, prediction execution, prediction output, then evaluation construction. `evaluate_model` precedence is union result type, task, then its exactly-once delegate. Plot precedence is evaluation-result type, complete evaluation schema/table/metric consistency, then Figure creation.

Malformed means any missing/wrong field type; task/value vocabulary violation; duplicate/missing/unknown/non-finite member; non-tuple sequence; wrong position coverage/order; pipeline/estimator identity mismatch; invalid `SchemaReport`; invalid table columns/dtypes/rows; incorrect residual; wrong metrics names/order/value; or external mutation inconsistent with the frozen fields. No coercion, fill, deduplication, sort, recovery, estimator call, Figure creation, or native numpy/pandas/sklearn exception leakage is allowed. Estimator `fit`/`predict` failures wrap their original `Exception` as the matching stable `ValueError` with the cause retained; `KeyboardInterrupt`, `SystemExit` and `GeneratorExit` are never caught.

## 7. Determinism、测试与允许文件

With default estimator and identical non-`None` random state, split positions, feature order, schema snapshot, pipeline column order, holdout table, metrics, plot order and metadata are deterministic. A custom estimator's internal randomness remains the caller's responsibility. No timestamp, generated ID, random jitter, hash-order iteration, network input, filesystem-derived name, sampling or hidden truncation is used.

Task 12 tests must cover exact signatures/top-level exports/keyword-only parameters/frozen field order and annotations; default Ridge and custom regressor; target/exclusion/time validation; train-only schema/ID-like/eligibility and holdout-only categories/extremes; duplicate warnings; clone/ownership/custom randomness; input/result immutability; estimator fit/predict failure and malformed output (including bool/object/string-numeric/array-valued/mixed-object/non-finite/wrong-shape/wrong-length predictions); malformed training/evaluation tables/results; MAE/RMSE/R² parity and table/position/residual alignment; constant holdout target with finite deterministic `force_finite=True` R²; strict `evaluate_classifier`/`evaluate_regressor` task rejection and one-time `evaluate_model` dispatch; no estimator/metric/recomputation calls in plotting; two Figures, exact titles/axes/reference lines/metadata/global state/lifecycle; deterministic repeated output; no sampling/truncation; and Tasks 01--11 regression tests.

Task 12 implementation may modify only: `src/sharper/modeling.py`, `src/sharper/evaluation.py`, `src/sharper/visualization.py`, `src/sharper/__init__.py`, `tests/test_modeling.py`, `tests/test_evaluation.py`, `tests/test_visualization.py`, `tests/test_public_api.py`, `docs/api.md`, `README.md`, this contract, `SPEC.md`, `IMPLEMENTATION_PLAN.md`, and `AGENTS.md`. The Task 11 classification dispatcher clarification is in `docs/decisions/task11-classification-baseline-evaluation-visualization-contract.md`; Task 12 implementation must comply with it but must not modify that contract. It must not modify workflow, reporting, CLI, I/O, schema, summary, quality, analysis, features, Task 10 containers or chart behavior, dependencies, lock files, caches, build outputs or `docs/.DS_Store`. Task 13 remains unimplemented.
