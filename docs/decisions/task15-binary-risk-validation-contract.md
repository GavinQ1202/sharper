# Task 15 二分类风险验证与基础业务指标公共契约

## 1. 状态、身份与权威边界

**状态：Approved — Go。** 本记录冻结已经通过 final contract review 的 Task 15 合同；
合同批准不等于实现完成、版本发布或 current public surface 已发生变化。

**批准记录：** final contract review verdict：`Go`；P0：`0`；P1：`0`；implementation
blocker：`none`。Implementation 必须严格遵循第 18 节 allowlist。

**Implementation 状态：Implementation complete — review Go。** Bounded closure review
verdict：`Go`；P0：`0`；P1：`0`；implementation blocker：`none`。Implementation
已完成，但尚未 commit、tag、publish 或 release；v0.2 整体仍未完成或发布。

**正式名称：** Task 15 — Binary Risk Validation and Business Metrics。

**前置依赖：** v0.1 Tasks 01--14 的稳定基线，以及已批准的
`docs/decisions/v02-roadmap-contract.md`。本文服从根目录 `AGENTS.md`、`SPEC.md`、
`IMPLEMENTATION_PLAN.md` 与该 roadmap；本文不重新讨论或扩大 v0.2 范围。

Task 15 的唯一目标是在不改变 v0.1 分类、workflow、report、CLI 和 public result 的
前提下，为风险型二分类提供：显式正类与风险方向、可审查 validation folds、逐行一次
的 validation/OOF prediction、时间标签成熟度、ranking/probability 指标、calibration
诊断、调用者预声明 threshold 的分析证据，以及不涉及策略动作的基础风险/损失汇总。
Task 15 同时提供一个 opt-in、result-only 的静态可视化入口，只把本文冻结的结果表映射为
调用者持有的 matplotlib Figure；该入口不读取原始数据，也不重算任何指标。

本合同的 public 语义保持通用。信用风险是主要验证场景，但任何字段名、标签、分数、
单位、阈值或损失假设均不得硬编码为银行、贷款、违约、DPD、PD、EAD 或 LGD 语义。

## 2. 模块与单一执行模型

Task 15 的 public 编排 owner 是新模块：

```text
src/sharper/risk_validation.py
```

该模块拥有 public input/config validation、validation-plan 与 fold construction、label
maturity、prediction-source dispatch、OOF/validation row assembly、public result/fold
assembly、warnings 与 provenance；它也拥有 exposure/observed-loss/expected-loss input
validation 和基础 business-result assembly，并只消费下述 selection/threshold primitives。
它只编排下述两个 private owner，不实现或复制 evaluation-owned prediction validation、
discrimination/probability metrics、calibration、gains 或 threshold primitives。它不得依赖
`workflow`、`reporting`、`cli`、Task 16--20 模块或 private condition kernel；它也不得
import/call `visualization.py`。`analysis.py`、workflow、reporting 与 CLI 保持不变。

`src/sharper/modeling.py` 继续只拥有 estimator source：per-fold estimator clone、fold-
local schema/feature selection、preprocessor construction、fit 和 prediction-interface
execution。Task 15 实现时只允许在 `modeling.py` 抽取一个 private、classification-only 的
`_fit_classifier_fold(...)` helper：它接收已经冻结的 train/validation 原始行位置，
在当前 train rows 上重新执行 Task 11 已批准的 schema-dependent selection 和 Pipeline
构建，clone caller estimator，fit 一次，并返回 private fold prediction snapshot。现有
`train_classifier` 必须调用同一 private core；其 signature、estimator-selection、错误、
warnings、result fields、split 和 observable behavior 均不得改变。不得复制一套完整的
v0.1 preprocessing/modeling 流程到 `risk_validation.py`，也不得在 `modeling.py` 计算
Task 15 metrics、validation plan 或 public result。

`src/sharper/evaluation.py` 是 Task 15 纯 prediction validation/math 的唯一 owner。只允许
添加 private Task 15 helpers，以验证 normalized ranking score/event probability 并计算
ROC-AUC、average precision、normalized Gini、KS、Brier、固定 log loss、ECE/calibration
bins、gains/lift/capture 与 threshold primitives。它不构造 fold、不 fit/clone estimator、
不选择 features、不处理 label maturity、不组装 public result，也不得改变任何 v0.1 public
symbol、signature、classification/regression dispatch 或 observable behavior。

`src/sharper/visualization.py` 是 Task 15 result-only Figure 的唯一 owner。它只接受冻结的
`BinaryRiskValidationResult`，验证并读取对应的 `gains`、`calibration` 或
`threshold_analysis` 表，然后创建静态、确定性的 matplotlib Figure；不接受 raw target、
ranking score、probability、DataFrame、estimator 或 Pipeline，不调用 evaluation/math
helper，不修改 result，也不创建指标、bin、cutoff 或 policy。`risk_validation.py` 不创建
或持有 Figure，`evaluation.py` 不绘图；既有 v0.1 visualization functions、containers、
exports、Figure ownership 与 observable behavior 均保持不变。

Task 15 只有一个 public 验证执行入口 `validate_binary_risk`，以及一个不执行验证、只消费
已冻结结果的 public 绘图入口 `plot_binary_risk_validation`。验证入口只有一条共享的下游
评估管线，并要求以下 prediction source **恰好一个**：

1. `estimator`：由上述 private modeling helper 在每 fold 独立 clone、构建 preprocessing、
   fit 和产生 fold prediction；
2. `external_predictions`：调用者提供已冻结的 validation/OOF scores/probabilities 和
   每 fold fit-row provenance；Sharper 重建同一 validation plan 并逐项核对 row/fold/
   maturity membership，再进入与 estimator 路径相同的 prediction validator 和指标
   管线。

这不是两套指标实现。两种 source 只能产生同一个 private normalized prediction table；
`risk_validation.py` 将它一次性交给 `evaluation.py` private primitives，指标数学不得在
两处出现。两个 source 同时提供或都未提供均以 `ValueError` 失败。`estimator=None` 只
表示 external mode；Task 15 不提供任何隐式 estimator，也不得从 Task 11 借用或构造一个。

External provenance 只能证明声明的 row membership 与本次 plan 一致，不能检查调用者
的 opaque 外部 fit state。合法 external 路径因此固定记录 warning
`external_fit_not_verifiable` 和同名 limitation；不得声称 Sharper 已证明外部模型无
泄漏。缺失、重复、矛盾或不符合本次 plan 的 provenance 必须失败，不能降级为 warning。

## 3. Public API 与 exports

Task 15 只新增以下顶层 public symbols，顺序固定为类型后函数；不增加 alias、overload、
`**kwargs`、generic registry、manager、公共 fold helper 或 Task 16--20 类型：

```python
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import matplotlib.figure
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

RiskLabel = str | int | bool  # private type alias; not exported
RiskLabelInput = RiskLabel | np.generic  # private input alias; not exported


@dataclass(frozen=True)
class BinaryRiskValidationConfig:
    validation_mode: Literal[
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
        "time_holdout",
        "time_forward",
    ]
    n_splits: int | None = None
    test_size: float | None = None
    random_state: int = 42
    group_column: str | None = None
    observation_time_column: str | None = None
    event_time_column: str | None = None
    outcome_end_time_column: str | None = None
    label_available_time_column: str | None = None
    maturity_source: Literal[
        "label_available_time", "observation_horizon", "outcome_end"
    ] | None = None
    prediction_horizon: timedelta | None = None
    prediction_horizon_column: str | None = None
    reporting_delay: timedelta = timedelta(0)
    fold_cutoffs: tuple[datetime, ...] = ()
    validation_end: datetime | None = None
    analysis_as_of: datetime | None = None
    thresholds: tuple[float, ...] = ()
    threshold_kind: Literal["ranking_score", "event_probability"] | None = None
    operating_metric: Literal[
        "sensitivity", "specificity", "precision", "negative_predictive_value", "f1"
    ] | None = None
    calibration_bins: int = 10
    gain_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
    exposure_column: str | None = None
    observed_loss_column: str | None = None
    observed_loss_available_time_column: str | None = None
    observed_loss_is_mature_snapshot: bool = False
    loss_fraction: float | str | None = None
    exposure_unit: str | None = None


@dataclass(frozen=True)
class ExternalRiskPredictions:
    row_positions: tuple[int, ...]
    fold_ids: tuple[int, ...]
    fold_fit_row_positions: tuple[tuple[int, tuple[int, ...]], ...]
    ranking_scores: tuple[float, ...] | None
    ranking_direction: Literal["higher_risk", "lower_risk"] | None
    event_probabilities: tuple[float, ...] | None
    probability_positive_label: RiskLabelInput | None
    probability_provenance: Literal[
        "predict_proba", "fold_safe_calibrated", "external_declared"
    ] | None


@dataclass(frozen=True)
class BinaryRiskValidationResult:
    target: str
    positive_label: str | int | bool
    validation_mode: Literal[
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
        "time_holdout",
        "time_forward",
    ]
    config: BinaryRiskValidationConfig
    prediction_scope: Literal["validation", "oof"]
    score_source: Literal[
        "estimator_predict_proba",
        "estimator_decision_function",
        "external_ranking_score",
        "external_event_probability",
        "external_ranking_and_probability",
    ]
    score_direction: Literal["higher_positive_event_risk"]
    probability_provenance: Literal[
        "predict_proba", "fold_safe_calibrated", "external_declared"
    ] | None
    input_n_rows: int
    eligible_n_rows: int
    predicted_n_rows: int
    evaluable_n_rows: int
    requested_threshold_count: int
    actual_threshold_count: int
    observed_loss_maturity_mode: Literal[
        "not_provided", "availability_column", "mature_snapshot"
    ]
    observed_loss_analysis_as_of: datetime | None
    observed_loss_mature_n: int
    observed_loss_excluded_n: int
    folds: pd.DataFrame
    predictions: pd.DataFrame
    excluded_rows: pd.DataFrame
    metrics: pd.DataFrame
    gains: pd.DataFrame
    calibration: pd.DataFrame
    threshold_analysis: pd.DataFrame
    operating_point: pd.DataFrame
    business_metrics: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


def validate_binary_risk(
    df: pd.DataFrame,
    target: str,
    *,
    positive_label: RiskLabelInput | None = None,
    config: BinaryRiskValidationConfig,
    estimator: ClassifierMixin | None = None,
    external_predictions: ExternalRiskPredictions | None = None,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
) -> BinaryRiskValidationResult: ...


def plot_binary_risk_validation(
    result: BinaryRiskValidationResult,
    *,
    kind: Literal["gains", "lift", "calibration", "threshold"],
) -> matplotlib.figure.Figure: ...
```

字段顺序、参数顺序、keyword-only 边界、Literal vocabulary 与 DataFrame schema 均为
冻结 public contract。五个 symbols 精确追加到当时完整、已验证的 v0.1 `__all__` 尾部，
顺序为 `BinaryRiskValidationConfig`、`ExternalRiskPredictions`、
`BinaryRiskValidationResult`、`validate_binary_risk`、
`plot_binary_risk_validation`；既有 export 的顺序和值不变。三个 public dataclass 都是
shallow frozen；`validate_binary_risk` 不修改 config、external object 或其中 tuple，
`plot_binary_risk_validation` 不修改 result 或其中 tables。结果不含 raw full DataFrame、
NumPy array、estimator、
Pipeline、Figure、path、timestamp-of-run、duration 或随机 artifact ID。结果 DataFrame
是独立 deep copy，但 pandas object cell 不承诺递归复制；调用者可修改自己的结果副本，
后续 consumer 必须先验证 observable schema。

Result 中的 `config` 是逐字段规范化后新建的等值 frozen snapshot：thresholds 与 gain
fractions 保留 caller requested tuple（包括重复值）以支持审计，时间/数值 scalar
规范化为本文声明的 built-in 类型；它不是 caller config 的同一对象。结果采用 pandas
DataFrame 以延续现有 Sharper 明细表模式，因此不是 JSON-native。
Task 20 如需 JSON 必须只消费并验证这些 frozen schemas，在其自己的合同中冻结序列化；
Task 15 不提供 `to_dict`、JSON encoder 或文件输出。

`eligible_n_rows` 固定表示通过 target-missing 与 plan role/membership checks、可参与 split
计划的行数，不表示 model-feature eligibility。Estimator feature availability 只在各 fold
的 private modeling core 中判断；external source 因而不会用 feature 条件改变该 count。

Task 15 一经批准并实现，这五个 symbol、signatures、dataclass field order/type hints、
table schemas、vocabularies 与默认值即属于 current release public contract；任何删除、
改名或语义变更必须先同步修订并 review 本文、`SPEC.md`、
`IMPLEMENTATION_PLAN.md`、API 文档与 compatibility tests。后续 Task 不得直接扩写这些
result fields 来承载自己的语义。

## 4. 输入、角色列与不可变性

`df` 必须是 pandas DataFrame，列名必须是唯一字符串。所有 public validation 和
config schema checks 在 split、clone、fit 或 prediction 之前完成。`target` 与 config 中
所有实际使用的非空列参数必须遵守 string-name/present 规则。

Feature validation 是 estimator source 的专属步骤。只有 estimator source 才验证
`features`、`exclude_columns`、schema eligibility、preprocessing 与 modeling input；它们
遵守 Tasks 11/12 的 string-name、present、unique sequence 规则。External source 要求
`features is None` 且 `exclude_columns == ()`，除此之外不执行 feature selection、schema
eligibility、preprocessor 或 modeling validation；external input 可以只含 target、
prediction provenance 需要的 fold/time/group 列以及可选业务列。即使 DataFrame 没有任何
eligible model feature，合法 external source 也不得失败，每个 fold 固定记录
`feature_columns=()`。External source 中非默认 `features`/`exclude_columns` 使用
`binary risk validation config has invalid schema` 失败，不检查其中列名或 eligibility。

下列 role columns 永远不进入 feature selection、schema eligibility 或 estimator input：

- target；
- group、observation/event/outcome-end/label-available/horizon 列；
- exposure、observed-loss、observed-loss-available-time 和 string-valued loss-fraction 列；
- caller `exclude_columns`。

Estimator source 的 `features=None` 时，每 fold 只用该 fold train rows 对剩余列执行 Task 11 已批准的
train-only schema/ID-like/eligibility 选择；不同 folds 可以记录不同的最终
`feature_columns`。显式 `features` 保留 caller 顺序，不能包含任何 role/excluded 列，
仍须在每 fold train rows 上通过 dtype/constant/ID-like checks。不得从完整 DataFrame、
validation rows 或后续 fold 学习类别词表、imputer、scaler、selection 或其他 state。

函数不得修改输入 DataFrame 的 index、columns、dtypes、values 或 object cells；不得修改
caller estimator，也不得复用任何 fitted estimator/preprocessor/calibrator state。
duplicate DataFrame index 允许，因为所有 identity 均使用零基原始 row position；结果
不存储 index label，并固定记录 `duplicate_index` warning。完全重复行同样不自动删除，
固定记录 `duplicate_rows` warning。Task 15 不声称这两个 warning 已解决 entity leakage。

## 5. 二值 target 与 positive label

Target 规则按以下顺序冻结：

1. 目标缺失用 `pd.isna` 判定。部分缺失 target rows 在任何 split/maturity/fit/metric
   之前过滤，并以原始位置写入 `excluded_rows` reason `missing_target`，同时记录
   `missing_target_rows_excluded` warning；它们不得获得 prediction。全缺失 target 失败。
2. 在 class inference、homogeneity 或 matching 前，对 target 中每个非缺失 scalar 和显式
   `positive_label` 执行同一规范化：任何 `np.generic` 先调用 `.item()`，因此
   `np.bool_(True) -> True`、`np.int64(1) -> 1`、`np.float64(1.0) -> 1.0`、
   `np.str_("1") -> "1"`。随后才按精确 built-in type 判断；这使 NumPy boolean、
   integer 与 string scalar 分别等同于对应 built-in，而 normalized float 仍因下一条规则
   被拒绝。不得只规范化 target 或只规范化显式 label。
3. 规范化后的非缺失标签只接受 homogeneous scalar exact built-in `str`、non-bool `int`
   或 `bool`；pandas categorical 只按其非缺失 scalar values 判断。float、bytes、datetime、
   tuple、mixed kind，以及 Python equality 相等但 type 不同的混合标签均失败。
4. 过滤后必须恰有两个 type-aware classes。class discovery 使用原始 row position 的
   first-appearance order，但该顺序不决定正类。
5. 规范化后的显式 `positive_label` 必须是合法 `RiskLabel`，并按
   `(type(value), value)` 与其中一个
   class 精确相等；`True` 不等于整数 `1`。
6. `positive_label=None` 只允许两个 canonical cases：classes 是 type-aware
   `{False, True}` 时推断 `True`；classes 是 type-aware `{0, 1}` 的整数时推断 `1`。
   class first appearance 为 `1, 0` 不改变该结果。字符串、其他整数 pair 和所有其他
   标签必须显式提供；不得按数值大小、词典序、频率、模型 class order 或名称猜测。

`ExternalRiskPredictions.probability_positive_label` 使用同一 NumPy-scalar 规范化与
type-aware matching；这不改变 Task 11 已冻结的 label 行为。Target missing filtering 只
表示该行不能用于监督验证，不是 imputation。每个 fold、overall
metric 和 business result 都记录实际有效样本量；不得把过滤前行数作为 denominator。

## 6. Score、probability 与 source mapping

所有内部和返回的 `ranking_score` 在进入任何排序或 threshold 逻辑前统一为：

```text
higher score = higher positive-event risk
```

Estimator source 必须是 cloneable binary classifier，具备 callable `fit` 且至少具备
callable `predict_proba`、`decision_function` 之一。每 fold fitted `classes_` 必须与
type-aware target classes 完全一致：

- callable `predict_proba` 优先。输出必须是 `(n_validation, 2)`、finite、每值 `[0,1]`，
  每行和与 1 的绝对误差不超过 `1e-12`；按 fitted `classes_` 精确定位
  `positive_label` 列。该列同时成为 `event_probability` 和 `ranking_score`；
- 只有 `decision_function` 时，只接受 finite one-dimensional length-
  `n_validation` 输出。若 positive label 是 `classes_[1]`，原值即 normalized
  `ranking_score`；若是 `classes_[0]`，逐值取负；其他 mapping 失败；
- 不调用 `predict`，不从 predicted class 构造 score；`decision_function`、margin 或
  `[0,1]` 范围内的一般 score 永远不自动成为 probability。

External source 的 `row_positions` 必须严格递增、唯一并精确等于本次 plan 按原始顺序
生成的 predicted positions；匹配只按零基原始 row position，禁止 pandas index alignment、
label-based join、reindex 或 index coercion。`fold_ids` 等长并与本次 membership 完全一致。
`fold_fit_row_positions` 按 fold id 升序，每项 train positions 严格递增、唯一，并与
Sharper 重建的 fold train positions 精确相等。

External source 至少提供 `ranking_scores` 或 `event_probabilities` 之一：

- `ranking_scores` 每值可为任意 finite real；同时必须提供 `ranking_direction`。方向为
  `higher_risk` 时原值保留，`lower_risk` 时逐值取负。不得从列名或数值猜方向；
- `event_probabilities` 每值必须 finite 且在闭区间 `[0,1]`；
  `probability_positive_label` 必须 type-aware 等于 resolved positive label，并必须提供
  `probability_provenance`；
- 仅提供 probability 时，它同时成为 normalized ranking score；同时提供两者时，
  ranking metrics 只用显式 ranking score，probability metrics 只用 probability；二者
  排序不一致允许但记录 limitation `ranking_probability_order_may_differ`；
- probability provenance `fold_safe_calibrated` 只是 caller declaration。Task 15 验证
  range、mapping、row/fold provenance，不验证 opaque calibrator state。

任何 NaN、`+inf`、`-inf`、越界 probability、长度/shape 错误、错误 class mapping 或
fold mapping 均失败；不得 drop、clip、fill、排序修复或反转未知方向。只有 ranking
score 时，probability metrics、calibration 和 expected loss 以结构化 unavailable
结果表达，不返回伪数值。

## 7. Validation plan 与 fold membership

### 7.1 共同规则

`random_state` 必须是 non-bool、非负 built-in integer；不接受 `None`。随机模式在相同
输入、config 和 estimator 可控随机性的条件下 fold membership 确定。fold id 是从 0
开始的连续整数。所有 train/validation identity 使用原始 row position。

Fold 上限固定为 20。repeated folds、nested CV、bootstrap、sample weights、resampling、
group + time 联合切分和 caller-supplied arbitrary folds 不属于 Task 15。任何同时声明
group 与 time 的 config 都失败，不得静默退化为普通 random/group/time split。
Unused split fields 不得被静默忽略：stratified modes 要求 `group_column` 与 label-maturity
split fields 为 `None`/空 tuple；group modes 要求且只要求一个 `group_column`，label-
maturity split fields 为 `None`/空 tuple；time modes 要求 `group_column is None` 并遵守
7.4。Observed-loss maturity metadata 独立于 split mode，适用第 13 节；非-time mode 的
`analysis_as_of` 只允许用于 observed-loss maturity，未提供 observed loss 时必须为
`None`。Holdout modes 要求 `n_splits is None`，K-fold modes 要求 `test_size is None`，time
modes 两者都为 `None`。违反任一组合均使用 config-schema error。

`prediction_scope`：holdout modes 固定为 `validation`；K-fold 与 time-forward 固定为
`oof`。这里的 OOF 表示每个 plan-defined validation candidate 恰好获得一次由未包含该
行的 train fold 产生的 prediction，不表示每个原始输入 row 必有 prediction。未进入
prediction scope 的行必须出现在 `excluded_rows`。

任何 fold 的 `train_row_positions` 与 `validation_row_positions` 在构造后都必须按数值
严格升序存储，并分别唯一且互斥；K-fold 的
validation sets 必须互斥。Estimator path 的每个 predicted row 恰好调用一次 fold
prediction；external path 每个 predicted row 恰好出现一次。不得静默跳过失败 fold。

### 7.2 Stratified modes

- `stratified_holdout`：`n_splits is None`；`test_size` 是 finite non-bool real 且严格在
  `(0,1)`；membership 精确使用
  `train_test_split(eligible_positions, stratify=y, test_size=test_size,
  random_state=random_state)`。train rows 在 `excluded_rows` 标为 `training_only`。
- `stratified_kfold`：`test_size is None`；`n_splits` 是 `2..20` 的 non-bool integer；
  精确使用 `StratifiedKFold(n_splits=n_splits, shuffle=True,
  random_state=random_state)`。每个 eligible row 恰好一次 OOF coverage。

Estimator source 的每个 train fold 必须包含正负两类且每类至少一行；split infeasible 或
任何 estimator train fold 单类均失败。External source 仍核对 plan-defined train positions
与 fit provenance，但不运行 feature/modeling eligibility。Validation fold 可以单类；按第
10 节逐 metric 判定，不能使用 fold-wide status，prediction 仍保留；pooled overall 若包含
两类，其 discrimination metrics 仍可用。

### 7.3 Group modes

Group column 必须存在、完整，且每值是 scalar `str`、integral non-bool `int` 或 `bool`；
identity 同样 type-aware，不做 string coercion。至少有两个 groups。

- `group_holdout`：`n_splits is None`，合法 `test_size` 同上；精确使用
  `GroupShuffleSplit(n_splits=1, test_size=test_size,
  random_state=random_state)` 的唯一 split；
- `group_kfold`：`test_size is None`；`n_splits` 是 `2..20` 且不大于 unique groups；
  精确使用 `StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
  random_state=random_state)`。

每 fold 计算 train/validation type-aware group sets 并要求交集为空；发现 overlap 失败。
Group split 后 estimator source 的 train fold 为空或单类失败；external source 只要求合法
非空 plan membership 与精确 fit provenance。Validation 单类按上一节逐 metric 语义。
Sharper 不自动把 duplicate index、ID-like 列或重复行推断为 group/entity。

### 7.4 Time modes

Time mode 要求 `group_column is None`、`n_splits is None`、`test_size is None`，且必须
提供 `observation_time_column`、`maturity_source`、一个明确 outcome definition、non-empty `fold_cutoffs`、
`validation_end` 和 `analysis_as_of`。所有时间列必须是 timezone-naive pandas
datetime dtype；所有 cutoff/as-of 是 timezone-naive `datetime`。不接受 object/string
自动解析、timezone-aware value、mixed timezone、NaT（target-missing rows 除外）或
date-only coercion。`reporting_delay` 与 horizon 必须 non-negative；prediction horizon
必须严格大于零。

`fold_cutoffs` 必须严格递增，`validation_end > last cutoff`，且
`analysis_as_of >= validation_end`：

- `time_holdout` 要求恰好一个 cutoff `C`，validation observation window 是
  `[C, validation_end)`；
- `time_forward` 要求 `2..20` 个 cutoffs。fold `i` 的 validation window 是
  `[C_i, C_{i+1})`，最后一个是 `[C_last, validation_end)`。各 window 左闭右开，
  `observation_time == C_i` 进入 validation，不进入该 fold training。

每个 time fold 的 train membership 精确为：

```text
observation_time < C
and label_available_time <= C
```

`label_available_time == C` 可训练；晚于 C 的 row 即使 event 已发生，也从该 fold train
purge。每 fold 分别记录所有 `observation_time < C` 的 candidate 数、mature train 数、
`immature_train_n`、`purged_train_n`；后二者在 Task 15 等价，均指仅因
`label_available_time > C` 未进入 fit 的非缺失-target rows，保留两个列名是为了
roadmap/report vocabulary，不重复计数。

Validation prediction 对 window 内所有 target-nonmissing rows 产生，不读取其 target。
Fold metrics/calibration denominator 只使用
`label_available_time <= analysis_as_of` 的 validation rows；等号成熟。晚于 as-of 的
rows 仍保留 prediction，但 `target_value` 返回为 `pd.NA`、`is_evaluable=False`、
`unevaluable_reason="immature_validation_outcome"`，并计入
`immature_validation_n`。因此允许 partial-maturity fold；若 fold 零 evaluable rows，
fold metrics 全部 `undefined/no_evaluable_rows`，但 prediction coverage 保留。

任何 time fold 若 validation window 无 candidate rows，整个调用稳定失败并指出 fold id；
estimator source 还要求 purge 后 train 非空、train target 双类且 fold-local modeling 至少
一个 eligible feature；external source 不执行这三项 modeling/feature eligibility，但仍
精确核对 plan train positions、label maturity 与 fit-row provenance。不得跳 fold。仅
observation time 排序、event 已发生或 target 当前可见均不证明
label-safe。Task 15 只验证本文的 label maturity 与 fold isolation，不执行 Task 16 的
一般 point-in-time feature availability/leakage audit。

## 8. Label maturity derivation 与一致性

`maturity_source` 是 label-availability derivation 的唯一 selector；Task 15 不按“哪个列
存在”猜测优先级。无论 selector 为何，time mode 都必须通过
`outcome_end_time_column` 或 `prediction_horizon`/`prediction_horizon_column` 提供 outcome
definition；`reporting_delay` 不能替代 outcome definition：

1. `label_available_time`：要求 `label_available_time_column`；该列是 authoritative；同时
   要求 outcome-end column 或 scalar/row horizon。该路径用 configured reporting delay
   验证 `label_available_time >= outcome_end + reporting_delay`，允许实际 availability 更晚；
2. `observation_horizon`：要求 `prediction_horizon` 与
   `prediction_horizon_column` 恰好一个；逐行
   `outcome_end = observation_time + horizon`，
   `label_available_time = outcome_end + reporting_delay`；
3. `outcome_end`：要求 `outcome_end_time_column`；逐行
   `label_available_time = outcome_end_time + reporting_delay`。

Outcome definition 的 cardinality 固定为：outcome-end column 与 horizon 可以提供其一或
同时提供；两者都没有失败。若同时提供，必须逐行满足
`outcome_end_time == observation_time + horizon`，不得按容差、日期或排序近似；horizon
的 scalar 与 column 仍恰好一个。显式 label availability 可以晚于 outcome end。

若提供非 authoritative 的 `outcome_end_time_column` 或
`label_available_time_column`，它成为 consistency check：其值必须与 authoritative
derivation 逐行完全相等，否则失败。`observation_horizon` 路径若同时提供显式 outcome
end，亦须完全相等。`label_available_time` 路径不要求 availability 等于 outcome end；
它必须满足下面的 ordering，允许更晚的 reporting/settlement availability。

任何提供 horizon 的路径都要求 scalar `prediction_horizon` 与
`prediction_horizon_column` 恰好一个；完全不提供 horizon 时两者都必须为 `None`。Event
time、label availability 或 reporting delay 均不能代替 outcome definition。

共同一致性：`outcome_end_time > observation_time`；每个非缺失 `event_time` 必须满足
`observation_time < event_time <= outcome_end_time`；negative target row 不得有
event_time；positive row 可因调用者未记录具体发生时点而为空。显式
显式 availability 路径满足
`label_available_time >= outcome_end_time + reporting_delay`；其余两条派生路径满足精确
等式。所有路径有 event time 时还必须
`label_available_time >= event_time`。任何 authoritative maturity value 缺失、溢出或
不一致均失败，不得把该 row 当成熟或无事件。

Scalar `prediction_horizon`/`reporting_delay` 和逐行 horizon column 只允许标准
`datetime.timedelta` 或 pandas timedelta dtype 所代表的精确 duration；不接受 month/
year calendar offset、字符串或“当前日期”。所有 derivation 使用 pandas nanosecond
timestamp arithmetic；溢出失败。Result folds 明确记录 outcome source、maturity source、
horizon source、reporting-delay source/value、cutoff、window 与 analysis-as-of。

## 9. Frozen result table schemas 与 ordering

所有表列名、顺序和 row ordering 固定如下。空表仍保留完整 columns/dtypes；不得用
`None` 代替表。Nullable numeric 使用 pandas nullable dtype；labels/object tuples 使用
`object`。所有 built-in scalar 输出在表构造前规范化，不保留 NumPy scalar。

### `folds`

```text
fold_id, cutoff, validation_start, validation_end, analysis_as_of,
outcome_end_source, prediction_horizon_source, prediction_horizon_value,
prediction_horizon_column, reporting_delay_source, reporting_delay,
train_row_positions, validation_row_positions, evaluable_validation_row_positions,
feature_columns, train_candidate_n, train_n, train_positive_n, train_positive_rate,
validation_n, validation_mature_n, validation_excluded_n,
evaluable_validation_n, evaluable_positive_n, evaluable_positive_rate,
immature_train_n, purged_train_n, immature_validation_n, maturity_source
```

按 `fold_id` 升序。每个 position tuple 内数值严格升序。`outcome_end_source` 只允许
`not_applicable`、`derived_horizon`、`column`；time mode 禁止 `not_provided`。
若提供 `outcome_end_time_column`，即使 horizon 同时存在并通过逐行一致性检查，source
也固定为 `column`；只有未提供该列、完全由 horizon 推导时才是 `derived_horizon`。
`prediction_horizon_source` 只允许 `not_applicable`、`scalar`、
`column`、`not_provided`。Scalar horizon 放入 `prediction_horizon_value`，column 路径只在
`prediction_horizon_column` 放列名；另一 cell 为 `pd.NA`。`reporting_delay_source` 只允许
`not_applicable`、`config_derivation`、`config_minimum_check`；后两者分别对应派生
availability 与显式 availability 验证。`reporting_delay` 是精确 `timedelta`。
`train_candidate_n` 是 observation-before-
cutoff 且 target 非缺失的行数；`train_n` 是再经过 label maturity 的实际 fit/provenance
行数。`validation_mature_n == evaluable_validation_n`，`validation_excluded_n ==
immature_validation_n`，两组名称分别保留 maturity/provenance 与 metric vocabulary，不能
遗漏或交叉计数。非-time mode 的所有 time/maturity cells 使用 `pd.NA`，对应 maturity
counts 为 0，`maturity_source="not_applicable"`。External fold 的 `feature_columns` 固定为空
tuple；position 与 feature cells 都是 tuples。

### `predictions`

```text
row_position, fold_id, target_value, is_evaluable, unevaluable_reason,
ranking_score, event_probability
```

按 `row_position` 严格升序；每个 predicted position 恰好一行。Immature validation 的
`target_value` 是 `pd.NA`。Ranking score 必有 finite float；probability absent 时整列
为 `pd.NA`，不能用 ranking score 填充。

### `excluded_rows`

```text
row_position, reason
```

按 `row_position`、再按固定 reason precedence 排序。Reason vocabulary/precedence：

```text
missing_target
training_only
before_first_validation_window
outside_validation_window
```

同一 row 最多一条；`missing_target` 优先。Fold 内 immature training rows 不进入此表，
因为同一 row 可在另一 time fold 成为 validation/training；它们只进入 fold counts。

### `metrics`

```text
scope, fold_id, metric, statistic, value, at_threshold, status, reason,
n_rows, n_positive, n_negative
```

Metric 顺序固定为：

```text
roc_auc
average_precision
normalized_gini
ks_statistic
brier_score
log_loss
expected_calibration_error
```

先按 fold id 输出 `scope="fold", statistic="direct"`；再输出
`scope="overall", fold_id=pd.NA, statistic="direct"` 的 pooled direct OOF/validation
结果；最后按 metric 顺序输出 `scope="fold_summary"` 的 `mean`、`sample_std`。Fold mean
只聚合 status `available` 的 fold values；零个可用值为 `unavailable/no_available_folds`；
sample std 使用 `ddof=1`，少于两个可用 folds 为
`undefined/insufficient_available_folds`。Overall 永远直接用 pooled evaluable rows
计算，不是 fold mean。

`status` 只允许 `available`、`undefined`、`unavailable`。`value` 只在 available 时为
finite float，否则为 `pd.NA`。`reason` 只在非-available 时非空，固定 vocabulary：

```text
no_evaluable_rows
single_class
probability_absent
no_available_folds
insufficient_available_folds
```

Direct metric reason precedence 固定为：先 `no_evaluable_rows`；非空后，四个
discrimination metrics 检查 `single_class`；三个 probability metrics 再检查
`probability_absent`。因此同一合法 scope 的每个 metric 都得到唯一 status/reason。

`at_threshold` 只在 direct、available `ks_statistic` row 中保存取得最大 KS 的最高
finite normalized score；其他 direct metrics、fold summary 以及 undefined/unavailable
KS 均为 `pd.NA`。因此 KS 的值、正负样本支持和阈值都可由 result 审查。

Status 是逐 metric row 独立计算，禁止 fold-wide status propagation。Single-class
evaluable scope 中只有 `roc_auc`、`average_precision`、`normalized_gini` 与
`ks_statistic`（包括其 `at_threshold`）为 `undefined/single_class`；若合法 probability
存在，`brier_score`、`log_loss`、`expected_calibration_error` 仍按各自公式 available。
Threshold counts 始终按实际 evaluable rows 计算；各 rate 只按自己的 denominator/status
规则判定。无正类不自动令 event rate、predicted-positive rate 或有定义的 counts/rates
不可用。

### `gains`

```text
scope, fold_id, requested_fraction, target_count, boundary_score,
selected_n, actual_fraction, total_positive_n, selected_positive_n,
event_rate, event_rate_status, event_rate_reason,
capture, capture_status, capture_reason,
lift, lift_status, lift_reason
```

先 fold、后 overall；各 scope 内按 deduplicated `requested_fraction` 升序。Requested
fraction `q` 的 target count 是 `ceil(q * n)`，至少 1；先按 normalized score 降序找到
target rank boundary，再以 `score >= boundary_score` 原子纳入全部 ties。因此
`selected_n` 可大于 target count，`actual_fraction` 是实际值。Counts、target count、
boundary 与 selected fraction 使用自身确定性数学，不共享 metric status。无 evaluable rows
时固定 `target_count=0`、`boundary_score=pd.NA`、`selected_n=0`、
`actual_fraction=pd.NA`，三个 metric value/status/reason 分别为
`pd.NA`/`undefined`/`no_evaluable_rows`；只要 evaluable denominator 非零，
`event_rate` 就是 available（无正类时精确为 `0.0`）。`capture` 以 total positives 为
denominator，`lift` 再使用 actual fraction；无正类时两者分别为
`undefined/zero_denominator`，不能污染 event-rate status。每个 status 只允许
`available`、`undefined`，available 对应 reason 为 `pd.NA`；reason precedence 固定为
`no_evaluable_rows` 后 `zero_denominator`。不得恢复 row-level `status`/`reason`。
`top-fraction capture` 即 `capture`；cumulative gains 即 capture curve；lift 是
`capture / actual_fraction`。本表同时是 Task 15 唯一自动构造的 cumulative score-band
证据，不另做自动 quantile cutoff 搜索。

### `calibration`

```text
scope, fold_id, bin_id, lower_bound, upper_bound, upper_inclusive,
n_rows, mean_predicted_probability, observed_event_rate,
absolute_gap, weighted_gap, status, reason
```

先 fold、后 overall、再 bin id 升序。固定 equal-width bins：bin `i` 为
`[i/B, (i+1)/B)`，最后一 bin 右闭以包含 1。所有 bins 都返回；empty bin 的三项 value
为 `pd.NA`、status `undefined`、reason `empty_bin`。只有 ranking score 时返回完整 schema
的零行表；probability metrics 在 `metrics` 中标记 unavailable。

### `threshold_analysis`

```text
scope, fold_id, threshold_kind, threshold, tp, fp, tn, fn,
sensitivity, sensitivity_status, sensitivity_reason,
specificity, specificity_status, specificity_reason,
precision, precision_status, precision_reason,
negative_predictive_value, negative_predictive_value_status,
negative_predictive_value_reason,
f1, f1_status, f1_reason,
accuracy, accuracy_status, accuracy_reason,
predicted_positive_rate, predicted_positive_rate_status,
predicted_positive_rate_reason
```

先 fold、后 overall；每个 scope 内 threshold 数值升序。每个 rate cell 使用 nullable
float，并拥有自己的 status/reason；status 只允许 `available`、`undefined`，reason 只允许
`no_evaluable_rows`、`zero_denominator`。零分母时只将对应 rate 置 `pd.NA` 并标为
`undefined/zero_denominator`；counts、其他有定义 rate 和 row 保留。只要 n > 0，accuracy
与 predicted-positive rate 即 available；single-class 不形成 fold-wide undefined。

### `operating_point`

```text
threshold_kind, operating_metric, threshold, metric_value,
candidate_count, status, reason
```

无 thresholds 或未声明 operating metric 时是完整 schema 零行表。若声明 metric，则只在
overall threshold rows 中比较该 metric 的非缺失值；最大值胜出，同值取更高 normalized
threshold。全部 undefined 时返回一行，threshold/value 为 `pd.NA`、
status `undefined`、reason `objective_undefined`。合法选择 status `available`、reason
`pd.NA`。它只是分析证据，不修改 config、result thresholds 或任何 Task 17 input。

### `business_metrics`

```text
segment_kind, segment_value, metric, value, status, reason,
n_rows, n_evaluable_rows, n_observed_loss_mature_rows, unit
```

先 `all`，再按 gains requested fraction 升序的 `top_fraction`，再按 threshold 升序的
`threshold_selected` segments；每 segment 的 metric 顺序固定为：

```text
event_rate
predicted_positive_rate
exposure_sum
observed_loss_sum
expected_loss_sum
```

Result 顶层 `observed_loss_maturity_mode`、`observed_loss_analysis_as_of`、
`observed_loss_mature_n` 与 `observed_loss_excluded_n` 按第 13 节记录独立 observed-loss
provenance；未提供 observed loss 时分别是 `not_provided`、`None`、0、0。所有表的字符串
vocabulary、counts 和 ordering 都确定；不得依赖 hash/order、原始 index label、locale
或并行 completion order。

## 10. Metric 数学合同

所有 discrimination、gains 和 threshold 计算使用 normalized higher-risk score；所有
target 转为 `y_i = 1[target_i type-aware equals positive_label]`。每个 fold 和 overall
只使用 `is_evaluable=True` rows。

- **ROC-AUC**：精确使用 `sklearn.metrics.roc_auc_score(y, ranking_score)`；ties 采用
  sklearn/Mann–Whitney 平均秩语义。两类都存在时范围 `[0,1]`；单类 undefined。
- **Average precision**：精确使用
  `sklearn.metrics.average_precision_score(y, ranking_score)`，名称只能是
  `average_precision`。但 single-class scope 不调用 sklearn，固定为
  `undefined/single_class`。Task 15 不公开 trapezoidal PR area，不使用模糊 `pr_auc` 名称。
- **Normalized Gini**：`2 * roc_auc - 1`，范围 `[-1,1]`；ROC undefined 时同 reason
  undefined。
- **KS statistic**：对每个 distinct normalized score threshold，以 `score >= threshold`
  计算 cumulative TPR/FPR，取 `max(TPR - FPR)`；ties 不拆开；两类存在时范围 `[0,1]`。
  最大值如多处相同，诊断 threshold 取最高 score。结果 fold metadata 已记录正负样本量。
- **Brier score**：仅 probability，`mean((p_i-y_i)**2)`，范围 `[0,1]`；只要至少一个
  evaluable row，single-class 仍有定义。
- **Log loss**：仅 probability，先把 caller/estimator 输出转为 `float64`，并在任何 clipping
  前验证原始值全部 finite 且位于闭区间 `[0,1]`；越界或 non-finite 直接失败。固定
  `epsilon = 1e-15`，逐值计算
  `p_clipped = min(max(p, epsilon), 1 - epsilon)`，再用
  `mean(-(y * log(p_clipped) + (1 - y) * log(1 - p_clipped)))`。输入 0 和 1 合法；不得调用
  sklearn 的版本相关 clipping default，也不得将原始越界值 clip 回合法区间。只要至少一个
  evaluable row，single-class 仍有定义且结果 finite non-negative。
- **Expected calibration error**：仅 probability，使用本文 equal-width non-empty bins，
  `sum_b (n_b/n) * abs(observed_rate_b - mean_probability_b)`，范围 `[0,1]`；single-class
  仍有定义。
- **Event rate**：positive target rows / target-evaluable rows；零 rows undefined。

Threshold decision 固定为 normalized `score >= threshold` 表示 predicted positive；等于
threshold 进入 positive，ties 不拆开。Confusion counts 定义为标准 TP/FP/TN/FN。
Sensitivity=`TP/(TP+FN)`；specificity=`TN/(TN+FP)`；precision=`TP/(TP+FP)`；negative
predictive value=`TN/(TN+FN)`；F1=`2TP/(2TP+FP+FN)`；accuracy=`(TP+TN)/n`；
predicted-positive rate=`(TP+FP)/n`。各自零分母为 `pd.NA`，不强制改为 0。Accuracy 只
是普通 threshold evidence，不进入主 metric 排序或默认 operating objective。

Single-class scope 的固定 status matrix 是：ROC-AUC、average precision、normalized Gini、
KS/KS threshold 为 `undefined/single_class`；Brier、fixed-epsilon log loss、ECE 与非空
calibration bins 在 probability 存在时 available；event/non-event counts、event rate、
threshold confusion counts、accuracy 与 predicted-positive rate 在各自 denominator 非零时
available；其他 threshold rates 只按自己的 denominator 判定。不得捕获一个 sklearn
single-class exception 后把整个 fold 或所有 metrics 标为 undefined。

空 overall evaluable input 不可能被静默接受：非-time path 在 split 前失败；time path
可以因 partial maturity 产生零 evaluable overall，此时 metrics 为 structured undefined，
但必须仍有合法 predictions。完全没有 validation candidates 则属于 invalid fold，直接
失败。

## 11. Calibration 边界

Task 15 **只做 calibration diagnostics，不拟合 calibration model**。不得在 Task 15
内部创建、fit、返回或序列化 Platt/sigmoid、isotonic、beta 或其他 calibrator，也不得
用 validation/test label 做 in-sample calibration。

Estimator `predict_proba` 和 caller-declared external probability 可进入 diagnostics；
它们是否在统计意义上校准必须由结果 provenance 和 calibration evidence解释，不能因
值域合法而声称 calibrated。External `fold_safe_calibrated` 必须有与本 plan 一致的
per-fold fit-row provenance，但 opaque state 仍不可验证并保留 limitation。未来如需
library-owned calibration fitting，必须先修订本文，冻结训练内层 split/cross-fit、方法、
小样本、ownership 和 leakage tests。

`calibration_bins` 是 non-bool integer `2..50`；只影响 diagnostics，不影响 model fit、
ranking score、threshold 或 probability 本身。

## 12. Threshold 候选与 operating evidence

Thresholds 必须由 caller 在 fit/prediction 前通过 config 预声明；Task 15 不扫描 score
unique values、不生成网格、不优化搜索空间。Requested count 是原 tuple 长度，上限 100；
每值必须 finite non-bool real。去重按数值 equality，返回升序；发生重复时记录
`duplicate_thresholds_removed` warning。

- 非空 thresholds 必须显式提供 `threshold_kind`；空 thresholds 要求 kind 为 `None`；
- `event_probability` threshold 必须在 `[0,1]` 且结果确有 probability；
- `ranking_score` threshold 是 normalized higher-risk scale 上的任意 finite real；调用者
  提供 lower-risk external score 时应先按本文的取负语义理解返回 threshold scale；
- `operating_metric` 只能在 thresholds 非空时提供，且只从 frozen metric vocabulary
  选择；没有 action/cost/capacity/exposure/budget/rate constraints；
- candidate analysis 只使用本调用的 fold validation/OOF rows。Public API 不接受 final
  test/holdout provenance；Task 15 不提供“用 final test 选择后再评估”的入口。

`operating_point` 不叫 optimal/business cutoff。它不自动保存、部署、修改模型、写文件、
传入 Task 17 或成为 policy。Task 17 只能接收 caller 在其自己的流程中另行明确冻结的
cutoff/bands，不能把本表的存在解释为采用授权。

## 13. “Business Metrics” 精确边界

Task 15 的 business 字样只指模型无关、无 action 的 population risk summaries：event
rate、predicted-positive rate、top-fraction capture/gains/lift、threshold confusion/rates、
calibration、exposure sum、mature observed loss sum 和 probability-based expected loss
sum。它们是离线描述或显式假设下的算术，不是 action effect、策略收益或业务最优。

`gain_fractions` 原始 count 上限 100，每值 finite non-bool 且在 `(0,1]`；去重后升序，
重复记录 `duplicate_gain_fractions_removed`。至少一个 fraction，必须包含 `1.0`，从而
固定 overall population baseline。

Business optional inputs：

- exposure 列必须 real non-bool non-complex numeric；所有 predicted rows 值 finite、
  non-negative；
- 未提供 `observed_loss_column` 时，`observed_loss_available_time_column` 必须为 `None`、
  `observed_loss_is_mature_snapshot` 必须为 `False`；result provenance 固定为
  `not_provided`；
- `observed_loss_is_mature_snapshot` 只接受 exact built-in `bool`，不得用整数 truthiness；
- 提供 `observed_loss_column` 时，必须同时提供 `analysis_as_of`，并在下列两种独立
  observed-loss maturity provenance 中恰好选择一个；两者均未选或同时选择都以
  `binary risk business inputs are invalid` 失败：
  1. **availability-column mode**：提供 `observed_loss_available_time_column` 且
     `observed_loss_is_mature_snapshot=False`。availability 列必须是 timezone-naive pandas
     datetime dtype，在全部 predicted rows 非缺失；仅
     `observed_loss_available_time <= analysis_as_of` 的 rows 是 observed-loss mature；
  2. **mature-snapshot declaration**：`observed_loss_available_time_column is None` 且
     `observed_loss_is_mature_snapshot=True`。Caller 明确声明本次 snapshot 中全部 predicted
     rows 的 observed loss 到 `analysis_as_of` 已成熟；Task 15 记录该声明但不推断 full-life
     loss maturity。
- Observed-loss `analysis_as_of` 必须是 timezone-naive `datetime`；availability-column 与
  time-validation 同时使用时，两者共享 config 中同一个 as-of 值，但 maturity 判断与
  counts 仍相互独立；
- observed-loss 列必须 real non-bool non-complex numeric；只读取并验证上述 provenance
  判定为 observed-loss mature 的 rows，值必须 finite、non-negative。Availability 晚于
  as-of 的 row 即使 input cell 可见也不得读取、验证或汇总；snapshot mode 没有此类排除。
  Target/label maturity、event occurrence、outcome-end 或 target evaluability 不能替代或
  推断 observed-loss maturity；两套 maturity provenance 彼此独立；
- 顶层 result 精确记录 mode、analysis-as-of、本次 predicted scope 的 mature count 与
  excluded count，且 `observed_loss_mature_n + observed_loss_excluded_n == predicted_n_rows`。
  每个 business segment 另记录 `n_observed_loss_mature_rows`；
- `loss_fraction` 是 finite non-bool scalar `[0,1]` 或一个 real numeric column name；列值
  在所有有 probability 的 predicted rows 上 finite 且 `[0,1]`；
- 提供 exposure、observed loss 或 loss fraction 任一个时必须提供 non-empty
  `exposure_unit`；该字符串只记录单位，不做 currency conversion；
- expected loss 只有在 `event_probability`、exposure 和 loss fraction 三者齐全时计算：
  `sum(event_probability * exposure * loss_fraction)`；它可以覆盖尚未成熟但已有合法 OOF/
  validation probability 的 predicted rows，并在 `n_evaluable_rows` 单列披露 observed
  outcome support；
- observed loss 只汇总 caller-provided observed-loss 值和 independently mature rows；不从
  target、target maturity、probability 或 `event * exposure * loss_fraction` 伪造 observed
  loss；
- 缺少相应 optional input 时保留 business metric row，status `unavailable`，reason 分别
  是 `exposure_absent`、`observed_loss_absent`、`probability_absent` 或
  `loss_fraction_absent`。

`all` 与 `top_fraction` segments 用 normalized ranking score；threshold-selected segment
只对应 config threshold kind。Business `top_fraction` 在全部 predicted rows 上独立使用
与 gains 相同的 `ceil`/tie-boundary 规则，从而 exposure/expected loss 可以覆盖 immature
outcomes；它不复用只基于 evaluable rows 的 statistical gains boundary。
`predicted_positive_rate` 只在 threshold segment available，其他 segment 为
`unavailable/not_threshold_segment`。Threshold selection 同样先作用于全部 predicted
rows；event rate denominator 再限制为该 segment 内 target-evaluable rows，observed loss
则独立限制为该 segment 内 observed-loss-mature rows。所有结果必须记录 `n_rows`、
`n_evaluable_rows`、`n_observed_loss_mature_rows` 和 unit。

Task 15 不拥有 approve/decline/review/request-information、action name/role mapping、rule
hit/path、manual-review capacity、policy constraint、selection/rejection semantics、action
cost、revenue、profit、payoff、expected profit、策略优化或 causal effect。Task 17 独占
这些 action/policy semantics。Task 15 的 expected loss primitive 不接受 action，也不与
Task 17 双重拥有策略收益。

## 14. Result-only visualization contract

Task 15 只新增一个 opt-in 静态绘图入口：

```python
def plot_binary_risk_validation(
    result: BinaryRiskValidationResult,
    *,
    kind: Literal["gains", "lift", "calibration", "threshold"],
) -> matplotlib.figure.Figure: ...
```

`result` 的 exact public type、keyword-only `kind`、closed Literal 与返回类型均冻结。该函数
位于 `visualization.py`，只返回一个裸 `matplotlib.figure.Figure`；这是单 kind、单 Figure
的定向 API，不改变或复用 `PlotResult`/`PlotCollection` 字段，也不改变任何 v0.1 绘图入口。
它以 built-in `ValueError` 表示调用错误。Runtime validation precedence 固定为：
`result` exact type；`kind` 是 exact built-in `str` 且属于 closed vocabulary；对应 frozen
table 的 exact schema；table ordering/status/reason/value invariants；证据可绘制性；全部
检查通过后才创建 Figure。失败不得留下部分 Figure。

Plot-kind inventory 与 deterministic ordering 精确为 `gains`、`lift`、`calibration`、
`threshold`。四个 kind 的 source、row selection 与 artists 精确冻结如下。所有 source rows 都只使用
`scope == "overall"` 且 `fold_id` 为缺失值的行，保持 result table 的既有顺序；函数不得
in-place sort，也不得绘制逐 fold curve。

| kind | 唯一 source | x | y / 固定 legend 顺序 | 固定 reference |
|---|---|---|---|---|
| `gains` | `result.gains` | `actual_fraction` | `capture` / `Event capture` | `(0,0)` 到 `(1,1)` 的 `Reference` diagonal |
| `lift` | `result.gains` | `actual_fraction` | `lift` / `Lift` | `(0,1)` 到 `(1,1)` 的 `Reference` horizontal line |
| `calibration` | `result.calibration` | `mean_predicted_probability` | `observed_event_rate` / `Observed event rate` | `(0,0)` 到 `(1,1)` 的 `Reference` identity line |
| `threshold` | `result.threshold_analysis` | `threshold` | `predicted_positive_rate`、`sensitivity`、`precision`、`specificity`，legend 依次为 `Predicted positive rate`、`Recall`、`Precision`、`Specificity` | none |

函数不得从 `predictions`、`metrics`、`operating_point`、`business_metrics`、config 或其他表
补造绘图数据。`sensitivity` 只在显示 label 中写作 `Recall`，不新增或重命名 frozen schema
column。Gains/lift 不生成新 bands，calibration 不重新 bin，threshold 不扫描、插值、优化或
挑选 cutoff。Task 15 不提供 ROC 或 precision-recall curve：现有 result 没有冻结其点列，
不得为绘图增加 score arrays、curve tables、large result fields 或 private recomputation。

每次调用精确创建一个 Figure 和一个 Axes。结构冻结为：

| kind | title | x label | y label | Line2D 数量与顺序 |
|---|---|---|---|---|
| `gains` | `Cumulative gains` | `Selected fraction` | `Cumulative event capture` | `Event capture`，`Reference` |
| `lift` | `Lift` | `Selected fraction` | `Lift` | `Lift`，`Reference` |
| `calibration` | `Calibration` | `Mean predicted probability` | `Observed event rate` | `Observed event rate`，`Reference` |
| `threshold` | `Threshold metrics` | `Threshold` | `Rate` | `Predicted positive rate`，`Recall`，`Precision`，`Specificity` |

三条 primary single-series curve 固定为 `#4C78A8`、solid line、circle marker；reference
固定为 `#9E9E9E` dashed line、无 marker。Threshold 四条 curve 按上述顺序固定使用
`#4C78A8`、`#F58518`、`#54A24B`、`#E45756`，全部 solid line、circle marker。Gains 与
calibration 的 x/y limits 固定为 `[0,1]`；threshold y limits 固定为 `[0,1]`；lift x
limits 固定为 `[0,1]`，y limits 只由上述 frozen plotted values 的 matplotlib autoscale
决定。所有图均显示固定顺序 legend。不得调用或修改全局 matplotlib/seaborn style、theme、
palette 或 rcParams API；line data、artist/axes structure 和上述显式样式不得依赖 caller 的
global style state。不使用随机数、当前日期、locale、filesystem path、model name、环境
变量或 pandas index label 决定 title、labels、artists 或 line data，也不得切换 matplotlib
backend 或产生 GUI side effect。合同测试断言 axes、
line-data、label、legend 和 ownership，不使用 pixel/hash snapshot。

Availability 规则逐 kind 固定：

- `gains` 要求 overall rows 非空，并且每行 `capture_status == "available"`、reason 缺失、
  x/y finite；`lift` 对 `lift_status` 使用同一规则；任一 required row undefined 即整个调用
  失败，不丢行或降级；
- `calibration` 允许且只允许完整 frozen table 中的结构性
  `undefined/empty_bin` rows 不形成 plot point；这不是 silent fallback，而是本文冻结的
  empty-bin 表示。其余 overall rows 必须 `available`、reason 缺失且 x/y finite，至少一个
  available bin；任何其他 undefined/unavailable 或零个 available bin 均失败；
- `threshold` 要求 overall rows 非空，且每行上述四个 rate 的各自 status 都是
  `available`、reason 缺失、value finite；任何一个 required cell undefined 即整个调用
  失败，不画断线、placeholder 或部分 metric；
- 对应 table 是合法零行表、缺少 overall rows 或没有可绘制 points 时均失败；函数不返回
  `None`、空 tuple、空 Figure、skip result、warning-only result 或 placeholder text。

Exact error messages 冻结为：wrong result type 使用
`result must be a BinaryRiskValidationResult`；invalid kind 使用
`binary risk validation plot kind is invalid`；对应表不是 DataFrame、不是本文 exact frozen
schema、ordering/status/reason/value invariants 不合法时使用
`binary risk validation plot table has invalid schema: {kind}`；合法表但 empty、缺少 overall
evidence 或 required evidence undefined/unavailable 时使用
`binary risk validation plot evidence is unavailable or undefined: {kind}`。函数不调用
`warnings.warn`，不把这些错误改成 skip/warning。

绘图函数不得修改 result 或任何 result DataFrame，不得 `show()`、`savefig()`、写文件、
`close()`，也不得把 Figure 写入 result/dataclass。返回 Figure 从创建时起由 caller 独占
保存与关闭责任；库内不缓存 Figure。Workflow、reporting 与 CLI 不调用该函数；Task 20
如需集成，必须在其独立合同中显式消费这个 public result-only API，并继承 caller-owned
Figure 生命周期。

## 15. 错误、不可用、undefined 与 warnings

Task 15 不新增 public exception class。无效 config/input/provenance、无法构造安全 fold、
fit/prediction execution 或 malformed estimator output 使用 built-in `ValueError`；底层
estimator 的普通 `Exception` 以对应稳定 `ValueError` 包装并保留 cause，不捕获
`KeyboardInterrupt`、`SystemExit` 或 `GeneratorExit`。

区别固定为：

- **configuration/programmer error**：错误类型/source 组合、字段冲突、超预算；立即
  `ValueError`，无部分 result；
- **invalid input**：target、time、group、score、probability、business value 非法；立即
  `ValueError`；
- **fold unusable**：validation empty、group overlap，或 estimator source 的 train
  empty/single-class/无 eligible feature/fit 失败；整体 `ValueError`，不得丢 fold；external
  source 不执行这些 modeling eligibility checks；
- **mathematically undefined**：数据合法但 denominator/class support 不足；保留 row，
  status `undefined`、value `pd.NA`、精确 reason；
- **unavailable**：合法 source 没有 probability 或 optional business input；保留适用
  metric row，status `unavailable`、精确 reason；
- **excluded/unevaluable**：逐行 reason 与 counts 记录，不等于异常或 warning；
- **nonfatal warning**：只以 ordered `warnings: tuple[str, ...]` 返回，不调用
  `warnings.warn`，不写 stderr。

关键稳定错误消息：

| condition | exact message |
|---|---|
| wrong DataFrame/names | `df must be a pandas DataFrame` / `DataFrame column names must be unique strings` |
| wrong source cardinality | `exactly one of estimator and external_predictions must be provided` |
| malformed config/external/result | `binary risk validation config has invalid schema` / `external risk predictions have invalid schema` / `binary risk validation result has invalid schema` |
| target absent/type/classes | `target column not found: {target!r}` / `binary target labels must be homogeneous string, integer, or boolean values` / `binary target must contain exactly two non-missing classes` |
| all target missing | `binary target has no non-missing labels` |
| positive label required/absent | `positive_label must be provided for non-canonical binary labels` / `positive_label is not present in binary target` |
| invalid role/feature column | `{role} column not found: {name!r}` / `model features must not contain role or excluded columns` |
| invalid split | `binary risk validation split is infeasible` |
| group overlap | `validation fold {fold_id} has group overlap` |
| empty/single-class train | `validation fold {fold_id} has no eligible training rows` / `validation fold {fold_id} training target must contain both classes` |
| empty validation | `validation fold {fold_id} has no validation rows` |
| time metadata/maturity | `time validation metadata is invalid` / `label maturity metadata is missing or inconsistent` |
| external membership | `external prediction provenance does not match validation plan` |
| ranking/probability | `ranking scores must be finite real values with explicit direction` / `event probabilities must be finite values in [0, 1]` / `event probability positive-label mapping is invalid` |
| estimator interface/clone | `estimator must be a cloneable binary classifier with a score interface` / `binary risk estimator could not be cloned` |
| fold execution/output | `binary risk estimator fit failed in fold {fold_id}` / `binary risk estimator prediction failed in fold {fold_id}` / `binary risk estimator has invalid output in fold {fold_id}` |
| threshold config | `threshold candidates are invalid` / `event-probability thresholds require event probabilities` |
| business input | `binary risk business inputs are invalid` |
| plot result/kind | `result must be a BinaryRiskValidationResult` / `binary risk validation plot kind is invalid` |
| plot table/evidence | `binary risk validation plot table has invalid schema: {kind}` / `binary risk validation plot evidence is unavailable or undefined: {kind}` |

Validation precedence：DataFrame type；config/external dataclass outer type；prediction-source
cardinality；scalar config schema 和预算；DataFrame names；target/positive label；实际使用的
role columns；time/group/business column dtype/value；validation plan construction；external
source 则直接核对 row/fold/time/maturity provenance，estimator source 才验证 feature/
exclusion names、interface/clone capability、逐 fold train eligibility、fit 与 prediction；
normalized prediction validation；metrics/threshold/business construction；result schema
validation。任何失败发生在 result 返回前；caller inputs 保持不变。External source 不得因
无 eligible feature、feature dtype/constant/ID-like 或 preprocessing 条件提前失败。

Warnings 固定 vocabulary 与顺序：

```text
duplicate_index
duplicate_rows
missing_target_rows_excluded
external_fit_not_verifiable
duplicate_thresholds_removed
duplicate_gain_fractions_removed
large_input
```

`large_input` 在 `input_n_rows > 1_000_000` 时记录，只披露 Task 15 是内存内单机分析，
不采样、不截断、不改变结果。Limitations 固定 vocabulary 与顺序：

```text
random_or_group_validation_not_time_safe
entity_isolation_not_checked
time_validation_not_general_feature_audit
external_fit_not_verifiable
ranking_probability_order_may_differ
probability_metrics_unavailable
calibration_diagnostic_only
partial_validation_maturity
single_class_validation_fold
observed_association_not_causal
```

## 16. 资源预算、确定性与依赖

固定 hard budgets：最多 20 folds、100 requested thresholds、50 calibration bins、100
requested gain fractions。超过即失败，不自动截断。Config 同时保留 requested threshold
count；threshold/gain 去重原因进入 warning。Task 15 不提供并行 fit、distributed backend、
chunking、sampling、GPU 或 cache。

相同 input values/row order、config、external predictions 或 deterministic estimator 时，
fold ids、positions、normalized scores、tables、ties、metrics、warnings 和 limitations 必须
完全确定。每个 fold 的 train/validation position tuple 数值升序，fold id 从 0 连续稳定；
external mapping 只按 position，不按 pandas index。每个 status、reason、warning、limitation
与 table row 使用本文固定 vocabulary/precedence/order；任何因 maturity 排除的行不进入
metric denominator，但必须进入相应 fold/result provenance count。

Task 15 的 estimator 必须由 caller 显式提供，并在每 fold 新建 clone。Estimator
内部随机性仍由 caller 负责。Private helper 不改写 custom estimator
random state；若其 randomness 不可由 caller 固定，结果额外使用既有 Task 11 语义的
`custom_estimator_random_state_not_managed` warning 与
`custom_estimator_determinism_not_guaranteed` limitation；这两个 vocabulary 作为 Task 11
兼容值附加在上述固定序列末尾。

不新增 core 或 optional dependency，不修改 `pyproject.toml`。只使用现有 Python、pandas、
NumPy、scikit-learn 和 matplotlib。Validation/math 路径不读写文件、网络、环境变量或
系统当前时间，不持有 raw DataFrame、不保存 estimator、不创建 Figure；只有第 14 节的
result-only 绘图入口创建并向 caller 返回 Figure。

## 17. 测试合同

Task 15 implementation tests 必须把以下每项映射到小型、可手算、确定性 fixture，并
断言 exact signature、frozen field order、表 schema/order、stable error 和输入不变性。

### 17.1 标签与方向

- bool 与 0/1 canonical inference；1/0 first appearance 仍选整数 1；
- `np.bool_`/`np.integer`/`np.str_` 在 target 与显式 positive label 两侧同样规范化；
  `np.floating` 先 `.item()` 后仍按 built-in float 稳定拒绝；
- `np.bool_(True)` 参加 canonical bool 推断，`np.int64(1)` 参加 canonical 0/1 integer
  推断，`np.float64(1.0)` 不升级为 integer，`np.str_("1")` 仍要求显式 positive label；
- string 与其他 integer pair 需要显式 positive label；合法 pandas categorical；
- explicit label type-aware matching，`True`/`1` 不混同；positive label absent；
- three-class、mixed-kind、全缺失与部分缺失 target；部分缺失计数/reason；
- external higher/lower direction 得到相反 normalization；不得从名称猜方向；
- ties、全部相同 score，以及输入行重排后只由 position tie-break 保持确定性。

### 17.2 Score 与 probability

- 任意负数/>1 finite ranking score；合法 0/1 boundary probability；
- NaN、正负 infinity、越界 probability、wrong length/shape；
- estimator class order 与 explicit positive column mapping；
- one-dimensional decision margin 的正/反向 mapping；
- probability 优先于同时存在的 decision function；
- ranking-only 产生 unavailable probability metrics、零行 calibration 和 unavailable
  expected loss；值域在 `[0,1]` 的 ranking score 仍不升级为 probability；
- external probability wrong positive mapping/provenance 失败；both-score 路径分别计算。
- estimator 与 external source 同时提供、两者都未提供分别命中同一 exact cardinality error；

### 17.3 Validation 与 leakage

- stratified holdout/K-fold 固定 seed 与 exact membership；
- group holdout/K-fold 的 train/validation group disjoint；伪造 external overlap/provenance
  失败；
- repeated/nested/group+time/sample-weight 不存在 public path，相关 config 稳定失败；
- OOF/validation 每个 expected position 恰好一次，row order 和 duplicate index 保留；
- external fixture 只含 target、prediction、fold/time provenance 且无 feature column 仍通过，
  `feature_columns=()`；DataFrame 无 eligible feature 不得触发 modeling validation，external
  路径的非默认 `features`/`exclude_columns` 在列名检查前稳定报 config-schema error；
- duplicate rows/index warnings 不自动删除数据；
- spy estimator 证明每 fold clone 不同、fit 只看到该 fold train rows；validation-only
  category/extreme/group 不改变 train imputer/encoder/scaler/schema/selection state；
- v0.1 `train_classifier` 在 private helper 抽取前后 result、errors 和 fit isolation 不变；
- 所有 fold train/validation position tuples 严格升序、fold ids 稳定；external 即使 duplicate
  index 也只按 row position 核对，不发生 pandas index alignment；
- 2 与 20 folds 的合法边界、21 folds，以及 thresholds/calibration bins/gain fractions 的
  各 hard budget 边界；
- caller estimator、DataFrame、config、external tuples 完全不变。

### 17.4 Time 与 maturity

- time holdout/forward 的左闭右开 validation window；`observation_time == C` 不训练；
  `label_available_time == C` 可训练；
- explicit availability、scalar/row horizon + delay、outcome-end + delay 三条路径；
- explicit label availability 但缺少 horizon/outcome-end 的 outcome definition 稳定失败；
  reporting delay 单独存在不能补足；
- horizon 与 outcome-end 同时提供时逐行完全一致通过，任一 row mismatch 稳定失败；
- explicit availability 使用 reporting delay minimum-check，允许晚于 derived minimum；
- reporting delay 未结束时 event 已发生仍 purge；逐行 horizon；
- authoritative/non-authoritative metadata exact consistency；缺失、timezone-aware、
  overflow、event/outcome ordering 错误；
- mixed mature/immature validation、`label_available_time == analysis_as_of`、partial fold、
  zero-evaluable fold、target masking 和 counts；
- estimator source 在 purge 后 empty/single-class train 稳定失败；external source 对相同
  plan 只核对 sorted fit-row provenance；empty validation window 对两种 source 都稳定失败；
- observation time 单调但 label availability 穿过 cutoff 的 leakage regression；
- later validation row 可进入后续 train 只在其 label 到后续 cutoff 已成熟时成立；
- group + time 组合稳定拒绝，不声称 Task 16 point-in-time feature audit。

### 17.5 Metrics、calibration 与 threshold

- 手算 ROC-AUC/Gini/KS/AP，与 sklearn 基准在绝对容差 `1e-12` 内一致；
- ties/constant score、single-class fold、pooled overall、fold mean 与 sample std；
- hand-calculated gains/lift/capture，tie boundary 扩张与 actual selected count；
- gains wide schema 的 event-rate/capture/lift 各自 status/reason；无正类时 event rate
  available=0、capture/lift 分别 `undefined/zero_denominator`；
- confusion counts、sensitivity/specificity/precision/NPV/F1/accuracy/predicted-positive rate，
  `>=` boundary 和每类 zero denominator；
- perfect、over-、under-calibration，0/1 probabilities、empty bins、Brier/fixed-epsilon log
  loss/ECE；手算 0、1、`epsilon/2`、`epsilon`、`1-epsilon`、`1-epsilon/2` 的 fixed-
  epsilon boundary/near-boundary log loss，并证明原始越界 probability 失败而非 clip；
- 单类 scope 逐 metric matrix：ROC-AUC/AP/Gini/KS/KS threshold undefined，而合法
  probability 的 Brier/log loss/ECE、calibration、counts 与有定义 rates 仍 available；
- caller thresholds 的去重/排序、domain、100 budget、ranking vs probability；
- operating metric max、objective tie 取更高 threshold、全部 undefined；
- threshold 不来自 final test、不写回 config/result policy、不调用 Task 17。

### 17.6 Business 与兼容性

- event rate、threshold selected rate 和 unit；
- exposure 在 predicted rows 按自身规则验证；observed loss 使用独立 availability-column
  maturity，只读取 `available_time <= analysis_as_of` rows，sentinel 证明排除 row 没有被
  访问，且 target unevaluable 不会替代该判断；
- mature-snapshot declaration 的 mode/as-of/mature/excluded counts；提供 observed loss 但
  两种 provenance 均未声明或同时声明分别稳定失败；target maturity 不替代 loss maturity；
- expected loss `p * exposure * loss_fraction` scalar/column 手算；ranking-only unavailable；
- negative/missing/non-finite exposure/loss、loss fraction 越界和 unit 缺失；
- result 不出现 action、rule、constraint、profit/payoff 或自动 cutoff；
- v0.1 全部 exports/signatures/dataclass fields/workflow/report/CLI/default behavior 不变；
- directed `tests/test_evaluation.py` 证明 private evaluation primitives 与 public
  `validate_binary_risk` orchestration 对同一 normalized fixture 的 metrics/gains/calibration/
  threshold 结果一致，不存在第二套数学；
- current release surface 只增加本文五个 symbols，版本在 Task 15 实现阶段是否 bump 仍由
  后续 release governance 决定，本文不修改当前 `0.1.0`；
- 无新依赖，wheel/sdist build 与 full v0.1 compatibility suite 通过。

### 17.7 Result-only visualization

- public API test 冻结 `plot_binary_risk_validation` 的 exact name、signature、keyword-only
  `kind`、Literal、`matplotlib.figure.Figure` return annotation、docstring 和第五个 Task 15
  export order；既有 v0.1 exports 顺序和值完全不变；
- 四个 kind 分别使用手工构造的 frozen result tables，断言唯一 source table、overall-only
  row selection、source row order、Axes count、Line2D count/order/data、固定 title/labels、
  legend、colors、linestyles、markers、reference line 与 limits；threshold 显示顺序固定为
  predicted-positive rate、recall、precision、specificity；
- spies/monkeypatch 证明函数不接收或读取 raw target/score/probability/DataFrame/estimator，
  直接传入 raw DataFrame、Series、NumPy array、sequence 或 estimator 均命中 wrong-result
  exact error；函数
  不调用 `evaluation.py` primitives、`validate_binary_risk`、modeling、feature selection、
  calibration/binning/threshold computation，也不读取 unrelated result tables；
- wrong result type、四种以外 kind、missing/extra/reordered columns、malformed ordering/status/
  reason/value、零行 table、缺少 overall row、gains/lift/threshold required undefined cells、
  calibration 零 available bin 与非法 non-empty-bin undefined 分别断言 exact `ValueError`；
  ranking-only result 请求 calibration、无正类导致 gains/capture undefined、缺失或零行
  threshold table 都失败且不产生 placeholder；`undefined/empty_bin` calibration rows 按合同
  显式不形成 points；
- 输入 result 和每张 DataFrame 在成功/失败路径均 byte-for-byte/equality 不变；duplicate
  pandas index 与故意不同的 index labels 不影响 line data，函数不 in-place sort；
- monkeypatch `show`、`savefig`、`close` 证明库不调用它们；返回值是 caller-owned Figure，
  caller 可自行保存/关闭，result/dataclass 不出现 Figure；重复调用创建不同 Figure 且无
  library cache；
- 相同 result 重复调用得到相同 axes/line/label/legend structure；调用前后 rcParams、现有
  seaborn/matplotlib global state 不变，不使用 pixel/hash snapshot；
- `plot_distributions`、`plot_classification_evaluation`、`plot_regression_evaluation` 及所有
  v0.1 visualization containers、errors、ownership 和 tests 保持兼容；workflow、report、
  CLI 不调用新入口。

## 18. Future implementation allowlist

本文通过 review 后，Task 15 implementation 只允许修改或创建以下精确文件：

```text
src/sharper/risk_validation.py                         (new; Task 15 domain owner)
src/sharper/evaluation.py                              (only private Task 15 prediction-validation/
                                                        metric primitives; no public behavior change)
src/sharper/modeling.py                                (only private fold-fit core extraction/addition;
                                                        no v0.1 public behavior change)
src/sharper/visualization.py                           (only the approved result-only Task 15 Figure
                                                        entry and narrow private plot validation/helpers;
                                                        no v0.1 plot behavior change)
src/sharper/__init__.py                                (only five approved opt-in exports)
tests/test_risk_validation.py                          (new; Task 15 contract tests)
tests/test_evaluation.py                               (only directed private-primitive/orchestration
                                                        consistency evidence; no broad rewrite)
tests/test_modeling.py                                 (only private-core/v0.1 regression evidence)
tests/test_visualization.py                            (only directed Task 15 result-only plot and v0.1
                                                        visualization compatibility evidence)
tests/test_public_api.py                               (current surface plus permanent v0.1 invariants)
README.md                                              (only approved Task 15 opt-in usage)
docs/api.md                                            (only approved Task 15 public API)
docs/decisions/task15-binary-risk-validation-contract.md
SPEC.md                                                (status sync only)
IMPLEMENTATION_PLAN.md                                 (status sync only)
```

`src/sharper/modeling.py` 的允许变更仅为抽取/新增 private
`_fit_classifier_fold`、让现有 `train_classifier` 复用相同 core、以及支持该 private result
所必需的 private type/helper；不得改变或新增 public symbol、签名、dataclass field、
estimator-selection、feature eligibility、错误、warning 或 v0.1 holdout split。若不能在此窄范围内
安全复用，implementation 必须停止并回报 blocker，先修订/review 本合同；不得复制完整
modeling pipeline 或临时扩大 allowlist。

`src/sharper/evaluation.py` 的允许变更仅为新增 private Task 15 normalized prediction
validation/math helpers，以及在不改 v0.1 public dispatch/结果/错误的前提下复用已有 private
纯函数；`tests/test_evaluation.py` 只添加这些 private primitives 与 Task 15 orchestration 的
定向一致性证据。不得新增 export、改变 `evaluate_classifier`/`evaluate_regressor`/
`evaluate_model` 或重写既有测试。若需要更宽改动，implementation 必须停止并先修订合同。

`src/sharper/visualization.py` 的允许变更仅为新增
`plot_binary_risk_validation` 与实现第 14 节所必需的窄范围 private schema/artist helpers；
不得增加计算指标、fold、bin、band、curve points、threshold 或 policy 的逻辑，不得改变
`PlotResult`、`PlotCollection` 或任何 v0.1 public/private observable plot behavior。
`tests/test_visualization.py` 只添加第 17.7 节定向证据与必要的永久 v0.1 compatibility
assertions，不得大范围重写既有 visualization tests。若需要 result schema、raw data 或
更宽 visualization 改动，implementation 必须停止并先修订/review 本合同。

默认禁止修改：其他 `src/` 模块、workflow、reporting、CLI、I/O、schema、
summary、quality、analysis、features、evaluation public behavior、Tasks 01--14 contracts/tests
（上述四个定向 compatibility test files 的窄 addition 除外）、Task 16--20 文件、examples、
`pyproject.toml`、dependencies、lock file、版本号、
cache、generated reports/build artifacts 与 `docs/.DS_Store`。实现不 commit、push、tag、
publish 或部署。

## 19. 明确排除与后续 ownership

Task 15 明确不实现或承诺：

- v0.1 quality rule 变更、Task 16 data quality/leakage audit、missingness drift 或 private
  condition kernel；
- 一般 point-in-time feature availability audit、multi-table/entity lifecycle engine；
- pre-loan eligibility/rule/action/constraint/policy、post-loan signal/warning/alert/lifecycle；
- approve/decline/review mapping、策略 cutoff 自动选择/保存/部署；
- expected revenue/profit/payoff、action-dependent loss/cost、causal/action effect；
- reject inference、WOE、target encoding、supervised binning、learned group aggregate；
- 新 estimator family、tree catalog、hyperparameter tuning、nested/repeated CV、resampling、
  sample weights、SHAP、drift、explainability/model comparison/governance；
- library-owned calibration fitting、final-test model/threshold selection；
- ROC curve、precision-recall curve、interactive plot、dashboard、multi-panel 自动组合、自动
  kind 选择、custom style/theme/palette API、model explanation、贷前策略、贷后预警、反欺诈，
  或任何第 14 节之外的图；
- workflow/report/CLI 集成、Figure 保存/关闭、JSON/YAML/TOML spec、serialization、
  file/network I/O；
- generic framework、registry、plugin、DSL、任意 code/callable 执行、server/dashboard；
- 反欺诈，以及真实审批、账户操作、客户联系或催收动作。

Task 16 只消费本文 frozen label/score/fold semantics，不重算 metrics/folds。Task 17 独占
action/policy/profit/payoff，并只接受 caller 另行冻结的 cutoff/bands。Task 18 独占
point-in-time warning/lifecycle。Task 19 只消费 frozen results 做自己的 comparison/
governance；Task 20 才拥有 opt-in workflow/report/CLI/serialization integration。

## 20. Review gate、完成定义与 blocker

本合同当前没有已知架构 blocker：现有 v0.1 modeling 可以通过本文严格限定的 private
core extraction 安全复用；但这项可行性必须在 contract review 中明确确认。若 review
认为无法在不改变 v0.1 behavior 的情况下抽取该 helper，结论必须是 Conditional Go 或
No-Go，且 Task 15 implementation 不得开始。

Task 15 只有在以下 gates 全部完成后才可从 implementation diff review 获得最终 Go：

1. 本合同先独立 review 并改为 Approved — Go；
2. public signatures、frozen fields、table schemas/vocabulary/order 全部 contract-tested；
3. validation membership、group isolation、time maturity、OOF coverage 和 fold-state
   isolation 全部通过；
4. ranking/probability 隔离、positive mapping、non-finite/range errors 全部通过；
5. 手算/sklearn metrics、calibration、ties、threshold 与 loss primitives 全部通过；
6. 四种 result-only plot 的 frozen source/structure/failure/ownership tests 全部通过，且没有
   metric recomputation、raw-data path 或 ROC/PR curve；
7. v0.1 permanent compatibility tests 未删除、搬移、弱化或改变；
8. `bash scripts/verify-uv-env.sh` 通过，且所有 Python 命令使用 `.venv/bin/python`；
9. `.venv/bin/python -m pytest`、`.venv/bin/python -m ruff check .`、
   `.venv/bin/python -m ruff format --check .`、
   `.venv/bin/python -m build --no-isolation` 全部通过；
10. `git diff --check` 与 `git diff --cached --check` 通过；
11. diff 只在 implementation allowlist，且无 Task 16--20 越界、依赖/版本/lock/cache/
    artifact 改动。

本文仍为 Approved — Go。Task 15 implementation 已完成，full verification 与 bounded
closure review 均已通过，当前状态为 `Implementation complete — review Go`；P0：`0`，
P1：`0`，implementation blocker：`none`。尚未 commit、tag、publish 或 release，v0.2
整体仍未完成或发布。
