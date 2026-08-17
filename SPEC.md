# Sharper 项目规格

## 1. 项目定位与 MVP 边界

### 1.1 定位

Sharper 是面向结构化表格数据的轻量级综合分析工具包。它把数据读取、模式推断、质量检查、画像、关系挖掘、候选特征发现、任务导向可视化、可选的基线建模与报告导出串成一条可复现流程。

Sharper 不是单纯的 EDA 工具，也不是 sklearn 的薄封装。v0.1 的核心价值是：用户拿到一个 CSV 或 Excel 后，能快速形成对数据质量、变量结构、变量关系、潜在特征与基线可预测性的整体判断，并导出可审阅的分析报告。

### 1.2 目标用户

- 数据分析师：快速完成数据理解、质量审查、分组比较与图表输出。
- 数据科学家和建模人员：发现候选特征，并建立无泄漏的分类或回归基线。
- 研究人员：获得可复现的描述统计、关联分析和报告。
- 需要快速理解表格数据并交付分析报告的其他用户。

### 1.3 主要场景

1. 读取 CSV/Excel，识别数值、类别、日期、ID 与疑似目标列。
2. 检查缺失、重复、常量、高基数、类型异常与范围异常。
3. 分析单变量分布、异常值、缺失模式和变量间关联。
4. 按分组或目标变量比较特征，发现潜在预测信号。
5. 提议 ratio、difference、interaction、日期、分箱和组聚合候选特征。
6. 生成与分析问题对应的静态图表。
7. 按需训练 sklearn 分类或回归基线，并独立评估。
8. 通过 Python API 或 CLI 导出 Markdown/HTML 报告。

### 1.4 v0.1 MVP

v0.1 提供一个受控但完整的轻量闭环：

- 文件读取：本地 CSV 与 Excel 单表读取；明确透传常用 pandas 参数。
- 数据理解：schema 推断、DataFrame 摘要、数值/类别画像。
- 数据质量：缺失、重复、常量、近常量、高基数、疑似 ID、无穷值和基础类型问题。
- 分析挖掘：Pearson/Spearman 相关、数值异常值、受限分组比较，以及分类/回归目标关系分析。
- 特征发现：生成结构化候选建议；v0.1 只物化 ratio、difference、product 和确定性日期特征。分箱、组聚合及 target-aware 候选只提示，不在 v0.1 transform。
- 可视化：优先使用 seaborn 生成围绕分布、缺失、相关性、异常值、分组/目标关系和模型评估的有限统计分析型图表，并以 matplotlib Figure 作为返回契约。
- 建模：一个分类基线和一个回归基线入口，使用 Pipeline、ColumnTransformer、明确 train/test split。
- 评估：分类与回归分离的指标结果；分类支持二分类和多分类的保守默认指标。
- 报告：将已有分析结果组织成 Markdown；HTML 为 Markdown 的静态渲染，不嵌入交互式应用。
- CLI：单条 `sharper analyze` 命令执行读取、分析、可选建模、图表与报告导出。
- Workflow：Python API 与 CLI 共用一个薄编排入口，保证完整流程的章节、警告和预算行为一致。

为保持可靠性，v0.1 不做穷举特征搜索、统计显著性“自动结论”、模型调参或多模型竞赛。默认行为应可解释、确定且有计算上限。

### 1.5 Non-goals

v0.1 明确不包含：

- AutoML、超参数搜索、模型排行榜或自动部署。
- 深度学习、分布式计算、数据库连接器、云部署。
- Web dashboard、复杂交互式 BI 或 notebook 小部件。
- feature store、MLflow、SHAP。
- 时间序列、文本、图像或嵌套/流式数据的专用分析。
- 因果推断、实验分析、自动业务结论或自动数据修复。
- 插件系统、注册表、依赖注入框架或深层 class hierarchy。
- Excel 多工作表联合分析、远程 URL 读取或超内存数据处理。

### 1.6 假设与约束

- PyPI 分发名与 import 名均冻结为 `sharper`，license 冻结为 MIT，初始版本冻结为 `0.1.0`。
- Python 3.10+；内存内 pandas DataFrame 是 v0.1 的统一数据边界。
- 主要面向中小型单表数据。默认相关性与绘图应设置列数/类别数上限，避免意外的二次复杂度。
- v0.1 默认预算：相关矩阵最多 50 列、每个类别分析最多显示 20 个水平、每类图最多 20 张、绘图最多采样 10,000 行、特征建议总数最多 50 个。调用者可在合理范围内调低或调高；任何截断或抽样都必须记录在结果和报告中。
- 目标列由用户显式确认。`infer_schema` 可以给出 `target_candidates`，但不得自动把候选当作 target。
- 所有 public functions 必须有完整 type hints 和 docstring。

## 2. 架构决策

| 决策 | 选择与理由 | 暂不采用 |
|---|---|---|
| 布局 | `src/sharper/`，避免从仓库根目录误导入 | 扁平包布局 |
| API 风格 | 以纯函数和不可变结果 dataclass 为主；仅训练产物与 sklearn transformer 持有拟合状态 | 服务对象、manager、深层继承 |
| 分层 | `io` 在入口，`schema/quality/analysis/features/modeling` 为领域能力，`visualization/reporting/cli` 为展示与编排边缘 | 一个巨型 analyzer 类 |
| 结果契约 | 分析函数返回具名 dataclass，表格明细使用 DataFrame；结果应可转成报告 | 无结构 dict、直接打印 |
| 绘图 | 输入数据或分析结果，优先使用 seaborn 实现统计分析型图表，返回 matplotlib `Figure`，默认不 `show()`/不写文件 | 按 matplotlib 原语逐层包装、多后端系统 |
| 建模 | 显式 split 后，在训练集内构造并 fit `Pipeline(ColumnTransformer, estimator)` | 先全量编码/插补再切分 |
| 报告 | 报告消费结构化结果，不重新计算隐藏分析 | 把分析逻辑塞入模板 |
| CLI | 薄编排层，只调用 public API；配置保持为命令参数 | v0.1 引入复杂 YAML 配置系统 |
| Workflow | 单一高级流程入口组合领域结果；CLI 与 Python 共用 | 在 CLI 中隐藏第二套分析流程 |
| 依赖 | pandas、numpy、scikit-learn、matplotlib、seaborn、scipy、typer；Excel 引擎作为 `excel` extra | plotly、altair、bokeh、polars 等额外依赖 |

依赖方向：

```text
cli -> io + workflow + reporting
workflow -> schema + summary + quality + analysis + features
         -> visualization + modeling + evaluation

reporting -> result types + visualization outputs
visualization -> result types (必要时读取 DataFrame，但不执行完整分析)
modeling -> schema contracts + sklearn
features -> schema contracts
analysis/quality -> schema contracts
schema -> pandas/numpy only
```

禁止 `schema`、`quality`、`analysis`、`features` 反向依赖 workflow、CLI、报告或绘图。`workflow` 只能编排 public API，不得复制领域算法；CLI 不得自行组合完整领域流程。模块之间共享的稳定结果类型放在其所属领域模块中，不创建泛化 `utils.py`。

### 2.1 规格与实施计划的职责

`SPEC.md` 定义产品定位、模块边界、公共原则和已批准版本路线。`IMPLEMENTATION_PLAN.md` 是任务执行依据：Tasks 01--14 记录已完成的 v0.1；Tasks 15--17 implementation 均已完成且 review 为 `Go`；Task 18 contract已批准为`Approved — Go`，final bounded contract closure为`Go`，implementation为`Implementation complete — review Go`且 final validation 已完成；Task 19 original contract checkpoint为`Approved — Go`，当前amended contract为`Approved — Go v2`，implementation为`Implemented — Post-Review Closure Complete`，Task19为`Complete`；Task 20 唯一一次 Full Contract Review 已以`NO-GO — Permanently Closed`结束，C1–C4 targeted fixes 与 bounded closure 已完成，合同治理为`Approved — Go`，冻结 findings 为`T20-CR-01..15`（15 Closed / 0 Open），implementation为`Not Started`，已 ready for implementation。每个 Task 的精确 API、允许文件、错误、排序和验收行为仍须由独立决策记录冻结。两者出现阶段划分或交付顺序冲突时，应先同步并评审治理文件，不得在实现中自行合并、跳过或扩大 Task。

## 3. 推荐目录结构

以下是设计目标树；本设计阶段不创建其中的实现或测试文件。

```text
sharper/
├── pyproject.toml
├── README.md
├── SPEC.md
├── AGENTS.md
├── CHANGELOG.md                  # 首次发布前创建
├── docs/
│   ├── quickstart.md
│   ├── analysis-guide.md
│   ├── leakage.md
│   └── api.md
├── examples/
│   ├── basic_analysis.py
│   └── baseline_modeling.py
├── src/
│   └── sharper/
│       ├── __init__.py
│       ├── _types.py
│       ├── io.py
│       ├── schema.py
│       ├── summary.py
│       ├── quality.py
│       ├── analysis.py
│       ├── features.py
│       ├── visualization.py
│       ├── modeling.py
│       ├── evaluation.py
│       ├── workflow.py
│       ├── reporting.py
│       └── cli.py
└── tests/
    ├── fixtures/
    ├── test_io.py
    ├── test_schema.py
    ├── test_summary.py
    ├── test_quality.py
    ├── test_analysis.py
    ├── test_features.py
    ├── test_visualization.py
    ├── test_modeling.py
    ├── test_evaluation.py
    ├── test_workflow.py
    ├── test_reporting.py
    ├── test_cli.py
    ├── test_public_api.py
    └── test_distribution.py
```

## 4. 模块职责

| 模块 | Owns | May depend on | Must not do |
|---|---|---|---|
| `sharper.__init__` | 版本与稳定 public exports | 各公开领域模块 | 包含业务逻辑或导出内部 helper |
| `sharper._types` | 跨模块小型类型别名、协议和枚举 | 标准库 | 变成无边界的 common/utils |
| `sharper.io` | CSV/Excel 输入与读取错误归一化 | pandas、路径类型 | 推断 schema、清洗或分析 |
| `sharper.schema` | 列角色、逻辑类型、疑似 ID/target 候选及 schema 结果 | pandas、numpy、`_types` | 自动确认 target 或修改数据 |
| `sharper.summary` | 数据集级与列级描述摘要 | pandas、numpy、schema | 质量判定、绘图或写报告 |
| `sharper.quality` | 质量规则、问题严重度与结构化发现 | pandas、numpy、schema、summary | 静默修复数据 |
| `sharper.analysis` | 数值/类别分析、相关性、异常值、分组及 target 关系 | pandas、numpy、scipy、schema | 训练模型或衍生特征 |
| `sharper.features` | 候选特征规格、建议与安全的无状态物化 | pandas、numpy、schema | 在 v0.1 拟合分箱/聚合、使用全量 target 统计或穷举组合 |
| `sharper.visualization` | 分析任务对应的 Figure 生成与统一视觉约定 | seaborn、matplotlib、pandas、领域结果类型 | 调用 `show()`、默认写文件、重跑完整分析或构建多后端系统 |
| `sharper.modeling` | split、预处理器、Pipeline 与分类/回归训练产物 | pandas、numpy、sklearn、schema、features | 评估混用、模型搜索、split 前 fit |
| `sharper.evaluation` | 分类/回归指标、预测诊断及评估结果 | numpy、pandas、sklearn metrics | 训练或选择模型 |
| `sharper.workflow` | 将领域 API 组合为 `AnalysisRun`，统一预算、警告和可选建模 | schema、summary、quality、analysis、features、visualization、modeling、evaluation | 实现领域算法、读取文件或写报告 |
| `sharper.reporting` | 将 `AnalysisRun` 与图表资产渲染为 Markdown/HTML | workflow 结果、领域结果类型、matplotlib、模板/标准库 | 隐式重算、训练模型或改变输入 |
| `sharper.cli` | 文件参数、workflow 调用、报告输出、退出码与用户消息 | io、workflow、reporting、Typer | 自行编排领域步骤、复制算法或吞掉异常 |

## 5. 数据分析能力设计

### 5.1 Schema 与摘要

- 物理 dtype 与逻辑角色分离：`numeric`、`categorical`、`datetime`、`boolean`、`text`、`identifier`、`unknown`。
- 推断基于 dtype、唯一率、可解析率、名称弱提示和样本值；每个推断包含置信度与理由。
- ID 和 target 只作为候选。低基数类别、二元列和末列不能被静默认定为 target。
- 摘要包含形状、内存、缺失率、唯一数，以及按逻辑类型选择的统计量。

### 5.2 质量检查

v0.1 规则固定且可单测：重复行、全缺失、缺失率阈值、常量/近常量、数值无穷、类别高基数、疑似 ID、混合 Python 类型、datetime 解析失败提示。每个问题包含代码、严重度、列、计数/比例、说明和建议；建议不自动执行。

### 5.3 分析挖掘

- 数值：分位数、偏度、零值率、Task 07 冻结的 IQR 异常值摘要。
- 类别：频数、比例、稀有水平与截断后的 top categories。
- 关联：数值 Pearson/Spearman；类别-类别和混合类型指标推迟，v0.1 可在 target 分析中提供有限的合适统计量。
- 分组比较：一个或多个数值列按一个类别列分组，返回 count、missing count、mean、median 和分位数；最多展示 20 个组，超限时按频数截断并披露。v0.1 不支持多重 group key 或透视表 DSL。
- target 关系：
  - 分类：数值按类别分组摘要与效应提示；类别特征交叉表、比例和可选卡方统计。
  - 回归：数值特征相关；类别特征分组 target 摘要与可选检验。
- 统计检验必须返回样本量、统计量和 p 值，并声明探索性、多重比较风险；不自动把 p 值翻译成业务结论。
- 异常值只标记/汇总，不删除。

## 6. 特征衍生能力设计

### 6.1 候选类型

- 数值对：按 DataFrame 顺序对 real numeric non-boolean columns 枚举有限的
  ratio、difference 和 product；不做 correlation-based candidate search、
  反向 pair、self-pair 或多项式爆炸。
- 日期：只对 timezone-naive pandas datetime dtype 建议 year、month、quarter、dayofweek、
  is-weekend；仅在 caller 提供显式 reference date 时建议 days-since。
- 单变量：learned binning candidate 在 v0.1 只建议，不物化。
- group aggregate 与 target encoding candidate 在 v0.1 只作为结构化 review
  suggestion，标记 `requires_fit=True`，不提供 transformer 或物化。

### 6.2 建议而非盲目生成

`suggest_feature_derivations` 返回冻结的 `FeatureSuggestionReport`，其 suggestions
含稳定名称、封闭类型、来源列、canonical formula/parameters、封闭 reason/risk、
是否需要 fit 和类型优先级。默认只建议且不修改 DataFrame。每类和总候选数有
固定预算；排除 target、显式 exclusions、ID-like、全缺失、常量、unsupported
dtype 和精确重复内容 columns。

ratio、difference、product、datetime components 和基于显式 reference date 的
days-since 可由 `derive_features` 无状态物化。任何依赖 target、数据驱动分箱
边界、类别集合或组统计映射的建议在 v0.1 均不可物化。

### 6.3 v0.1 收缩

数据驱动/固定分箱物化、group aggregate transformer、target encoding、WOE、
监督分箱、自动筛选最佳交互、correlation-based search 和模型驱动特征选择
推迟。涉及 target 的候选只依据 feature dtype/schema 输出高风险提示，不读取
target values，也不物化。Task 09 的完整冻结规则见
`docs/decisions/task09-feature-engineering-contract.md`。

## 7. 可视化能力设计

v0.1 使用任务型专用函数，优先通过 seaborn 实现统计分析型图表。matplotlib 保留为 seaborn 的底层 backend、`Figure`/`Axes` 对象契约和 seaborn 不适用时的低级 fallback。public plot functions 按已冻结契约返回 `PlotResult`（具名 matplotlib Figure、图型、来源、项目和稳定 metadata）或 `PlotCollection`；不返回裸 `Axes`。Task 10 的完整字段、图型、预算、metadata、错误、Figure 生命周期和全局状态规则以 `docs/decisions/task10-visualization-contract.md` 为准。图表范围包括：

- 分布：数值直方图、类别 top-N 条形图。
- 缺失：列缺失率图；复杂缺失组合图推迟到 v0.2。
- 相关：受 50 列上限约束的相关热图。
- 异常值：消费 `OutlierAnalysis.summary` 的每特征异常率条形图。
- 分组比较：消费 `GroupComparison` 的组间统计图。
- target relationship：消费 `TargetAnalysis` 的分类或回归专用图。
- 模型评估：分类的混淆矩阵/ROC（适用时）；回归的残差/预测对比图。

原则：

- API 表达分析问题，而不是镜像 `plt.hist` 等绘图原语。
- 优先使用 seaborn 完成统计分析型图表；matplotlib 用作底层 backend、Figure/Axes 契约和低级 fallback。
- 返回 `PlotResult` 中的 matplotlib Figure，不调用 `plt.show()`；保存由调用者或报告层负责。
- 不在函数中随意修改全局 matplotlib 或 seaborn style；其他后续任务若接受少量稳定 style/figsize 参数，样式影响必须局部且可恢复。该规则不适用于 Task 10：Task 10 仅支持冻结的六个 public signatures，不接受 `style`、`figsize`、`palette`、Axes 或 `**kwargs`。
- v0.1 只有 seaborn + matplotlib 这一实现栈，不建立多可视化后端抽象，不引入 Plotly、Altair、Bokeh 或 dashboard。
- 高基数类别、过多列与大样本按默认预算截断/抽样，并在结果元数据和报告中披露预算、实际数量及原因。
- 绘图函数优先消费已计算的分析结果；不得为了绘图隐藏重算统计。
- collection 图在无可绘制项目时返回空 `PlotCollection`；单 `PlotResult` API 返回正常但无 data artists 的 Figure。参数、类型或冻结 result schema 非法时抛出稳定 `ValueError`。

## 8. 建模与评估设计

### 8.1 训练契约

`train_classifier` 与 `train_regressor` 接受 DataFrame、显式 target、可选特征列、用户标记的 exclusions、可选 `time_column`、测试比例、随机种子和可选 estimator；二者遇到任一声明的时间列或任一 datetime/timedelta 列均稳定拒绝，不提供 time-aware split。分类只在 Task 11 支持 stratified split；回归在 Task 12 使用非分层随机 holdout。内部顺序固定：

1. 验证 target 与任务；
2. 分离 `X/y`；
3. 划分训练/测试（分类尽可能 stratify）；
4. 只基于训练分区确定/确认预处理列；
5. 构造 `ColumnTransformer`：数值插补与可选缩放，类别插补与 `OneHotEncoder(handle_unknown="ignore")`；
6. 将任何需拟合的特征变换放入 Pipeline；
7. 仅在训练集 fit；
8. 返回包含 fitted pipeline、split 索引、任务和 schema 快照的 task-specific training result。

默认 estimator 保持简单：分类使用 Logistic Regression，回归使用 Ridge。Task 11 的 `TrainingResult` 是分类专用；Task 12 的回归结果使用独立的 `RegressionTrainingResult`，不为回归伪造分类标签字段。树模型、模型比较和调参推迟。

### 8.2 独立评估

- `evaluate_classifier`：accuracy、balanced accuracy、macro F1；二分类且概率可用时增加 ROC AUC。返回混淆矩阵和类别标签。
- `evaluate_regressor`：MAE、RMSE、R²，并返回预测/残差表。
- `evaluate_model`：便利分派函数，根据 `TrainingResult` 或 `RegressionTrainingResult` 的冻结 `task` 调用严格分离的实现；不允许猜测任意 estimator 的任务类型。
- 默认只评估 holdout。训练指标可作为明确标记的诊断项，不能与测试指标混淆。

## 9. Data leakage 防护

以下是架构不变量：

1. 在 train/test split 前不得 `fit` 插补器、编码器、缩放器、分箱器、特征选择器或组统计映射。
2. target-aware 变换只能位于 Pipeline 内，并仅在训练折拟合；未来交叉验证时必须使用 out-of-fold 训练编码。
3. target 列、其直接变体、未来信息、后验字段和用户标记的排除列不得进入特征。Task 11 的 raw-DataFrame 训练入口以冻结的 `exclude_columns` 参数接收用户已知的后验/未来/entity 风险列；语义无法由 schema 可靠识别的风险必须由调用者显式标记，且这些列在 split、schema、feature selection 与 fit 前移除。
4. schema 的纯 dtype 识别可在全量 `X` 上执行，但任何数据驱动的阈值、类别词表或列选择必须由训练集决定。Task 11 更严格：其 `infer_schema`、ID-like、dtype eligibility 与最终 feature eligibility 只读取 split 后的 `X_train`，绝不以完整 DataFrame 决定模型特征。建模入口默认在 split 后确认这些决策。
5. v0.2 以后若实现 group aggregate，其映射也只能从训练行计算；未知组使用训练期全局统计或明确缺失策略。
6. 日期特征需要用户提供观察时点/参考日期时，不得默认使用当前时间生成不可复现或未来信息。
7. 报告必须披露 split 策略、随机种子、拟合范围、排除列和潜在泄漏警告。
8. 测试必须使用“测试集含独有类别/极值/组”的夹具证明这些值未影响 fitted state。
9. split 前检查重复索引与重复行；发现可能跨分区的实体重复时警告。v0.1 不声称解决 entity/group leakage。
10. v0.1 默认仅支持随机 holdout。Task 11 的 `time_column` 非 `None`，或输入含 datetime/timedelta 列时，必须停止使用 v0.1 建模入口并报稳定错误；预切分/time-aware 建模推迟到 v0.2。纯数值时间编码不能可靠自动识别，调用者必须通过 `time_column` 声明。不得将随机切分描述为时间安全。
11. holdout test set 只用于最终评估，不得用于特征筛选、阈值选择、模型选择或重试决策。
12. 默认 estimator 与 split 使用同一个 `random_state`。自定义 estimator 的随机状态由调用者设置，并记录在 `TrainingResult` 警告中。

## 10. Public API 草案

结果类型均为公开、只读倾向的 dataclass；具体字段可在实现前的小型 API 决策记录中冻结。函数默认不修改输入。Task 03 的冻结合同见
`docs/decisions/task03-schema-summary-contract.md`；Task 04 的冻结合同见
`docs/decisions/task04-quality-contract.md`；Task 05 的冻结合同见
`docs/decisions/task05-workflow-report-cli-contract.md`；Task 06 的冻结合同见
`docs/decisions/task06-excel-io-contract.md`；Task 07 的冻结合同见
`docs/decisions/task07-analysis-contract.md`；Task 08 的冻结合同见
`docs/decisions/task08-group-target-analysis-contract.md`。实现、测试和 API
文档不得偏离对应记录。

结果类型不在 Task 01 预先冻结，而是在拥有相应功能的 Task 中与行为、测试和文档一起冻结：

- `SchemaReport` 及列 schema 结果在 Task 03 冻结。
- `DataFrameSummary` 在 Task 03 的 `summarize_dataframe` 实现中冻结。
- `QualityReport` 及 `QualityIssue` 已由 Task 04 API 决策记录冻结，并在
  `check_data_quality` 实现中首次提供。
- `NumericAnalysis`、`CategoricalAnalysis`、`CorrelationAnalysis` 和
  `OutlierAnalysis` 在 Task 07 冻结。
- `GroupComparison` 和 `TargetAnalysis` 在 Task 08 冻结。
- `RegressionTrainingResult` 和 `RegressionEvaluation` 在 Task 12 冻结；其精确字段、回归验证、评估、图和 `evaluate_model` 分派规则见 `docs/decisions/task12-regression-baseline-evaluation-visualization-contract.md`。
- 其他结果类型在 `IMPLEMENTATION_PLAN.md` 指定的对应功能 Task 中冻结。

v0.1 不定义公共自定义异常体系，也不创建 `exceptions.py`。文件不存在、不可读或读取失败使用保留底层因果链的 `OSError`；无效参数、缺失列、非法列类型及其他用户输入错误使用可操作消息的 `ValueError`。若未来需要公共自定义异常，必须通过 v0.2 或单独的 SPEC 修改评审后引入。

### 10.1 I/O、schema 与摘要

```python
def load_csv(path: str | Path, **read_options: Any) -> pd.DataFrame: ...
def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    **read_options: Any,
) -> pd.DataFrame: ...
def infer_schema(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    id_threshold: float = 0.98,
) -> SchemaReport: ...
def summarize_dataframe(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
) -> DataFrameSummary: ...
```

- `load_*` 读取单一本地表，不修改列；文件缺失、解析错误和不支持参数以具因果链的 `OSError`/`ValueError` 报告。Excel 未安装 extra 时给出明确安装提示。唯一副作用是文件读取。
- Task 06 的 `load_excel` 只承诺本地 `.xlsx` 单 sheet 读取，使用
  optional `excel` extra 中的 `openpyxl`，返回 `pd.DataFrame`，拒绝
  多 sheet 返回、`engine` 覆盖和未冻结的 pandas `read_excel` 参数。
  Task 06 不做 schema、summary、quality、workflow、reporting 或 CLI
  集成；`sharper analyze` 的 Excel 支持推迟到 Task 13。
- `ColumnSchema` 字段冻结为 `name`、`pandas_dtype`、`logical_type`、
  `nullable`、`missing_count`、`missing_rate`、`unique_count`、
  `unique_rate`、`is_constant`、`is_id_like`、`confidence`、`reasons`。
  `logical_type` 只允许 `numeric`、`categorical`、`datetime`、`boolean`、
  `text`、`identifier` 和 `unknown`。
- `TargetCandidate` 字段冻结为 `name`、`suggested_task_type`、`confidence`、
  `reasons`；任务建议只允许 `classification`、`regression` 和 `unknown`。
- `SchemaReport` 字段冻结为 `n_rows`、`n_columns`、`columns`、
  `logical_type_counts`、`target_candidates`。
- `DataFrameSummary` 字段冻结为 `n_rows`、`n_columns`、
  `memory_usage_bytes`、`total_missing_cells`、`total_missing_rate`、`schema`、
  `column_summary`。
- `column_summary` 列顺序冻结为 `column`、`pandas_dtype`、`logical_type`、
  `non_null_count`、`missing_count`、`missing_rate`、`unique_count`、
  `unique_rate`、`is_constant`、`is_id_like`、`min`、`max`、`mean`、`std`、
  `q25`、`median`、`q75`。各列 dtype、包括空结果的 dtype，以 Task 03
  决策记录为准；`min`/`max` 不适用时为 `None`，其余 numeric stats
  不适用时为 `NaN`。
- Task 03 只支持字符串列名；非字符串列名报 `ValueError`，消息包含
  `DataFrame column names must all be strings`，且不得自动转换列名。
- `infer_schema` 返回逐列角色、置信度、理由与候选 target；候选不确认
  target，也不触发 target-aware 行为。日期字符串只做不修改输入的
  80% 可解析率检测；全缺失、空行表中的列和无法归类的混合 object 为
  `unknown`。完整优先级、ID 规则、rate 分母和 target candidate 规则以
  Task 03 决策记录为准。
- `ColumnSchema.confidence` 只使用 `1.0`、`0.9`、`0.85`、`0.8` 和
  `0.5`；`TargetCandidate.confidence` 在 v0.1 只产生 `0.9` 和 `0.75`，
  `0.6` 仅为未来明确的弱信号预留。两类 `reasons` 只能使用 Task 03
  决策记录冻结的 reason codes。
- object/string/category/StringDtype 的全部非缺失值经
  `str(value).strip().casefold()` 后只包含 `"true"`、`"false"` 时判为
  boolean，confidence 为 `0.8`，reason 为 `boolean_values_only`；不接受
  yes/no、y/n、字符串 0/1 或数字 0/1，也不修改原值或 dtype。
- mixed object unknown 必须在 identifier 前判断。跨多个 Python 类型族且
  未命中 direct、boolean-token 或 datetime-string 规则的列固定为
  unknown（confidence `0.5`、reason `mixed_object_unknown`），即使全唯一、
  唯一率达到阈值或列名含 id/uuid/key 也不得判为 identifier。
- 显式 target 必须进入候选并包含 `explicit_target`；不存在时报错消息
  包含 `target column not found`。
- `infer_schema` 与 `summarize_dataframe` 均接受 0 行或 0 列 DataFrame，
  并返回结构完整的空结果。重复列名、非法 `id_threshold`、不存在的显式
  target 或不匹配的传入 schema 报 `ValueError`。
- Task 03 不包含 duplicate rows、broader data quality、outlier、
  correlation 或 target relationship；这些能力留给 Task 04 或后续
  analysis tasks。

### 10.2 质量与分析

```python
def check_data_quality(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    missing_threshold: float = 0.40,
) -> QualityReport: ...
def analyze_numeric_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> NumericAnalysis: ...
def analyze_categorical_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    top_n: int = 10,
) -> CategoricalAnalysis: ...
def compute_correlations(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    max_columns: int = 50,
    min_periods: int = 2,
) -> CorrelationAnalysis: ...
def detect_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> OutlierAnalysis: ...
def analyze_target_relationships(
    df: pd.DataFrame,
    target: str,
    *,
    task: Literal["classification", "regression"],
    features: Sequence[str] | None = None,
) -> TargetAnalysis: ...
def compare_groups(
    df: pd.DataFrame,
    group_by: str,
    *,
    values: Sequence[str] | None = None,
    max_groups: int = 20,
) -> GroupComparison: ...
```

- Task 04 的完整冻结合同见
  `docs/decisions/task04-quality-contract.md`。
  `QualityIssue` 是 `dataclass(frozen=True)`，字段顺序冻结为 `code`、
  `severity`、`scope`、`column`、`count`、`ratio`、`threshold`、
  `message`、`suggestion`；对应类型依次为 `str`、`str`、`str`、
  `str | None`、`int | None`、`float | None`、`float | None`、`str`、
  `str`。
- `QualityReport` 是 `dataclass(frozen=True)`，字段顺序冻结为
  `n_rows: int`、`n_columns: int`、`issue_count: int`、
  `severity_counts: dict[str, int]`、`issues: list[QualityIssue]`。
  它不包含 schema、summary、时间戳、路径或随机 ID。
- Task 04 severity 只允许 `info`、`warning` 和 `error`；issue code
  只允许 `empty_dataframe`、`duplicate_rows`、
  `all_missing_column`、`high_missing_column`、`constant_column`、
  `near_constant_column`、`high_cardinality_categorical`、
  `identifier_like_column`、`infinite_values`、`mixed_python_types`
  和 `datetime_parse_failures`。
- Task 04 只做 minimal data quality reporting，只报告问题和建议，不修改
  输入 DataFrame。它不做 outlier、correlation、target relationship、
  feature engineering、visualization、modeling、evaluation、reporting、
  CLI、自动清洗、重复 index/实体、leakage 或业务规则检查。
- `missing_threshold` 必须满足 `0 < missing_threshold <= 1`。非字符串
  列名、非法阈值和不匹配的 schema 使用决策记录冻结的可操作
  `ValueError`。未提供 schema 时可以调用 `infer_schema`，但 Task 04
  不调用 `summarize_dataframe`。
- Near-constant 的固定阈值为 `0.95`；high-cardinality categorical
  只适用于 categorical schema，要求 `unique_count > 50` 且
  `unique_rate > 0.50`。各规则的分母、边界、互斥、稳定文本和排序以
  Task 04 决策记录为准。
- Task 07 只实现 non-target feature analysis：numeric feature analysis、
  categorical feature analysis、numeric pairwise correlations 和 numeric
  outlier detection。Task 07 不做 target relationship、grouped analysis、
  feature engineering、visualization、modeling、evaluation、reporting、
  workflow 或 CLI integration。
- `NumericAnalysis`、`CategoricalAnalysis`、`CorrelationAnalysis` 和
  `OutlierAnalysis` 的 dataclass 字段、输出表 schema、skipped reason
  vocabulary、错误行为和 deterministic ordering 以
  `docs/decisions/task07-analysis-contract.md` 为准。
- Task 07 的所有 analysis 函数验证列存在且适合任务；无可分析列时返回带
  skipped reasons 的空结果，不伪造统计量。`columns=None` 基于 pandas
  dtype 自动选择列，不调用 `infer_schema`、`summarize_dataframe` 或
  `check_data_quality`。
- Task 07 的 correlation 使用 long-form pairwise table，默认
  `method="pearson"`、`max_columns=50`、`min_periods=2`；Task 07 不返回
  p-values 或 heatmap。
- Task 07 的 outlier detection 只支持 IQR method，默认
  `threshold=1.5`；不删除异常值，不修改输入。
- Task 08 的完整冻结合同见
  `docs/decisions/task08-group-target-analysis-contract.md`。
  `GroupComparison` 和 `TargetAnalysis` 均为 `dataclass(frozen=True)`；字段、
  输出表 columns/dtypes、skipped reason vocabulary/precedence、errors、预算和
  deterministic ordering 以该记录为准。
- `compare_groups` v0.1 仅支持一个 categorical group key 和 numeric value
  columns；默认最多 20 个 groups，按频数和首次出现顺序截断并披露。
- Task 08 的 numeric target/value/feature 必须是 real numeric non-boolean；
  complex 不进入 Task 08 numeric 路径。该收缩使用 Task 08 专用 private
  predicate，不改变 Task 07 numeric dtype 合同。
- `analyze_target_relationships` 要求 target 无歧义且 task 显式；固定四条路径
  为 classification × numeric Kruskal-Wallis、classification × categorical
  Chi-square/Cramér's V、regression × numeric Pearson、regression ×
  categorical Kruskal-Wallis。它不会训练模型。
- Task 08 target analysis 固定最多 50 个 eligible features、每个
  categorical feature 最多 20 个 categories、classification target 最多
  20 个 classes。categorical feature category budget 只基于 target/feature
  complete cases；超限 feature 整体跳过，不截断 category 或创建 Other。
- Task 08 固定内部 `TASK08_MIN_GROUP_SIZE=2`，只用于 classification × numeric
  和 regression × categorical 的 Kruskal-Wallis group retention；它不是参数，
  也不进入 result metadata。SciPy statistic、p-value 或 effect size 非有限时，
  feature 使用 `statistical_test_not_applicable`。
- Task 08 limitations 使用决策记录冻结的封闭 vocabulary 和确定顺序。统计结果
  保留 retained 有效样本量、缺失处理和探索性限制。
- Task 08 不修改输入且无外部副作用，不接入 workflow、reporting、CLI 或
  I/O，也不改变 Task 07 non-target analysis 合同；Task 08 不调用 Task 07
  public analysis functions。
- 接受标准小型 DataFrame 时，结果中的计数、排序、相关系数和异常标记必须可重复。

### 10.3 特征与可视化

```python
def suggest_feature_derivations(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    target: str | None = None,
    exclude_columns: Sequence[str] = (),
    reference_date: str | date | datetime | pd.Timestamp | None = None,
    max_suggestions: int = 50,
) -> FeatureSuggestionReport: ...
def derive_features(
    df: pd.DataFrame,
    suggestions: Sequence[FeatureSuggestion],
    *,
    copy: bool = True,
) -> FeatureDerivationResult: ...
def plot_distributions(
    df: pd.DataFrame,
    *,
    max_plots: int = 20,
    sample_size: int = 10_000,
) -> PlotCollection: ...
def plot_missingness(df: pd.DataFrame, *, max_columns: int = 50) -> PlotResult: ...
def plot_correlations(result: CorrelationAnalysis) -> PlotResult: ...
def plot_outliers(result: OutlierAnalysis, *, max_plots: int = 20) -> PlotCollection: ...
def plot_group_comparison(result: GroupComparison) -> PlotCollection: ...
def plot_target_relationships(result: TargetAnalysis) -> PlotCollection: ...
def plot_classification_evaluation(
    result: ClassificationEvaluation,
) -> PlotCollection: ...
def plot_regression_evaluation(
    result: RegressionEvaluation,
) -> PlotCollection: ...
```

Task 10 的 `PlotResult` 与 `PlotCollection` 是 frozen dataclass，其精确字段与
visualization 输入/输出合同以
`docs/decisions/task10-visualization-contract.md` 为准；Task 10 不增加 feature
suggestion plotting API。

- 建议报告必须标记 leakage 风险、拟合需求、per-type/global budget 与确定性
  截断；target 和 explicit exclusions 不得成为 source columns。Task 09 不调用
  Task 07/08 analysis API，也不计算 correlation。
- `derive_features` 只接受 Task 09 白名单中的无状态 suggestion；返回 data、
  applied names、保留的 empty skipped fields 和 copy mode。除零、missing 和
  non-finite arithmetic result 按决策记录转为 `NaN`；arithmetic sources 必须在
  运算前转为 `float64`。任一 `requires_fit=True` suggestion 使整次调用
  fail-fast `ValueError`。所有 validation 和临时计算成功后才统一追加新列；
  `copy=False` 的任一失败也不得留下部分修改。`copy=True` 使用 pandas
  `df.copy(deep=True)` 语义，不承诺递归复制 object-cell Python objects。
- 绘图函数默认不显示、不保存；输入结果类型、冻结 result schema 或预算范围错误报稳定 `ValueError`。不适用 collection 图返回空 collection；单项图返回正常但无 data artists 的 Figure。

### 10.4 建模、评估与报告

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
def evaluate_classifier(result: TrainingResult) -> ClassificationEvaluation: ...
def evaluate_regressor(result: RegressionTrainingResult) -> RegressionEvaluation: ...
def evaluate_model(
    result: TrainingResult | RegressionTrainingResult,
) -> ClassificationEvaluation | RegressionEvaluation: ...
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

- Task 05 的最薄 workflow、Markdown 和 CLI 行为以
  `docs/decisions/task05-workflow-report-cli-contract.md` 为准；Task 13 的完整
  集成、扩展 `AnalysisRun`、Markdown/HTML、assets、CLI 和 errors 以
  `docs/decisions/task13-full-workflow-static-html-cli-contract.md` 为准。
  Task 13 后 `AnalysisRun` 是 `dataclass(frozen=True)`，字段顺序冻结为
  `schema`、`summary`、`quality`、调用配置、四个 non-target analysis result、
  optional group/target result、`feature_suggestions`、optional
  training/evaluation、`plots`、`skipped`、`warnings`、`limitations`；精确
  annotations、None union 与 ownership 以 Task 13 决策记录为准。它不保存 raw
  input DataFrame。
- `ReportArtifact` 是 `dataclass(frozen=True)`，字段顺序冻结为
  `path: Path`、`format: str`、`title: str`。Task 05 只支持 Markdown，
  不包含 content、asset、时间戳或其他非确定字段。
- `run_analysis` 是唯一完整 Python 编排入口：它只调用 Tasks 03--12 已冻结
  public APIs，不读写文件、不复制算法、不保留 raw DataFrame，并以 result 嵌套
  metadata 披露 budgets。target/task 从不自动推断；模型只在二者显式确认且
  `include_model=True` 时执行，分类和回归严格分派。
- `generate_analysis_report` 只消费 `AnalysisRun`，生成确定性 UTF-8 Markdown
  或标准库静态 HTML，并将既有 Figure 导出为相对 PNG assets；它不重算分析、
  training、evaluation、prediction、metric 或 plot。两种 format 都是报告与
  assets bundle；精确 section、asset、overwrite、deterministic
  staging/backup/rollback、preflight 和 Figure-close 行为以 Task 13 决策记录为准。
- 训练入口返回 fitted Pipeline 与 holdout 数据/索引元数据，不修改输入；小样本、缺失 target、单类分类或非法 split 报 `ValueError`。
- 分类/回归评估拒绝错误任务类型；仅消费未参与拟合的 holdout。
- 未显式提供 target/task 时绝不执行 target-aware analysis 或建模；未指定
  `group_by` 时不执行 group comparison。领域结果不继承通用 renderer hierarchy。

最小工作流示例：

```python
df = load_csv("data.csv")
run = run_analysis(df)
generate_analysis_report(
    run,
    output_path="report.md",
)
```

内部 helper、模板渲染器、推断启发式、sklearn transformer 构造器和 CLI 编排函数不从 `sharper.__init__` 导出。

## 11. CLI 设计

入口点：`sharper = "sharper.cli:app"`。该入口与 CLI help 在 `IMPLEMENTATION_PLAN.md` 的 Task 05 创建和验收；Task 01 不创建 `cli.py`、不注册脚本入口，也不验收 CLI help。Task 13 冻结 root eager `sharper --version`：在首个 `analyze` 前发现 `--version` 时，stdout 为 `sharper {__version__}` 加一个换行、stderr 为空、exit code 为 0，且不读取输入或调用 workflow/reporting；`analyze` 后的 `--version` 是 exit 2 的 parser error。Figure ownership、其他 CLI precedence 与输出细节以 Task 13 决策记录为准。

Task 05 只实现 Typer `analyze` 命令的 CSV → schema → summary → quality
→ Markdown 垂直切片：

```text
sharper analyze INPUT --output report.md
```

Task 05 的参数、默认值、help 最小内容、stdout/stderr 和 exit code 完整
冻结在 `docs/decisions/task05-workflow-report-cli-contract.md`。CLI 只调用
`load_csv`、`run_analysis` 和 `generate_analysis_report`，不直接组合领域
步骤或写 Markdown。

Task 06 只新增 Python API `load_excel`，不修改 Task 05 CLI。Task 06
完成后，`sharper analyze INPUT` 仍不接受 `.xlsx` 作为已支持输入；Excel
CLI 支持属于 Task 13 完整 workflow/CLI 收口。

以下是 Task 13 完成后的完整 CLI，不属于 Task 05：

```text
sharper analyze INPUT
  --output report.html
  --format html
  --target TARGET
  --task classification|regression
  --id-column COLUMN          # 可重复
  --exclude-column COLUMN     # 可重复
  --model / --no-model
  --random-state 42
```

Task 13 的 CLI 才默认执行 schema、摘要、质量、单变量分析、相关、异常、
特征建议、基础图表和报告。只有同时显式提供 `--target`、`--task` 与
`--model` 时才训练；target candidate 永远不会自动启用 target-aware
流程。

## 12. Packaging 与工具配置

`pyproject.toml` 应按 `IMPLEMENTATION_PLAN.md` 分 Task 配置：

- `[build-system]`：轻量 PEP 517 backend（建议 hatchling）；配置 `src` 包发现。
- `[project]`：分发名 `sharper`、描述、README、MIT license、作者、分类器、`requires-python = ">=3.10"` 和以 `sharper.__version__` 为单一来源的初始版本 `0.1.0`。
- 核心依赖：pandas、numpy、scikit-learn、matplotlib、seaborn、scipy、typer。可视化是 v0.1 核心能力，因此 seaborn 不拆分为 optional extra。版本下界应在首次实现与 CI 验证后确定，不凭空锁定。
- `[project.optional-dependencies]`：
  - `excel`：openpyxl，用于 Task 06 冻结的 `.xlsx` 单 sheet 读取；
  - `dev`：pytest、pytest-cov、ruff、build；
- v0.1 HTML 使用标准库和内部受控静态模板，不增加 renderer 依赖；复杂主题和 Markdown 扩展推迟。
- `[project.scripts]`：`sharper = "sharper.cli:app"`，仅在 Task 05 创建 `cli.py` 时加入；Task 01 不配置不存在的 CLI 入口。
- `[tool.pytest.ini_options]`：测试路径、严格 markers、短 traceback。
- `[tool.ruff]`：目标 Python 3.10、lint 规则、格式化约定。
- coverage 阈值在形成真实基线后设置；v0.1 发布前要求核心领域模块的分支有覆盖，不以单一百分比替代关键契约测试。

不提交 lock file。库项目发布 sdist 与 wheel；开发依赖不进入运行时依赖。

## 13. 测试策略与模块清单

| 模块/风险 | 测试清单 | 类型与位置 | 成功证据 |
|---|---|---|---|
| `io` | CSV 编码/选项、Excel sheet、缺失文件、坏格式、缺 Excel extra | 单元/集成，`test_io.py` | 数据不被隐式改写，错误可操作 |
| `schema` | 各逻辑类型、nullable dtype、混合值、ID 候选、target 候选、重复列、空表、确定性 | 单元，`test_schema.py` | 角色、置信度与理由符合夹具 |
| `summary` | 形状、内存、缺失、唯一值、分位数、空行/全缺失/常量 | 单元，`test_summary.py` | 计数与 pandas 基准一致 |
| `quality` | 每条规则、阈值边界、严重度、重复、inf、高基数、无静默修改 | 参数化单元，`test_quality.py` | issue code 与证据稳定 |
| `analysis` | 数值/类别统计、top-N、Pearson/Spearman、IQR outliers、分类/回归 target、NaN、常量、小样本 | 单元/数值，`test_analysis.py` | 与手算或 scipy 基准容差内一致 |
| 分组比较 | 单分组列、缺失组、超 20 组截断、非法 value、多 key 拒绝 | 单元，`test_analysis.py` | 统计正确且截断被披露 |
| `features` | 冻结结果类型、列 eligibility/exclusion、每类/全局预算、命名/去重/排序、reference date、arithmetic/datetime dtypes、copy/fail-fast、拒绝需 fit suggestion | 单元/性质，`test_features.py` | 无 target/statistical leakage、无组合爆炸、无部分 mutation，结果符合 Task 09 决策记录 |
| `visualization` | 每个任务函数、Figure 类型、图数量/采样上限、标签、空/常量/高基数、无 `show()`、Figure 清理、禁止重算 | headless 单元/视觉结构，`test_visualization.py` | Agg backend 下稳定，不泄漏 Figure，预算元数据完整 |
| `modeling` | 分类/回归默认 pipeline、自定义 estimator、未知类别、缺失值、单类、小样本、随机复现 | 集成，`test_modeling.py` | fitted state 只来自训练集 |
| leakage | 测试集独有类别/极值/组、重复行/索引警告、实体/时间风险、target/后验列排除、split 前 fit、禁止 test-set selection | 专项集成，`test_modeling.py`/`test_features.py` | transformer 统计不含测试数据，风险被拒绝或披露 |
| `evaluation` | 二/多分类指标、无概率 estimator、回归指标、错误任务、holdout 标签对齐 | 单元，`test_evaluation.py` | 与 sklearn metrics 一致 |
| `workflow` | 无 target、带 target 不建模、显式分类/回归建模、非法参数、预算/警告聚合、与 CLI 章节一致 | 集成，`test_workflow.py` | 一个 DataFrame 产生完整且确定的 `AnalysisRun` |
| `reporting` | Markdown/HTML、转义、图像资产、覆盖策略、只消费显式结果、I/O 失败 | 快照/集成，`test_reporting.py` | 文件可打开且内容/链接完整 |
| `cli` | 最小流程、Excel、带/不带模型、非法组合、退出码、帮助文本 | Typer runner 集成，`test_cli.py` | 输出产物完整、错误清晰 |
| public API | 只导出批准符号、签名与 docstring、输入不变性 | 合约，`test_public_api.py` | `__all__`、文档和类型契约一致 |
| 分发 | clean env 安装 wheel/sdist、import、CLI、metadata、缺 optional extra | 构建 smoke，`test_distribution.py`/CI | wheel/sdist 均可安装运行 |
| 示例 | README 与 examples 只使用 public API | doctest/subprocess/CI | 示例在小夹具上无错误 |

所有随机测试固定 seed；浮点比较使用明确容差。报告快照只锁定稳定语义，避免因时间戳、路径或 matplotlib 小版本产生脆弱测试。

共享最小夹具应覆盖：混合类型表（nullable boolean、日期字符串、ID、全缺失、常量、inf）、分类表（不平衡与测试独有类别）、已知线性关系回归表、重复实体分组表、时间顺序/未来字段表、零分母/日期 reference/列名冲突特征表，以及包含 skipped、warnings、plots 和可选 model 的 `AnalysisRun`。

## 14. 文档策略

Tasks 01--14 已完成。Task 13 提供完整 workflow、Markdown/HTML report + PNG
assets bundle 与 CLI；Task 14 完成文档、examples、wheel/sdist 构建与独立
clean-install 发布准备验证，不新增 API 或改变 Tasks 03--13 行为。release
readiness 已完成，但尚未发布到 PyPI；详细范围以
`docs/decisions/task14-release-readiness-contract.md` 为准。

- README：一句话价值、目标用户、能力边界、安装与 quickstart、项目状态。
- `docs/quickstart.md`：读取到报告，以及可选建模两条完整流程。
- `docs/analysis-guide.md`：每个分析结果如何解释及其限制。
- `docs/leakage.md`：split、Pipeline、target-aware 变换和时间字段的安全规则。
- `docs/api.md`：仅列 public API，并由签名/docstring 生成或校验。
- `examples/`：一个无 target 分析、一个显式 target 基线建模；示例必须在 CI 运行。
- 版本采用 SemVer。0.x 仍通过 changelog 记录 API 变更；首次发布前创建 changelog、license 和贡献说明。

## 15. 推荐实现顺序

实施顺序严格以 `IMPLEMENTATION_PLAN.md` 的有序 Task 为准，不把多个 Task 合并成新的“阶段”：

1. Task 01：package skeleton、`pyproject.toml`、`src` layout 与最小 import/version/`__all__` 契约；不冻结领域结果类型，不创建 CLI。验收不包含 CLI help。
2. Task 02：CSV I/O。
3. Task 03：schema 与 `summarize_dataframe`，并冻结 `SchemaReport`、列 schema 结果和 `DataFrameSummary`。
4. Task 04：`check_data_quality`，并冻结 `QualityReport` 与 quality issue 结果。
5. Task 05 及以后：依次按实施计划完成 workflow/Markdown/CLI、Excel、分析、特征、可视化、建模、评估、HTML 与发布准备。

每个 Task 都必须同时实现对应测试和最小文档。不得把 Task 02 改成 summary/quality，也不得将 Task 03/04 的公共结果类型提前到 Task 01。

## 16. v0.2 已批准路线

v0.2 roadmap 已通过统一 review；Task 15 implementation review 已通过且 Task 15 已完成；
Task 16 contract 已批准为 `Approved — Go`，其 implementation 已完成且 review 为 `Go`；Task 17
contract 已批准为 `Approved — Go`，implementation 已完成且 review 为 `Go`。Task 18唯一一次full
contract已批准为`Approved — Go`，final bounded contract closure为`Go`，implementation为`Implementation complete — review Go`且 final validation 已完成；
Task 19 original contract checkpoint已批准为`Approved — Go`；当前amended contract为`Approved — Go v2`，implementation为`Implemented — Post-Review Closure Complete`，Task19为`Complete`；
Task 20 唯一一次 Full Contract Review 已为`NO-GO — Permanently Closed`，C1–C4
targeted fixes 与 bounded contract closure 已完成，当前合同为`Approved — Go`，frozen
findings 为`T20-CR-01..15`（15 Closed / 0 Open），implementation为`Not Started`且
ready for implementation；v0.2 整体尚未完成或发布，当前 package version 仍为 `0.1.0`。
其权威路线合同为
`docs/decisions/v02-roadmap-contract.md`；本节冻结产品和架构边界，不冻结任何
Task 15--20 public symbol、签名或 dataclass 字段。

### 16.1 产品目标与三条独立路径

v0.2 按以下顺序交付：

1. Binary Risk Validation and Business Metrics；
2. Data Quality and Leakage Audit；
3. Pre-loan Eligibility Rules and Decision Strategy Simulation；
4. Post-loan Early Warning and Lifecycle Monitoring；
5. Explainability, Champion/Challenger and Governance；
6. v0.2 Integration and Release Readiness。

这些能力形成三条可独立启用、最终可组合报告的路径：风险分数验证、贷前准入、
贷后预警。贷前只执行 caller-defined 准入规则、回测和离线策略模拟，输出 simulated
action、rule path 和假设下的业务结果；贷后只在 `entity × observation time` 上执行
point-in-time 预警、alert lifecycle 和生命周期分析。两者不共享结果对象，也不执行
真实审批、账户操作、客户联系或催收动作。

v0.2 仍是通用表格分析能力，信用风险只是主要验证场景。代码、API、CLI 和测试不得
硬编码信用字段、正类标签、分数方向、动作名称、DPD/MOB 状态或 Kaggle schema。
反欺诈完全排除，且不属于本路线的延期项。

### 16.2 时间成熟度、分数和阈值语义

监督时间验证必须显式区分 `observation_time`、可空 `event_time`、
`outcome_end_time`、`label_available_time`、prediction horizon 和 reporting delay。
fold cutoff 为 `C` 时，训练行必须同时满足：

```text
observation_time < C
label_available_time <= C
```

未成熟标签必须从 fit、fold metrics、calibration 和 loss denominator 中排除，并记录
mature、immature、purged 和 unevaluable 数量。仅有 observation time 顺序不能证明
标签无泄漏；event 已发生也不能替代 label availability。

风险输入必须显式声明为以下一种语义：

- `ranking_score`：任意有限、方向明确的实数，只用于排序指标、gains/lift、score
  bands 和 caller 预声明 threshold 候选分析；
- `event_probability`：必须是显式 positive event 的概率且位于 `[0, 1]`，才可用于
  calibration、expected loss、expected revenue 或 expected payoff。

`decision_function`、raw margin 或一般 score 不得被自动解释为 probability。只有
ranking score 时，概率型业务结果必须 unavailable/undefined，而不是产生伪数值。
Task 15 只在 train/validation/OOF 上分析 caller 预声明的有界 threshold 候选；final
test 不用于选择，分析点不自动成为业务 cutoff。Task 17 只消费 caller 另行冻结的
cutoff/bands，不自动采用、部署或优化策略。

### 16.3 Ownership 与依赖 DAG

```text
Task 15 ─┐
         ├─> Task 17 ─┐
Task 16 ─┤             ├─> Task 19 ─> Task 20
         └─> Task 18 ─┘
```

- Task 15 拥有 risk validation、OOF、ranking/probability metrics、calibration、
  analytical threshold evidence 和基础 business metrics；
- Task 16 拥有 quality/leakage audit、input/missingness profiling、missingness drift，
  以及 Tasks 17/18 唯一共享的 private closed condition kernel；
- Task 17 使用独立聚焦的 pre-loan policy 模块，拥有 eligibility rule evaluation、
  simulated action、constraints、rule trace、backtest 和 policy comparison；
- Task 18 使用独立聚焦的 post-loan monitoring 模块，拥有 point-in-time signals、
  warning rules、alert history/episodes、backtest 和 lifecycle analysis；
- Task 19 使用独立聚焦的 explanation/comparison/governance owner，只消费 Tasks
  15--18 frozen results，并只新增自身的 model explanation、model comparison、
  prediction drift 和 performance stability；
- Task 20 只编排 opt-in workflow、静态 report、CLI、文档、examples 和发行准备。

### Task 19 contract stage

**状态：Contract Approved — Go v2；Implementation Implemented — Post-Review Closure Complete。**

**合同：** `docs/decisions/task19-explainability-champion-challenger-governance-contract.md`。

Task 19唯一一次full contract review已完成且verdict为`No-Go`；第一次bounded contract-review closure同样为
`No-Go`。`T19-CR-04/07/08/13/14`经residual targeted repair后在bounded contract-review re-closure中取得
`Go`；原approved checkpoint的`T19-CR-01..16`历史状态保持`Closed`。随后T19-CR-13 blocker adjudication确认
`invalid_owner_result`与`privacy_unsafe_value`不可达；targeted post-approval amendment已应用，bounded amendment closure为`Go`并已`Closed`，
当前amended contract为`Approved — Go v2`，implementation为`Implemented — Post-Review Closure Complete`，Task19为`Complete`。Task 19只消费Tasks 15--18 frozen owner facts，不重算其metrics、
missingness drift、rules、actions、alerts、backtests或lifecycle tables；它另以closed structured
declarations承载model coefficient/native/permutation attribution、prediction histograms、bounded
performance slices和governance metadata，并拥有带fixed-seed/count-only bootstrap uncertainty的
prediction TVD、带fixed-seed vector bootstrap uncertainty的model performance stability及
result-only plots。它不接受或执行estimator/callable/raw feature matrix，不新增mandatory dependency，
不进入Task 20 workflow/report/CLI/release范围。Task 20 implementation仍未开始。

Task19 FIX-B后直接、单向消费四个typed predecessor-result tuples；每个candidate以本次调用内的
0-based result position和owner-approved candidate locator绑定specific source。Model和strategy result
各最多绑定一个candidate；同一Task18 result可通过不同scenario key绑定多个warning candidates。
Criteria必须与pair family及closed owner-source registry一致；Task16只作diagnostic/explanation，不参与
directional comparison。Evidence refs使用privacy-safe compound locators与owner-authoritative fingerprint，
禁止pandas index/object identity/raw entity joins。Snapshot/alignment只有同一owner-result内部proof可为
verified；跨result/cross-owner默认unverified，Task16 config fingerprint、equal as-of/row count/position均
不能升级为raw snapshot proof。`T19-CR-01/06/07/11`均已`Closed`。

Task19 FIX-C冻结Tasks15/17/18的92个exact directional metric keys；caller direction仅是assertion，
不得反转registry，diagnostics/attributions/metadata保持not-directional。Numeric delta始终为challenger
minus champion；exact higher/lower/closed-range truth tables仅在owner status、snapshot/alignment及
per-owner normalized support identity通过后执行。Decision criteria显式记录role与promotion requirement；
policy冻结minimum comparable count和`promotion_only/all_recommendations`人工复核模式。必须恰有一个
approved champion，允许zero challenger，explicit pairs必须恰好覆盖challengers一次；challenger state及
promotion completeness进入closed五值recommendation matrix。`T19-CR-02/03/09/10`已在第一次bounded closure中
`Closed`。

Task19 FIX-D冻结governance exact datetime和naive/aware discipline、38-entry registry的authoritative
source-time mapping及`source_time <= analysis_as_of` fail-closed gate；time verification与snapshot/alignment
保持三个独立证明维度。Task15只在frozen time mode或observed-loss as-of可证明时提供absolute time，Task16
无result-level authoritative time且只能作time-unverified diagnostic，Task17/18分别使用唯一owner
`evaluation_time/analysis_as_of` provenance。Structured attribution/profile/performance declarations显式提供
evidence time，metadata time为not-applicable。

FIX-D同时冻结exact-type tagged canonical JSON encoder、11个semantic fingerprints、38/92/5三个registry
version identities、35-row privacy-safe provenance、76-key exact ValueError registry、5 statuses/15 reasons及
十张table各自ordinal finding-key family。禁止repr/str/Python-hash fallback、current clock、raw payload和
invalid-source-to-status降级。`T19-CR-04/12/13`均已`Closed`。

Task19 FIX-E冻结十表exact row units/columns/dtypes/nullability/identities、17个reachable caller-variable resource
gates、9个fixed cardinality/projection invariants及完整normative implementation test matrix；pair/comparison/
evaluation/recommendation/summary maxima分别为15/960/960/15/16，provenance固定35。SPEC/PLAN中所有Task19
状态已同步；`T19-CR-08/14/15/16`均已`Closed`。Task19 amended contract为`Approved — Go v2`，implementation为
`Implemented — Post-Review Closure Complete`；唯一一次Full Implementation Review为`NO-GO — Permanently Closed`，冻结的`T19-IR-01..06`为`6 Closed / 0 Open`，bounded implementation closure与post-review final validation均为`PASS`；A–O与16/16 roadmap traceability均已Closed；Implementation Stage已Complete；Task19为`Complete`；下一阶段唯一为`TASK19 FINAL IMPLEMENTATION CHECKPOINT COMMIT`。

### Task 20 contract stage

**状态：Approved — Go；Contract Governance Complete；Full Contract Review NO-GO — Permanently Closed；Implementation Not Started；Task20 Ready for Implementation。**

**合同：** `docs/decisions/task20-v02-integration-release-readiness-contract.md`。

Task 20 的 scope freeze、C1、C2、C3 与 C4 targeted amendments 冻结了独立 opt-in `v02_workflow`/static
report surface、score validation/pre-loan/post-loan 三条可独立启用路径、显式 optional
Task 16 audit handoff、显式 optional Task 19 governance final step、Tasks 15--19 的
exact-call matrix、result-only Markdown/HTML、两个 versioned closed pure-data JSON
schemas、Task 13 Figure/asset ownership、独立 `sharper v02-run` adapter、永久 v0.1
compatibility manifest、append-only exports、0.1.0→0.2.0 metadata gate、docs/examples/
CI/distribution acceptance 和 release-readiness-only boundary。C1 进一步冻结了：
score CLI 的 closed external-prediction Path A predicate 与 Task 15 caller mapping、
Task 19 governance dependency matrix 与 exactly-once gate、primary/secondary raw
DataFrame carrier roles、Task 15 closed `positive_label` domain，以及 warning/
limitation token grammar、source inventories、ordering、deduplication 和 parity。
精确 symbols、字段、errors、resources、report sections、CLI 参数和 A–X acceptance
matrix 只以该 decision record 为准；C1 不提前处理 C2 findings，C2 不提前处理 C3
findings，C3 不提前处理 C4 findings，C4 不提前处理 bounded closure。C2 进一步冻结了 policy JSON 中所有 Task17 tuple/sequence 的唯一
array representation 与 exact arity/order/duplicate semantics、Task17/18 JSON temporal
fields 的 exact six-digit UTC/naive grammar 与 integer-microsecond duration carrier，
以及 12-row resource registry、per-object/per-array/depth/condition counting、report
figure-entry pre-acquisition gate 和 staged aggregate PNG-byte gate。C3 进一步冻结了
exact12 report-section provenance matrix、enabled-empty/not-requested distinction、
nine-slot result-only plot registry、fixed ordinal filenames、Task13 title/path/overwrite
transaction、ReportArtifact field mapping、四套 API-specific validation precedence tables
与28-key error reachability matrix。

C4 进一步冻结唯一版本源 `src/sharper/__init__.py::__version__`、`dynamic` Hatchling
version linkage、approved checkpoint 仍为 `0.1.0`、唯一 `0.1.0→0.2.0` transition gate、
wheel/sdist 分离的 exact content matrix、Hatchling sdist include/exclude table、
CHANGELOG existing heading/bullet contract、31-entry exact future tracked allowlist 和
roadmap-to-file matrix。wheel 仅承载 runtime package；sdist 承载 buildable source、
approved docs、decisions、tests、CHANGELOG、license、README 与七个 examples；二者均
禁止缓存、生成物、notebooks、内部机器文件、credentials 和发布配置。

唯一一次 Full Contract Review 已以`NO-GO — Permanently Closed`结束，第二次 Full Contract
Review 被禁止。当前 C1 的 `T20-CR-01..05`、C2 的 `T20-CR-06..08`、C3 的
`T20-CR-09..12` 与 C4 的 `T20-CR-13..15` 均为`CLOSED`，bounded contract closure 为
`PASS`，OPEN finding 数为 0；approved contract checkpoint 提交后允许进入
implementation kickoff。Task20 不扩充
`AnalysisRun`、`run_analysis`、`sharper analyze`，不复制 Task16 private kernel，不重算
Tasks 15--19 facts，不新增 mandatory dependency，也不 tag/push/upload/创建 release 或
实际发布。

**下一阶段：** TASK20 — PHASED IMPLEMENTATION KICKOFF。第二次 Full Contract Review：Forbidden。

```text
Contract Governance:          Complete
Bounded Contract Closure:     PASS (15 CLOSED / 0 OPEN)
Approved Contract Checkpoint: This exact three-document commit
Current Package Version:      0.1.0
Implementation Target:        0.2.0
Task20 Release Goal:          Release Ready — Not Released
```

Tasks 17 和 18 并列且不相互依赖；Task 18 仅在使用模型分数时可选消费 Task 15。
Task 19 不重算 Tasks 15--18 的指标、missingness drift、规则、动作、alerts、backtest
或 lifecycle tables；Task 20 不承载领域算法。现有 `analysis.py` 保持 v0.1 职责，
不得扩张为 risk、policy、warning 或 lifecycle catch-all。

### 16.4 规则、动作与 CLI 边界

Task 16 的 shared condition kernel 必须是 private、dependency-light、封闭 operator
inventory、三值 `true/false/unknown`、确定性且有 missing/effective/expiration 和资源
预算语义。它不是 public DSL，不接受 `eval`、任意 Python、callable、函数、脚本、
插件或动态 operator；Tasks 17/18 只能消费，不能复制实现。

`action_name` 与 `action_role` 必须分离。动作名称由 caller 定义；只有存在显式 role
mapping 时才计算 selection/rejection/review/request-information 等业务指标。没有
mapping 时只输出通用 action distribution，不从名称猜测批准、拒绝或人工审核含义。
constraints 只评价 caller-frozen action scenario，不生成、改写或优化动作。

Task 20 的 CLI 规划版本化、封闭、纯数据输入：Path A 只接受显式
external-ranking-score 或 external-event-probability 列、严格 boolean validation
mask、显式 typed positive label、ranking direction/probability provenance，以及固定
`stratified_holdout` 的 score options；Path B/C 继续使用版本化 JSON policy/warning
spec。Path A 不自动推断 target、score、mask、label、direction、probability 或
maturity；Governance 保持 Python-only。未知 schema/version、字段、operator、score
kind 或 label encoding 必须失败；不接受 YAML/TOML、Python 表达式、函数、脚本、模板、
include、`$ref`、环境变量或路径展开，也不得演变为通用规则 DSL。

### 16.5 Public API、依赖与 v0.1 兼容

v0.2 使用新的 opt-in 入口和优先 frozen、具名、可序列化的结果，不向已有 v0.1
dataclass 追加语义关键字段，也不建立 manager、registry、plugin framework 或深层
继承。每个新增 public symbol 必须先由对应 Task contract 冻结，再同步文档和
contract tests 后加入 `sharper.__all__`。

v0.1 public signatures、dataclass 字段顺序、默认行为、errors、warnings、reports、
CLI、Figure ownership 和既有 exports 必须保持兼容。测试必须分为永久的
`v0.1 compatibility invariants` 和随版本演进的 `current release surface`；后者负责
当前版本、完整有序 `__all__`、v0.2 opt-in exports 和 distribution metadata。不得通过
删除、搬移或弱化 v0.1 行为测试完成迁移。

v0.2 core 不新增强制第三方依赖，也不以 LightGBM、XGBoost、CatBoost、SHAP 或其他
optional extra 作为完成门槛。每个 API 继续要求输入不可变性、确定性、有界预算、
有效样本量和限制披露。

### 16.6 延期与明确排除

以下旧 v0.2 草案能力已延期为后续独立研究，不属于 Tasks 15--20：新增 tree model
families、supervised binning、WOE、target encoding、learned group aggregation、
SHAP-specific public API、非目标类别—类别 Cramér's V、混合变量效应量、校正后的
多重检验、通用报告主题/章节配置和通用轻量配置文件。Task 08 已冻结的 target
analysis Cramér's V 不受影响。

动态风险定价、自动额度优化、reject inference、因果/action-effect、uplift、controlled
exploration、催收行动/渠道优化、多期动态规划、实时规则或决策引擎、自动贷款审批和
监管合规认证只保留为长期研究方向，不构成已批准版本承诺。

AutoML、深度学习、大规模 stacking、server/dashboard、反欺诈、任意代码执行规则
DSL、多表关系引擎、分布式后端、生产部署和实际版本发布明确不属于 v0.2。

### 16.7 后续规模与工作流研究

原 v0.3 研究方向继续保留，但不构成版本承诺：

- 多表但非数据库的数据关联分析，需先定义键与基数契约；
- 可选 Plotly 交互式静态 HTML extra；
- 分块 CSV profiling、采样策略与更明确的性能预算；
- 模型持久化、模型卡和可复现运行 manifest；
- 可选 statsmodels 或专用统计 extra（有明确用户需求后再决定）。

AutoML、深度学习、Web dashboard、分布式系统、feature store、MLflow 和云部署在
后续路线中仍不是承诺能力。

## 17. 风险、开放决策与验收标准

### 17.1 主要风险

- **v0.1 面过宽**：通过每类只支持少量固定算法、单一完整 CLI 和结构化结果契约控制，不以删除分析或建模整块来缩小。
- **自动推断误导**：输出置信度与理由，target 必须确认，质量建议不自动修复。
- **特征组合爆炸与伪发现**：强预算、排除规则、稳定排序和探索性统计免责声明。
- **泄漏**：以 Pipeline、专项测试和报告披露作为发布阻断条件。
- **报告耦合**：报告只消费结果，分析模块不感知渲染。

### 17.2 发布前开放决策

以下分发决策已经冻结，不再是开放项：distribution name 为 `sharper`，import name 为 `sharper`，license 为 MIT，初始版本为 `0.1.0`。

仍需在对应 Task 中完成的决策：

1. 通过最小支持矩阵确定依赖下界。
2. 在各功能 Task 中冻结对应公开结果 dataclass 字段；0.1 发布后字段删除/改名需弃用路径。

### 17.3 v0.1 完成标准

- Python API 与 CLI 均能对代表性 CSV 完成读取、理解、质量、分析、特征建议、绘图和 Markdown/HTML 报告。
- CLI 与 Python workflow 对相同输入产生相同章节、警告和预算披露。
- 显式 target 时可分别完成分类和回归基线，且 leakage 专项测试通过。
- 所有 public functions 有 type hints、docstring、错误契约和测试。
- Ruff、pytest、构建、clean install、示例和 CLI smoke checks 全部通过。
- 不包含 non-goals，核心安装不引入未说明的 heavy dependency。
