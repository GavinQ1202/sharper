# AGENTS.md

## 项目目的

Sharper 是结构化表格数据的综合分析工具包，覆盖读取、质量、画像、关系挖掘、候选特征、任务型可视化、可选基线建模和报告/CLI。它不是单纯 EDA，也不是 sklearn wrapper。`SPEC.md` 是范围、架构与 public API 的权威设计合同。

## 仓库与依赖方向

采用 `src/sharper/` 布局，测试位于 `tests/`。模块职责：

- `io`：文件读取。
- `schema`、`summary`：类型/角色推断与描述摘要。
- `quality`、`analysis`、`features`：质量、分析挖掘与特征发现。
- `modeling`、`evaluation`：训练与独立评估。
- `visualization`、`reporting`：图表与报告。
- `workflow`：唯一完整分析编排层，返回 `AnalysisRun`。
- `cli`：读取参数、调用 workflow、写报告。

领域模块不得依赖 workflow、`cli`、`reporting` 或 `visualization`。报告不得隐藏重算分析；workflow 只组合 public API；CLI 不得自行组合领域步骤或复制领域逻辑。不要创建 catch-all `utils`、manager、registry、插件系统或深层 class hierarchy。

## 开发规则

1. 优先函数式 API 和具名结果 dataclass；不要引入框架式抽象。
2. 每次实现或修复必须同步新增/更新测试。没有测试的行为变更不算完成。
3. 只修改当前任务需要的文件；禁止顺手重构、大范围格式化或无关依赖升级。
4. 不提交 lock file、生成报告、构建产物、缓存或本地数据。
5. 不得在未经 SPEC 更新和评审时扩大 v0.1 范围或新增 public API。
6. 运行时依赖必须有明确用户价值；开发工具不得进入核心依赖。
7. 不静默修改用户 DataFrame。任何 copy/in-place 行为必须在签名和 docstring 中明确。
8. 统计与推断必须记录有效样本量、缺失处理和限制；不得把探索性结果表述为因果结论。
9. 任何截断、top-N、列预算或采样必须记录请求值、实际值和原因，并进入 `AnalysisRun` 与报告。
10. 若 Task 存在已接受的 `docs/decisions/` API 决策记录，实现、测试和文档必须遵守该记录；需要改变冻结字段或行为时，先同步更新并评审决策记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`，不得直接在代码中偏离。

## 任务执行与专项 skill 边界

- `IMPLEMENTATION_PLAN.md` 是 Tasks 01--14 已完成 v0.1 与 Tasks 15--20 已批准 v0.2 的任务拆分、验收边界和实现顺序依据；`SPEC.md` 定义产品与最终能力。治理文件冲突时先同步并评审文档，不在实现中自行扩大、合并或跨 Task 提前实现。
- Task 01 只建立打包、工具配置和最小 import/version/`__all__` 契约；不冻结领域结果类型，不创建自定义异常体系或 CLI。
- `SchemaReport`、列 schema 结果和 `DataFrameSummary` 在 Task 03 冻结；`QualityIssue` 与 `QualityReport` 已由 Task 04 API 决策记录冻结，并在 Task 04 首次实现。
- Task 04 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task04-quality-contract.md`；改变冻结字段、code、severity、规则、阈值、文本或排序前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。
- Task 05 实现、测试、Markdown 和 CLI 文档必须遵守已接受的 `docs/decisions/task05-workflow-report-cli-contract.md`；改变冻结字段、签名、章节、文本、参数、输出通道或 exit code 前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。
- Task 06 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task06-excel-io-contract.md`；改变 `load_excel` 签名、`.xlsx` 单 sheet 范围、`read_options` 白名单、optional dependency、错误类型或稳定消息前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 06 不修改 CLI、workflow 或 reporting。
- Task 07 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task07-analysis-contract.md`；改变 analysis 函数签名、结果 dataclass 字段、输出表 schema、skipped reason codes/precedence、错误消息、排序或 non-target 范围前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 07 不修改 CLI、workflow 或 reporting，不实现 target relationship、grouped analysis、visualization、feature engineering、modeling 或 evaluation。
- Task 08 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task08-group-target-analysis-contract.md`；改变 `compare_groups`/`analyze_target_relationships` 签名、`GroupComparison`/`TargetAnalysis` 字段、输出表 schema、四条统计路径、effect size、Task 08 real-numeric 规则、固定 minimum group size、complete-case category budget、limitations vocabulary、skipped reason codes/precedence、错误消息、缺失/常量/infinity/小样本行为或排序前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 08 必须使用专用 private real-numeric predicate，complex 不进入 Task 08 numeric path；不得改变 Task 07 `_is_numeric_non_bool` 或 Task 07 public behavior。Task 08 不修改 workflow、reporting、CLI、I/O 或 Task 07 合同，不调用 Task 07 public analysis functions，不实现 visualization、feature engineering、modeling 或 evaluation。
- Task 09 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task09-feature-engineering-contract.md`；改变两个函数签名、三个 frozen result dataclass、feature/reason/risk vocabulary、requires-fit 映射、列 eligibility/exclusion、reference date、预算、pair enumeration、命名、去重、排序、错误、物化 dtype 或 copy 行为前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 09 只依赖 pandas、numpy、Task 03 schema contracts 和 `infer_schema`；Task 07 只是 sequencing prerequisite，不得 import/call Task 07/08 public analysis functions 或计算 correlation。Task 09 不修改 workflow、reporting、CLI、I/O、analysis、pyproject 或 Tasks 01–08 合同。
- Task 10 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task10-visualization-contract.md`；改变六个函数签名、`PlotResult`/`PlotCollection` 字段、图型、数据来源、算法、预算、metadata、排序、错误、空结果、Figure 生命周期或全局状态前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 10 只使用 seaborn + matplotlib；Task 09 仅为 sequencing prerequisite，不得绘制 feature suggestions。仅 `plot_distributions` 和 `plot_missingness` 接受 raw DataFrame，且不得调用 Task 07/08/09 public API；result-only 图不得重算统计。Task 10 不修改 workflow、reporting、CLI、I/O、analysis、features、pyproject、dependency groups 或 Tasks 01–09 contracts，也不保存文件。
- Task 12 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task12-regression-baseline-evaluation-visualization-contract.md`；改变 `train_regressor`/`evaluate_regressor`/`evaluate_model`/`plot_regression_evaluation` 签名、`RegressionTrainingResult`/`RegressionEvaluation` 字段、validation precedence、stable errors、holdout-only 数据流、leakage 边界、metrics、预测表、图、metadata、排序、Figure 生命周期或全局状态前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 12 的 `RegressionTrainingResult` 独立于 Task 11 分类 `TrainingResult`；仅扩展 `evaluate_model` 的回归分派，不改变分类结果字段或分类分支行为。Task 12 不修改 workflow、reporting、CLI、I/O、schema、summary、quality、analysis、features、Task 10 containers/图行为、依赖或 Task 13 内容，也不保存文件。
- Task 13 实现、测试和 API/CLI 文档必须遵守已接受的 `docs/decisions/task13-full-workflow-static-html-cli-contract.md`；改变扩展后的 `AnalysisRun` 字段、`run_analysis`/`generate_analysis_report` 签名、workflow call order 或次数、classification/regression dispatch、no-recomputation、Markdown/HTML section/asset/file 行为、CLI 参数/输出/exit code、stable errors、determinism、Figure ownership/lifecycle 或 allowlist 前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 13 只编排 Tasks 03--12 public APIs；workflow 不读写或保存 raw DataFrame，reporting 不训练/评估/predict/recompute，CLI 不含领域算法；reporting 仅在冻结的 figure-ownership acquisition point 后负责关闭 workflow stored Figures，之前由 Python caller 持有且 CLI 失败路径在 `finally` 关闭。
- Task 14 实现、测试和发布文档必须先评审并遵守 `docs/decisions/task14-release-readiness-contract.md`；它只审计既有 public surface、文档、examples、distribution 和 CI evidence，不新增 API 或改变 Tasks 03--13 behavior。Task 14 allowlist 仅为 `README.md`、`CHANGELOG.md`、`LICENSE`（仅核验）、`docs/quickstart.md`、`docs/analysis-guide.md`、`docs/leakage.md`、`docs/api.md`、`docs/decisions/task14-release-readiness-contract.md`、`examples/basic_analysis.py`、`examples/baseline_modeling.py`、`tests/test_public_api.py`、`tests/test_distribution.py`、`.github/workflows/ci.yml`、`pyproject.toml`、`SPEC.md`、`IMPLEMENTATION_PLAN.md` 与 `AGENTS.md`；不得使用“其他必要文件”等开放兜底。不得修改 `src/`、Tasks 03--13 contracts、其他 workflow files、lock file、生成物、cache 或 `docs/.DS_Store`；Task 14 不发布、tag、push 或修改核心依赖范围。
- v0.1 默认以 `OSError` 表示文件读取失败，以 `ValueError` 表示无效参数、缺失列和非法列类型等用户输入错误；没有单独 SPEC 修改不得新增公共自定义异常体系。
- `analytics-workflow-builder` 最早可用于 Task 03 或 Task 04，不用于 Task 01 或 Task 02。
- `feature-engineering-builder` 不用于 Task 01、Task 02、Task 03 或 Task 04；仅在 `IMPLEMENTATION_PLAN.md` 进入 feature engineering Task 后使用。
- `visualization-system-builder` 不用于 Task 01，也不用于尚未进入 visualization Task 的工作。

## v0.2 长期治理不变量

- v0.2 开发必须遵守已批准的 `docs/decisions/v02-roadmap-contract.md`、`SPEC.md`、`IMPLEMENTATION_PLAN.md` 和当前 Task 独立合同；每个新 Task 严格按 contract -> review -> implementation -> diff review -> final Go 推进。
- Tasks 17、18、19 必须使用独立聚焦模块；现有 `analysis.py` 不得扩张为 risk、policy、warning 或 lifecycle catch-all。Task 19 只消费 Tasks 15--18 frozen results，Task 20 只编排 workflow/report/CLI，不得承载领域算法。
- point-in-time 监督验证必须显式验证 label maturity；time fold 训练行同时满足 `observation_time < fold_cutoff` 与 `label_available_time <= fold_cutoff`，未成熟标签不得进入 fit、metrics、calibration 或 loss denominator。
- `ranking_score` 只用于排序、bands 和 cutoff 分析；只有 `[0, 1]` 内且对应显式正类的 `event_probability` 可用于 calibration 和 expected loss/revenue/payoff。不得把 margin、`decision_function` 或一般 score 自动当作概率。
- Task 16 是 private closed condition kernel 与 missingness drift 的唯一 owner；Tasks 17/18 只消费 kernel，Task 19 只消费 frozen missingness evidence，不得重复实现或重算。
- condition kernel 不得成为 public DSL，也不得接受 `eval`、任意 Python、callable、函数、脚本、插件或动态 operator；Task 20 CLI 只允许版本化、封闭、纯数据 JSON 载体。
- v0.2 不研究、不规划、不实现反欺诈，且不把反欺诈列为本路线延期项。不得执行真实审批、账户操作、客户联系或催收动作。
- v0.1 signatures、dataclass fields、默认行为、errors、reports、CLI 和既有 exports 必须保持兼容；v0.2 使用 opt-in 入口，不得通过删除或弱化 v0.1 测试完成迁移。

## Leakage 不变量

- 禁止在 train/test split 前 fit 任何 preprocessing object，包括 imputer、encoder、scaler、binner、selector 和聚合映射。
- 先分离 `X/y`，再 split；只在训练分区构建数据驱动状态。
- 所有需拟合的特征工程必须置于 sklearn Pipeline，并在训练集/训练折内 fit。
- target-aware 特征必须使用训练折；未来支持交叉验证时必须 out-of-fold/cross-fit。
- target、target 的直接派生、后验字段、未来信息、ID 和显式排除列不得进入特征。
- v0.2 以后若实现 group aggregate，其映射只允许用训练行拟合；测试集独有组不得影响统计。
- 任何依赖“当前日期”的特征都必须改用显式 reference date。
- 每个相关改动必须包含测试集独有类别、极值或组的 leakage 回归测试。
- split 前检查重复索引和重复行，并对潜在跨分区重复发出警告。
- v0.1 不声称解决 entity/group leakage；存在同一实体多行时必须警告或拒绝建模。
- v0.1 不声称随机 holdout 对时间数据安全；发现时间顺序/未来字段风险时必须停止建模。预切分/time-aware 建模属于 v0.2。
- test set 只用于最终评估，禁止用于特征、阈值、模型选择或失败后的反复试验。
- 默认 estimator 与 split 使用统一 `random_state`；自定义 estimator 的随机状态责任必须记录。

## Feature engineering 规则

- `suggest_feature_derivations` 默认只建议，不物化。
- 所有建议使用 Task 09 决策记录冻结的来源列、canonical formula/parameters、封闭 reason/risk、requires-fit 和稳定名称。
- 按冻结的 per-type/global budgets、pair direction、去重和 deterministic ordering 生成；禁止无界笛卡尔组合、随机抽样或 correlation-based search。
- v0.1 仅可直接物化 ratio、difference、product、pandas datetime components 和基于显式 reference date 的 days-since。
- 需要拟合的 suggestion 不得通过普通 DataFrame helper 在全量数据执行。
- learned/fixed binning、group aggregate、target encoding、WOE、监督分箱和 target-aware candidates 在 v0.1 只能建议，不提供 transform。
- target values 不参与候选评分；target、显式 exclusions、ID-like、all-missing、constant 和 exact duplicate-content columns 不作为 source。
- 只有 timezone-naive pandas datetime 可作为 datetime source；weekday/weekend、reference-date dispatch 和 timezone-aware rejection 必须遵守 Task 09 决策记录。
- Arithmetic source 必须在运算前转为 `float64`；除零、无穷、large-integer overflow 和 datetime missing 必须遵守决策记录。
- `derive_features` 必须 validation/computation-before-mutation；`copy=False` 的任何失败不得部分修改输入，`copy=True` 只承诺 pandas `df.copy(deep=True)` 而非递归复制 object cells。

## Visualization API 规则

- API 围绕“分布、缺失、相关、目标关系、模型评估”等分析任务，不镜像 matplotlib 原语。
- v0.1 统计分析型图表优先使用 seaborn；matplotlib 保留为底层 backend、Figure/Axes 对象契约和低级 fallback。
- 按 SPEC 冻结的 public contract 返回包含 `matplotlib.figure.Figure` 的具名结果或集合；不得自行改为裸 `Axes`，库代码不得调用 `show()`。
- 默认不写文件；保存由报告层或调用者负责。
- 不在函数中随意修改全局 matplotlib 或 seaborn style。高基数、过多列和大样本必须受预算约束并披露截断/抽样。
- v0.1 不建立多可视化后端系统，不引入 Plotly、Altair、Bokeh 或 dashboard。
- 绘图函数应消费已有分析结果；禁止为了绘图隐藏重算统计。
- 每张图必须创建独立 Figure；库代码不得 `show()` 或 `close()`，不得切换 backend、修改 rcParams 或调用全局 matplotlib/seaborn style/theme/palette API。
- v0.1 必须分别覆盖分布、缺失率、相关、异常值、分组比较、target relationship、分类评估和回归评估图。
- 空列、全缺失、常量和不适用图型应明确跳过或报可解释错误。
- 绘图测试使用 headless backend，并关闭 Figure 防止资源泄漏。

## Public API 规范

- 只通过 `sharper.__init__` 和文档明确导出的符号视为稳定 public API。
- 每个 public function/class/dataclass 必须有完整 type hints 和 docstring。
- Docstring 至少说明参数、返回值、异常、副作用、缺失值策略和最小示例。
- 不返回无结构、字段不稳定的嵌套 dict；优先具名 dataclass 和 DataFrame 明细。
- 输入错误使用具体异常和可操作消息，不吞掉底层异常因果链。
- public API 的签名或结果字段变化必须更新 SPEC、README/API 文档和合约测试。
- 内部 helper 使用前导下划线且不得被 README/examples 导入。

## 测试与质量检查

测试目录按领域契约组织，不要求机械复制源码树。至少覆盖正常路径、空/缺失/常量、小样本、非法列、混合类型、日期、ID-like、异常值、缺失模式、输入不变性、确定性和错误消息。数值结果与 pandas/scipy/sklearn 基准比较并使用明确容差。workflow 与 CLI 必须对同一输入产生一致章节；报告测试必须验证 Markdown/HTML 和图像链接；绘图测试必须使用 headless backend。

## Review Scope, Full Audits, and Closure

### 默认范围与 scope declaration

- 所有审查默认采用能够可靠验证当前变更的最小范围；范围必须匹配工作类型、修改文件和风险，不得仅为继续寻找问题而重复 full audit。已通过独立审查且未被当前修改触及的领域默认保持关闭，后续修复优先使用 targeted review 或 bounded closure review。
- 每次审查开始时，prompt/报告必须明确 review mode、in-scope files/findings、允许重新打开的结论、明确排除领域、阻塞条件和终止条件；未声明 scope 时不得自行扩张为 full audit。
- 最小审查范围只限制 review 扩围，不削弱当前合同要求的测试、uv 环境、allowlist、compatibility 或 release-readiness 门禁。

### Review modes

- **Targeted review**：用于单个或少量已知 finding、单一行为/错误/schema 边界、定向测试或文档/状态/distribution gate 修复。只验证 finding 是否关闭、直接行为、直接回归及必要门禁，不重审无关合同或模块。
- **Bounded closure review**：用于一次完整审查后的收口。scope 冻结为未关闭 findings；已关闭结论不得重开，只检查直接回归。除新 P0、与修改直接相关且有证据的 P1 或已接受假设变化外，新观察记为非阻塞 backlog；满足终止条件后必须给出 `Go`。
- **Full audit**：重新检查完整合同、跨模块 ownership、实现、测试、兼容性和必要门禁。仅在新 Task 精确合同首次批准、实现完成后的首次独立 review、跨领域架构变更、public API/schema/持久化格式/依赖/版本/distribution contract 变化、validation/OOF/point-in-time/label-maturity/row-alignment 等系统性语义变化、有证据表明遗漏系统性 P0/P1、major phase boundary、release-readiness/实际发布前或用户明确要求时使用。
- **Release-readiness audit**：在版本或阶段发布前检查完整 tests、lint/format、build、wheel/sdist clean install、distribution smoke、version/metadata、文档状态、compatibility invariants 和 release checklist；除发现真实 P0/P1 外，不重新设计已批准合同。
- “可能还有问题”“测试可以更严格”或“再全面看看”不是 full-audit 触发条件。修复已知 findings、加强定向测试、修改 error/reason/validation precedence、单一 plot/dtype/resource/status/Markdown/format/distribution-environment 问题、批准 allowlist 内调整、不改变 public API/schema/dependency/ownership 的局部重构，以及上一轮 full review 后的剩余收口，默认使用 targeted 或 bounded closure review。

### 已关闭结论与证据标准

- 已接受的 finding、模块或合同条款仅在当前修改直接触及、有可复现 P0/P1、先前假设变化、public API/schema/ownership/dependency/execution model 变化、用户明确要求或正式 release-readiness audit 时可重开。不得仅因 reviewer 想到更强的测试方式而否定已接受证据。
- 验收要求在合同或首次 full audit 中冻结。已接受的实现与测试不得仅因还能增加 spy、fixture、边界值或 artist assertion 而重新阻塞；只有能给出现有测试会让错误实现错误通过的具体示例时，才可要求补充阻塞证据。可选测试、防御性检查和维护建议记为 backlog；测试数量或证明形式不是目标。

### Finding 等级与证据

- **P0 — Critical blocker**：包括 train/validation、时间标签或 observed-loss 泄漏，prediction/target 错位，ranking score 被当作 probability，核心数学系统性错误，数据损坏，严重安全问题或 v0.1 核心 public behavior 严重破坏；任何 P0 均为 `No-Go`。
- **P1 — Required blocker**：包括批准的 public API/schema 不一致、当前核心功能违反合同、必需 validation/boundary 缺失、ownership 冲突/重复实现、必需门禁失败、明确 compatibility 回归或 distribution/artifact 隔离实质削弱。P1 必须与当前 scope 直接相关并有可复现证据。
- **P2 — Non-blocking backlog**：包括不影响正确性的额外测试、防御性校验、非关键文档/状态/错误文本、维护性重构或 bounded scope 外的边缘观察；默认不阻塞，除非其正是当前文档/状态任务的验收目标。
- **P3 — Style or optional improvement**：包括命名、格式、注释、非必要重构、convenience API 或视觉偏好；不得阻塞。
- 每个 P0/P1 必须同时给出准确文件和行范围、可复现输入/触发方式、预期行为、实际行为、违反的合同/治理/兼容要求、为何属于当前 scope 以及最小修复方向。推测、偏好或无复现的风险不得列为 P0/P1。

### Task 生命周期、收口与扩围

每个 Task 默认遵循：

```text
roadmap/governance approval
-> contract drafting
-> one full contract review
-> targeted contract fixes
-> bounded contract closure
-> contract approval
-> implementation
-> one full implementation review
-> targeted implementation fixes
-> bounded implementation closure
-> final Task Go
```

- contract 和 implementation 阶段各最多一次开放式 full review；后续默认 targeted/bounded。只有新的系统性 P0/P1、大范围变更或用户明确要求时才可再次 full audit。Task 最终 Go 后，除非后续修改触及该领域，不得在其他 Task review 中重审。
- bounded closure 在列明 findings 全部关闭、无直接 P0/P1 回归、要求的测试/门禁通过且 public API/schema/dependency/version 无未授权变化时必须结束并给出 `Verdict: Go`。无关 P0 可阻塞；与修改直接相关且有证据的 P1 可阻塞；其他观察只能记为 P2/P3，不得延迟收口。
- reviewer 若认为 targeted/bounded review 必须扩为 full audit，不得直接扩围；必须先报告具体原因、命中的 full-audit 条件、涉及模块、额外成本以及当前工作是否应暂停。除用户已授权或发现紧急 P0 外，应等待确认。

## Mandatory uv environment

All Python-related work in this repository must use the uv-managed project virtual environment at `.venv`.

The canonical interpreter is:

```text
<repo-root>/.venv/bin/python
```

Before running tests, builds, Ruff, examples, CLI commands, or temporary Python scripts, run:

```bash
bash scripts/verify-uv-env.sh
```

All project Python commands must use one of these forms:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build --no-isolation
.venv/bin/python -m sharper.cli ...
.venv/bin/python examples/<script>.py ...
```

Do not use `python`, `python3`, `pip`, `pip3`, system Python, Conda Python,
another venv, `/usr/bin/python`, or `/usr/local/bin/python`. Do not treat
missing packages in system Python as a project blocker.

If the project `.venv` is missing a required test, build, lint, or development
tool, internet installation is allowed only with:

```bash
uv pip install --python .venv/bin/python <explicit-packages>
```

Such installation must target only `.venv`, install only the minimum explicitly
required packages, and must not use `uv add`, `uv sync`, `uv lock`, `--system`,
`--user`, `sudo`, global tool installation, unrelated package upgrades, changes
to `pyproject.toml` or a lock file, runtime dependency metadata, or the build
backend. After installation, verify that each installed module is imported from
`.venv`.

The only permitted alternative interpreters are temporary wheel/sdist virtual
environments created by distribution tests. They may be used only for artifact
clean-install verification. The controlling test process, artifact build, Ruff,
normal pytest, examples, and development CLI commands must still use
`.venv/bin/python`.

Any test, build, Ruff, example, or CLI result produced with the wrong
controlling interpreter is invalid and must be rerun in `.venv`.

When installing or using key tools, verify that each current-task module comes
from `.venv`, for example:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import build
import hatchling

venv = Path(".venv").resolve()
for name, module in {"build": build, "hatchling": hatchling}.items():
    path = Path(module.__file__).resolve()
    try:
        path.relative_to(venv)
    except ValueError as exc:
        raise SystemExit(f"{name} is outside .venv: {path}") from exc
    print(f"{name}: {path}")
PY
```

This is an example pattern: check the key tools actually used by the current
task; it does not require `build` or `hatchling` for every task.

Do not modify or delete `.DS_Store`, `docs/.DS_Store`, tracked `.pyc`, or
unrelated existing workspace changes.

For Task 01 clean-environment verification, use the same project-local
interpreter unless the task explicitly requires a separate clean environment.

在 `pyproject.toml` 完成对应配置后，每个实现任务结束前运行：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

涉及打包、public API、CLI 或发布时，另运行：

```bash
.venv/bin/python -m build --no-isolation
```

Task 01 的干净环境验证范围仅包括：wheel/sdist 可以构建、安装后可以 `import sharper`、当前公共契约要求时可以读取 `sharper.__version__`，以及 pytest smoke tests 通过。`sharper --help` 仅从 Task 05 或其他明确的 CLI task 开始验证。若当前 Task 范围内的命令或入口尚未在 `pyproject.toml` 配置，不得声称相应检查通过；应明确报告未运行原因。

## 任务交付

交付说明应列出：行为变化、测试变化、执行过的命令及结果、未解决风险。不要用“覆盖率高”替代关键契约与 leakage 测试证据。
