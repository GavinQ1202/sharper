# Sharper v0.2 Roadmap Contract

## 1. 状态、身份与权威边界

**状态：Approved — Go。** 本文冻结 Sharper v0.2
的产品目标、范围、任务顺序和完成定义；它不是 Task 15 的实现合同，也不授权修改
代码、测试、CLI、依赖、版本或 v0.1 行为。

**研究基线：** Sharper `0.1.0`，稳定提交 `0b86986`，Tasks 01–14 已完成并通过
Go。研究输入为 `docs/research/kaggle-credit-risk-methods.md`、
`docs/research/credit-risk-decision-strategies.md` 和
`docs/research/credit-policy-and-early-warning.md`；三者分别回答“数据如何形成可验证
风险分数”“分数如何进入可审计离线策略”“贷前准入与贷后预警如何形成不同的规则、
时间、回测和治理对象”。

`SPEC.md` 仍是产品范围、架构和 public API 的最终权威，
`IMPLEMENTATION_PLAN.md` 仍是任务执行依据。本文通过 review 后，开始 Task 15 前
必须先让 `SPEC.md`、`IMPLEMENTATION_PLAN.md` 与 `AGENTS.md` 的 v0.2 边界与本文
一致；若冲突，先修订并 review 文档，不在实现中自行选择。

本文使用以下规范词：

- “必须/不得”是 v0.2 roadmap 的冻结要求；
- “可”表示任务合同可以在本文边界内选择；
- 每个 Task 的精确 public signatures、dataclass 字段、errors、排序、allowlist
  和测试 fixture 必须由该 Task 的独立决策记录冻结，本文不提前伪造这些细节。

### 1.1 与现有 `SPEC.md` v0.2 草案的差异处置

现有 `SPEC.md` 第 16 节是 v0.1 阶段形成的初步路线，不得被本文静默覆盖。roadmap
review 必须逐项接受下表中的吸收或延期提案；在 `SPEC.md` 同步前，这些提案不具有
implementation authority：

| 现有草案条目 | 本文处置 | Tasks 15–20 归属 |
|---|---|---|
| supervised binning、target encoding、WOE、learned group aggregate | 提议延期；策略闭环优先，v0.2 不新增 learned transform API | 长期 roadmap；须修订 `SPEC.md` |
| group/time validation 与 CV | 保留 | Task 15 |
| 有限 tree baseline | 提议延期；v0.2 消费 v0.1 caller-supplied estimator 或外部分数 | 长期 roadmap；须修订 `SPEC.md` |
| 更完整缺失模式、共线性、两个显式数据集的 drift | 保留并收紧：Task 16 唯一计算 input/missingness profile 与 missingness drift，Task 19 只消费其 frozen evidence 并拥有 prediction/performance stability | Tasks 16、19 |
| score band、cutoff、贷前准入动作、贷后 alert/lifecycle、基础成本收益/约束与规则治理 | 新增；作为本 roadmap 的通用离线分析闭环 | Tasks 15、17–20；须同步 `SPEC.md` |
| 非目标类别—类别 Cramér's V、混合变量效应量、校正后的多重检验 | 提议延期，不进入本轮优先路线 | 后续 roadmap；须修订 `SPEC.md` |
| 报告主题、章节选择、通用轻量配置文件 | 提议延期；Task 20 只集成固定章节，并仅为 Task 17/18 CLI 接受 versioned closed JSON policy/warning spec | 后续 roadmap；窄 JSON carrier 属 Task 20；须修订 `SPEC.md` |

上述延期不删除 Task 08 已冻结的 target analysis Cramér's V 行为，也不禁止 v0.2
报告复用 v0.1 已有报告能力；它只是不新增相应 public capability。若 roadmap review 不
接受任一延期，必须先重新划分 Tasks 15–20，而不是在实施中把能力塞入相邻 Task。

## 2. v0.2 产品目标

v0.2 的产品目标是：在不把 Sharper 变成信用评分专用包、业务审批/预警执行系统、
反欺诈系统或 AutoML 框架的前提下，为二分类风险型表格问题提供可审查的 validation、
business metrics、leakage audit、贷前准入策略模拟、贷后 point-in-time 预警与生命
周期监测、解释与治理，并接入现有静态报告工作流。

v0.2 必须形成三个可独立启用、最终可组合报告的路径：

```text
single DataFrame
  -> schema / summary / v0.1 quality
  -> v0.2 quality + leakage audit

  A. score validation
     -> explicit target + positive label + frozen ranking scores and/or event probabilities
     -> explicit score kind, direction, class mapping and provenance
     -> stratified/group/time metrics + calibration + business curves

  B. pre-loan eligibility
     -> explicit evaluation time + rules + caller actions
     -> optional ranking scores/event probabilities + caller-frozen bands
     -> rules/bands/precedence/fallback produce simulated actions
     -> optional outcomes/costs/constraints evaluate the frozen replay
     -> rule/policy backtest + feasibility evidence

  C. post-loan early warning
     -> entity x observation time + prior-only signals/rules
     -> optional future events for matured-horizon backtest
     -> alert episodes + vintage/MOB/roll-rate/cure/lifecycle

  -> model/rule/policy provenance + comparison inventory + audit/governance
  -> result-only Markdown/HTML report and opt-in CLI workflow
```

路径 A 才必须有 binary target/positive label；路径 B 只有使用 score bands 时才要求
score，只有评估 outcome impact 时才要求 outcome；路径 C 可以在无 future label 时生成
离线 alert/history，但不得计算 precision、capture、lead time 等监督回测指标。

v0.2 继续保持 Sharper 的定位：结构化表格综合分析工具包，建模只是分析闭环的一部分；
它不是 sklearn wrapper、信用审批系统、竞赛复现框架或模型部署平台。

### 2.1 in-scope

v0.2 只包含：

1. 二分类风险 validation 与风险指标；
2. 数据质量、有限缺失模式/共线性与 group/time/entity leakage audit；
3. ranking score discrimination、event-probability calibration、score banding、
   threshold/cutoff analysis、Gini、KS、PR、gains、lift 和基础风险/业务指标；
4. 通用 `scores + rules + actions + costs + constraints + observed outcomes` 输入上的
   deterministic offline strategy simulation；其中 score 必须声明为 ranking score 或
   positive-event probability，二者可计算范围不同；
5. 数据完整性、资格、产品适用性、信用政策、敞口、affordability、历史状态、产品
   互斥、补件和人工审核等 caller-defined 贷前准入规则；
6. `pass/approve/decline/refer/request_information/limit/exclude` 仅作为
   caller-supplied symbolic action-name 示例的多动作策略；业务指标通过显式通用
   action-role mapping 计算，不从名称猜测；
7. 基础 expected loss/expected payoff、observed outcome replay、manual-review
   capacity、exposure/budget/rate 等约束与可行性报告；expected 指标只接受合法
   event probability；
8. `customer/account × observation date` 上的 level/change/trend/persistence/
   combination/state-transition/peer/history 贷后预警规则和 alert lifecycle；
9. 显式声明的多期宽表、单表 entity-event point-in-time 描述、vintage/MOB、
   roll-rate、cure、early delinquency 和 cohort/time monitoring；
10. 模型/规则/策略 reason provenance、rule path、override/policy audit、离线
   champion/challenger comparison；
11. 系数/原生/置换重要性、reference/current drift 和 time/group performance
   stability；
12. 预算、样本量、fold、observation/horizon、时间边界、maturity/censoring、outcome
   support、假设、缺失处理和限制披露；
13. 对上述结果的 opt-in workflow、静态 Markdown/HTML 和 CLI 集成；
14. 文档、examples、distribution/clean-install/CI 的 v0.2 release readiness。

### 2.2 数据形态支持

| 数据形态 | v0.2 支持级别 | 冻结边界 |
|---|---|---|
| 单表静态数据 | 完整支持 | binary audit、validation、score/business metrics、pre-loan eligibility/policy simulation |
| 多期宽表 | 显式支持 | 调用者传入有序列组；不从列名猜月份；支持 prepared lifecycle/alert signals |
| entity-event longitudinal | 单表支持 | 显式 entity、observation/event time、cutoff、window、horizon、cohort/state；point-in-time safe |
| 多表关系型数据 | 不直接支持 | 只消费外部预聚合的单表；不 join、不推断 key/cardinality |
| 时间漂移与稳定性数据 | 支持 | reference/current 或显式 time buckets；只做诊断，不解释因果 |

### 2.3 通用能力，不硬编码信用字段

任何 v0.2 public API、private helper、测试主路径、CLI 或报告逻辑都不得硬编码
信用数据字段，包括但不限于 `TARGET`、`customer_ID`、`SK_ID_CURR`、`S_2`、
`WEEK_NUM`、`MONTH`、`DAYS_*`、`loan_status`、`default`、`grade` 和 `interest_rate`。

所有语义只能通过通用参数或结构化规格显式提供，例如：

- target column 与 positive label；
- score column、`ranking_score` 或 `event_probability` kind、score direction、
  positive/event label 与 probability class mapping；
- entity/group column；
- event/time/cutoff column；
- application evaluation time 或 post-loan observation time、available time、event time、
  `outcome_end_time`、`label_available_time`、prediction horizon 与 reporting delay；
- caller-defined rule type、closed conditions、Boolean composition、priority、stop-on-hit、
  effective/expiration date、missing/conflict behavior 和 policy version；
- caller-defined `action_name`、显式 `action_role` mapping、rule priority、band
  boundaries、ties 和 fallback；
- caller-defined alert levels、persistence、cooldown、resolution/reopen 和 event matching；
- caller-supplied costs、utility units、constraints、exposure 和 observed outcomes；
- cohort/origination time、observation time、period unit、state column 与 state order；
- ordered temporal column groups；
- exclude/post-outcome/identifier columns；
- caller-supplied range、allowed-values、special-values 和 availability rules；
- aggregation/window/metric/threshold objective。

文档和 benchmark 可以说明如何把特定数据集映射到这些参数，但映射不得进入库代码。

## 3. 明确 out-of-scope

以下能力不属于 v0.2，任何 Task 15–20 都不得顺手实现或承诺：

- AutoML、自动模型/特征/超参数搜索；
- 深度学习、神经网络、RNN、CNN、Transformer 或 tabular foundation model；
- 大规模 stacking/blending、leaderboard ensemble 或伪标签；
- 身份欺诈研究或检测；
- 设备指纹、IP/VPN/代理检测；
- 申请 velocity fraud；
- 团伙、关系或图网络欺诈；
- 外部黑名单接口或身份一致性核验；
- 实时交易欺诈、欺诈模型或欺诈规则引擎；
- 实时流式拦截；
- 自动规则生成；自动生成、部署或采用业务 cutoff；自动 business-policy
  selection/optimization；
- 任意规则 DSL、表达式执行、动态代码执行或 production rules runtime；
- reject inference、accepted-only bias 修复或响应策略学习；
- 自动贷款审批或任何线上 action execution；
- 动态风险定价、自动额度优化、真实利润优化、催收行动自动优化；
- action-effect/causal response models、uplift modeling、controlled exploration；
- 多期动态规划、reinforcement learning、实时决策引擎或在线 experiment allocation；
- 公平性、因果性、监管合规、模型验证合规或适当性结论；
- server、Web dashboard、交互式 UI、在线监控服务；
- 特定 Kaggle 字段、竞赛 metric 或 submission 文件硬编码；
- 多表关系发现、join planner、key/cardinality engine；
- chunked/distributed I/O、数据库、Polars/Spark/Dask backend；
- feature store、experiment tracker、model registry、MLflow；
- SHAP 依赖或 SHAP-specific public API；
- 新增 tree model families、supervised binning、WOE、target encoding 或通用 learned
  group-aggregate transformer；
- 非目标类别—类别 Cramér's V、混合变量效应量和校正后的多重检验；
- 新的报告主题系统、任意章节配置、多格式配置解析或通用配置框架；Task 20 所需的
  窄范围、版本化、纯数据 JSON policy/warning spec 载体不在此排除项内；
- 模型持久化、生产部署或实际版本发布；
- tag、push、PyPI upload、GitHub release 或 publishing credentials。

其中新增模型族、learned transforms、Cramér's V/效应量/多重检验与报告配置是对
现有 `SPEC.md` 初步 v0.2 路线的**延期提案**，roadmap review 后必须同步修订
`SPEC.md`；其余多表关系型能力、
chunked CSV/performance engine、持久化/model card/manifest 等仍按现有 `SPEC.md`
保持在 v0.3 或更晚，除非先单独修订并 review roadmap。

反欺诈不是本 roadmap 的延期项：v0.2 不为其收集案例、设计 API、设置 benchmark、
分配 Task 或安排长期版本。若未来用户另行提出，必须从独立 research/roadmap review
重新开始，不能复用本合同作为授权。

### 3.1 长期 roadmap（非 v0.2 承诺）

以下主题保留为后续独立研究方向，不表示已批准实现、版本或时间表：

- 动态风险定价与自动额度优化；
- action-effect/因果响应模型、uplift modeling 与 controlled exploration；
- 催收 treatment/channel 自动优化；
- reject inference 算法及其选择偏差评估；
- 多期动态规划、reinforcement learning 与 sequential policy；
- 实时规则/决策引擎、在线 experiment allocation 与自动贷款审批；
- 监管合规认证、法律通知生成和自动审批授权；
- SHAP-specific public API 与相应 optional dependency；
- 新增 tree model families、supervised binning、WOE、target encoding 与 learned
  group aggregate。

任何后续立项都必须重新证明数据识别条件、因果/选择偏差边界、安全性、依赖价值、
隐私与监管适用性；v0.2 的离线分析结果不能作为上述能力的实现授权。

## 4. 向后兼容原则

### 4.1 v0.1 public surface

v0.1 已冻结的 public functions、signatures、keyword defaults、dataclass 字段顺序、
errors、warnings、limitations、排序、报告章节、CLI 参数/exit code 和 Figure ownership
必须保持兼容。

特别是：

- `train_classifier` 的 v0.1 random holdout 行为不得静默变为 CV/group/time split；
- `TrainingResult` 和 `ClassificationEvaluation` 不得直接增加、删除或重排字段；
- `check_data_quality` 不得因 v0.2 新规则改变已有 issue 集合、阈值或排序；
- `derive_features` 不得开始在全量 DataFrame 物化 requires-fit suggestions；
- `run_analysis`、`AnalysisRun`、`generate_analysis_report` 与 `sharper analyze` 的
  既有默认调用必须产生与 v0.1 一致的行为和章节。

### 4.2 新行为的承载方式

需要新元数据或新不变量的能力必须使用新的具名 frozen result dataclass 和新的
opt-in public path，或仅在不改变旧调用语义的新增 keyword-only 参数下扩展。不得：

- 根据输入内容让同一旧签名返回不同结果类型；
- 在旧 dataclass 上追加“可选但语义关键”的字段；
- 用不稳定 nested dict 代替具名结果；
- 建立 manager、registry、plugin framework 或深 class hierarchy；
- 通过 `**kwargs` 把未冻结的 sklearn/第三方参数变成 Sharper public contract。

Task 20 优先采用独立的 v0.2 binary-risk workflow/result/report path，再决定是新增
CLI subcommand 还是在 `analyze` 上增加完全 opt-in 的参数；精确选择由 Task 20
合同冻结。无论选择哪种，旧 CLI invocation、stdout/stderr、exit code 和输出文件
行为都必须由回归测试证明不变。

## 5. Public API 演进原则

1. 每个新增 public function/class/dataclass 在实现前必须有独立 Task 决策记录，
   并同步 `SPEC.md`、`IMPLEMENTATION_PLAN.md`、API 文档和 public-contract tests。
2. 风险型二分类 API 必须显式区分 `ranking_score` 与 `event_probability`，记录
   `positive_label`、score kind 和“数值越大是否表示正类事件风险”；不得依赖 label
   排序、first appearance 或 estimator 内部列顺序推断业务正类。`ranking_score`
   可为任意有限实数；`event_probability` 必须位于 `[0, 1]` 且有明确的正类映射。
3. 结果必须记录实际有效样本量、缺失/非有限处理、ties、class counts、事件率、
   fold/split、time/group boundary、requested/actual budget、warnings 和 limitations。
4. validation、metric、calibration、threshold、drift 和 explanation 结果必须是不同
   具名对象或清晰组合，不返回字段随参数变化的自由 dict。
5. 预测模型、score evaluation、pre-loan eligibility/decision policy、post-loan
   early-warning policy 与 strategy simulation 必须可独立版本化和审计；策略/预警
   API 不 fit estimator，建模 API 不返回业务动作或 alert。
6. score kind/direction、positive/event label、probability class mapping、动作名称与
   action-role mapping、规则优先级、band 边界、成本单位、约束、evaluation/
   observation time、有效期、missing/conflict behavior、alert horizon/lifecycle 必须
   显式；不得根据列名、标签值、动作文字或分数分布猜测。
7. observed replay、model-based expectation 和外部 randomized experiment log 必须
   使用不同结果语义；离线模拟不得命名为线上实验或因果收益。
8. `DecisionPolicySpec`、`PolicyRuleSpec`、`EligibilityRuleSpec`、
   `EarlyWarningRuleSpec`、`RuleCondition`、`RuleAction`、`RuleHit`、
   `EligibilityPolicy`、`EarlyWarningPolicy`、`AlertEvent`、`AlertHistory`、
   `PolicySimulationResult`、`PolicyComparisonResult`、`VintageAnalysis`、
   `RollRateAnalysis`、`ReasonCode` 和 `PolicyAudit` 等都只是研究候选，本文不冻结
   public symbol；Tasks 16–19 必须与既有候选去重并收敛最小 public surface，其中
   private condition kernel 不要求任何顶层 public symbol。
9. API 优先是纯函数和小型、封闭的具名规格；不得接受任意规则代码、表达式执行、
   通用 DSL、optimizer callback 或不稳定 nested dict。Task 16 拥有的 condition kernel
   是 private shared foundation，不是 public rules API，不能被导出为通用 DSL。
10. eligibility result 与 alert/history result 必须分离：前者是一行候选记录的离线
    action/path，后者是 entity × observation time 的 alert state/episode；共享 condition
    vocabulary 不允许产生字段随 policy 类型变化的通用结果容器。
11. 输入 DataFrame 默认不修改；copy/in-place 行为必须在签名、docstring 和原子失败
   测试中冻结。
12. 错误继续使用 `ValueError` 表示无效输入、缺列或非法类型，`OSError` 表示 I/O
   失败；没有单独 SPEC 修订不得新增公共异常体系。
13. 新 symbol 只在实现、文档和 contract tests 同时稳定后加入 `sharper.__all__`。
14. public API 不暴露特定 Kaggle metric 名。若实现 top-percent capture 或 stability
   trend，使用通用参数和通用结果名。
15. 同一计算只能有一个 ownership layer；plot/report 只消费已有结果，不隐藏
    predict、fit、metric、drift 或 explanation 重算。
16. 任何 expected loss、expected revenue、expected payoff/profit 或概率校准计算只可
    消费合法 `event_probability`；`decision_function`、未经校准 margin 或一般
    `ranking_score` 不得被自动解释为概率。只有 ranking score 时，相关概率型结果必须
    是 unavailable/undefined 并记录原因，而不是返回伪数值。

## 6. Optional dependency 原则

### 6.1 core-first

v0.2 的完成不得依赖新增第三方 runtime dependency。优先使用现有 pandas、NumPy、
SciPy、scikit-learn、matplotlib 和 seaborn：

- binary metrics、calibration、policy simulation、lifecycle tables、permutation
  importance 和静态 plots 使用现有依赖实现；
- Task 20 的 closed JSON spec carrier 只使用标准库解析与 Tasks 16–18 冻结的显式 schema
  validation；不为此新增 `jsonschema`、YAML/TOML parser 或模板依赖；
- LightGBM、XGBoost、CatBoost、imbalanced-learn、SHAP、Polars 不进入 v0.2 core；
- caller 仍可通过既有/新合同允许的 sklearn-compatible estimator 接口自行传入
  第三方 estimator，但 Sharper 不承诺其安装、参数空间或确定性。

### 6.2 未来 extra 的门槛

任何未来 optional estimator、optimizer、规则引擎或 lifecycle backend 都不属于
v0.2 Done。若长期 roadmap 证明某个 extra 具有 core 无法提供的明确用户价值，必须
先单独修订并 review roadmap/Task 合同；最多引入按能力命名的 optional extra，且
必须同时满足：

- core import 和所有非该能力路径不 import optional package；
- 缺包错误稳定、可操作，并包含安装 extra 的提示；
- wheel/sdist metadata、license、Python/platform support 和 CI matrix 已验证；
- core clean-install 和无 extra 测试继续通过；
- 不因“教程常用”或“提高 leaderboard”而引入；
- v0.2 Done 不以该 extra 存在为条件。

本 roadmap 不批准任何新增依赖或 `pyproject.toml` 修改。

## 7. Leakage 与 validation 不变量

v0.1 的全部 Leakage 不变量继续有效，并增加以下 v0.2 冻结要求：

每条进入监督验证的记录必须区分四个时间角色：

```text
observation_time
event_time
outcome_end_time
label_available_time
```

- `observation_time` 是生成特征和预测的 as-of 时点；
- `event_time` 是目标事件实际发生时点，可为空；它本身不能证明标签已可用，仍须由
  `label_available_time` 与目标定义决定 maturity；
- `outcome_end_time` 是判定该行标签所需观察窗口的结束时点；
- `label_available_time` 是训练时最早可合法使用该标签的时点，须包含适用的 reporting
  delay。

调用者必须通过以下一种方式提供 label maturity：显式 `label_available_time`；
`observation_time + prediction_horizon + reporting_delay`；或从明确的
`outcome_end_time` 推导的规则。不得默认历史标签在 `observation_time` 当日已经可用。
Task 15 合同必须冻结时区、缺失值、逐行 horizon、推导规则以及端点的 inclusive/
exclusive 语义，并验证 `outcome_end_time`、`event_time` 与 `label_available_time` 的
一致性。

1. 先确定 outer split/folds，再在每个训练 fold 内 fit schema-dependent selection、
   imputer、encoder、scaler、calibrator 和 estimator；外部提供的 score 也必须携带
   OOF/validation/holdout 与训练标签在各 fit cutoff 前成熟的 provenance。
2. 任何 target-aware state 的训练行输出必须使用 OOF/cross-fit；训练行不得看到自己
   的 target，也不得看到 validation/test target。
3. group validation 中同一 entity/group 不得跨 train/validation。time fold cutoff 为
   `C` 时，每条训练记录必须同时满足 `observation_time < C` 和
   `label_available_time <= C`；验证记录必须落在合同冻结的 observation window，且其
   outcome horizon/maturity 必须独立记录。任何训练 outcome window 或 reporting delay
   穿过 `C` 的记录均不得用于该 fold；purged/immature n 和由此形成的 gap 必须记录。
   训练标签不得与 validation observation window 重叠而把未来结果带入 fit。
4. entity-event 或 lifecycle analysis 只能消费调用者允许的 point-in-time records；
   `event_time` 与 cutoff/observation time 的 inclusive/exclusive 规则必须由 Task 18
   合同冻结并测试。
5. reference/current drift 的 reference bins/category support 只从 reference 拟合；
   current 不得反向改变 bins、top categories 或 smoothing state。
6. calibrator fit 只能发生在训练内层，calibration 诊断和预声明 threshold 候选的
   operating-point 比较只能使用训练或 validation/OOF。最终 holdout/test 不用于选择
   阈值、模型、特征或失败后的重试；分析性候选不得被自动写入 Task 17 policy。
7. train/test 不得拼接以学习 category vocabulary、rank、imputation、aggregation、
   frequency、dtype-dependent eligibility、cutoff、policy、cost 或 constraint。
8. test-only category、极值、group、entity、时间段和 drift category 必须有专项
   leakage 回归测试。
9. 若调用者声明 entity/time 风险但所选策略不能保证隔离，必须拒绝，不得只 warning
   后继续建模。
10. entity/group/time 风险、cutoff 和数据可用性由调用者负责声明；Sharper 的 audit
    可以发现结构迹象，但不得声称推断出业务上的 future/post-outcome 字段。
11. 调用者预声明的 bounded cutoff/band 候选只能在训练或独立 validation/OOF 上进行
    分析比较；final holdout 只做一次调用者已冻结策略的评估，不得用于调整 bands、
    规则、成本、约束或 tie-break。Sharper 不自动生成、部署或采用业务 cutoff。
12. 历史 rejected/unselected rows 没有 observed outcome 时必须记录 missing support；
    不得用 model probability 冒充已观察标签，不得在 v0.2 内做 reject inference。
13. historical action/policy 与 outcome 有选择关系时，离线结果只描述 observed replay
    或 assumption-based expectation；不得从非随机日志声称 action effect。
14. 贷前 rule evaluation 只能读取 `available_time <= evaluation_time` 的字段；policy
    version 由 evaluation time 与显式 effective/expiration boundaries 选择。
15. 贷后 signal/alert 在 observation time 只能读取当时及之前可用的数据；未来 event
    只在 backtest 阶段按冻结 horizon 匹配，不能进入 signal、rule 或 alert state。
16. peer-deviation baseline 只从训练/reference period 拟合；customer-history baseline
    只使用本 entity 的 prior observations；current/test 不改变 threshold/support。
17. Task 15 中未成熟标签不得进入 fit、fold metric、calibration 或 loss denominator，
    必须标记 purged/immature/unevaluable；Task 18 中未成熟 prediction horizon 必须标记
    censored/unevaluable，不得计为无事件、false alert、resolved alert 或 constraint
    passed。二者是不同结果语义，不能互相替代。
18. `ranking_score` 只要求有限值与显式风险方向，可用于排序、ROC-AUC、Gini、KS、
    gains、lift、top-k/top-fraction capture、bands 和 cutoff replay；它不得进入概率校准、
    expected loss、expected revenue 或 expected payoff/profit。
19. `event_probability` 必须是 `P(y == positive_label)`，值域为 `[0, 1]`，并携带
    positive-label mapping 与来源/校准 provenance。`predict_proba` 的列、调用者声明的
    概率或合法 calibrator 输出可以提供该角色；`decision_function`、raw margin 和一般
    score 不能被自动提升为概率。值域合法只说明可解释为概率，不证明已经校准。

## 8. Tasks 15–20

### Task 15 — Binary Risk Validation and Business Metrics

**目标**

建立显式正类/事件、可审查 folds/OOF、风险型二分类评估和模型无关的基础业务指标，
为后续策略模拟提供唯一 ranking-score、event-probability、label-maturity 与 outcome
数学语义。

**依赖**

Tasks 03、09、11、13、14。Task 15 与 Task 16 是两项基础能力；Task 17 使用分数或业务
指标时消费 Task 15，Task 18 使用模型分数时可消费 Task 15，但两者均不得反向依赖或
改变 Task 15 结果。完整 Task 19/20 集成等待本 Task 完成。

**范围**

- 显式 `positive_label`、score semantic kind 与风险方向；来源路径与语义角色正交：
  caller-supplied frozen OOF/validation/holdout 值或由 `modeling` 在 fold 内产生的值，
  均须声明为 `ranking_score` 或 `event_probability` 并携带 provenance；
- `ranking_score` 可为任意有限实数，只用于 discrimination/ranking、bands 与 cutoff
  replay；`event_probability` 必须位于 `[0, 1]`、对应显式 `positive_label`，可来自
  `predict_proba`、训练折内合法 calibration 或调用者明确声明的概率；
- stratified holdout/CV、group holdout/CV、time-ordered holdout/forward validation；
  time 路径必须消费 `observation_time` 与合法 label-maturity metadata，并按 fold cutoff
  执行 `observation_time < C`、`label_available_time <= C`；
- frozen fold membership、OOF predictions 和 fold-level metadata；
- ROC-AUC、normalized Gini、KS、average precision、明确命名的 PR curve/area、
  gains、lift、top-k/top-fraction capture；
- 只对 `event_probability` 计算 log loss、Brier score、calibration table/curve；
- 对调用者预声明的有界 threshold/band 候选，在训练或 validation/OOF 上输出 curve、
  按显式指标或仅引用已计算 threshold-curve 列的 metric-only diagnostic guardrail
  比较并报告 deterministic analytical operating point 及依据；不消费 action/cost/
  capacity/exposure/budget/rate 等业务约束，不发现搜索空间，不写入策略，也不声称
  真实业务最优；
- exposure 可按任一 ranking/probability band 或候选 threshold-selected population
  汇总；observed loss 只使用成熟、已观察 outcome；expected loss 只使用合法
  `event_probability` 与调用者提供的 exposure/loss fraction。不得在本 Task 分配业务
  动作、使用 action cost 或计算 revenue/profit/payoff；
- 只消费结果的 risk evaluation plots。

**指标语义要求**

- Gini 与 ROC-AUC 的关系、score direction、ties 和 degenerate class 行为必须冻结；
- `average_precision` 与 trapezoidal PR area 必须分别命名，不得都叫模糊的
  `pr_auc`；Task 15 可以选择只公开其中一个，但必须准确命名；
- gains/lift/capture 必须记录基准事件率、population fraction、tie boundary 和
  实际选中行数；
- KS 必须记录正负类有效样本量和最大差异对应阈值；
- `ranking_score` 的方向与有限值、`event_probability` 的范围、positive-label mapping
  和 provenance 必须分别验证；不得把 `decision_function`、未经校准 margin 或一般
  score 自动解释为概率；
- calibration/analytical operating-point 结果不得从最终 test 拟合或选择；报告候选点
  不等于采用业务 cutoff，Task 17 只能接收调用者另行明确传入或已冻结的 cutoff/bands；
- expected 与 observed loss 必须分栏，记录 outcome support、单位、时点、损失假设、
  有效样本和 unevaluable reason；`PD × exposure × loss_fraction` 只是信用示例，
  public contract 必须使用通用 event-probability/exposure/loss 语义。只有 ranking score
  时 probability calibration 与 expected loss/revenue/payoff 均为 unavailable/undefined。
- time validation 必须记录每 fold cutoff、validation observation window、outcome
  horizon、label-availability 推导方式、mature/immature/purged n 和端点语义；正事件已
  发生但 reporting delay 未结束仍不得提前使用标签。

**不属于 Task 15**

数据质量新规则、动作/规则/cost/constraint/profit/payoff simulation、feature
materialization、新 estimator、drift/explainability；`evaluation` 不 fit/clone
estimator 或建立 preprocessing state；
准入规则执行、预警规则、通用规则 DSL、完整策略 optimizer、自动生成/部署/采用业务
cutoff；workflow/reporting/CLI、AutoML、重采样算法目录、新 tree model family、WOE、
target encoding、新依赖。

**验收条件**

- 手算小样本与 sklearn/SciPy 基准在明确容差内一致；
- label 为 `0/1`、`1/0` first appearance、bool、string 时显式正类均正确；
- ties、常量分数、单类 fold、缺失/非有限 score、小样本和非法 split 有稳定行为；
- 合法 probability、`0/1` 边界、超出 `[0,1]`、非有限值、错误 positive-label mapping、
  `decision_function` margin、风险方向反转均有稳定结果；一般 ranking score 可包含
  负数或大于 1 的有限值，但不生成 calibration 或 expected loss；
- score kind/direction、positive/event label、loss direction、单位和 expected/observed
  semantics 不得推断；event-probability expected loss 与 exposure/observed loss 的手算
  fixture 一致，ranking-only input 不生成概率型损失；
- 调用者预声明 thresholds 的 ties、重复 objective 值和 deterministic evidence 正确；
  metric-only guardrail 只引用 frozen threshold-curve columns；action/cost/capacity/
  exposure/budget/rate constraint 明确拒绝并归 Task 17。final test 不参与候选比较，
  分析点不会自动传入 Task 17；
- group 不交叉、OOF 每行恰好一次且 fold provenance 完整；time fixtures 覆盖
  observation time 合法但 label 未成熟、event 已发生但 reporting delay 未结束、
  mature/immature 混合、`label_available_time == C`、outcome horizon 与 validation
  window 重叠，以及 observation time 单调但仍有 label leakage；
- time fold purge 后的空/单类训练集、逐行 horizon、缺失或不一致 maturity metadata、
  group + time 联合隔离和 final-holdout immature rows 均有稳定失败或 unevaluable 结果；
- spy/leakage tests 证明 estimator-driven 路径只有 `modeling` 在当前训练 fold fit，
  calibrator 也只在当前训练 fold fit，`evaluation` 只消费 frozen predictions；external-
  score 路径拒绝缺失、矛盾或 label-maturity 不安全的 provenance；
- v0.1 classification API、result、plot 和 holdout metrics 回归测试完全不变；
- 独立 Task 15 决策记录、SPEC/plan/API 文档与 tests 同步通过 review。

### Task 16 — Data Quality and Leakage Audit

**目标**

在不改变 v0.1 `check_data_quality` 的前提下，提供 opt-in、声明式的数据质量和
leakage audit，覆盖 binary target、entity/group 和 time 数据结构；同时提供 Tasks
17/18 唯一共享的 private closed condition-evaluation kernel。

**依赖**

Tasks 03、04、14；Task 16 与 Task 15 可独立实施。当调用者提供 target/score/folds 时
消费 Task 15 已冻结的正类、score-kind、label-maturity provenance 和有效样本语义，
不得重新实现 Task 15 metrics 或 time-fold construction。纯 rule-input/time audit 不
要求 target 或 score；Task 16 是 Tasks 17、18 的共同前置。

**范围**

- target class counts/event rate、稀有类和按 slice 的可用性；
- duplicate row/index、entity/group overlap、time ordering、cutoff violations；
- caller-supplied ID、post-outcome、future、exclude 和 availability declarations；
- score、outcome、historical action/policy、cost/exposure/constraint fields 的
  availability、missing support、selection dependence 和时间可用性；
- eligibility/early-warning rule input availability、evaluation/observation/event time、
  duplicate entity-time keys、window/horizon maturity 和 post-event fields；
- 通用 range、allowed-values、special-values、cross-column 和 monotonic-time rules；
- missingness、unknown category、constant/near-constant、high-cardinality 和
  train/reference-vs-validation/current 的结构差异；
- missingness profiling 与 missingness drift 的唯一计算：reference/current missing
  rates、absolute/relative change、new all-missing columns、recovered columns、schema/
  missing-pattern differences，并记录各侧样本量与预算警告；
- 为 Task 19 治理汇总冻结有界的 reference/current numeric/categorical input-profile
  evidence；Task 19 不重新扫描 raw inputs 生成第二套 feature-distribution 结果；
- 有预算的 missingness co-occurrence/pattern evidence 与 numeric collinearity
  evidence；不得建立第二套与 Task 07 冲突的相关性语义；
- advisory 的 ID-like column/near-copy/target-proxy evidence，必须记录
  方法、阈值和 false-positive limitation。
- private、dependency-light 的 closed condition-evaluation kernel：三值
  `true/false/unknown`，AND/OR/NOT，数值与日期比较，集合成员判断、缺失判断，封闭
  operator inventory、missing propagation、effective/expiration time、deterministic
  ordering、输入不可变性及 conditions/depth/detail 计算预算。

**边界**

- audit 只报告，不自动删列、填值、截尾、重采样、改 target 或选择特征；
- 不通过字段名字判定 future/post-outcome；
- public audit 只审计 caller 声明的 rule inputs，不执行 policy 或生成 action/alert；
  private kernel 只验证 condition schema 与计算三值 truth result，不理解 rule priority、
  policy conflict、action、alert、cost 或 constraint；
- private kernel 不是 public rules API 或 DSL，不允许任意 Python 表达式、callable、
  `eval`、函数、插件代码或动态 operator；Task 17/18 只能消费它，不能各自实现副本；
- 不声称相关或高预测性等于 leakage；
- 不把 rejected/unselected outcome 缺失自动填成好/坏样本，不执行 reject inference；
- 不修改 Task 04 frozen codes、severity、rules、thresholds 或排序。

**验收条件**

- 新 audit 使用新的结果类型/入口；v0.1 quality output bit-for-bit/结构等价不变；
- 合成数据覆盖重复实体跨 fold、未来记录、时间逆序、test-only category、特殊值、
  合法高相关、缺失模式预算截断、伪 proxy、historical action-assignment dependence 和
  outcome support 缺口；
- 每个 issue 有 row/column/slice 范围、有效样本量、证据、severity、limitation 和
  deterministic ordering；
- missingness drift 的 reference/current rates、absolute/relative change、new all-
  missing/recovered columns 与 schema/pattern differences 通过手算，frozen result 自身
  记录两侧 n、budgets 和 warnings；
- 缺失模式和共线性结果有明确算法、有效样本、阈值、requested/actual budget，且
  与 Task 07 既有 correlation contract 的 ownership 在 Task 16 决策记录中唯一化；
- private kernel 的 atomic operators、完整三值 truth tables、unknown propagation、
  AND/OR/NOT、日期/数值/集合/缺失、effective/expiration exact boundaries、unknown
  operator、nesting/depth/budget 与 stable errors 通过手算；
- input DataFrame 不变，条件规格不变，预算和截断进入结果；相同输入/spec 的 ordering
  与结果确定；
- 独立 Task 16 决策记录与文档通过 review。

### Task 17 — Pre-loan Eligibility Rules and Decision Strategy Simulation

**目标**

在 candidate/application rows 上执行 caller-defined 贷前准入规则，并可选结合冻结
scores/bands 形成 deterministic simulated actions；再用可选 costs、outcomes 和
constraints 对已形成的 action replay 做 assumption-aware 评估、规则回测和策略比较。
constraints 不生成或改写 action；本 Task 不训练模型或执行真实审批。

**依赖**

Task 16；使用 score/target/outcome/business metrics 时依赖 Task 15 的唯一数学语义。
纯规则、无 score 或无 outcome 的 replay 必须仍可运行并相应限制可报告指标。

**范围**

- 数据完整性、基础资格、产品适用性、信用政策、现有敞口、收入/负债能力、历史
  状态、产品互斥、必须补件和人工审核等 caller-defined rule types；
- hard/soft/refer rules；只消费 Task 16 private kernel 的 AND/OR/NOT、数值/日期比较、
  集合匹配、缺失判断与三值 condition result，不实现第二套 operator/truth semantics；
- priority、stop-on-hit、multi-hit accumulation、effective/expiration date、policy
  version、missing behavior、conflict resolution 和 deterministic fallback；
- caller-defined `action_name` 与显式 `action_role` mapping；通用语义角色至少覆盖
  `selected/rejected/review/request_information/limited/other`，但本文不冻结 public
  enum 或类型名。`approve/accept/book/decline/reject/manual_review/refer` 等只作为
  名称示例，不具有可推断语义；Task 17 合同必须冻结包含上述角色的 closed inventory，
  一个名称只能映射一个 role，多个名称可映射同一 role；
- 先规则后模型、先模型后规则、hard-rule override、review score band、score +
  eligibility、cutoff + constraints 和 bounded multi-action matrix；
- caller 明确传入或已冻结的 ordered decision bands/cutoffs、score kind/direction、bound
  closure/ties、generic utility/cost、manual capacity、exposure/budget/rate constraints
  和三种 replay semantics；Task 15 analytical candidate 不会自动成为本 Task policy；
- simulated action 只由 caller rules、bands、precedence、override 和 fallback 形成；
  business constraints 只对 frozen policy/scenario 返回 feasible/violated/unevaluable、
  demand/gap 与 violation magnitude，不改变 action 或自动选择替代 policy；
- ranking score 或 event probability 均可驱动 bands/cutoffs 和 action-distribution
  replay；只有 event probability 可驱动 expected loss/revenue/payoff，只有成熟 observed
  outcome 可驱动 observed loss/payoff；
- rule hit/sole-or-first unique hit、overlap、conflict、通用 action distribution、target
  capture、fixed-order/leave-one-out marginal contribution、incremental action count、
  capacity demand/gap、unknown/missing volume 和 segment/time stability；
- 仅在 action-role mapping 存在时计算相应 selection/approval、rejection、review、
  request-information rates、selected-population event/bad rate、selected exposure 与
  review-capacity usage；结果记录实际 action set、role set、denominator 和 mapping；
- 同一 frozen rows/support 上的 reference/challenger pre-loan policy comparison；
- bounded evaluated/hit/not-evaluated rule path、base/final simulated action、override、
  reason provenance、assumptions、warnings 和 limitations；
- result-only rule-impact、action-mix、cutoff、constraint 和 payoff plots。

**边界**

- 只执行/枚举 caller 给定并冻结的 rules/policies/cutoffs/bands；不自动生成或采用
  规则/业务阈值，不自动选择 winner，不建立通用 DSL、solver、optimizer 或 production
  rules engine；
- 不重新实现 Task 16 condition kernel；rule priority/action mapping 属于 Task 17，
  condition truth semantics 属于 Task 16；
- 使用 score/bands 时复用 Task 15 frozen score direction、bounds/closure、ties 和有效样本
  语义；Task 17 只增加 action mapping，不重算另一套 score/threshold 数学；
- `exclude` 不删除输入，`limit` 不计算额度，`approve/decline` 不执行真实审批；
- historical accepts 或 observed action 子集不得外推到无 outcome/common-support rows；
  ranking score 不冒充 event probability，二者都不冒充 observed outcome；
- 缺少 action-role mapping 时只输出通用 action distribution，不根据 action 名称包含
  `approve`、`decline`、`refer` 等文本猜测角色，也不计算固定审批/拒绝/审核指标；
- mapping 缺少某个业务指标所需 role 时该指标 unavailable；不在 Task 17 closed
  inventory 的 role、同一 action name 的重复冲突 mapping 必须明确失败；
- final holdout 不用于修改 rules、order、policy、cutoff、cost、constraint 或 tie-break；
- dead/over-broad/duplicate/conflicting/order-sensitive rule suggestion 只是人工复核证据；
- 不生成法律通知、合规结论、真实利润结论或任何反欺诈能力；
- 不修改 Task 15 metrics、Task 16 audit 或 v0.1 workflow/reporting/CLI。

**验收条件**

- hand-worked fixtures 覆盖 hard decline、soft/refer/request-information、missing、
  overlap/conflict、Task 16 condition parity、priority、stop-on-hit 和 not-evaluated
  distinction；
- action 示例、两/三/多段 bands、升/降分增险、open/closed bounds、ties、fallback、
  effective/expiration boundary 和 version selection 均有稳定结果；
- rule hit/unique/overlap/conflict、generic action distribution、target capture、marginal
  contribution、incremental action count、capacity/missing volume 与手算一致；
- arbitrary action names、两个名称映射同一 role、缺少必要 role、未知 role、重复/冲突
  mapping 和无 mapping 均有稳定结果；映射后的 selection/rejection/review/request-
  information rates、selected-population event rate/exposure 与 review capacity 手算一致；
- 缺少 event probability 时不输出 probability-based expected loss/revenue/payoff；缺少
  成熟 observed outcome 时不输出 observed bad/loss/profit。observed replay 披露
  support，event-probability model-based expectation 明示假设；unevaluable 不写成
  passed/zero；
- ranking score 与 outcome availability 正交：ranking score + mature observed outcome 可
  产生 observed replay，但仍不产生 probability-based expectation；
- 相同 rules/bands/actions 下改变 constraint 只改变 feasibility/violation evidence，
  不改变 simulated action；infeasible scenario 不被自动替换或命名为 rejected policy；
- reference/challenger 使用相同 rows/denominators，paired action transitions 可审查；
- validation/holdout spy 证明选择隔离，test-only outcome/cost/extreme 不改变 policy；
  input DataFrame 不变，rules/conditions/pairs/details/grid 有预算；
- 独立 Task 17 决策记录冻结最小 API、Task 16 condition-kernel compatibility、action/
  role vocabulary、result schemas、comparison ownership、排序和 errors 后才可实现。

### Task 18 — Post-loan Early Warning and Lifecycle Monitoring

**目标**

在 `customer/account × observation date` 上用当时及之前可用的数据执行贷后预警规则，
形成 alert episodes，回测未来冻结 horizon 内的 events，并提供 vintage/MOB/roll-rate/
cure 等生命周期监测；不执行催收或账户动作。

**依赖**

Task 16；使用 model scores/outcomes 时依赖 Task 15。Task 18 不依赖 Task 17 的
eligibility/action result；它只消费 Task 16 private condition kernel 的三值结果并返回
独立 alert/history result，不能导入贷前动作语义。无模型分数的纯规则路径不硬依赖
Task 15。

**范围**

- explicit entity、observation/available/event time、lookback/recent/historical windows、
  prediction horizon、left/right closure、period unit、timezone 和 maturity/censoring；
- 逾期状态/连续性/恶化、还款缺失或下降、余额/使用率增长、超限、caller-defined
  行为比率、现金流下降、多账户恶化、查询/新增负债、无正常还款、波动和个人历史
  偏离等通用 signal categories；不硬编码字段、阈值或方向；
- level/change/trend/persistence/combination/state-transition/peer-deviation/
  customer-history rules；peer baseline 只从 train/reference 拟合，个人基线 prior-only；
- rule 的 atomic/Boolean/missing/date condition truth 只由 Task 16 private kernel 计算；
  Task 18 拥有 signal alignment、persistence、state transition、alert level 与 episode
  语义，不实现另一套 condition evaluator；
- caller-defined alert levels；`none/watch/warning/high/critical` 只作示例，不冻结 enum；
- first/repeated alert、persistence、episode、cooldown、resolution 和 reopen semantics；
  raw rule hits、notifications 与 episodes 分别计数，cooldown 只抑制 notification，
  missing observation 不得自动关闭 episode；
- rule hits、alert level、first time、duration、recent change、supporting indicators、
  reason、resolved/repeated/reopen flags 和 policy/rule version；
- vintage/cohort age/MOB、state migration、roll-rate count/weight、roll-forward/back、
  cure、early delinquency、cohort comparison、status transitions 和 period risk trend；
- alert rate、raw rule-hit rate、entity coverage、event capture/recall within horizon、
  precision、false-alert share、false-positive rate、mean/median lead time、warning burden、
  alerts per case、duplicate/unresolved rate、
  severity、segment/time/vintage performance 和 descriptive roll-rate difference；
- no-alert、single threshold、current rules、challenger rules、model score 和
  model+rules 在相同 mature observations/horizon 上的 backtest comparison；
- result-only alert timeline、capture/lead-time、vintage、roll-rate 和 lifecycle plots。

**边界**

- 只接受一个 raw/prepared DataFrame；不读文件、不 join 多表、不推断关系；
- observation `t` 的 signal/rule/alert 不能读取未来 event 或 feature；event 仅用于
  backtest matching；未成熟 horizon 不能当无事件/false alert；
- 不硬编码 MOB、DPD、状态、cure、alert level 或贷款日期；
- 不实现 survival/causal/action-effect model，不优化催收 action/channel，不联系客户、
  不修改账户、不建立 real-time streaming/online warning engine；
- 不实现反欺诈、设备/IP/交易欺诈或图网络信号；
- 不修改 Task 09 features，不实现 learned transforms 或新 estimator；
- 不导入 Task 17 policy/action result，不复刻 Task 16 condition kernel；
- alert 与后续 migration/loss 的差异只描述 association，不声称 alert 产生效果。

**验收条件**

- fixtures 覆盖单次/连续恶化、recovery/cure、first/repeated alert、persistence、
  cooldown、resolution/reopen、多账户 prepared observations 和 duplicate entity-time；
- 与 Task 17 共用的 atomic/Boolean/missing/date condition fixtures 产生完全一致的 Task 16
  三值结果，但 alert state/episode 与 eligibility action 结果 schema 保持分离；
- observation/window/horizon exact boundaries、timezone、missing periods/time/state、
  non-consecutive periods、exit/re-entry、multiple events 和 zero denominator 稳定；
- future-only row/value 不影响 current signal、peer/history baseline 或 alert；
- mature/immature vintage、right-censored alerts、同 MOB 对齐、roll-forward/back/cure
  transition 与手算/pandas 基准一致；
- alert/raw-hit/capture/precision/recall/false-alert-share/false-positive/lead-time/burden/
  duplicate/unresolved metrics 记录 observation/entity/event denominator 和 matured/
  censored n 并与手算一致；false-alert share 与 false-positive rate 不得混名，no-alert
  baseline 的 precision、false-alert share 和 lead time 必须为 undefined 而不是 0；
- 所有 baselines 使用相同 observations、entity population、horizon/maturity policy；
  historical non-random comparison 不称为 A/B test；
- input 不变，columns/rules/windows/entities/episodes/details 有预算和 provenance；
- 独立 Task 18 决策记录冻结 exact APIs、time/event matching、alert episode/state、
  lifecycle schemas、comparison ownership、排序和 errors。

### Task 19 — Explainability, Champion/Challenger and Governance

**目标**

把模型解释、模型层 champion/challenger，以及 Tasks 17/18 已计算的规则/策略比较、
reason、override、stability evidence 汇总为治理诊断；不得重复执行规则或回测。

**依赖**

Tasks 15、16、17、18。不得在本 Task 重算 scores、input/missingness profiles、rules、
actions、alerts、policy comparisons、lifecycle tables 或基础/backtest metrics。

**范围**

- linear coefficients、支持时的 estimator native importance、holdout/OOF
  permutation importance 与 source-feature provenance；
- 同一 frozen fold/row set 上的 model-score champion/challenger metric comparison；
- 消费 Task 17 frozen pre-loan policy comparison 和 Task 18 frozen warning-policy
  comparison，形成统一 comparison inventory，不重新计算 paired deltas；
- model-based、policy-based、alert-based、override-based reason provenance、mapping
  coverage、unmapped/fallback 和 bounded audit summary；
- 消费 evaluated/hit/order/not-evaluated、base/final action、alert episode、override
  flag/type 等 frozen facts；不重新执行 condition 或改变结果；
- 外部实验/历史日志的 assignment mechanism、time、segment、common support 和
  randomized/non-randomized provenance；只描述日志，不分流实验；
- prediction drift 与 performance-by-time/group；消费 Task 16 frozen missingness drift 和
  feature-distribution evidence 做治理性汇总，不重新计算 reference/current missing
  rates、schema/pattern differences 或 raw feature profiles；
- model/rule/policy stability comparison，以及 Tasks 17/18 已有 action/alert/override/
  reason stability 的治理汇总；
- purpose、owner、version、materiality、assumptions、limitations、monitoring thresholds、
  issue/remediation status 等通用 governance metadata；
- result-only explanation/comparison/drift/audit plots。

**边界**

- 不引入 SHAP，不把 reason table 作为 adverse-action notice generator；
- 不把 offline comparison 命名为 A/B test，不从非随机 assignment 声称 causal lift；
- 贷前 rule/policy backtest 与 time/segment stability 只属于 Task 17，贷后 alert/lifecycle
  backtest 与 time/vintage stability 只属于 Task 18；Task 19 只消费其 frozen results；
- missingness profile/drift 与 input feature-profile evidence 只属于 Task 16；Task 19 只
  解释、排序和展示。Task 19 自己计算的 drift 仅限 prediction drift 与 model
  performance-by-time/group，不反向修改上游 reference state；
- 不做公平性认证、监管/model-validation 合规、自动审批授权或 drift root-cause 判断；
- current/test 不改变 reference bins/support、importance model、policy 或阈值；
- reporting/visualization 不重算 predict、metric、policy、drift 或 importance。

**验收条件**

- model champion/challenger 使用相同 frozen fold/denominator；policy/alert comparison
  inventory 保留 Task 17/18 的 denominators、paired transitions、support 和 limitations；
- randomized/unknown/non-random assignment 明确区分，缺少 action/outcome support 时不
  产生因果或业务提升结论；
- reason source、priority、mapping、override 和 fallback coverage 与 Task 17/18 frozen
  trace 一致，spy 证明未重新执行 rules；不得包含未预算的全量 PII details；
- permutation importance、prediction drift 和 slice metrics 保留 v0.2 所需 seed、n、
  reference state、方向、不确定性与小样本 skipped behavior；Task 16 missingness/
  feature-profile results 的 n、budgets 与 warnings 原样保留，spy 证明无重算；
- governance fields 是 provenance/monitoring evidence，不输出“compliant/approved/safe”；
- 独立 Task 19 决策记录冻结最小 explanation/governance/inventory surface、上游 result
  compatibility 和稳定 vocabulary。

### Task 20 — v0.2 Integration and Release Readiness

**目标**

把 Tasks 15–19 的 public results 接入唯一 opt-in v0.2 workflow、静态报告和 CLI，
完成文档、examples、distribution 和 CI readiness；不实际发布。

**依赖**

Tasks 15–19 全部完成并通过各自合同。

**范围**

- 冻结新的 v0.2 workflow result，而不是扩充 v0.1 `AnalysisRun`；
- workflow 为 score validation、pre-loan eligibility 和 post-loan early warning 提供
  三条独立 opt-in 路径；可以组合报告，但不强迫共享 target/score/action/result；
- workflow 只编排 Tasks 03–19 public APIs，每个启用的上游步骤 exactly once；
- Markdown/HTML 增加 validation/business metrics、leakage audit、policy assumptions、
  pre-loan rule hits/action/constraint simulation、post-loan alert history/backtest、
  vintage/MOB/roll-rate/cure、comparison inventory、reason/override audit、explanation
  和 stability 章节；
- 报告只消费 frozen results，PNG asset/Figure ownership 沿用 Task 13 原则；
- CLI 采用显式 opt-in 路径并把参数原样传给 workflow，不含领域算法；Task 17/18 的
  唯一文件载体是窄范围、版本化、纯数据 JSON policy/warning spec；
- 更新 SPEC/plan/README/API/leakage/analysis guide/examples/changelog 和 distribution
  tests；
- 允许在 Task 20 独立合同 review 后准备 `0.2.0` metadata，但不 tag、push、upload
  或创建 release。

**兼容策略**

- 旧 `run_analysis`、`generate_analysis_report` 和 `sharper analyze` 默认输出不变；
- v0.2 报告不得让 v0.1 report 新增空章节或改变 asset 路径；
- CLI 新参数/命令、stable errors、exit codes、output layout 和 allowlist 必须由
  Task 20 决策记录精确冻结；
- Python caller 与 CLI 的 Figure close/rollback 继续遵守 Task 13 ownership。

**v0.1 测试迁移原则**

Task 20 必须把 public/distribution contract tests 明确分成两层，而不是删除旧断言：

1. `v0.1 compatibility invariants` 永久验证 v0.1 signatures、dataclass 字段、默认
   行为、report/CLI 路径和全部 v0.1 exports 仍存在；旧行为断言不得搬移、弱化或删除。
2. `current release surface` 验证当前 package version、当前完整有序 `__all__`、v0.2
   opt-in exports 和当前 distribution metadata。

v0.1 symbols 应作为固定 ordered compatibility manifest 在当前 `__all__` 中 append-only
保留，但“完整 `__all__ == v0.1 列表`”和字面 `0.1.0` 只能是 v0.1 release-surface
断言，不能继续作为永久兼容不变量。版本文本、artifact filename 和 metadata 只可在
Task 20 合同批准后同步迁移；Tasks 13/14 contracts 作为历史合同不反向改写。

**验收条件**

- Python workflow 与 CLI 对同一 input/spec 产生一致 results、sections、warnings、
  limitations、budgets 和 assets；
- policy/warning spec 仅接受 closed JSON schema：必须有 schema version，只允许 Tasks
  16–18 合同列出的字段、类型和 condition operators；未知 version、字段或 operator
  明确失败，不得静默忽略；
- JSON parsing/CLI adapter 只属于 Task 20；condition/rule/alert 语义和 schema contract
  分别由 Tasks 16–18 冻结。不得接受 YAML/TOML、Python 表达式、函数、脚本、模板、
  comments、include/URL/`$ref`、环境变量或路径展开、任意代码；不得建立通用 DSL；
- Task 20 合同冻结 JSON bytes/string、rules/conditions/nesting/bands/windows budgets、
  duplicate-key 与 stable I/O/parse/schema errors。conceptual CLI 入口可为
  `--policy-spec <json-file>`、`--warning-spec <json-file>`，实际参数名留给合同；
- spy 证明 workflow 不重复计算，reporting 不 fit/predict/evaluate，CLI 不实现领域逻辑；
- v0.1 compatibility invariants 与 current release-surface tests 分层通过；v0.1 行为、
  signatures、result fields、默认 report/CLI 和 exports 均保留，仅 current version/
  complete-export/distribution expectations 按 Task 20 合同迁移；
- CLI 安全 tests 覆盖未知 schema version/field/operator、任意代码字符串、路径或环境
  变量展开尝试、非 JSON、duplicate key、超预算嵌套和 deterministic round-trip；等价
  Python spec 与 JSON spec 产生相同领域结果；
- 新 score-validation、pre-loan eligibility、post-loan early-warning/lifecycle examples
  确定运行且只使用 public API；示例 rules/actions/alerts/costs 必须标记为
  synthetic/illustrative；
- pytest、Ruff check、Ruff format check、build、wheel/sdist 独立 clean-install、
  examples、CLI smoke 和支持 Python matrix 通过；
- 未增加 forbidden capability、lock file、生成报告、Kaggle data 或 publishing 权限；
- Task 20 只达到 release readiness，不执行 tag/push/upload/release。

## 9. 依赖顺序与跨任务 ownership

```text
Task 15 ─┐
         ├─> Task 17 ─┐
Task 16 ─┤             ├─> Task 19 ─> Task 20
         └─> Task 18 ─┘
```

Task 15、16 是可独立 review/实施的两项基础能力。Task 17 硬依赖 Task 16，并在使用
score/outcome/business metrics 时消费 Task 15；Task 18 硬依赖 Task 16，可选消费 Task
15 的 frozen score semantics，但不因纯规则/生命周期路径强制要求 score。Task 17 与
Task 18 逻辑并列，二者不 import/call 或要求对方 result。Task 19 等待 Tasks 15–18
冻结结果，Task 20 等待 Task 19；不得通过后续 Task 临时复制前置能力绕过依赖。

以下冻结的是逻辑 ownership；聚焦模块的实际文件名在各 Task contract 与 `SPEC.md`
中最终确认，但不得改变职责或 DAG：

- Task 15 拥有 risk validation 与 business-metric semantics：`modeling` 唯一拥有 fold 内
  estimator/pipeline/calibrator fit 与 OOF/holdout predictions；`evaluation` 只验证
  frozen external/modeling score provenance 并计算 ranking/probability metrics、
  analytical threshold evidence 和基础 business primitives，不 fit estimator；
- Task 16 拥有 quality/leakage evidence、input feature profiles、missingness profiling/
  drift，以及唯一的 private closed condition-evaluation kernel；它不清洗数据、不选择
  policy、不生成 action/alert。private kernel 可位于 Task 16 合同批准的聚焦内部模块，
  但不得成为 public API；
- `features` 保持 v0.1 feature specification/materialization ownership；v0.2 不新增
  learned transform；
- Task 17 必须有独立、聚焦的 pre-loan policy 模块（`strategy` 只是工作名），拥有
  pre-loan eligibility rule evaluation、simulated action、constraints、rule trace、
  backtest 和 pre-loan policy comparison；它只消费 Task 16 private kernel，不 fit
  model、不执行 action、不成为通用规则引擎；
- Task 18 必须有独立、聚焦的 post-loan monitoring 模块，拥有 point-in-time signals、
  warning rules、alert history/episode/backtest、vintage/MOB/roll-rate/cure/lifecycle 和
  warning-policy comparison；它只消费 Task 16 private kernel，不依赖 Task 17，不训练
  模型或执行 alert action。现有 `analysis.py` 保持 v0.1 职责，不承载 Task 18；
- Task 19 必须有独立的 explanation/comparison/governance inventory owner，只单向消费
  Tasks 15–18 frozen results；它可计算 Task 19 自有的 model explanation、model-score
  comparison、prediction drift 和 performance stability，但不得重算 Task 15 metrics、
  Task 16 missingness/input profiles、Task 17 policy comparison 或 Task 18 alert/lifecycle
  comparison。Tasks 15–18 的领域模块不得反向 import Task 19 owner；
- `visualization` 只消费结果或合同明确允许的 raw DataFrame，不保存文件；
- Task 20 的 `workflow` 只编排 public APIs，既不承载 condition/policy/alert/governance
  计算，也不成为跨域 inventory owner；
- `reporting` 不重算；`cli` 不含领域逻辑。

若某项最终需要新模块，必须先在该 Task 决策记录和 SPEC 中证明单一职责与依赖方向；
不得创建 catch-all `risk`、扩张 `analysis.py`、`validation_manager`、registry 或 plugin
framework。import DAG 必须证明无环：Task 16 private foundation 只被 Tasks 17/18
消费，Task 19 只消费上游 results，Task 20 只编排。

## 10. Benchmark 数据集矩阵

Kaggle/UCI 数据不得进入 sdist、wheel 或默认 CI。benchmark 分三级：

- Tier A：小型 deterministic synthetic，强制 CI/release gate；
- Tier B：许可允许的中型 public dataset，开发者显式下载后离线运行；
- Tier C：大型/关系型 Kaggle dataset，手工 benchmark 和文档证据，不是 CI gate。

| 数据集 | identity/形态 | 层级 | 主要验证 | v0.2 输入边界 |
|---|---|---|---|---|
| Synthetic static risk | 单表静态 | A | label maturity、ranking/probability、正类、ties、metrics、calibration、bands/business primitives | 仓库内小 fixture |
| Synthetic eligibility policy | rules + optional score/action/cost/constraint/outcome | A | condition parity、hard/soft/refer、missing/conflict/version、action-role metrics/payoff | 完全 synthetic；Task 17 correctness 权威 fixture |
| Synthetic early-warning history | entity × observation date + rules/events | A | alert episodes、horizon/censoring、capture/lead time、vintage/roll-rate/cure | 完全 synthetic；Task 18 correctness 权威 fixture |
| Synthetic policy audit | model/rule/policy versions + hits + overrides + reasons | A | provenance、comparison inventory、governance no-recompute | 无 PII；上游已知 frozen results |
| Synthetic wide periods | 多期宽表 | A | ordered windows、change/trend、缺期、vintage age | 通用列名 |
| Synthetic drift slices | reference/current + time | A | Task 16 missingness/input profiles、prediction drift、model/policy/alert stability、new categories | 通用 time/group/action/alert 名 |
| Give Me Some Credit | 单表静态 | B | 特殊值、缺失、不平衡、Gini/KS/PR/lift、illustrative eligibility | 用户下载；不硬编码列；无真实 policy log |
| Default of Credit Card Clients | 六期宽表 | B | ordered temporal/lifecycle signals、calibration | UCI/Kaggle mirror；无真实 alert log |
| FICO HELOC Explainable ML data | 单表静态 | B | model explanation、reason provenance schema | 不把 challenge explanation 当法定 reason |
| Credit Risk Dataset | 单表混合类型 | B | prepared eligibility rules、unknown categories、provenance | 固定为 `laotse` identity；rules/actions/costs 仅 illustrative |
| Loan Default Prediction Dataset | 25 万级单表 | B | memory/runtime/reproducibility | 固定为 `nikhil1e9` identity；无真实 policy log |
| Home Credit Default Risk | 多表关系型 | C | 外部聚合后 score metrics/explanation | 只接受 prepared flat table；策略假设不作效果证据 |
| American Express Default Prediction | entity-event longitudinal | C | prepared observation history、group/time isolation、lifecycle/memory | 可取有界 entity subset；无真实 warning policy |
| Home Credit Model Stability | 多表 + time drift | C | prepared time/vintage drift/stability/memory | 只接受 external flat/time slices；无真实 alert episodes |

每个 Tier B/C benchmark runner 必须记录：来源 URL、数据版本/文件 hash、许可、下载不
属于测试、target/positive/entity/time/exclusion 映射、row/column counts、抽样方式、
seed、硬件/软件环境、wall time、peak memory、结果和 limitations。仓库不得提交原始
Kaggle 数据、派生大表、模型、notebook output 或 leaderboard submission。

benchmark 不以复现某个 public/private leaderboard score 为门槛。可接受条件是：
数据流正确、无 leakage、结果可复现、指标/样本量可审计、规模行为没有非预期无界
增长。性能比较只能在相同数据版本、fold plan 和环境下解释。

公开 Kaggle/UCI 风险数据通常没有真实 rule versions、actions、alert episodes、
costs、constraints、override、randomized assignment 或 rejected outcomes。它们只能
验证 score/lifecycle 数据流和带明确假设的示例，不能作为 policy/alert correctness、
策略收益或在线预警效果 benchmark。Tasks 17/18 的 correctness 只以 Tier A synthetic
为权威；Tier B/C 仅为开发者本地可选的兼容性、规模和方法 evidence，不是 release
gate。若未来引入真实脱敏 policy/alert log，必须另行评审许可、selection、隐私与
实验 provenance。

## 11. Synthetic test strategy

合成测试是 v0.2 的权威 correctness evidence，至少覆盖：

### 11.1 binary metrics

- 标签 first appearance 为正类/负类相反、bool/string/int 正类；
- 完美、反向、随机、常量、ties 和 top-fraction 边界分数；
- 单类、无正类、无负类、缺失/非有限 score；
- 手算 ROC/Gini、KS、average precision/PR、gains/lift/capture、Brier/log loss；
- ranking score 覆盖负数、大于 1、`decision_function` margin、风险方向反转和严格单调
  变换；排序指标保持正确，但 probability metrics 与 expected loss 为 unavailable；
- event probability 覆盖合法值、0/1、接近边界、小于 0、大于 1、NaN/Inf、错误
  positive-label mapping，以及 `p`/`1-p` 配合相反正类的手算 Brier/log loss；
- `predict_proba` 与 `decision_function` 同时存在时 role/provenance 分离；合法值域但
  失准的概率与合法校准结果分开测试；
- event-probability expected loss 手算一致，ranking-only 不生成 expected loss；
- 预声明阈值恰在 ties 上、多个候选同目标值和 deterministic tie-break；final-test
  sentinel 不改变 analytical operating point，诊断 band 不自动成为 action band。

### 11.2 validation and leakage

- stratified folds 保留可行类分布；
- 同一 entity 多行，证明 group 不跨 fold；
- observation time 有序但 outcome window 穿过 fold cutoff、event 已发生但 reporting
  delay 未结束、mature/immature labels 混合和单调时间仍 label-leaking 的哨兵；
- `label_available_time == C`、outcome horizon 与 validation observation window exact
  overlap、逐行 horizon、显式 outcome-end 推导及 inclusive/exclusive boundary；
- missing/non-monotonic/inconsistent label-availability metadata、purge 后空或单类 fold、
  group + time 联合隔离和 final-holdout immature/censored rows；
- 有序时间、future-only category/extreme/outcome，证明训练只使用 cutoff 前已成熟标签，
  validation/test future outcome 改变不影响 fold、fit、calibrator 或 threshold state；
- duplicate row/index、同实体多 target、time reversal 和 insufficient groups/time；
- test-only category/extreme/group/entity/action/outcome 不进入 imputer、encoder、
  calibrator、threshold、policy、cost、constraint 或 estimator state；
- historical action-assignment dependence、rejected outcome missing、post-action/post-
  outcome field 均产生 evidence 而不是自动修复；
- evaluation-time rule version、future-only rule input、future-only peer baseline、
  customer-history future value、horizon endpoint 和 test-only alert threshold sentinels；
- 同 customer 多 accounts 按 caller group 隔离；missing/misordered available time、
  observation time 和 event time 有稳定 evidence。

### 11.3 pre-loan eligibility and decision strategy

- single hard decline、soft/refer/request-information 和 caller-defined symbolic actions；
- AND/OR/NOT、numeric/set/missing/date/multi-field conditions、nesting/condition budgets；
- 同一 closed condition 经 Task 17/18 消费时得到相同 `true/false/unknown`，未知 operator、
  arbitrary code/callable 与超预算嵌套明确失败；
- overlap、conflict、same priority、stop-on-hit、not-evaluated、multi-hit 和 fallback；
- effective/expiration boundaries、policy version、disabled rule 和 missing behavior；
- 升分增险/降分增险、score band + rule、open/closed bounds、ties、overlap/gap；
- action-only/no-outcome、observed-outcome replay、model-based expectation 三种结果语义
  及其可并存/claim 边界；
- utility/cost、exposure、manual capacity、rate/count/sum/metric constraints 的
  feasible/violated/unevaluable；
- 任意自定义 action names、两个 names 映射同一 role、缺少 role、未知 role、重复/冲突
  mapping、仅通用 action distribution，以及 mapping 后的 selection/rejection/review/
  request-information、selected-population event rate/exposure、capacity metrics 手算；
- 手算 hit/unique/overlap/conflict、target capture、ordered/leave-one-out marginal
  contribution、incremental action count、capacity/missing volume；
- reference/challenger paired actions 使用相同 rows/support；
- historical accepts only、outcome missing、action support disjoint 时披露 selection；
- caller policy development/comparison 只用 OOF/validation evidence，final holdout 不改变
  rules/order/policy，也不由 Sharper 自动选择 winner；
- Task 17 只消费 caller 明确传入/冻结的 cutoff/bands；Task 15 analytical candidate 不会
  隐式进入 policy；ranking score + mature outcome 只生成 observed metrics，仍不生成
  expected loss。

### 11.4 post-loan early warning and lifecycle

- single/continuous deterioration、recovery/cure 和 level/change/trend/persistence/
  combination/state-transition/peer/history rules；
- first/repeated alert、persistence、episode、cooldown、resolution、reopen；
- 多账户 prepared customer observations、duplicate entity-observation key、missing periods；
- exact observation/lookback/recent/history/horizon left/right boundaries、timezone；
- future-only row/value/event 不影响 current signal、peer/history baseline 或 alert；
- mature/immature horizon、right-censored alert、multiple alerts/events matching；
- caller-defined state stay/worsen/roll-forward/roll-back/cure/exit/missing transitions；
- vintage/cohort age/MOB 对齐、不同 cohort size 和 zero denominator；
- 手算 alert rate、raw rule-hit rate、entity coverage、capture/recall、precision、
  false-alert share、false-positive rate、mean/median lead time、burden、alerts per case、
  duplicate/unresolved rate；
- no-alert、single-threshold、current/challenger rules、model-score、model+rules 使用相同
  observations/entity population/horizon/maturity policy；
- 输入不变性、stable schemas、rule/window/entity/episode/detail budgets。

### 11.5 explainability, champion/challenger and governance

- 已知线性信号与噪声，验证 coefficient/permutation direction 和 provenance；
- model/policy/alert/override reason source、unmapped/fallback coverage、rule path 与版本；
- Task 17/18 frozen comparison/inventory 原样保留 denominators、support、paired
  transitions 和 limitations；spy 证明 Task 19 不重新执行 rules/alerts；
- randomized、non-randomized、unknown assignment 分开，只有外部日志不触发 experiment
  allocation 或因果 claim；
- 完全相同 reference/current、location shift、scale shift、new/dropped category、
  missingness shift、constant/all-missing；
- Task 16 missingness drift 的 reference/current rates、absolute/relative delta、new all-
  missing/recovered columns 与样本量手算一致；spy 证明 Task 19 只消费，不重算；
- time slices 事件率稳定但 feature drift、feature 稳定但 model/policy/alert performance
  drift；
- 小 slice/单类 slice 正确 skipped，不用零或全局指标伪装；
- reference bins/support 固定，current 不反向影响 state；
- governance purpose/version/materiality/limitations/issue status 只记录 evidence，不生成
  approved/compliant/safe 结论。

### 11.6 determinism and budgets

- 相同 seed/spec 得到相同 fold、metrics、actions、alerts、episodes、lifecycle tables、
  排序、warnings 和 report；
- row reorder 在合同允许时语义不变，tie-break 依赖稳定原始 row position；
- 超过 columns/rules/conditions/bands/cutoffs/details/entity/window/horizon/episode/cohort/state/slice/importance
  budgets 时记录 requested、actual、reason；
- 有界 scale test 证明没有无界 rule × cutoff × action expansion、pair/window expansion
  或整表 train/test concat。
- Task 20 JSON carrier 覆盖未知 schema version/field/operator、duplicate key、任意代码
  字符串、路径/环境变量展开、非 JSON、超预算嵌套和 deterministic round-trip；同义
  Python spec 与 JSON spec 产生相同 frozen domain result。

## 12. Report 与 CLI 集成策略

Task 20 之前，Tasks 15–19 只交付领域 public APIs、results 和适用的 result-only plots，
不得修改 workflow、reporting 或 CLI。

Task 20 集成必须遵守：

1. 新 v0.2 workflow result 不持有 raw full DataFrame；只保存 frozen metadata、必要
   holdout/OOF rows 的有界结果和 Figures；
2. workflow call order 和每个 public API 的调用次数由 Task 20 合同冻结；
3. reporting 从 result 获取所有数值和表，不重新 split、aggregate、fit、predict、
   calibrate、compare analytical thresholds、adopt cutoff/policy、execute rules、simulate
   actions、reconstruct alerts、compute backtest/lifecycle/drift/importance；
4. Markdown/HTML 章节一致，所有图为相对 PNG assets，仍不引入交互式 renderer；
5. 报告显式区分 exploratory audit、validation estimate、final holdout、action-only/
   no-outcome replay、observed replay、model-based expectation、simulated pre-loan action、
   reconstructed post-loan alert、external experiment log、observed association、drift
   和 limitation；不得把 simulated action/alert 写成真实执行结果，不使用“合规”
   “批准”“安全”等结论性语言；
6. CLI 只解析 input/spec/output、调用 workflow/reporting、管理 errors/exit code 和
   Figure cleanup；
7. 旧 `sharper analyze` smoke、帮助、版本和输出 bundle 测试保持；
8. 新 opt-in CLI 必须分别冻结三条最短路径：score-validation 要求 target/task/
   positive label、score kind/provenance/direction，time validation 另要求 label-maturity
   metadata；pre-loan eligibility 要求 semantic roles、action-role mapping、evaluation
   time 和 versioned policy spec，score/outcome/cost/constraint 仅在相应能力启用时要求；
   post-loan warning 要求 entity、observation time 和 warning spec，只有监督 backtest
   才要求 event/horizon；
9. Task 17/18 spec 的 CLI 文件载体只允许单一、versioned、closed JSON schema，必须含
   schema version。JSON decode/文件读取属于 Task 20；Tasks 16–18 分别冻结 condition、
   policy/action 与 warning/alert schema semantics。conceptual 入口可为 `--policy-spec`
   与 `--warning-spec`，精确参数留给 Task 20 contract；
10. CLI 不接受 YAML/TOML、多格式配置、Python callable/expression、函数/脚本、模板、
    comments、include/URL/`$ref`、环境变量或路径展开、第三方 rules DSL；未知 schema
    version、字段、operator 或超预算嵌套必须明确失败。不得自动采用 target、猜
    score/alert direction 或 action role、内置贷款 actions/levels/costs，或出现反欺诈
    配置入口。

## 13. v0.2 完成定义

只有同时满足以下条件，才可称为 v0.2 release-ready：

- Tasks 15–20 各有已接受的独立决策记录；Tasks 15/16 是基础能力，Task 17/18 均在
  Task 16 后且互不依赖，Task 19/20 按第 9 节 DAG 等待全部前置完成；
- 第 1.1 节的保留/延期处置已经 roadmap review 明确接受，并已同步到 `SPEC.md`；
- `SPEC.md`、`IMPLEMENTATION_PLAN.md`、`AGENTS.md`、API/guide/leakage 文档、README
  和 changelog 与实现一致；
- v0.1 public API、CLI、reports、results 和 defaults 保持兼容；tests 已拆为永久 v0.1
  compatibility invariants 与 current release surface，旧行为断言未被删除或弱化；
- `ranking_score` 与 `event_probability` 的值域、方向、positive-label mapping、
  provenance 和 allowed metrics 分离；fold/OOF、Gini/KS/PR/gains/lift、probability-only
  calibration 与预声明 threshold 候选的 analytical operating point 通过手算和基准；
- time validation 对每个 fold 同时执行 `observation_time < C` 与
  `label_available_time <= C`，记录 outcome window/reporting delay、mature/immature/
  purged n，并通过未来标签成熟度 leakage sentinels；
- Task 15 expected/observed loss primitives 与 Task 17 action-dependent payoff
  simulations 分属唯一 ownership，均记录单位、假设、outcome support 和 unevaluable
  reason；只有 event probability 进入 expected loss/revenue/payoff，且不把期望值表述
  为已实现业务结果；
- Task 16 private condition kernel 的 closed operators、three-valued truth、missing、
  effective/expiration time、budgets 与 Task 17/18 parity 测试通过；Task 16 missingness
  profile/drift 是唯一计算结果，Task 19 no-recomputation spy 通过；
- Task 17 的 hard/soft/refer、priority/stop-on-hit/conflict/version、caller action names 与
  explicit action-role mapping、caller-frozen score bands/cutoffs、constraints、rule
  metrics、deterministic policy simulation/comparison 通过手算和边界测试；
- group/time/entity/cutoff、calibration、analytical-threshold isolation、caller-frozen
  policy 和 historical outcome support 的 leakage regression tests 通过；
- Task 18 的 observation/available/event time、prior-only rule state、alert episode/
  cooldown/resolution/reopen、horizon matching、maturity/censoring、alert metrics、MOB/
  vintage/roll-rate/cure 有 stable schemas、provenance、budgets 和手算证据；
- model/policy/alert/override reasons、domain-owned offline comparisons、governance
  inventory、explanation/drift/stability 记录 n、方向、reference state、assignment
  provenance 和 limitations，Task 19 no-recomputation spies 通过；
- Task 17、Task 18 与 Task 19 各有独立聚焦 ownership，现有 `analysis.py` 未扩张为
  warning/lifecycle catch-all，import DAG 无环；
- score validation、pre-loan eligibility、post-loan early warning 三条 Python/CLI
  opt-in workflow 均可独立运行；Task 17/18 CLI 只接受 versioned closed JSON spec，
  安全/round-trip tests 通过，组合 Markdown/HTML 结果一致且无重算；
- Tier A synthetic matrix 全部通过；Tier B/C 若运行则有版本/环境/限制记录，但其
  数据不可得或未运行不阻断 release readiness；
- `bash scripts/verify-uv-env.sh` 后，项目 `.venv` 的 pytest、Ruff check、Ruff format
  check、build、wheel/sdist clean-install、examples、CLI smoke 和 CI support matrix
  全部通过；
- core 无新增强制 dependency，v0.2 不批准 optional extra；
- 反欺诈关键词审计证明相关词只出现在明确排除说明；没有 fraud/device/IP/VPN/
  velocity/graph/blacklist/transaction-fraud model、API、benchmark、CLI 或 report path；
- 未实现本文 out-of-scope 能力，未提交数据、模型、报告、cache、lock file 或密钥；
- 只达到 release readiness；没有 tag、push、upload、PyPI/GitHub release 或实际发布。

## 14. Roadmap review 与 implementation gate

本文具备 roadmap review 的前提是 review 同时核对：

1. 与 `SPEC.md` 已有 v0.2/v0.3 边界是否一致；
2. v0.1 frozen contracts 是否有被隐式破坏；
3. Tasks 15–20 是否职责单一、依赖顺序清楚、验收可测试；
4. multi-table、dependency、CLI 和 release 边界是否足够封闭；
5. benchmark 是否把 correctness evidence 与 leaderboard evidence 分开；
6. leakage、label maturity、ranking score/event probability、positive label、calibration、
   analytical threshold candidate、business cutoff 和 reference/current state 是否有
   明确 ownership；
7. action-only/observed/expected/experiment semantics、caller-frozen policy、outcome
   support 和 common-support limitations 是否不可混淆；
8. score validation、pre-loan eligibility 和 post-loan warning/lifecycle 是否是三条
   required inputs 与结果对象不同的独立 opt-in 路径；
9. Task 16 是否唯一拥有 private condition kernel 与 missingness drift；Task 17/18 是否
   各有独立模块并各自拥有领域规则、backtest、comparison/stability，现有
   `analysis.py` 是否未成为 catch-all；Task 19 是否只消费 frozen upstream results
   做其自有 explanation/comparison/governance 而无重算；
10. evaluation 是否从不 fit estimator，observation/horizon/peer/history state 是否
    point-in-time safe；
11. 反欺诈是否只出现在明确排除说明，且没有 Task、API、benchmark 或长期路线；
12. strategy/lifecycle/reason/override 能力是否保持通用且不构成线上执行、自动审批、
    自动催收或合规结论。
13. action name/role 是否显式分离且未按字符串猜测业务含义；Task 17/18 CLI 是否只用
    versioned closed JSON spec，拒绝未知 schema/operator 与任意代码载体；
14. v0.1 compatibility invariants 与 current release-surface tests 是否分层，且旧行为
    断言没有因 v0.2 export/version 迁移被删除或弱化。

在 roadmap review 给出 Go 且 `SPEC.md`/`IMPLEMENTATION_PLAN.md`/`AGENTS.md` 已同步
前，Task 15 不得开始。即使 roadmap 获得 Go，Task 15 仍必须先创建并 review 自己的
精确 API 决策记录；roadmap Go 不等于 Task 15 implementation Go。
