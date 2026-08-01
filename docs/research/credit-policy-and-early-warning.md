# Sharper v0.2 贷前准入规则与贷后预警研究

## 1. 文档身份与边界

本文是 Sharper v0.2 roadmap 的研究输入，不是 public API、Task 合同或实现授权。
研究基线为 Sharper `0.1.0` 稳定提交 `0b86986`，资料检索截止日为
2026-07-31。

本文专门回答两个问题：

1. 如何在离线表格数据上表示、执行、回测和比较贷前准入规则；
2. 如何在 `entity × observation time` 数据上回测贷后预警规则并监测生命周期。

本文与其他研究的分工为：

- `docs/research/kaggle-credit-risk-methods.md` 研究如何形成可验证的风险分数；
- `docs/research/credit-risk-decision-strategies.md` 研究 score、actions、costs、
  constraints 与 observed outcomes 的通用离线策略分析；
- 本文细化准入 policy rules 和 post-loan alert lifecycle，供
  `docs/decisions/v02-roadmap-contract.md` 划分 Tasks 17–19。

所有动作和警报都只是离线 simulation/backtest 输出。Sharper 不执行真实审批、
额度变更、催收、客户联系、账户处置或在线拦截，也不输出监管合规结论。

### 1.1 证据与项目建议

本文沿用策略研究的证据等级：

| 等级 | 定义 | 本文用途 |
|---|---|---|
| `A` | 监管/公共机构原文、同行评审研究、可复现实验 | 支持边界、治理和可测试不变量 |
| `B` | 公开代码或完整技术方案 | 支持数据流、规则结构和结果 schema 候选 |
| `C` | 厂商客户案例 | 只证明出现过某类流程；成效必须为 `vendor-reported` |
| `D` | 博客、演示或个人经验 | 只用于发现术语或缺口 |

文中使用：

- `OBS`：从来源直接观察到的做法；
- `REC`：Sharper 基于来源形成的通用抽象或测试建议。

来源中的字段、阈值、风险等级、产品政策和法律要求不得直接复制为 Sharper API。

### 1.2 反欺诈完全排除

v0.2 不研究、不规划、不实现反欺诈。排除身份欺诈、设备指纹、IP/VPN/代理检测、
申请 velocity fraud、团伙/图网络欺诈、外部黑名单接口、实时交易欺诈、欺诈模型、
欺诈规则引擎和实时流式拦截。本文不为这些能力设置 future Task、候选 API、benchmark
或长期 roadmap 位置。

边界相邻但仍允许的能力只有：

- 通用数据完整性、类型、范围和缺失审计，不判断身份真伪；
- caller 已提供的信用行为序列分析，不识别可疑设备、网络、商户或交易；
- 同一 entity 的 point-in-time 生命周期分析，不做团伙关系或图传播；
- caller 声明的 external exclusion flag 可作为普通输入，但 Sharper 不查询、生成或
  解释黑名单。

## 2. 方法证据

1. `A` [OCC Retail Lending, Version 2.0](https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/retail-lending/pub-ch-retail-lending.pdf)：
   underwriting/operating criteria、credit/documentation exceptions、high/low-side
   overrides、exception volume/trend、multiple exceptions 和 booked-loan performance。
2. `A` [OCC Installment Lending](https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/installment-lending/pub-ch-installment-lending.pdf)：
   underwriting criteria、verification、exception/override tracking、early-warning
   signs、early delinquency、vintage、roll-rate、cure 和组合趋势。
3. `A` [EBA Guidelines on loan origination and monitoring](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring?version=2020)
   与[官方 PDF](https://www.eba.europa.eu/sites/default/documents/files/document_library/Publications/Guidelines/2020/Guidelines%20on%20loan%20origination%20and%20monitoring/884283/EBA%20GL%202020%2006%20Final%20Report%20on%20GL%20on%20loan%20origination%20and%20monitoring.pdf)：
   credit-granting criteria、credit decision、持续 monitoring、quantitative/qualitative
   early-warning indicators、watch lists、follow-up 与 escalation。
4. `A` [Basel general credit-risk management principles](https://www.bis.org/basel_consolidated_guidelines/chapter/CRI/10.htm)：
   ongoing monitoring、delinquency identification、risk-rating deterioration、watchlist
   与责任/报告。
5. `A` [Basel expected-credit-loss guidance](https://www.bis.org/basel_consolidated_guidelines/chapter/PAP/20.htm)：
   信用风险随时间变化、风险驱动因素、monitoring intensity 与 deterioration evidence。
6. `A` [World Bank Credit Scoring Approaches Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)：
   score/model 与 downstream decision 分层、human review、data accountability 和
   governance。
7. `A` [A Submodular Optimization Approach to Accountable Loan Approval](https://ojs.aaai.org/index.php/AAAI/article/view/30310)：
   证明 score 与 approval rule 是不同对象；其自动规则优化本身不进入 v0.2。
8. `A` [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)：
   purpose、materiality、outcome analysis、ongoing monitoring、limitations、change 和
   exception governance。
9. `B` [FinRegLab Explainability & Fairness in ML for Credit Underwriting](https://finreglab.org/wp-content/uploads/2023/12/FinRegLab_2023-12-07_Research-Report_Explainability-and-Fairness-in-Machine-Learning-for-Credit-Undewriting_Policy-Analysis.pdf)：
   model-based 与 strategic/policy reasons 的分层。
10. `B` [OMG Decision Model and Notation 1.5](https://www.omg.org/spec/DMN/1.5/About-DMN)：
    提供 unique、first、priority、rule-order、collect 等 hit-policy 参考；v0.2 只借鉴
    最小确定性冲突/排序语义，不实现 DMN、解析器或规则 DSL。
11. `A` [FDIC Risk Management Examination Manual for Credit Card Activities, Chapter V](https://www.fdic.gov/regulations/examinations/credit_card/pdf_version/ch5.pdf)：
    vintage 按相似 origination cohort 和相同 age 比较，roll-rate 描述状态迁徙；旧资料
    只用于方法定义，不用于当前合规判断。
12. `B` [OCC Estimating Conditional Mortgage Delinquency Transition Matrices](https://www.occ.treas.gov/publications-and-resources/publications/economics/working-papers-new-frontiers-bank-risk-mgmt/pub-econ-working-paper-est-cond-mort-del-trans-matrices.pdf)：
    月度 loan-level panel、`t` 到 `t+1` 状态、MOB、prepayment/default/maturity/censoring；
    论文不代表 OCC policy，产品状态不得硬编码。
13. `A` [Consumer credit-risk models via machine-learning algorithms](https://doi.org/10.1016/j.jbankfin.2010.06.001)
    与[Risk and risk management in the credit card industry](https://doi.org/10.1016/j.jbankfin.2016.07.015)：
    使用严格 as-of inputs、future outcome windows、precision/recall 和 threshold tradeoff；
    专有数据的数值结果、成本或阈值不可外推。

`OBS`：监管材料要求机构形成信用准入、例外、监控、watchlist 和 escalation 流程，
但不会给出适用于所有产品的统一字段、阈值或软件 schema。

`REC`：Sharper 只抽象可验证的离线数据流、规则语义、时间边界、回测指标和审计证据；
不把监管检查程序改写为软件认证，也不抽象机构专属 risk appetite。

## 3. 贷前准入规则

### 3.1 分析单位与输入

贷前准入的最小分析单位是一行 candidate/application 与一个显式 `evaluation_time`。
规则只可访问该时间点已经可用的输入。若数据没有时间语义，结果必须披露无法验证
point-in-time availability，而不能声称不存在 leakage。

通用输入为：

```text
rows
+ semantic column roles
+ evaluation time
+ eligibility/policy rules
+ optional frozen ranking score or event probability
+ caller-defined action names and action-role mapping
+ costs/constraints
+ observed outcomes and support
```

不得硬编码申请、收入、负债、逾期、核销、产品、额度或客户标识字段名。

### 3.2 规则类型

| 规则类型 | 通用问题 | 允许的离线示例 | 不得推断 |
|---|---|---|---|
| 数据完整性 | 决策所需输入是否存在/可用 | required columns、missing check、freshness | 身份真实性或欺诈 |
| 基础资格 | 是否满足 caller 定义的基本条件 | range、set membership、date eligibility | 法定资格或阈值 |
| 产品适用性 | row 是否适用于某 caller-defined product/segment | category combination、mutual applicability | 自动产品推荐 |
| 信用政策 | 是否满足 caller 的 risk policy | score band、历史状态、caller threshold | “高风险”的固定标签 |
| 现有敞口 | caller 提供的 exposure 是否超过限制 | numeric comparison、aggregate input | 新额度或最优额度 |
| 收入与负债能力 | caller 提供的 affordability inputs 是否满足规则 | ratio/range/missing | 法定口径或还款能力结论 |
| 历史逾期/核销 | caller 提供的历史状态是否触发政策 | set/window check | DPD/charge-off 固定定义 |
| 产品互斥 | 多产品/状态组合是否冲突 | set intersection、mutual-exclusion rule | 产品营销或定价建议 |
| 必须补件 | 哪些缺失/过期输入需要补充 | `request_information` candidate | 自动外部数据查询 |
| 人工审核 | 哪些 cases 需要 refer | soft/refer rule、score middle band | 人工审核结论 |

`REC`：规则类型是 metadata vocabulary 候选，不是 public enum。Task 17 合同必须决定
是否公开、如何扩展以及未知 rule type 的错误行为。

### 3.3 规则动作

研究以下 caller-defined symbolic actions：

```text
pass
approve
decline
refer
request_information
limit
exclude
```

语义边界：

- `pass` 表示通过当前 gate，不等于最终批准；
- `approve`、`decline` 和 `refer` 只是 final simulated action 候选；
- `request_information` 只记录需补充输入，不调用外部接口；
- `limit` 只表示调用者定义的受限候选动作，不计算额度或修改账户；
- `exclude` 只从当前策略 denominator 中按显式规则排除，并保留原因；不得删除原始行；
- 动作名称和顺序不得冻结为信用专用 enum，调用者可以提供其他安全字符串。

动作名称与业务指标角色必须分开。通用 `action_role` 至少包括
`selected/rejected/review/request_information/limited/other`；一个名称只能映射一个
role，多个名称可映射同一 role。没有 mapping 时只报告通用 action distribution，
不得从 `approve`、`decline`、`refer` 等名字猜测 selection/rejection/review 语义。

### 3.4 规则语义

| 语义 | 研究边界 |
|---|---|
| hard rule | 命中后可按显式 priority/stop policy 覆盖 score 或其他规则 |
| soft rule | 记录命中、累积信号或改变候选 action；不得隐式升级为 hard decline |
| refer rule | 形成 caller-defined review/request-information action |
| AND / OR / NOT | 仅组合封闭原子条件；限制 nesting/depth/condition budget |
| 数值比较 | 明确 operator、单位、inclusive/exclusive boundary 和非有限值行为 |
| 集合匹配 | 明确 exact membership、unknown value 和 dtype normalization |
| 缺失判断 | missing 是独立状态；不得默认当作 0、pass 或 decline |
| 日期窗口 | 显式 anchor、timezone、left/right closure；不得读取 evaluation time 之后数据 |
| 多字段组合 | 组合已声明的原子条件，不允许任意 Python/eval 表达式 |
| priority | 明确排序、稳定 tie-break 和同 priority 行为 |
| stop-on-hit | 记录未执行规则；不得把“未执行”写成“未命中” |
| 多规则累计 | 记录所有 evaluated/hit rules、aggregate signal 和 budget |
| effective/expiration date | 用 `evaluation_time` 选择唯一有效 policy version；边界必须冻结 |
| missing behavior | caller 显式选择 error/unknown/no-hit/refer 等封闭行为 |
| conflict behavior | explicit precedence、conflict table 或 stable error；不得静默覆盖 |

原子 condition 应研究 `true/false/unknown` 三值结果。missing/invalid input 产生
`unknown`，再由 policy 显式映射为 no-match、match、request-information/refer 或
error；不得在 condition 层静默把 unknown 当作 false。hit policy 至少区分 unique、
first/stop-on-hit、explicit priority 和 collect/cumulative，不得依据 action 文本或
severity 猜默认优先级。

v0.2 不建立任意规则 DSL、通用 workflow engine、动态代码执行或 production rules
runtime。Task 16 唯一拥有 private、dependency-light 的 closed condition-evaluation
kernel，冻结三值 truth、原子 operators、AND/OR/NOT、missing、effective/expiration
boundary、deterministic ordering 和预算；它不生成 action 或 alert。Tasks 17/18 只消费
该 kernel 并各自冻结 policy/alert 语义，不能实现副本或把 kernel 导出为 public DSL。

### 3.5 模型、规则、约束与动作组合

必须分别记录：

```text
prediction
policy rule
business constraint
final simulated action
```

支持研究以下离线 sequencing：

1. 先规则后模型：eligibility gate 先排除/转人工，再对剩余 rows 使用 score bands；
2. 先模型后规则：score 形成 base action，hard policy rule 可以显式 override；
3. 硬规则覆盖模型：记录 base action、override rule 和 final simulated action；
4. 分数区间触发人工审核：两个或多个 boundaries 映射 caller actions；
5. score band + eligibility rule：分别记录 band 与 rule contribution；
6. cutoff + policy constraints：约束只判定 feasible/violated/unevaluable；
7. 多动作决策矩阵：dimension、precedence 和 fallback 必须有预算。

final simulated action 只由 rules、bands、precedence、override 与 fallback 形成；business
constraint 在 action 冻结后评价 scenario，不生成/改写 action、不选择替代 policy。

不得根据 score direction、label 值、action 名称或规则文字猜测风险方向。任意有限
`ranking_score` 可用于排序、bands 与 cutoff replay；只有明确对应 positive event 且
位于 `[0,1]` 的 `event_probability` 可用于 calibration 与 expected loss/payoff。
`decision_function` margin 不得自动解释为概率。模型输出不能直接等同最终决策；规则
命中也不能自动改写 observed historical action。

### 3.6 准入规则回测指标

| 指标 | 推荐定义与限制 |
|---|---|
| rule hit rate | `hit rows / eligible evaluated rows`；记录 missing/unevaluated |
| unique hit rate | 明确是 sole-hit、first-hit 还是 leave-one-out；三者不得混名 |
| rule overlap | pair/combination hit counts；受 pair/combination budget 限制 |
| rule conflict | 同一 row 产生不兼容 actions/precedence 的 count/rate |
| action distribution | 所有 caller action names 的 count/rate；不需要 role mapping |
| selection/approval、rejection、review、request-information rate | 只在对应 action roles 显式映射时计算；不是实际业务率 |
| selected-population event/bad-rate impact | 只在 selected role 与成熟 observed-outcome support 上计算并披露 selection bias |
| target capture | 被规则/动作覆盖的 observed events / supported events |
| marginal rule contribution | fixed-order incremental 或 leave-one-out delta，必须命名方法 |
| incremental action volume | caller-selected role/action subset 的 paired count delta |
| review/request-information volume | 显式 role 对应 count/rate 与 capacity gap |
| missing-input volume | 因 required input missing 而 unknown/refer/error 的 rows |
| segment-level impact | caller-defined slices，记录 n、support 和小样本 limitation |
| time-period stability | frozen rule version 在显式 periods 上的 hit/action/outcome trend |

所有 policies 必须在相同 frozen rows、score provenance、action-role mapping、cost/
constraint assumptions 和 outcome support 上比较。Task 15 只在 train/validation/OOF
报告 caller 预声明 threshold 候选的 analytical operating point；Task 17 只消费 caller
明确传入或已冻结的 cutoff/bands，不自动生成、部署或采用业务 policy。final holdout
仅评估一次。

### 3.7 规则建议边界

v0.2 可以报告 dead rule、over-broad rule、overlap、conflict、order sensitivity 和
leave-one-out marginal contribution，形成“需要人工复核”的分析建议；不得自动生成
新业务阈值、自动删规则、修改数据、选择真实 policy 或执行 action。

## 4. 贷后预警规则与生命周期监测

### 4.1 分析单位与 point-in-time 不变量

贷后预警的权威分析单位只能是：

```text
customer × observation_date
```

或：

```text
account × observation_date
```

调用者必须显式给出 entity column、observation date、可用时间、event date/horizon 和
必要的 account-to-customer 映射。v0.2 只接受一个 DataFrame，不 join 多表；多账户
客户级输入必须由调用者在外部按 point-in-time 方式准备。

对 observation `t` 的任何 signal、peer reference、personal-history baseline、rule
state 和 alert state，只能使用 `available_time <= t` 的信息。用于回测的未来 event
可以在评估阶段读取，但不得参与 `t` 时的 alert 形成。

### 4.2 基础预警信号类别

| 信号类别 | 通用表达 | 边界 |
|---|---|---|
| 逾期状态变化 | caller-defined state transition | 不硬编码 DPD bucket |
| 连续逾期 | prior-only consecutive-hit count | 缺期与连续性规则显式 |
| 逾期程度恶化 | ordered state/change rule | state order 由 caller 提供 |
| 最低还款缺失 | required-payment flag/amount missing | 不定义产品法定最低还款 |
| 还款金额下降 | recent vs prior/reference change | 金额单位和分母显式 |
| 余额快速增长 | level/change/trend | 不自动解释原因 |
| 授信使用率快速增长 | caller-provided numerator/denominator trend | 不推断额度字段 |
| 超限 | caller-provided value 与 limit 比较 | 不调整或推荐 limit |
| 现金提取/高风险信用行为占比增长 | caller-defined behavior ratio trend | 不识别交易欺诈、设备/IP 或商户风险 |
| 收入/现金流下降 | caller-provided series change | 不推断收入真实性 |
| 多账户同时恶化 | prepared customer-level simultaneous signals | 不做关系发现或图分析 |
| 查询/新增负债快速增加 | caller-provided count/change | 不调用外部征信 API |
| 长期无正常还款 | elapsed periods since caller-defined normal payment | state/period 显式 |
| 稳定转为波动 | rolling dispersion/change-point diagnostic | 只描述，不解释因果 |
| 相对个人历史偏离 | prior-only personal baseline deviation | 不得用未来个人记录 |

这些是研究 taxonomy，不是默认启用的规则或 public enum。Sharper 不提供信用字段名、
阈值、等级或风险方向。

### 4.3 时间与窗口语义

| 概念 | 研究定义 |
|---|---|
| observation date | 形成本次 signal/alert 的 as-of 时间 |
| lookback window | observation date 之前允许读取的总窗口 |
| recent window | 与 observation date 相邻的 caller-defined 子窗口 |
| historical window | 与 recent 对照、仍严格早于/截至 observation date 的窗口 |
| prediction horizon | 回测时从 alert time 向未来寻找 event 的区间 |
| event date | caller-defined outcome 首次/目标发生时间；不参与当时 alert 计算 |
| lead time | qualifying alert 与后续 event 的显式时间差 |
| first alert | 一个 alert episode 的首次有效命中 |
| repeated alert | 同一 open episode 内再次命中，不自动当新 episode |
| alert persistence | 连续或允许 gap 后仍保持 active 的 observation periods |
| cooldown | alert/resolution 后禁止重新开启的 caller-defined period |
| alert resolution | caller-defined clear conditions 满足后的 episode close |
| alert reopen | resolution + cooldown 后再次命中形成新 episode |

所有窗口必须冻结 timezone、period unit、left/right closure、missing observation、
duplicate `(entity, observation_date)` 和 incomplete horizon 行为。未成熟 horizon 必须
标记 censored/unevaluable，不能计为无事件或 false alert。

cooldown 只抑制重复 notification，不抹去 underlying rule hit 或 alert state；因此 raw
hits、notifications 和 episodes 必须分别计数。缺失 observation 是 unknown/censored，
不得自动解释为 resolved；resolution 必须由显式 clear condition 或足够连续的已观察
clear periods 形成。

### 4.4 预警规则类型

| 类型 | 语义 | 关键测试 |
|---|---|---|
| level rule | 当前值跨越阈值 | boundary、missing、direction |
| change rule | 当前与上一有效期或 anchor 的变化 | gap、zero denominator、first period |
| trend rule | caller-defined 多期连续方向/斜率 | 最小 periods、缺期、ties |
| persistence rule | signal 连续/累计命中 | allowed gap、reset、cooldown |
| combination rule | 多个 signals 同期或窗口内组合 | AND/OR/NOT、time alignment |
| state-transition rule | caller-defined prior state → current state | order、exit/re-entry、unknown state |
| peer-deviation rule | 相对 train/reference-fitted peer baseline | reference leakage、unseen peer |
| customer-history rule | 相对当前 entity 的 prior-only baseline | first observation、future leakage |

peer baseline 只能从训练/reference period 拟合；customer history 只能使用本实体过去
记录。current/test 不得反向改变 peer bins、group support、threshold 或 baseline state。

### 4.5 预警等级与输出

可研究 caller-defined ordered levels，例如：

```text
none
watch
warning
high
critical
```

这些值不在研究阶段冻结为 public enum，也不具有监管、催收或业务处置含义。稳定输出
至少应表达：entity、observation date、rules evaluated/hit、level、first alert time、
episode ID、current duration、recent change、supporting indicators、reason provenance、
resolved flag/time、repeat/reopen flag、policy/rule version、warnings 和 limitations。

### 4.6 贷后生命周期分析

v0.2 纳入以下描述性分析：

- vintage analysis；
- months-on-book 或 caller-defined cohort age；
- delinquency/state migration；
- roll-rate count/weight matrix；
- roll-forward 与 roll-back；
- cure rate；
- early delinquency；
- cohort comparison；
- account status transitions；
- period-level risk trend；
- alert episode 的 open/persist/resolve/reopen lifecycle。

vintage 必须按相同 age/MOB 比较并披露 maturity/censoring。roll-forward、roll-back、
cure、exit 和 missing 的 state sets 由 caller 显式定义；不得硬编码 current、30/60/90、
charge-off 或 cure。alert 与后续 migration 的关系是 descriptive association，不是
预警造成或避免迁徙的因果结论。

### 4.7 预警回测指标

| 指标 | 推荐定义与限制 |
|---|---|
| alert rate | alerted eligible observations / eligible observations |
| raw rule-hit rate | 未受 cooldown/notification suppression 影响的 raw hits / eligible observations |
| customer/account coverage | 至少一次 alert 的 entities / eligible entities |
| event capture rate | horizon 前存在 qualifying alert 的 mature events / mature events |
| recall within horizon | 与 event capture 使用相同 event denominator；命名需唯一化 |
| precision | 后续 horizon 内发生 event 的 mature alerts / mature alerts |
| false-alert share | mature alerts 中 horizon 内无 event 的比例；等于 `1 - precision` 仅在相同定义下 |
| false-positive rate | alerted mature non-event units / all eligible mature non-event units；不得与 false-alert share 混名 |
| average/median lead time | event date - first qualifying alert；只在 captured events 上 |
| warning burden | alert observations/episodes 相对 entities、periods 或 review capacity |
| alerts per case | 每个 entity/event 的 alert count；明确 observation 或 episode 粒度 |
| duplicate-alert rate | 同一 episode 内 repeated alerts / alerts |
| unresolved-alert rate | 截止 evaluation end 仍 open 的 mature/eligible episodes；披露删失 |
| severity distribution | caller levels 的 count/rate；无默认好坏方向 |
| segment performance | caller slices 的 n、events、alerts、precision/recall/lead time |
| time stability | 相同 frozen rules/horizon 在 time buckets 上的指标 |
| vintage performance | 同 cohort age/MOB 上的 alert 与 event metrics |
| roll-rate impact | alerted/non-alerted 的后续 transition 差异，只描述 association |

Task 18 决策记录必须决定用户术语 `false-alert rate` 对应哪个稳定 public 名称，或同时
公开上述两个不混淆的指标。所有指标都必须记录 observation/entity/event denominators、matured/censored counts、
missing event dates、multiple alerts/events matching policy、requested/actual horizon 和
skipped reasons。

比较基准至少包括：不预警、单一阈值规则、当前规则集、challenger 规则集、模型分数
预警、模型 + 规则组合。所有基准使用相同 frozen observations、entity population、
horizon、maturity policy 和 event definition。非随机历史策略不能称为 A/B test。
不预警 baseline 的 alert burden/capture 为 0，但 precision、false-alert share 和
lead time 为 undefined，而不是 0。

## 5. 贷前与贷后的边界及统一抽象

### 5.1 三层对象不能混合

| 层 | 分析单位 | 输入 | 输出 | 禁止混入 |
|---|---|---|---|---|
| prediction | row/entity observation | features + fitted model | ranking score 或 positive-event probability + provenance | policy action、alert、审批 |
| pre-loan eligibility/decision | candidate/application | rules + optional frozen score + constraints | final simulated action + rule path | 真实审批、alert episode |
| post-loan early warning | entity × observation date | prior-only signals + rules + horizon definition | alert state/episode + reason | 贷前 approve/decline、催收 action |

准入和预警共享 Task 16 private kernel 的原子条件、Boolean composition 与三值 truth，
并可共享 rule metadata、version 和 hit provenance 概念，但不能强行使用同一个结果
对象：准入产生 candidate action，预警产生 time-indexed alert state/episode。Task 16
kernel 不理解 action 或 alert；Tasks 17/18 不相互依赖。

### 5.2 通用输入模型

```text
input data
+ semantic column roles
+ rules
+ evaluation/observation time
+ optional ranking score or event probability
+ caller-defined action names/roles or alert levels
+ outcomes
+ constraints
```

共享不变量包括：显式 roles、无字段名猜测、Task 16 closed condition vocabulary、
deterministic priority、point-in-time availability、input immutability、budgets、stable
errors、版本和 provenance。信用模型预测、准入策略和贷后预警必须独立版本化。

监督 score validation 还必须区分 `observation_time`、可空 `event_time`、
`outcome_end_time` 与 `label_available_time`。time fold 训练行只有在 observation 早于
cutoff 且 label 在 cutoff 前可用时才合法；本研究的 alert-horizon censoring 不能替代
Task 15 的 label-maturity 合同。

### 5.3 候选概念评估

以下名字只用于研究，不冻结 public API、module、dataclass 字段或 enum：

```text
PolicyRuleSpec
EligibilityRuleSpec
EarlyWarningRuleSpec
RuleCondition
RuleAction
RuleHit
RuleEvaluationResult
EligibilityPolicy
EarlyWarningPolicy
AlertEvent
AlertHistory
PolicySimulationResult
PolicyComparisonResult
PolicyAudit
```

建议：

- `RuleCondition`/`RuleHit` 只是研究名；共享 condition truth 必须由 Task 16 private
  kernel 唯一计算，不因此冻结 public class；
- `EligibilityPolicy` 与 `EarlyWarningPolicy` 必须保留不同的时间、输出和 backtest
  语义；
- `RuleEvaluationResult` 不应成为字段随 policy 类型变化的自由容器；
- `AlertHistory` 必须有 bounded details，不能默认保存无界全量轨迹或 PII；
- 新候选必须与现有 `DecisionPolicySpec`、`DecisionRule`、`StrategySimulation`、
  `ReasonCode`、`PolicyAudit` 等候选合并评审，不能建立两套同义 public API。

精确 public/private surface 留给 Tasks 16–19 的独立决策记录。路线合同只冻结概念
边界，不提前冻结名字。

## 6. 规则治理

贷前准入和贷后预警都需要记录：

- rule ID、rule type、version、owner、description；
- priority、severity、effective date、expiration date；
- reason code、missing behavior、enabled/disabled；
- rule provenance、policy version、evaluation timestamp；
- matched-rule path、base/final result、override；
- score/model version（如使用）、semantic roles、timezone/window/horizon；
- reproducibility manifest、spec hash、budgets、warnings 和 limitations。

`evaluation timestamp` 是执行/生成分析的时间，不能代替 application evaluation time
或 post-loan observation date。相同 rule ID/version 若内容 hash 改变必须报错或产生
新版本，不能静默改写历史。

治理分析至少覆盖：

- 永远不命中的规则；
- 覆盖过宽的规则；
- 高度重复或等价的规则；
- 规则 conflict 与 unresolved rows；
- 规则顺序敏感性和 stop-on-hit shadowing；
- fixed-order incremental 与 leave-one-out marginal contribution；
- champion/challenger rule/policy sets；
- 不同 time/segment/vintage 的 hit、action/alert、outcome stability；
- override rate、reason coverage、unmapped/fallback 和 disabled/expired-rule usage。

治理结果只提供 evidence 和需要人工复核的建议，不判断 policy 合规、合理、适当或
可上线。

## 7. Benchmark 与 synthetic test strategy

### 7.1 贷前准入 synthetic scenarios

Tier A deterministic fixtures 至少包括：

- 单一 hard decline；
- soft/refer/request-information rules；
- missing input 的 error/unknown/refer 分支；
- 重叠 rules、同 priority conflicts 和 explicit resolution；
- stop-on-hit 与未执行 rule distinction；
- score band + eligibility rule、升分增险/降分增险；
- ranking-only replay 不生成 probability calibration/expected loss，合法 event
  probability 的 expected loss 与手算一致；
- reference/challenger rule sets 和 paired action transitions；
- effective/expiration boundary 与 policy version；
- arbitrary action names、多个 names 到同一 role、缺失/未知/冲突 role mapping、无 mapping
  时仅 action distribution，以及 mapping 后的 selection/rejection/review/capacity 手算；
- 手算 rule hit/unique/overlap/conflict、selected-population observed event rate、target
  capture、marginal contribution、incremental action 与 capacity volume；
- Task 15 analytical threshold candidate 与 caller-frozen Task 17 cutoff 分离，final-
  holdout 不影响两者；
- 与 Task 18 相同 closed conditions 得到完全一致的 Task 16 三值 truth；
- input immutability、rule/detail/pair budget 和 deterministic order。

### 7.2 贷后预警 synthetic scenarios

Tier A deterministic fixtures 至少包括：

- 单次 level deterioration；
- 连续 change/trend/persistence deterioration；
- recovery/cure、resolution、reopen；
- repeated alert、duplicate suppression、cooldown boundary；
- 多账户 prepared customer observation；
- exact observation-date/window left/right boundary；
- future-only row/value 不影响当前 signal/peer/history baseline；
- caller-defined roll-forward/roll-back/cure/exit transitions；
- mature/immature vintages 和 censored horizons；
- 手算 alert rate、raw rule-hit rate、coverage、event capture、precision/recall、
  false-alert share、false-positive rate、lead time、warning burden、duplicate/unresolved
  rate；
- no-alert、single-threshold、current/challenger、model-score、model+rule baselines；
- 与 Task 17 相同 atomic/Boolean/missing/date conditions 得到完全一致的 Task 16 三值
  truth，但 alert episode 与 action result 保持不同 schema；
- missing periods、duplicate observation key、timezone、multiple events 和 zero
  denominator；
- rule/window/entity/episode/detail budgets 与 stable ordering。

### 7.3 真实 benchmark 边界

真实公开数据只用于用户本地离线 benchmark，不进入仓库、sdist、wheel 或默认 CI：

- Give Me Some Credit、Credit Risk Dataset 等静态数据可验证 prepared score +
  synthetic eligibility rules，但没有真实 policy logs；
- Default of Credit Card Clients 可验证有序宽表和 illustrative lifecycle signals；
- American Express prepared bounded entity subset 可验证 observation history、group/time
  isolation 和 memory budget；
- Home Credit Stability prepared time slices 可验证 time/vintage stability；
- 这些数据通常没有 rule versions、alert episodes、review capacity、完整 actions、
  rejected outcomes 或 randomized assignment，不能作为 policy/alert 业务效果证据。

真实 benchmark 必须记录来源、版本/hash、许可、semantic role mapping、observation/
event/horizon 定义、抽样、seed、硬件、wall time、peak memory、outcome support 和
limitations。

## 8. 对 Tasks 15–20 的研究结论

1. Task 15 冻结 binary risk validation、ranking-score/event-probability、label maturity、
   预声明 threshold 候选的 analytical operating point 和基础 business metrics；不执行
   rules，也不采用业务 cutoff。
2. Task 16 审计 rule inputs、point-in-time availability、target proxy、entity/time
   leakage、missing/special values；唯一拥有 private closed condition kernel 与
   missingness profiling/drift，不自动修复或生成 action/alert。
3. Task 17 承载 pre-loan eligibility rules、score/policy combination、decision strategy
   simulation、action-name/role mapping、rule backtest 和 pre-loan policy comparison。
4. Task 18 承载 post-loan point-in-time signals、alert rules/history、alert backtest、
   vintage/MOB/roll-rate/cure、lifecycle monitoring 和 warning-policy comparison；它与
   Task 17 并列且不依赖其 action result。
5. Task 19 消费 Task 15 frozen model results 做 model comparison，并消费 Tasks 16–18
   已算结果，统一 explanation、version、provenance、override、comparison inventory 与
   stability；不得重算 condition、missingness drift、policy/alert backtest 或指标。
6. Task 20 才接入独立 opt-in workflow、static Markdown/HTML、CLI、examples 与 release
   readiness；Task 17/18 CLI 只使用 versioned closed JSON policy/warning spec，不实际
   发布或执行 action。

## 9. 结论

v0.2 的最小规则闭环应是：

```text
explicit semantic roles
  -> versioned, point-in-time-safe rules
  -> deterministic rule hits
  -> pre-loan simulated actions OR post-loan alert episodes
  -> support-aware backtest
  -> lifecycle/stability
  -> reason/override/version audit
  -> result-only static report
```

贷前准入回答“在这批候选记录、这些规则和假设下，会形成什么离线动作”；贷后预警
回答“在每个 observation date，当时可用的信息会形成什么 alert，之后在成熟 horizon
内观察到什么”。两者都不能回答真实 action 的因果效果、替代人工或机构政策判断，
也不构成真实贷款审批、催收建议或合规结论。
