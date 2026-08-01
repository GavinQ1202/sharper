# 信用风险决策策略调研

## 1. 文档身份与研究边界

本文是 Sharper v0.2 roadmap 的决策策略研究输入，不是 public API 合同，不授权
实现，也不构成贷款建议、自动审批或监管合规结论。资料检索截止日为
2026-07-31；研究基线为 Sharper `0.1.0` 稳定提交 `0b86986`。

本文与其他 v0.2 研究的分工如下：

- Kaggle 文档研究“数据如何变成可验证的风险分数”；
- 本文研究“分数、规则、成本、约束和已观察结果如何形成可审计的离线策略分析”；
- `docs/research/credit-policy-and-early-warning.md` 进一步冻结贷前准入规则与贷后
  alert/lifecycle 的研究边界；
- 三者都不得把 leaderboard、厂商宣传或历史规则外推为真实信贷建议。

反欺诈不属于当前研究或 roadmap：不研究身份/设备/IP/velocity/团伙图/黑名单/
实时交易欺诈、欺诈模型/规则引擎或流式拦截。相邻的数据完整性和 ID-like leakage
审计不构成身份核验或欺诈检测。

研究的通用输入固定表达为：

```text
scores + rules + actions + costs + constraints + observed outcomes
```

其中 score 必须声明为任意有限、带方向的 `ranking_score`，或明确对应 positive event 且
位于 `[0,1]` 的 `event_probability`。普通 score、rank 或 margin 不是概率，不能直接
进入 calibration、expected loss/revenue/payoff。action name 也不携带业务含义；需要
selection/rejection/review 等指标时必须由 caller 显式映射 action role。

信用风险是主要验证场景，但 Sharper 抽象不得依赖贷款字段、信用标签、特定分数方向
或监管辖区。自动批准、人工审核和拒绝只是文档示例；通用 API 必须允许调用者提供
其他动作名称。

### 1.1 证据等级

| 等级 | 定义 | 本文使用规则 |
|---|---|---|
| `A` | 监管/公共机构文件、同行评审研究、可复现实验 | 可支持方法边界、风险与可测试不变量 |
| `B` | 公开代码或完整技术方案 | 可支持可观察的数据流和实现方法，不自动证明外部有效性 |
| `C` | 厂商客户案例 | 只支持厂商采用过的场景和工作流；成效必须标记 `vendor-reported` |
| `D` | 博客、演示、个人经验或只有赛题背景 | 只用于发现术语、案例或研究缺口，不作为产品有效性证明 |

同一案例可以有多个等级。例如，SAS 的完整技术论文可作为 `B` 方法证据，SAS
客户故事中的收益仍是 `C` 且为 `vendor-reported`。FICO Explainable ML Challenge
的公开数据和论文实验可为 `A/B`，FICO 产品收益声明仍只能为 `C`。

来源选择优先使用检索日仍有效的官方原文、同行评审论文及作者公开代码；厂商材料
只用于观察工作流和产品边界，国内案例优先保留可检查代码或完整方案。只有二手转述、
缺少方法细节或无法识别原始证据的内容不用于冻结 v0.2 行为。

### 1.2 厂商数字与法律状态

- FICO、Experian、SAS 客户案例中的收益、通过率、效率或损失下降均标记为
  **vendor-reported**，不得作为独立验证或 Sharper benchmark target。
- CFPB Circular 2022-03 与 Circular 2023-03 已在 2025-05-12 被撤回。本文只把
  它们当作历史解释材料；截至检索日，当前规范性研究依据是 Regulation B
  §1002.9 及其 official interpretation。Sharper 不判断具体机构或决策是否合规。
- 2026 Revised Guidance on Model Risk Management（SR 26-2）于 2026-04-17 发布，
  取代 SR 11-7 和 SR 21-8；本文使用其 risk-based、materiality、validation、
  outcome analysis、ongoing monitoring 和 governance 原则，不把 supervisory
  guidance 转写成软件“认证清单”。

### 1.3 搜索主题覆盖

| 主题 | 主要证据 |
|---|---|
| application strategy、score cutoff、manual review | OCC Retail/Installment、CGAP、AAAI accountable approval |
| approval optimization、decision optimization、expected payoff、strategy simulation | EMP 研究、FICO Decision Optimizer、SAS collections paper |
| credit line management、risk-based pricing | OCC Retail/Installment、FICO/Experian/SAS 厂商材料 |
| champion/challenger、overrides、policy monitoring | OCC Retail、Experian PowerCurve、SAS/UOB、SR 26-2 |
| vintage、MOB、roll-rate、early warning | OCC Retail/Installment |
| collections strategy | OCC Installment、SAS 技术论文、FICO 客户案例 |
| reject inference | 同行评审综述、国内比赛公开方案；仅研究，不纳入 v0.2 算法 |
| adverse-action reasons、reason codes | Regulation B、FinRegLab、FICO xML Challenge |
| model and strategy governance | SR 26-2、World Bank、OCC、FinRegLab |

## 2. 来源清单与分级

### 2.1 监管与公共机构

1. `A` [OCC Comptroller's Handbook: Retail Lending, Version 2.0](https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/retail-lending/pub-ch-retail-lending.pdf)：
   credit cutoff、high/low-side overrides、exception monitoring、champion/challenger、
   risk-based pricing、vintage、roll-rate、scenario analysis 与全生命周期监控。
2. `A` [OCC Comptroller's Handbook: Installment Lending](https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/installment-lending/pub-ch-installment-lending.pdf)：
   scorecard front/back-end monitoring、cutoff/strategy feedback、early warning、MOB/vintage、
   roll-rate loss forecasting、collections workload 与策略评估。
3. `A` [World Bank Credit Scoring Approaches Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)：
   区分 score/model 与 downstream decision，强调 explainability、data accountability、
   model governance 和 human-centric review。
4. `A` [CFPB Regulation B §1002.9 official interpretation](https://www.consumerfinance.gov/rules-policy/regulations/1002/interp-9/)：
   adverse action 的 principal reasons 必须对应实际使用或评分的因素；“未达到最低
   分数”本身不是充分的具体原因。
5. `A` [CFPB withdrawn guidance index](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/)：
   证明 Circular 2022-03 与 2023-03 的撤回状态，防止把历史 circular 写成当前
   guidance。
6. `A` [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)
   与[正文](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)：
   model materiality、effective challenge、validation、outcome analysis、ongoing
   monitoring、inventory、documentation 和 third-party oversight。
7. `B` [CGAP Credit Scoring in Financial Inclusion](https://www.cgap.org/research/publication/credit-scoring-in-financial-inclusion)：
   score band、risk appetite、自动/人工流程和业务过程集成的公共技术指南。

### 2.2 独立研究

1. `B` [FinRegLab Explainability & Fairness in ML for Credit Underwriting](https://finreglab.org/wp-content/uploads/2023/12/FinRegLab_2023-12-07_Research-Report_Explainability-and-Fairness-in-Machine-Learning-for-Credit-Undewriting_Policy-Analysis.pdf)：
   区分 model-based 与 strategic reason codes，并分析 reason taxonomy、局部解释和
   adverse-action workflow。
2. `A/B` [FICO Explainable ML Challenge 获奖研究](https://arxiv.org/abs/2106.02605)
   与[公开交互原型](https://dukedatasciencefico.cs.duke.edu/)：可分解 risk model、
   globally consistent local summaries、case-based explanations 和公开 HELOC 数据。
3. `A` [Development and application of consumer credit scoring models using profit-based classification measures](https://doi.org/10.1016/j.ejor.2014.04.001)：
   Expected Maximum Profit 将 exposure、LGD 与收入纳入 cutoff 选择。
4. `A` [A Submodular Optimization Approach to Accountable Loan Approval](https://doi.org/10.1609/aaai.v38i21.30310)：
   已部署场景中的可解释 rule-base approval optimization；证明 score 与 approval rule
   是两个对象。
5. `A` [Reject inference methods in credit scoring](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041715/)：
   总结经典 reject inference 的隐含假设、选择偏差和难以真实评估的问题。
6. `A` [Credit scoring using three-way decisions with probabilistic rough sets](https://doi.org/10.1016/j.ins.2018.08.001)：
   accept / reject / acquire-more-information 三类动作的研究案例。

### 2.3 决策厂商案例

1. `C` [FICO Decision Optimizer strategy simulation](https://investors.fico.com/news-releases/news-release-details/fair-isaac-announces-version-50-decision-optimizer-enhanced/)：
   actions、action-effect models、objectives、constraints、多场景模拟；属于厂商产品
   描述，不证明方法独立有效。
2. `C` [FICO Česká spořitelna collections case](https://investors.fico.com/news-releases/news-release-details/czech-republics-ceska-sporitelna-uses-fico-prescriptive/)：
   collection capacity、segment/action intensity、champion/challenger；所有改进数字均
   为 **vendor-reported**。
3. `C` [Experian PowerCurve Strategy Management](https://www.experian.com/assets/strategy-management/product-sheets/powercurve-strategy-management.pdf)
   与[OneAZ automated decisioning case](https://www.experian.com/blogs/insights/case-study-automated-decisioning/)：
   rule flow、strategy monitoring、test-and-learn、manual review；26% funding-rate
   increase 和 25% manual-review decrease 均为 **vendor-reported**。
4. `B` [SAS Optimizing Collection Strategies with Intelligent Decisioning](https://support.sas.com/resources/papers/proceedings20/4301-2020.pdf)：
   用 risk、channel propensity、cost 形成 expected payoff，再将 segment 映射到
   action/channel；是 vendor-authored 完整方法，不是独立效果验证。
5. `C` [SAS/UOB lifecycle decisioning case](https://www.sas.com/en_gb/customers/united-overseas-bank.html)：
   behavior score、risk segment、scenario simulation、challenger、audit trail；收益与
   expected-loss 改善均为 **vendor-reported**。

### 2.4 国内竞赛与技术案例

1. `B` [天池贷款违约预测 Top 6 单模公开代码](https://github.com/caozichuan/TianChi_loadDefault)
   与 `B` [Top 11 公开代码](https://github.com/LogicJake/tianchi-loan-default-prediction-top11)：
   提供建模、特征和验证证据，但没有 actions、costs、constraints 或真实策略结果。
2. `B` [CCF BDCI 个贷违约预测二等奖完整方案](https://blog.csdn.net/DataFountain/article/details/126395609)
   与 `D` [官方赛题页](https://www.datafountain.cn/competitions/530)：涉及跨客群迁移、
   OOF、PSI、分布适配；其 AutoML、test feedback 和竞赛目标不进入 v0.2。
3. `B` [DataCastle CashBus 第一名公开代码](https://github.com/wepe/DataCastle-Solution)：
   可复查风险预测代码，但不提供从分数到真实行动的实验设计。
4. `B` [DataCastle/Rong360 公开竞赛方案索引](https://github.com/jackychancjcjcj/Competition_notebook)
   与 `D` [Top 7 个人复盘](https://www.cnblogs.com/jiading/articles/12840198.html)：
   展示多表申请/行为/账单数据与逾期预测；“是否通过贷款”的叙述不能替代可识别的
   policy experiment。

国内公开比赛的共同证据缺口是：大多数只提交风险概率或排序，不公开真实 cutoff、
人工审核容量、成本、拒绝样本结果、override、在线实验分流或策略后的生命周期结果。
因此它们是模型和数据 benchmark，不是策略收益 benchmark。

## 3. 重点案例提取

### 3.1 OCC Retail Lending：策略、例外与组合监控（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 贷前申请、账户管理、组合监控和贷后处置 |
| 2. 决策对象 | 申请、已开户账户、策略例外和组合 segment |
| 3. 可选动作 | approve/decline、policy exception、调整 cutoff/pricing/marketing、loss mitigation |
| 4. 预测量 | credit score、delinquency/loss、response、portfolio risk |
| 5. 业务目标 | 在 risk appetite 下管理增长、损失、收益和运营稳定性 |
| 6. 约束条件 | policy、risk tolerance、capital/liquidity、操作能力、适用法律 |
| 7. 策略逻辑 | score cutoff 加非评分规则；记录 high/low-side override；按 segment/vintage 反馈 |
| 8. 模拟或实验设计 | scenario/what-if；champion 对多数账户、challenger 对较小受控测试组 |
| 9. 模型指标 | score distribution、discrimination、actual-vs-expected、population stability |
| 10. 业务指标 | approval/decline、override rate、delinquency/loss、profitability、volume/exposure |
| 11. 监控方法 | exception trend、13+ month trend、vintage、roll-rate、chronology log |
| 12. 解释与审计 | rule/exception type、decision criteria、override performance、independent review |
| 13. 数据要求 | application、score、decision、rule hits、override、account outcome、time/exposure |
| 14. 可复现程度 | 方法定义与检查程序公开；无可运行数据，方法可用 synthetic 重现 |
| 15. 对 Sharper 的通用抽象 | rule path、base/final action、override、band/cutoff、policy monitoring |
| 16. 不适合直接抽象 | 特定贷款政策、risk appetite 数值、pricing、资本和监管判断 |

### 3.2 OCC Installment Lending：早期预警、vintage 与 roll-rate（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 贷前、账户管理、早期预警、催收与损失预测 |
| 2. 决策对象 | scorecard population、origination vintage、DPD state、collection queue |
| 3. 可选动作 | 修改 cutoff/strategy、调整账户管理、分配 collection workload；不规定单一动作 |
| 4. 预测量 | default/delinquency score、short-horizon loss、state migration |
| 5. 业务目标 | 风险/回报平衡、及时发现性能恶化、预测损失和资源需求 |
| 6. 约束条件 | 观察窗口、MOB 可比性、产品/客群差异、operational staffing |
| 7. 策略逻辑 | front-end population tracking + back-end actual/expected；roll rates 逐状态迁移 |
| 8. 模拟或实验设计 | development benchmark 对 production；季度 roll-rate 平滑；vintage curve comparison |
| 9. 模型指标 | score odds、population/characteristic stability、delinquency discrimination |
| 10. 业务指标 | approval/denial、bad/loss rate、MOB curve、balance migration、collection workload |
| 11. 监控方法 | 12/15/18 月 early-warning、chronology log、vintage/lagged/roll-rate reports |
| 12. 解释与审计 | cutoff/strategy change history、override、actual-vs-expected 与异常说明 |
| 13. 数据要求 | entity、origination time、observation time、state/DPD、balance、score、strategy version |
| 14. 可复现程度 | 公开了 roll-rate 算例；可用 synthetic state transitions 手算复现 |
| 15. 对 Sharper 的通用抽象 | `VintageAnalysis`、`RollRateAnalysis` 候选、MOB alignment、state transition matrix |
| 16. 不适合直接抽象 | 法定 charge-off timing、产品专用 cure program、具体催收动作 |

### 3.3 World Bank Credit Scoring Approaches Guidelines（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 评分开发、使用、决策解释和持续治理 |
| 2. 决策对象 | applicant/existing borrower 的评分及其 downstream decision |
| 3. 可选动作 | 指南不固定动作；强调 human-centric、review 与 correction avenue |
| 4. 预测量 | PD/delinquency risk 和可能的 EAD/LGD/ECL inputs |
| 5. 业务目标 | 风险评估、可解释透明、公平、数据责任和金融可及性 |
| 6. 约束条件 | 法律/伦理、隐私、数据质量、模型风险和组织能力 |
| 7. 策略逻辑 | 明确 score 如何进入流程；模型输出不能代替 decision rationale |
| 8. 模拟或实验设计 | 政策指南，不规定统一 cutoff 实验 |
| 9. 模型指标 | discrimination、robustness、performance within expectation |
| 10. 业务指标 | access、decision outcomes、complaints/appeals；无统一收益指标 |
| 11. 监控方法 | data/model governance、validation、ongoing performance review |
| 12. 解释与审计 | 数据来源、决策理由、review/correction、对监管者说明使用逻辑 |
| 13. 数据要求 | traditional/alternative data provenance、quality、purpose 和 outcome |
| 14. 可复现程度 | 原则公开，非实验，不提供实现代码 |
| 15. 对 Sharper 的通用抽象 | score/policy separation、provenance、reason/audit metadata、limitations |
| 16. 不适合直接抽象 | 国家法律判断、公平性结论、消费者通知文本和申诉流程 |

### 3.4 CGAP：score band 到 approve/review/reject（`B`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 贷前申请策略和运营集成 |
| 2. 决策对象 | 已评分的申请记录 |
| 3. 可选动作 | 自动批准、人工审查、拒绝 |
| 4. 预测量 | applicant PD/risk score |
| 5. 业务目标 | 在 growth goals 与 risk appetite 间选择可执行策略 |
| 6. 约束条件 | loan officer capacity、数据成熟度、IT integration、风险容忍 |
| 7. 策略逻辑 | 通过两个 cutoff 形成三段；中间段转人工而不是假设为正/负类 |
| 8. 模拟或实验设计 | score table/cumulative good-bad 分析，改变 cutoff 观察 approval 与 bad rate |
| 9. 模型指标 | predictive accuracy、score distribution、good/bad separation |
| 10. 业务指标 | approval rate、bad rate、manual-review volume、operational efficiency |
| 11. 监控方法 | model test/fine-tune、业务过程反馈、loan officer adoption |
| 12. 解释与审计 | scorecard logic、人工参与、process ownership |
| 13. 数据要求 | score、observed repayment、decision、manual review outcome、time |
| 14. 可复现程度 | 指南含示例表；可用 synthetic band/cutoff 重现 |
| 15. 对 Sharper 的通用抽象 | ordered bands、explicit boundary/tie policy、capacity、action distribution |
| 16. 不适合直接抽象 | 具体分数区间、risk appetite、贷款人员流程和机构文化 |

### 3.5 CFPB adverse-action reasons（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 申请拒绝、账户终止或不利条款变化后的通知 |
| 2. 决策对象 | 发生 adverse action 的具体申请或账户 |
| 3. 可选动作 | 软件研究只记录 decision/reason；不生成法律通知或判断合规 |
| 4. 预测量 | 可能使用 score/model factor，但 regulation 关注实际 action reasons |
| 5. 业务目标 | 给出与实际决策依据一致的 principal reasons |
| 6. 约束条件 | reason 必须源自实际 scored/considered factors；自动拒绝规则也要记录 |
| 7. 策略逻辑 | 分开 model factor、policy rule、final action；不能只说“未达 qualifying score” |
| 8. 模拟或实验设计 | 不适用；属于逐案 provenance 与通知要求 |
| 9. 模型指标 | 不规定 |
| 10. 业务指标 | notice coverage/accuracy 可作为内部 QA，但本文不定义合规率 |
| 11. 监控方法 | reason mapping、rule coverage、unmapped/fallback/override audit |
| 12. 解释与审计 | 保留实际因素、rule hit、priority、base/final action 和 override provenance |
| 13. 数据要求 | scored factors、rules evaluated/hit、decision version、reason mapping |
| 14. 可复现程度 | 法规解释公开；synthetic path audit 可复现，合规判断不可由库复现 |
| 15. 对 Sharper 的通用抽象 | `ReasonCode`/`PolicyAudit` 候选、model-vs-strategy reason 分层 |
| 16. 不适合直接抽象 | 法律适用性、通知措辞、reason 数量和消费者沟通 |

### 3.6 2026 Revised Guidance on Model Risk Management（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | model/decision component 全生命周期治理 |
| 2. 决策对象 | model、model use、overlay/adjustment 和受模型影响的业务流程 |
| 3. 可选动作 | approve use、limit use、overlay、adjust、recalibrate、redevelop |
| 4. 预测量 | 任意 material model output |
| 5. 业务目标 | 按 model exposure、purpose、materiality 管理风险 |
| 6. 约束条件 | 组织规模/复杂度、independence、third-party opacity、aggregate model risk |
| 7. 策略逻辑 | 风险与用途决定治理强度；正确模型也可能因 misuse 产生高风险 |
| 8. 模拟或实验设计 | validation、back-testing/outcome analysis、ongoing monitoring |
| 9. 模型指标 | 相对目标/用途的 performance threshold 与 deviations |
| 10. 业务指标 | model exposure、business impact、issue/remediation status |
| 11. 监控方法 | changes in product/exposure/client/data/market、limitations 与 deterioration |
| 12. 解释与审计 | inventory、documentation、recommendation/response/exception tracking |
| 13. 数据要求 | model purpose、owner、use、version、validation、monitoring、exceptions |
| 14. 可复现程度 | 原则公开；不提供单一实现或认证 checklist |
| 15. 对 Sharper 的通用抽象 | provenance、version、assumptions、limitations、challenge/status metadata |
| 16. 不适合直接抽象 | materiality judgment、组织角色批准、监管结论和 third-party validation |

### 3.7 EMP：利润驱动 cutoff（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 贷前模型选择和 cutoff 设定 |
| 2. 决策对象 | 有 PD/score 的申请 |
| 3. 可选动作 | grant/decline；论文不处理 manual review |
| 4. 预测量 | PD/rank score、exposure、LGD、operating income |
| 5. 业务目标 | 最大化期望利润而非 accuracy/AUC |
| 6. 约束条件 | 收入、损失和成本分布假设；概率校准与样本代表性 |
| 7. 策略逻辑 | 对 cutoff 枚举/求解，根据 expected profit 选择 operating point |
| 8. 模拟或实验设计 | 政府贷款数据上比较 EMP、accuracy、AUC 选模/选阈值 |
| 9. 模型指标 | AUC、classification measures |
| 10. 业务指标 | expected profit/monetary gain、optimal cutoff |
| 11. 监控方法 | 原论文重点不在持续 policy monitoring |
| 12. 解释与审计 | 明确成本收益假设和 cutoff；不提供逐规则 reason path |
| 13. 数据要求 | score/PD、observed outcome、exposure、LGD/income/cost assumptions |
| 14. 可复现程度 | 公式和实验公开；原始业务数据未完全公开时只能方法复现 |
| 15. 对 Sharper 的通用抽象 | generic outcome-state utility、expected payoff curve、assumption disclosure |
| 16. 不适合直接抽象 | 特定贷款现金流、LGD/EAD 定义、资金成本和利润口径 |

### 3.8 FICO xML + FinRegLab：模型理由与策略理由（`A/B`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | underwriting explanation、adverse-action reason workflow |
| 2. 决策对象 | individual score、模型贡献与策略拒绝原因 |
| 3. 可选动作 | 模型只预测；策略决定 action，reason layer 解释 final decision |
| 4. 预测量 | HELOC risk performance、局部/全局 feature contribution |
| 5. 业务目标 | 准确、可理解且与真实计算一致的解释 |
| 6. 约束条件 | sparsity/support、monotonicity、dependent features、reason taxonomy |
| 7. 策略逻辑 | strategic reasons 与 model-based reasons 分层，再映射到可用 taxonomy |
| 8. 模拟或实验设计 | HELOC 公开 challenge、可解释模型、局部 summary 和 case-based comparison |
| 9. 模型指标 | AUC、解释 sparsity/support、global/local consistency |
| 10. 业务指标 | reason coverage/consistency；没有公开真实 approval/loss impact |
| 11. 监控方法 | reason stability、mapping coverage、模型/策略版本变化 |
| 12. 解释与审计 | exact model contribution、rule source、taxonomy mapping、fallback |
| 13. 数据要求 | score/model explanation、rules、final action、source feature provenance |
| 14. 可复现程度 | HELOC 数据、论文和原型公开；实际 lender reason workflow 不公开 |
| 15. 对 Sharper 的通用抽象 | model reason 与 policy reason 分离；稳定 code + human label + source |
| 16. 不适合直接抽象 | 专有 FICO score、法律认可的 reason taxonomy、消费者文案 |

### 3.9 AAAI accountable approval rule optimization（`A`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 贷前 underwriting |
| 2. 决策对象 | applicant 与 rule-base approval decision |
| 3. 可选动作 | approve/decline；论文聚焦 rule set，不是人工审核工作流 |
| 4. 预测量 | PD/creditworthiness 以及规则覆盖/风险贡献 |
| 5. 业务目标 | 兼顾 risk 与 rule-base simplicity/accountability |
| 6. 约束条件 | 规则数量/复杂度、业务条件、可解释性和实际部署要求 |
| 7. 策略逻辑 | score 只是输入；用可解释规则组合形成最终 approval rule |
| 8. 模拟或实验设计 | 已部署应用研究，比较规则质量与基线 |
| 9. 模型指标 | prediction/rule quality；具体指标以论文为准 |
| 10. 业务指标 | approval rule usefulness/coverage；不作为 Sharper 收益基准 |
| 11. 监控方法 | 论文不是通用 monitoring contract |
| 12. 解释与审计 | 简洁 rule-base 可向 analyst/customer 说明 |
| 13. 数据要求 | applicant features、score/outcome、candidate rules、constraints |
| 14. 可复现程度 | 同行评审方法公开；业务数据/完整生产系统非公开 |
| 15. 对 Sharper 的通用抽象 | 分数与规则分层、规则 hit path、complexity/coverage metadata |
| 16. 不适合直接抽象 | 自动生成或优化规则、专有业务限制、生产审批 |

### 3.10 FICO 决策优化与催收案例（`C`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | acquisition、pricing、line management、collections |
| 2. 决策对象 | customer/account segment |
| 3. 可选动作 | 多个 offer/limit/contact/intensity；具体 actions 专有 |
| 4. 预测量 | score、response、action effect、collection/recovery outcomes |
| 5. 业务目标 | profit/risk/recovery/resource use |
| 6. 约束条件 | finite exposure、staff capacity、operating constraints |
| 7. 策略逻辑 | action-effect network + objectives/constraints + multi-scenario optimization |
| 8. 模拟或实验设计 | what-if/optimization；champion/challenger incremental testing |
| 9. 模型指标 | 未公开统一指标 |
| 10. 业务指标 | vendor-reported profit/recovery/efficiency improvement |
| 11. 监控方法 | strategy comparison 和 incremental challenger |
| 12. 解释与审计 | 产品声称可视化/策略导出；完整审计实现专有 |
| 13. 数据要求 | scores、actions、effect models、costs、constraints、outcomes |
| 14. 可复现程度 | 产品描述公开，模型/客户数据/算法不公开，低 |
| 15. 对 Sharper 的通用抽象 | scenario table、constraint feasibility、assumption sensitivity |
| 16. 不适合直接抽象 | action-effect optimization、自动额度/定价/催收、厂商 solver |

### 3.11 Experian PowerCurve / OneAZ（`C`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | application decision 与 customer lifecycle |
| 2. 决策对象 | applicant/account |
| 3. 可选动作 | approve/review/decline、line/precollections 等厂商模板动作 |
| 4. 预测量 | risk score、eligibility/policy inputs |
| 5. 业务目标 | approval/funding、loss、manual-review efficiency |
| 6. 约束条件 | decision criteria、data availability、operational review capacity |
| 7. 策略逻辑 | drag/drop rule flow、reusable score/segment、champion/challenger |
| 8. 模拟或实验设计 | test-and-learn、strategy performance mapped to executed strategy |
| 9. 模型指标 | 客户案例未完整披露 |
| 10. 业务指标 | 26% funding-rate increase、25% review decrease，均 vendor-reported |
| 11. 监控方法 | proactive strategy monitoring 与快速调整 |
| 12. 解释与审计 | rule/strategy trace；专有实现未公开 |
| 13. 数据要求 | input data、rules、decision、review status、outcome、strategy version |
| 14. 可复现程度 | 产品/案例公开，无数据和代码，低 |
| 15. 对 Sharper 的通用抽象 | rule trace、manual capacity、strategy version、offline comparison |
| 16. 不适合直接抽象 | 厂商模板、自动部署、实时 decision platform 和宣传收益 |

### 3.12 SAS expected-payoff collections 与 UOB（`B/C`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | account management、early warning、collections |
| 2. 决策对象 | delinquent customer/account |
| 3. 可选动作 | channel × treatment/action |
| 4. 预测量 | risk、channel propensity、payment/recovery amount |
| 5. 业务目标 | expected payoff、recovery、cost、resource use |
| 6. 约束条件 | channel cost、contact capacity、budget、customer segment |
| 7. 策略逻辑 | customer/action expected payoff 后按 segment 映射 action/channel |
| 8. 模拟或实验设计 | scenario simulation、champion/challenger；真实 causal design 未公开 |
| 9. 模型指标 | risk/propensity model performance，公开摘要不完整 |
| 10. 业务指标 | recovery/cost/ECL 改善为 vendor-reported |
| 11. 监控方法 | daily behavior score、risk segment、strategy performance/audit trail |
| 12. 解释与审计 | decision flow、rules、model version 和 audit trail |
| 13. 数据要求 | account history、risk、contact/payment outcome、channel cost、capacity |
| 14. 可复现程度 | 论文流程可方法复现；客户数据和产品执行不可复现 |
| 15. 对 Sharper 的通用抽象 | generic action-cost matrix、capacity constraint、observed action audit |
| 16. 不适合直接抽象 | propensity/action-effect、自动 channel optimization、实时催收编排 |

### 3.13 国内公开竞赛组合（`B/D`）

| 模板字段 | 提取结果 |
|---|---|
| 1. 业务阶段 | 多数为贷前违约预测，少量跨场景/新产品风险迁移 |
| 2. 决策对象 | applicant/user |
| 3. 可选动作 | 赛题通常不提供；提交物是 ranking score 或 event probability，不是真实 action |
| 4. 预测量 | default/overdue probability 或 risk rank |
| 5. 业务目标 | 竞赛 metric；背景文字可能提到 approval/loss，但未形成实验目标 |
| 6. 约束条件 | 不平衡、分布差异、匿名字段、小样本；业务 capacity/cost 通常缺失 |
| 7. 策略逻辑 | 特征工程、GBDT/ensemble、迁移/对抗验证；无完整 policy layer |
| 8. 模拟或实验设计 | CV/OOF 与 leaderboard；不是 controlled policy experiment |
| 9. 模型指标 | AUC 等竞赛指标 |
| 10. 业务指标 | 通常缺失；不能由 leaderboard 反推 approval/bad rate/profit |
| 11. 监控方法 | PSI/新旧分布在部分方案出现；缺少上线后 policy monitoring |
| 12. 解释与审计 | feature importance/业务解释有限；rule path/override/reason 多数缺失 |
| 13. 数据要求 | application/behavior/repayment tables；常无拒绝结果和 action logs |
| 14. 可复现程度 | 部分代码公开，数据获取/运行顺序不一，中等 |
| 15. 对 Sharper 的通用抽象 | score 输入、time/group leakage、drift benchmark；反证 policy 数据要求 |
| 16. 不适合直接抽象 | leaderboard tricks、字段清洗、AutoML/stacking、隐含“score=decision” |

## 4. 贷前、贷中、贷后与催收方法矩阵

| 方法 | 贷前申请 | 贷中账户管理 | 贷后监控 | 催收 | v0.2 结论 |
|---|---|---|---|---|---|
| score banding | 核心 | 行为分层可用 | cohort 切片 | DPD/risk segment 可用 | Task 17，通用 ordered bands |
| cutoff simulation | approval/review/reject | 只做静态 action replay | 监控旧/新 policy | 不做动作优化 | Task 17 |
| expected loss/payoff | 仅 event probability，假设驱动 | 同左 | actual-vs-expected | 只报告输入 action 的 payoff | Task 15 仅 loss primitives；Task 17 action payoff；ranking score 不可代入 |
| manual review capacity | 中间区间容量 | exception queue | review backlog/结果 | 非自动优化 | Task 17 |
| exposure/risk constraint | approval exposure | limit/exposure 只报告 | concentration | balance/cost 只报告 | Task 17；不优化额度 |
| champion/challenger | 离线同样本比较 | 可分析外部实验日志 | performance/stability | 只比较已提供策略 | Task 17 贷前、Task 18 预警、Task 19 模型与 frozen inventory |
| vintage/MOB | origination cohort | 同 MOB 比较 | 核心 | cure/charge-off cohort | Task 18 |
| roll-rate | 不适用 | state transition | 核心 | workload/loss input | Task 18；caller 定义 state |
| early warning | score/population shift | short-horizon behavior | drift/performance/rule/override | queue/state shift | Task 18 计算；Task 19 治理汇总 |
| reason codes | model + policy reasons | rule/action reason | mapping stability | observed action reason | Tasks 17/18 产生 trace，Task 19 汇总；非合规认证 |
| override audit | high/low-side 示例 | line/review exception | rate/trend/outcome | collector exception | Tasks 17/18 产生 base/final facts，Task 19 汇总 |
| dynamic pricing | 研究 | 研究 | 监控输入 | 不适用 | 长期 roadmap |
| automatic limit optimization | 不适用 | 研究 | 监控输入 | 不适用 | 长期 roadmap |
| collection action optimization | 不适用 | 不适用 | 状态输入 | treatment/channel optimization | 长期 roadmap |

## 5. 建模方法与策略方法的边界

| 层 | 回答的问题 | 合法输入 | 输出 | 不得混入 |
|---|---|---|---|---|
| prediction model | “怎样排序，或 positive event 概率是多少？” | features、target、train-only fitted state | ranking score 或 event probability + provenance、model explanation | action、业务批准、成本最优结论 |
| validation/evaluation | “分数在未见数据上是否可靠？” | y、score、fold/time/group provenance | discrimination、calibration、slice stability | 在 final test 上选 cutoff/policy |
| decision policy | “给定分数、规则和 bands，分配什么动作？” | frozen score、rules、bands、actions、precedence/fallback | deterministic proposed action 与 rule path | fit 模型、改标签、猜 score 方向、由 constraint 自动改动作 |
| business constraints | “冻结 action replay 是否满足调用者边界？” | frozen actions、capacity/exposure/budget/rate definitions | feasible/violated/unevaluable + gap | 生成/改写 action、solver 或自动 policy selection |
| strategy simulation | “候选 policy 在明确假设下会产生什么离线结果？” | policy outputs、constraints、cost/benefit、observed outcome/support | action mix、risk/loss/payoff curve、feasibility | 因果 action effect、拒绝样本真实结果 |
| post-loan early warning | “每个 as-of 时点产生什么 alert？” | prior-only signals、warning rules、mature horizon | alert state/episode 与 lifecycle metrics | 贷前 action、催收 action、因果效果 |
| governance/audit | “谁、为何、以哪个版本改变了什么？” | model/policy version、hits、override、reason mapping | trace、exceptions、monitoring evidence | 合规认证、审批授权或在线执行 |

模型 AUC 更高不保证策略 profit 更高；calibration、cost assumptions、constraints 和
action mix 会改变 operating point。反过来，策略离线收益更高也不证明模型更准确，
更不证明新动作造成了收益。所有 report 必须把 model estimate、policy replay、
assumption-based expectation、observed outcome 和 randomized experiment 分栏表达。

## 6. 通用策略方法论

### 6.1 分数到动作

1. 调用者显式给出 score column、score kind 和 direction；涉及 target/outcome 时还要
   给出 positive/event label。`ranking_score` 可为任意有限实数；
   `event_probability` 必须对应该 positive event 且位于 `[0,1]`。不得以 label 排序、
   列名、均值或经验猜“高分更高风险”，也不得把 margin 自动当概率。
2. 先执行无 score 的 eligibility/hard rules，再执行 score bands，再执行明确的
   tie/boundary/fallback 规则。priority、short-circuit 和 multi-hit 行为必须可审计。
3. 三段策略表示为两个有序边界和三个调用者动作，例如 approve/refer/decline；
   Sharper 不把这些名字或顺序设为默认业务语义。
4. missing/non-finite score、未命中规则、bands overlap/gap 必须稳定失败或进入调用者
   显式 fallback，不能静默批准或拒绝。

动作必须区分 `action_name` 与 `action_role`。名称完全由 caller 定义；通用角色至少
包括 `selected/rejected/review/request_information/limited/other`。没有显式 mapping
时只能报告 action distribution，不按 `approve`、`decline`、`refer` 等文本猜角色；
selection/approval、rejection、review、request-information、selected-population event
rate/exposure 和 capacity 指标只在所需角色已映射时计算。

### 6.2 cutoff 与 band simulation

对 caller 预先给定并有界的 cutoff/band grid 逐点输出：

- 每个 action 的 count/rate、score range 和 ties；
- observed-outcome coverage、事件数/率和有效样本量；
- exposure；仅在有合法 event probability 时输出 expected loss/payoff，仅在有成熟
  observed outcome 时输出 observed loss/payoff；
- manual-review demand 与 capacity gap；
- constraint pass/fail、violation magnitude 和 infeasible reason；
- reference policy 的 absolute/relative difference。

Task 15 只可在 train/validation/OOF 上按显式指标或 threshold-curve metric-only
guardrail 报告预声明候选的 analytical operating point；capacity、exposure、budget、
rate、action/cost 等业务约束只属于 Task 17。Task 15 不自动生成、部署或采用业务
cutoff。Task 17 只消费 caller 明确传入或已冻结的 cutoff/bands；final holdout 只做一次
冻结策略评估，不得看完 holdout 后修改 cutoff、cost、constraint 或 tie-break。

### 6.3 三种离线结果语义

| 模式 | 可计算 | 不可声称 |
|---|---|---|
| action-only replay | 无 event probability/observed outcome 时的 action mix、capacity、exposure、rule hits | expected/observed event rate、loss、profit |
| observed-outcome replay | 有 outcome 支持记录上的 observed event/loss/payoff | 被拒或不同动作下的反事实结果 |
| model-based expectation | 基于合法 event probability 和 utility assumptions 的 expected loss/payoff | 真实收益、action effect、因果改善 |

score kind 与 outcome availability 正交：`ranking_score + mature observed outcome` 可以
进入 observed-outcome replay；`event_probability` 可以进入 model-based expectation；
同一场景可以同时给出 observed 与 expected 分栏，但任一者都不能冒充另一个。

策略改变会改变未来可观察标签。历史拒绝样本通常没有 repayment outcome；把其 model
probability 当真实标签，或只在 historical accepts 上比较新 cutoff，都会产生选择偏差。
v0.2 必须披露 common support、outcome missingness 和 historical-policy dependence，
不实现 reject inference。

### 6.4 成本收益与约束

通用 utility 使用 outcome-state 表，而不是硬编码贷款公式：

```text
expected payoff(action_i)
  = sum_k P(outcome=k | row_i) * utility(action_i, outcome=k, row_i)
    - action_cost(action_i, row_i)
```

信用示例可把 expected loss 表达为 `PD × exposure × loss_fraction`，但 `PD`、exposure、
loss fraction、income、funding/operating cost 都必须由调用者提供并记录单位、时点、
缺失处理和假设。这里的 `PD` 必须是对应显式 positive event 的合法
`event_probability`；ranking score、decision margin 或未经声明的 `[0,1]` 值不能代入。
v0.2 只枚举/比较调用者给定的 policies，不做全局 optimizer。

通用约束候选包括：action count/rate 上下限、manual capacity、总 exposure 上限、
某 action 子集的 observed/expected event-rate 上限、budget 和分组 guardrail。约束必须
返回 feasible/violated/unevaluable，不能把缺失 outcome 当作 constraint passed。action
先由 caller rules/bands/precedence/fallback 冻结；constraint 只评价 scenario，不改写
action、不选择替代 policy，也不成为 optimizer。

### 6.5 champion/challenger

- **离线同样本比较**：同一 frozen row set、scores、costs、constraints 和 outcome
  support 上比较两个 deterministic policies；使用 paired action differences 和
  相同 denominator。
- **历史日志比较**：若 champion/challenger 分配不是随机，结果只描述 association；
  记录 assignment mechanism、时间、segment 和 common support。
- **真实在线实验**：需要预先设定 allocation、sample size、guardrail、stopping 和
  treatment exposure；Sharper v0.2 不分流、不执行实验，只可分析调用者提供的日志。

不得把 offline simulation 命名为 A/B test，也不得把 challenger 的 model metric
提升当作策略业务提升。

### 6.6 vintage、MOB 与 roll-rate

- vintage 由 caller 指定 cohort/origination time；MOB 是 observation time 与 cohort
  anchor 的显式周期差，period unit、partial period 和 timezone 行为必须冻结。
- 比较必须按相同 MOB 对齐，披露 cohort size、matured/eligible denominator、censoring
  和 outcome horizon；不能用未成熟 vintage 的低坏账率与成熟 cohort 直接比较。
- roll-rate 是 caller-defined states 在相邻 observation periods 的 count/weight
  transition matrix；不得硬编码 current/30/60/90/charge-off。
- stay、worsen、cure、missing/exit 的定义和 state order 必须显式；结果记录 transition
  denominator、权重、窗口和 skipped states。

### 6.7 rules、override 与 reason codes

每行审计至少需要：policy version、rules evaluated、rules hit、priority/order、base
action name/role、final action name/role、override flag/type、reason code source、score/
model version 和 warnings。reason 必须区分：

- model-based：解释风险分数的 feature contribution；
- policy-based：解释为何规则/band/precedence 形成某 action；
- constraint-based：解释 frozen scenario 为何 feasible/violated/unevaluable，不解释或
  改写 action 的形成；
- override-based：解释 base action 为何被改写。

`ReasonCode` 候选只能提供稳定 machine code、human label、source/provenance 和
mapping status；它不生成辖区法律文本，也不保证满足 adverse-action requirements。

### 6.8 通用价值与信用专有部分

| 通用、可进入 Sharper | 信用专有、仅作示例或长期研究 |
|---|---|
| score direction、rules、bands、actions、ties | approve/decline 的法律含义 |
| generic outcome utility、action cost、capacity | PD×EAD×LGD、APR、funding cost 的具体口径 |
| constraint feasibility、scenario comparison | risk appetite、资本、pricing 与额度政策 |
| champion/challenger offline comparison | controlled credit exploration 与 reject inference |
| cohort age、state transition、vintage | MOB/DPD/charge-off 的产品/法规定义 |
| reason provenance、rule path、override audit | adverse-action 通知和合规结论 |

## 7. Sharper 候选概念评估

以下名字只是研究词汇，不冻结 public API、字段或模块。Tasks 16–19 的独立决策记录
必须重新判断是否需要公开，优先合并而不是把每个名词都变成 class。

| 候选概念 | 通用价值 | 最小建议 | 主要风险 |
|---|---|---|---|
| `DecisionPolicySpec` | 高 | 一个有版本、方向、规则/band/action 顺序的具名 spec 候选 | 变成通用规则引擎或配置 DSL |
| `DecisionRule` | 高 | 只引用 Task 16 private closed condition vocabulary；可作为 spec 内部元素 | 任意 callable/表达式执行、安全与不可复现 |
| `DecisionBand` | 高 | 明确 bounds、closure、action 和 tie policy | overlap/gap、方向与边界歧义 |
| `PolicyOutcome` | 中 | 优先考虑稳定 row-level DataFrame schema，不急于逐行 dataclass | 大结果内存、混入反事实结果 |
| `StrategySimulation` | 高 | summary + bounded details + assumptions/limitations 的 frozen result 候选 | 把 expectation 写成真实 profit |
| `ConstraintSpec` | 高 | 封闭 count/rate/sum/metric guardrail 候选 | 发展成 optimization language |
| `CostBenefitSpec` | 中高 | generic outcome-state utility 与单位/时点 metadata | 硬编码信用现金流或单位混乱 |
| `ChampionChallengerResult` | 高 | paired offline differences + support/provenance | 暗示 causal/A-B 结论 |
| `VintageAnalysis` | 高 | cohort × age 的 count/rate/value result 候选 | censoring/MOB 定义不清 |
| `RollRateAnalysis` | 高 | caller state order 下的 count/weight transition result 候选 | 硬编码 DPD bucket、忽略 cure/exit |
| `ReasonCode` | 中 | 可先用稳定 table schema；若公开 class 必须极小 | 被误解为合规 reason generator |
| `PolicyAudit` | 高 | bounded rule path/override/reason coverage result 候选 | 保存 PII、全量无界 traces |

最小 public surface 不应同时暴露十二个顶层 constructors。更小的候选组合是：一个
policy specification、一个 simulation entry point、两个 lifecycle analyses、一个
policy comparison/audit entry point及其具名结果。精确 API 留给 Task 决策记录。

## 8. 对 v0.2 的研究建议

1. Task 15 先冻结 ranking-score/event-probability、positive-label/direction、label
   maturity、OOF/time/group validation、风险与 business metric 的数学语义；只报告
   预声明 threshold 候选的 analytical operating point，不训练新模型或采用业务 cutoff。
2. Task 16 审计 score/outcome/action availability、historical action-assignment dependence、
   post-outcome fields、entity/time leakage 和策略输入质量，并唯一拥有 private closed
   condition kernel 与 missingness drift。
3. Task 17 做 pre-loan eligibility rules 与 decision strategy simulation：规则语义、
   caller-frozen bands、action-name/role mapping、cost/constraint、准入回测和贷前 policy
   comparison；不执行审批。
4. Task 18 做 post-loan early-warning rules 与 lifecycle monitoring：observation-time
   signals、alert episodes/backtest、vintage/MOB、roll-rate/cure；不优化催收 action。
5. Task 19 消费 Task 15 frozen model results 做 model comparison，并消费 Task 16
   input/missingness evidence、Task 17 pre-loan comparison 与 Task 18 warning comparison
   的 frozen results，做 explanation、comparison inventory、override/provenance 和
   governance；不得重新执行 rules、backtest 或 missingness drift。
6. Task 20 才集成 opt-in workflow、static report 与 CLI；Task 17/18 的 CLI 载体只允许
   versioned closed JSON policy/warning spec，不建立通用 DSL；默认 v0.1 不变。

为控制范围，原 v0.2 草案中的新增 tree model families、supervised binning、WOE、
target encoding 和通用 learned group-aggregate transformer 应提议延期。v0.1 已支持
caller-supplied estimator，v0.2 可以消费外部 score/model result 验证策略；策略研究
不证明必须新增 estimator dependency 或训练 API。

## 9. 长期研究而非 v0.2 实现

- 动态风险定价和自动额度优化；
- action-effect、response、treatment-effect 或因果模型；
- uplift modeling 与 controlled exploration；
- collections treatment/channel 自动优化；
- reject inference 算法；
- 多期动态规划、reinforcement learning 和 sequential policy；
- 实时规则/决策引擎、在线 experiment allocation 和自动审批；
- 监管合规认证、法律 reason generation、fair-lending certification。

vendor adapter、规则 DSL、通用 solver 与 server/dashboard 不因本研究进入长期 roadmap；
若未来另行提出，必须重新 research/roadmap review，不能把本节视为实现授权。

## 10. 研究结论

风险模型与决策策略必须是两个可独立版本、验证和审计的对象。v0.2 有通用价值的
最小闭环是：显式 score 语义 → deterministic rules/bands/actions → assumption-aware
offline simulation → constraints/coverage → lifecycle outcomes → reason/override audit
→ result-only report。

该闭环可以回答“若在这批已知数据和这些假设上应用此策略，会得到怎样的动作分布、
可观察风险和期望 payoff”，但不能回答“该策略上线后一定产生多少利润”“被拒者实际
会怎样”“某动作造成了何种结果”或“该决策满足监管要求”。这些限制必须进入结果、
报告和 roadmap release gate。
