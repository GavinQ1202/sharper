# Kaggle 信用风险方法调研

## 1. 文档身份与研究边界

本文是 Sharper v0.2 roadmap 的研究输入，不是 public API 合同，也不授权实现。
研究基线为 Sharper `0.1.0` 稳定提交 `0b86986`，资料检索截止日为
2026-07-31。

本文研究“数据如何形成可验证的风险分数”。分数如何进入规则、动作、成本、约束、
vintage/roll-rate 和策略审计，见
`docs/research/credit-risk-decision-strategies.md`；贷前准入规则和贷后预警/alert
lifecycle 的详细边界见 `docs/research/credit-policy-and-early-warning.md`。三份研究共同输入
`docs/decisions/v02-roadmap-contract.md`。

研究对象固定为：

1. Give Me Some Credit；
2. Default of Credit Card Clients；
3. Home Credit Default Risk；
4. American Express – Default Prediction；
5. Home Credit – Credit Risk Model Stability；
6. Credit Risk Dataset；
7. Loan Default Prediction Dataset。

本文只提取可泛化到结构化表格分析工具的方法。竞赛分数、排名和投票数是来源
选择信号，不是 Sharper 功能的验收目标。任何贷款审批、收益优化、监管判断、
公平性结论或业务阈值都不从 Kaggle 结果外推。

### 1.1 证据标签

为避免把教程、榜单方案和项目建议混在一起，全文使用以下标签：

| 标签 | 含义 | 可以支持的结论 |
|---|---|---|
| `EDA` | 高点赞或被广泛复用的 EDA/tutorial notebook | 可读工作流、常见清洗、基础特征和基线模型 |
| `LB` | 明确标注排名的赛后 write-up 或官方获奖访谈 | 高 leaderboard 方案实际采用的策略 |
| `CODE` | 可检查的公开 solution/reproduction code | 代码中直接出现的数据流、特征、模型和资源策略 |
| `DATA` | 竞赛页、数据卡或原始数据仓库 | 数据形态、字段说明、规模、目标与官方指标 |
| `OBS` | 本文从上述来源直接观察到的做法 | 只描述来源实际展示的行为，不表示最佳实践 |
| `REC` | Sharper 根据多来源抽象出的建议 | 项目设计判断，不能反向归因给某个 notebook |

“高点赞”只按检索时可见的 Kaggle 页面快照判断；票数会变化。“高排名”只在
来源明确给出排名时使用。GitHub 仓库若未证明其榜单名次，只归为 `CODE`，不称为
获奖方案。

### 1.2 选择标准

来源按以下顺序选择：

1. Kaggle 官方竞赛页、官方指标页、赛后获奖 write-up 或官方访谈；
2. 高点赞、可复用且覆盖完整流程的 Kaggle notebook；
3. 获奖团队或作者公开的可检查代码；
4. 原始数据发布方，例如 UCI；
5. 为补齐代码可复现性而选取的公开复现仓库。

同一赛题优先同时保留 `EDA`、`LB` 和 `CODE` 三类证据。若对象只是 Kaggle
dataset 而不是正式竞赛，本文明确记录“无官方 leaderboard 证据”，不会用普通
notebook 的分数代替获奖名次。

### 1.3 研究限制

- Kaggle 页面是动态页面，部分 write-up 在未登录检索中只能确认标题、排名和
  链接，无法检查正文；这种来源不用于归纳未直接看到的实现细节。
- Default of Credit Card Clients、Credit Risk Dataset 和 Loan Default
  Prediction Dataset 在本文采用的 canonical identity 下主要是数据集/教学基准，
  不是具有可比官方 leaderboard 的 featured competition。
- 竞赛代码常允许 train/test 联合转换、榜单融合和依赖特定字段的技巧；这些做法
  可以作为 `OBS`，但只有满足 split-first、训练态隔离和通用字段契约时才可能转为
  `REC`。
- 多表关系型建模的研究价值被保留，但 Sharper v0.2 不实现多表关系引擎；相关
  benchmark 只能消费外部预聚合的单表。

## 2. 来源清单与证据等级

### 2.1 Give Me Some Credit

- `DATA`：[Kaggle competition](https://www.kaggle.com/competitions/GiveMeSomeCredit)。
- `EDA`：[Starter: Give Me Some Credit](https://www.kaggle.com/code/mostig/starter-give-me-some-credit)，
  检索快照为 46 votes，private score 0.86807。
- `LB`：[Kaggle 官方对第三名 Joe Malicki 的访谈](https://medium.com/kaggle-blog/credit-where-credits-due-joe-malicki-on-placing-third-in-give-me-some-credit-161e3a2d5661)。
- `CODE`：[DrIanGregory/Kaggle-GiveMeSomeCredit](https://github.com/DrIanGregory/Kaggle-GiveMeSomeCredit)。

官方访谈直接给出了分层采样、不同正负样本比例的 boosting/random forest、模型
平均、领域特征以及“依赖大样本交叉验证而不是 public leaderboard”的经验。公开
复现代码直接展示了 96/98 特殊值、年龄 0、收入/家属缺失、约 6.7% 正类和多模型
AUC 对比。

### 2.2 Default of Credit Card Clients

- `DATA`：[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)，
  30,000 行、23 个特征、六个月还款/账单/支付宽表，无声明缺失值。
- `DATA`：[Kaggle mirror](https://www.kaggle.com/datasets/tunguz/default-of-credit-card-clients-data-set)。
- `CODE`：[CC-Default-Prediction-Customer-Segmentation](https://github.com/datascisteven/CC-Default-Prediction-Customer-Segmentation)。

UCI 原始说明强调预测违约概率比只输出可信/不可信标签更有风险管理价值。公开代码
比较了多类基线、特征选择、欠采样/过采样/混合采样，并同时检查 accuracy、
ROC-AUC、precision、recall。该 canonical 数据集没有可用于“获奖方案”归纳的
官方 Kaggle leaderboard。

### 2.3 Home Credit Default Risk

- `DATA`：[Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk)。
- `EDA`：[Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction)。
- `EDA`：[Home Credit Risk with Detailed Feature Engineering](https://www.kaggle.com/code/mathchi/home-credit-risk-with-detailed-feature-engineering)，
  检索快照为 112 votes、private score 0.79297。
- `LB`：[1st place write-up](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821)。
- `LB` + `CODE`：[KazukiOnodera/Home-Credit-Default-Risk，2nd place](https://github.com/KazukiOnodera/Home-Credit-Default-Risk)。
- `LB` + `CODE`：[Cirice/4th-place-Home-Credit-Default-Risk](https://github.com/Cirice/4th-place-Home-Credit-Default-Risk)。
- `CODE`：[js-aguiar/home-credit-default-competition，7th place code](https://github.com/js-aguiar/home-credit-default-competition)。

第二名公开代码明确列出：主申请表的 delta/ratio，多张历史表的全历史与
1/2/3 年窗口聚合、first/last，以及再聚合特征。第四名代码明确记录大量 OOF、
一级 stacking 和 blending。后二者是竞赛观察，不是 v0.2 对大规模 stacking 的
建议。

### 2.4 American Express – Default Prediction

- `DATA`：[Kaggle competition](https://www.kaggle.com/competitions/amex-default-prediction)。
- `EDA`：[AMEX LightGBM Quickstart](https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart)。
- `LB`：[1st solution write-up](https://www.kaggle.com/competitions/amex-default-prediction/discussion/348111)。
- `LB` + `CODE`：[jxzly/Kaggle-American-Express-Default-Prediction-1st-solution](https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution)。
- `CODE`：[fintech-quagga-group/american-express-default-prediction](https://github.com/fintech-quagga-group/american-express-default-prediction)。

一等奖代码直接展示了：客户级 mean/std/min/max/sum/last、相邻期 diff、最近
3/6 期窗口、实体内 rank、月份横截面 rank、5-fold OOF、LightGBM DART、神经
网络与加权融合；也使用 Feather、中间特征文件和多进程减轻大数据处理成本。
官方指标是 normalized Gini 与 top 4% default capture 的均值。

该代码还在部分特征阶段合并 train/test。本文只把它作为竞赛 `OBS`；Sharper
不得通过 train/test 合并学习类别、rank、聚合、分箱或其他状态。

### 2.5 Home Credit – Credit Risk Model Stability

- `DATA`：[Kaggle competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)。
- `EDA`：[Home Credit Baseline](https://www.kaggle.com/code/greysky/home-credit-baseline)，
  检索快照约 1,400 votes。
- `LB`：[Yuuniee 1st place — My Betting Strategy](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337)。
- `CODE`：[Home Credit baseline training](https://www.kaggle.com/code/andreynesterov/home-credit-baseline-training)。
- `CODE`：[zulqarnainalipk/Home-Credit---Credit-Risk-Model-Stability](https://github.com/zulqarnainalipk/Home-Credit---Credit-Risk-Model-Stability)。
- `OBS`：[关于 weekly Gini slope 与 regime dependence 的公开讨论](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/505852)。

一等奖来源可直接确认排名与方案身份，但公开索引未提供足够正文，因此本文不向它
归因具体模型细节。公开代码直接显示多 Parquet 文件读取、类型转换、内存优化、
不同 data depth 的聚合、LightGBM/CatBoost 与 voting。竞赛讨论直接区分 concept
drift、covariate shift 和模型决策引起的 user adaptation，并指出短期 weekly
Gini slope 可能受非平稳 regime 影响。

### 2.6 Credit Risk Dataset

- `DATA`：[laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)，
  检索快照包含 148 个公开 notebooks。

本文固定使用该 identity，而不是其他同名数据集。它是单表混合类型教学数据，包含
年龄、收入、就业时长、贷款用途/等级、金额、利率、收入占比、历史违约和信用历史
长度。它没有 featured competition 的官方 leaderboard，因此只承担通用单表质量、
类别处理、比例特征和解释性 benchmark，不提供“获奖方案”证据。

### 2.7 Loan Default Prediction Dataset

- `DATA`：[nikhil1e9/loan-default](https://www.kaggle.com/datasets/nikhil1e9/loan-default)，
  255,347 行、18 列、CC0，检索快照为 149 votes、56 个公开 notebooks。

本文固定使用该 identity，避免与其他同名合成数据或旧竞赛混淆。它是较大的静态
单表，适合检查内存、混合类型预处理、类别不平衡和可复现运行；没有本文可验证的
官方 leaderboard/获奖方案。

## 3. 按数据形态归纳

| 数据形态 | 主要对象 | 直接观察到的核心难点 | Sharper 可泛化抽象 |
|---|---|---|---|
| 单表静态数据 | Give Me Some Credit、Credit Risk Dataset、Loan Default Prediction Dataset | 特殊编码、长尾/极值、缺失、类别不平衡、ID、混合类型 | 显式 target/positive label、规则化质量审计、风险/业务指标、冻结 score 输入 |
| 多期宽表 | Default of Credit Card Clients | 同一语义按月份展开，近期与历史状态并存，字段顺序携带时间信息 | 调用者声明有序列组；做 recent/first/last/delta/trend/窗口的描述性 lifecycle 摘要，不猜字段名 |
| entity-event longitudinal data | American Express | 一实体多行、观测长度不等、最近窗口、diff/rank、实体不能跨 fold | 显式 entity/time/cutoff；point-in-time 描述；group/time split；vintage/roll-rate |
| 多表关系型数据 | Home Credit Default Risk、Home Credit Model Stability | 一对多和二层关系、聚合窗口、key/cardinality、特征爆炸 | v0.2 只消费外部预聚合单表；关系发现与 join engine 继续留在 v0.3 |
| 时间漂移与稳定性数据 | Home Credit Model Stability；AmEx 的月份横截面仅作辅助观察 | 时间切片性能变化、协变量变化、regime、样本量波动 | reference/current 两数据集 drift；time-bucket 指标；趋势/波动与限制披露 |

`REC`：数据形态必须由调用者通过通用参数声明。Sharper 不识别
`customer_ID`、`SK_ID_CURR`、`S_2`、`WEEK_NUM`、`TARGET` 等特定名字，也不根据
正则表达式猜测信用语义。

## 4. 每个对象的核心方法

### 4.1 Give Me Some Credit

`OBS`：

- 先检查目标比例、缺失率、数值分布和逻辑异常；96/98 等特殊值不能当普通连续值；
- 收入缺失和异常 DebtRatio 联动，说明“缺失本身”及缺失子群值得单独检查；
- 使用 stratified sampling/CV，比较 logistic regression、random forest、GBDT、
  XGBoost；第三名对不同采样比例的模型做简单平均；
- 领域构造以 ratio、债务/收入、家属/可支配收入等可解释变换为主；
- private leaderboard 与本地大样本 CV 更一致，public leaderboard 不应充当调参集。

`REC`：特殊值规则必须由调用者提供或由通用分布审计提示，不能硬编码 96/98；
不平衡问题优先通过 PR-AUC、gains/lift、class/sample weight 和阈值曲线评估，采样
只能发生在训练 fold 内。

### 4.2 Default of Credit Card Clients

`OBS`：

- 六个月 repayment status、bill amount、payment amount 构成有序宽表；
- 原始数据无声明缺失，但仍有类别编码、ID 和时间顺序解释问题；
- UCI 研究目标强调 probability estimation，公开代码表明不平衡处理可能提高
  ROC-AUC/precision/recall，而不一定提高 accuracy；
- 常见建模比较覆盖 logistic、tree、random forest、AdaBoost/GBDT/XGBoost，
  但 calibration 与时间外推在教程中经常缺位。

`REC`：多期宽表需要显式 ordered column groups，并保留列到时间位置的 provenance；
模型评估同时报告 discrimination、probability-only calibration 和 analytical
operating-point 结果，不能用 accuracy 替代概率质量；分析点不是业务 cutoff。

### 4.3 Home Credit Default Risk

`OBS`：

- 主申请表与 bureau、previous application、POS cash、installments、credit card
  等一对多历史表形成关系型数据；
- 高排名公开代码大量使用 ratio/delta、first/last、count/mean/min/max/sum 和
  最近 1/2/3 年窗口；
- LightGBM 是主要强基线；高排名团队通过 OOF、多组特征和模型多样性融合；
- 关系聚合、特征数量和中间文件会显著增加内存与复现成本。

`REC`：v0.2 不复制竞赛的一整套关系型 join/stacking。外部预聚合数据可以验证
质量、指标、模型和稳定性；point-in-time 多表聚合、key/cardinality 合同和二层
关系引擎继续归 v0.3。

### 4.4 American Express – Default Prediction

`OBS`：

- 客户月度事件先压缩为客户级特征：mean/std/min/max/sum/last、观测数、类别
  nunique/one-hot summary；
- temporal information 通过最近 3/6 期、相邻期 diff、最后一期、实体内 rank
  和月度横截面 rank 表达；
- 以 group-aware folds 产生 OOF，使用 LightGBM 和其他模型融合；
- Feather/Parquet、压缩 dtype、中间结果缓存、分批/多进程是大数据可运行的关键；
- 竞赛指标直接奖励排序与有限资源下的 top-percent capture，而不是固定阈值 accuracy。

`REC`：v0.2 应先实现单表 entity-event 的有界、point-in-time 聚合和 group/time
validation；不实现神经网络、序列 Transformer 或大规模融合。任何横截面 rank
或类别词表都只能在训练分区/训练时点内拟合。

### 4.5 Home Credit – Credit Risk Model Stability

`OBS`：

- 数据同时具有多表、一对多 data depth、极大行数和时间稳定性目标；
- notebook/code 采用 Parquet、多文件读取、类型缩减、分层聚合和 boosting；
- 官方任务不是只追求全局 AUC/Gini，还惩罚随周次的性能退化；
- 公开讨论提醒：性能趋势可能混合 sampling noise、covariate shift、concept drift
  和 regime effect；单一 slope 不能自动成为因果或监管结论。

`REC`：Sharper 提供通用 reference/current drift 和 time-bucket performance，记录
每个切片样本量、事件率、指标、趋势和波动；不复刻特定 Kaggle stability score，
不声明识别 concept drift 的原因，更不实现 reject inference。

### 4.6 Credit Risk Dataset

`OBS`：

- 小型单表包含 numeric、categorical、比例、等级、历史状态与 ID-like 信息；
- 字段存在明显逻辑约束候选，例如年龄/就业时长、贷款金额/收入、利率和贷款等级；
- 数据卡有大量教学 notebooks，但没有统一 leaderboard 方案可比较。

`REC`：把它作为 caller-supplied range/rule、混合类型预处理、类别未知值、解释性
输出和静态 drift 的便携 benchmark；不把字段名写进库。

### 4.7 Loan Default Prediction Dataset

`OBS`：

- 25 万级单表适合暴露整表复制、object dtype、无界 one-hot 和不可重复 split 的
  成本；
- 数据卡强调二分类教学，但没有获奖方案或官方 metric 证据。

`REC`：把它作为离线规模 benchmark，检查峰值内存、运行时、确定性和未知类别；
不以公开 notebook accuracy 或任一随机 split 的分数作为 release gate。

## 5. 通用方法矩阵

下表中的“核心”表示 v0.2 应直接支持，“适用”表示由显式元数据触发，“外部”表示
v0.2 只消费已准备好的结果，“审计”表示只提示风险而不自动修复。

| 方法 | 单表静态 | 多期宽表 | entity-event | 多表关系型 | 时间漂移/稳定性 | v0.2 结论 |
|---|---|---|---|---|---|---|
| 数据质量 | 核心 | 核心 | 核心 + 实体/时间约束 | 外部扁平表 | 核心 + 切片 | Task 16 |
| 缺失与异常值 | 缺失率、特殊值、range | 按期缺失轨迹 | 观测长度、窗口缺失 | 外部聚合缺失 | 缺失率 drift | Task 16 唯一计算；Task 19 只治理汇总 |
| 类别不平衡 | 核心 | 核心 | 实体级事件率 | 外部 | 按时间切片事件率 | Task 15 |
| validation | stratified | stratified/time-aware | group/time-aware | 外部扁平表 | rolling/forward | Task 15 |
| group/time leakage | ID/entity 审计 | 列时序声明 | 核心阻断 | key/cutoff 外部保证 | 核心阻断 | Tasks 15/16/18 |
| entity aggregation | 不适用 | 可选列组摘要 | 核心 | 外部 | 按 cutoff | Task 18，限 point-in-time/lifecycle |
| temporal features | 显式日期组件 | recent/delta/trend | windows/recency/slope | 外部 | time bucket | Task 18，限 descriptive/lifecycle |
| 模型 | v0.1 logistic 或外部 score | 同左 | 同左 | 外部扁平后建模 | 同一 validation plan 比较 | v0.2 消费既有/外部 score；新增模型族延期 |
| calibration | 仅 event probability | 同左 | group-safe | 外部 | 按时间复核 | Task 15；Task 19 做 performance governance |
| thresholding | 预声明候选的 validation/OOF 诊断 | 同左 | 同左 | 外部 | 阈值稳定性只报告 | Task 15 报告 analytical candidate；Task 17 只消费 caller-frozen action cutoff |
| Gini | 核心，明确正类/方向 | 核心 | 核心 | 外部 | 切片 Gini | Task 15/19 |
| KS | 核心，披露 ties/n | 核心 | 核心 | 外部 | 切片 KS | Task 15/19 |
| PR-AUC | 不平衡首选摘要 | 核心 | 核心 | 外部 | 切片 PR-AUC | Task 15/19 |
| gains/lift | 核心 | 核心 | 实体级排序 | 外部 | 切片并记录基准率 | Task 15/19 |
| explainability | 系数/置换/原生重要性 | 同左 + 列组 provenance | 聚合特征 provenance | 外部 provenance | 重要性稳定性 | Task 19 |
| stability/drift | reference/current | 按期比较 | 按时点/实体 cohort | 外部扁平 | 核心 | Task 16 计算 input/missingness evidence；Task 19 计算 prediction/performance 并治理汇总 |
| memory-efficient processing | 有界复制/dtype | 列预算 | 排序/窗口预算 | 不实现 join engine | 分片结果表 | 全任务约束 |
| reproducibility | seed、row positions | 列组顺序 | fold/cutoff metadata | 外部 manifest | 时间桶边界 | 全任务约束 |

## 6. 横向比较与产品抽象

### 6.1 数据质量、缺失和异常值

`OBS`：教程 notebook 普遍先做 missing rate、分布、相关和特殊值检查；GMSC 的
96/98、年龄 0、收入缺失是“数值 dtype 不等于连续测量”的典型证据。多期/事件
数据还会出现观测长度不等、某期整体缺失和 last-value 语义。

`REC`：v0.2 增加声明式、通用的 range/special-value/availability rules 与按
target/time/entity 的质量切片。所有规则产生结构化 issue、有效样本量、分母和
限制；不自动清洗、不用字段名推断业务含义。

### 6.2 类别不平衡和 validation

`OBS`：GMSC 使用分层抽样；AmEx 使用实体级 folds；Home Credit Stability 说明
时间稳定性不能由随机 holdout 代表。公开教程大量尝试重采样，但常没有证明采样只在
训练 fold 内。

`REC`：v0.2 的顺序是先选择与数据生成过程匹配的 stratified/group/time fold，
再在训练 fold 内 fit preprocessing、resampling/weighting、calibration 和 estimator。
time fold 不能只保证 observation time 单调：训练标签还必须在 fold cutoff 前成熟，
显式记录 `outcome_end_time`/`label_available_time` 与 reporting delay。OOF/validation
只比较调用者预声明的 threshold 候选；最终 holdout/test 只使用一次，分析候选不会
自动成为业务 policy。

### 6.3 entity aggregation 和 temporal features

`OBS`：高排名方案的稳定模式是 count/mean/std/min/max/sum/first/last、最近窗口、
diff、ratio、recency 和 trend，而不是依赖某个信用字段名字。

`REC`：这些模式继续作为 point-in-time 与 lifecycle 分析依据，但修订后的 v0.2
优先路线只在 Task 18 承载显式 entity、event time、cutoff、window、ordered column
group、vintage/MOB 和 state transition 所需的有界描述性计算。supervised binning、
WOE、target encoding 和通用 learned group-aggregate transformer 提议延期。输出仍
必须保存 source columns、窗口、排序、requested/actual budget 和 skipped reason；
任一事件时间晚于 cutoff 都不得进入分析。

### 6.4 模型、校准和阈值

`OBS`：logistic regression 是解释性基线，GBDT/LightGBM/CatBoost/XGBoost 是
竞赛强基线；高榜单融合很常见。教程与 leaderboard 更常优化排序指标，calibration
和阈值的训练/验证隔离较少被完整记录。

`REC`：修订后的 v0.2 优先消费 v0.1 caller-supplied estimator 或外部系统产生的
冻结分数，不新增模型族；有限树基线提议延期。任意有限 `ranking_score` 在显式方向
下可用于 ROC-AUC/Gini、PR、KS、gains/lift、bands 和 cutoff 诊断；只有明确对应
positive label 且位于 `[0,1]` 的 `event_probability` 才能用于 Brier/log loss、
calibration 和 expected loss。`decision_function` margin 不得自动解释为概率。概率
calibrator 只能在训练内层拟合；Task 15 仅在 validation/OOF 上比较调用者预声明候选
并按 threshold-curve metric-only guardrail 报告 analytical operating point；业务
constraints 只属于 Task 17，且 Task 17 只接受调用者另行冻结的 decision cutoff。

### 6.5 explainability、stability 和 drift

`OBS`：公开方案常使用模型重要性筛选或解释特征，但重要性会随 fold、时间和相关
特征变化。Home Credit Stability 的公开讨论显示“指标下降”不等于已识别漂移原因。

`REC`：v0.2 提供系数、原生 importance 和 holdout/OOF permutation importance，
并保留 transformed feature 到 source feature 的 provenance。Task 16 唯一计算
reference/current input/missingness profile evidence；Task 19 只消费它，并计算 prediction
drift、时间/分组性能和 importance stability。输出是诊断，不是因果、公平性或监管
结论。

### 6.6 内存效率与可复现性

`OBS`：AmEx 和 Home Credit Stability 代码通过 Feather/Parquet、中间结果、dtype
缩减、分批/多进程处理大数据；高排名代码也常依赖大量缓存和手工运行顺序。

`REC`：v0.2 不新增 chunked I/O 或分布式引擎，但每个 API 必须有列/特征/窗口/切片
预算，避免隐式深拷贝、train/test 拼接和无界笛卡尔组合。结果记录 seed、fold、
row positions、时间边界、样本量、预算和 limitation；benchmark 记录 wall time 与
peak memory，但不以特定机器的绝对值作为 CI 门禁。

## 7. 对 v0.2 的功能输入

研究结论按优先级收敛为六组能力：

1. binary risk validation and metrics：显式正类、ranking-score/event-probability 角色、
   label maturity、stratified/group/time validation、OOF、Gini、KS、PR、gains/lift、
   probability-only calibration 和预声明 threshold 候选分析；
2. data quality and leakage audit：通用规则、实体/时间重叠、重复、特殊值、按切片
   缺失和可用性审计；
3. pre-loan eligibility rules and decision strategy simulation：执行 caller-defined
   准入规则，可选结合冻结分数、costs/constraints 做离线动作回放与规则回测；
4. post-loan early warning and lifecycle monitoring：在 entity × observation time 上
   做 prior-only signals、alert backtest、vintage/MOB、roll-rate/cure；
5. explainability, champion/challenger and governance：Task 19 做模型比较，并只消费
   Task 17 贷前比较和 Task 18 预警比较的 frozen results，汇总 reason provenance、
   override/policy audit、drift 与 stability；
6. integration and release readiness：新的 opt-in workflow/result/report/CLI 路径，
   保持 v0.1 默认行为和 no-recomputation/Figure ownership。

具体任务边界、依赖顺序和验收条件由
`docs/decisions/v02-roadmap-contract.md` 冻结。

为给新增策略闭环留出可验收边界，新增 tree model families、supervised binning、
WOE、target encoding 和通用 learned group aggregation 在本轮 roadmap 中提议转入
长期路线；该取舍必须由 roadmap review 接受并同步修订 `SPEC.md`，不能由实现静默
改变。

## 8. 不转化为 v0.2 功能的竞赛做法

- leaderboard 驱动的超参数搜索或反复使用最终 test；
- train/test 联合 fit、transductive rank、全量 target encoding；
- 自动欠采样/过采样算法目录或 AutoML 搜索；
- 神经网络、序列模型、Transformer；
- 数十到数百模型的 stacking/blending；
- 字段专用清洗、信用业务阈值或利润函数；
- reject inference、自动审批、监管/公平性合规结论；
- 多表关系引擎、server、dashboard、模型部署与实际版本发布。
