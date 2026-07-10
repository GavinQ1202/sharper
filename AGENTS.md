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

- `IMPLEMENTATION_PLAN.md` 是 v0.1 的任务拆分、允许文件、验收标准和实现顺序的执行依据；`SPEC.md` 定义产品与最终能力。两者冲突时先修订文档，不在实现中自行扩大或合并 Task。
- Task 01 只建立打包、工具配置和最小 import/version/`__all__` 契约；不冻结领域结果类型，不创建自定义异常体系或 CLI。
- `SchemaReport`、列 schema 结果和 `DataFrameSummary` 在 Task 03 冻结；`QualityIssue` 与 `QualityReport` 已由 Task 04 API 决策记录冻结，并在 Task 04 首次实现。
- Task 04 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task04-quality-contract.md`；改变冻结字段、code、severity、规则、阈值、文本或排序前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。
- Task 05 实现、测试、Markdown 和 CLI 文档必须遵守已接受的 `docs/decisions/task05-workflow-report-cli-contract.md`；改变冻结字段、签名、章节、文本、参数、输出通道或 exit code 前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。
- Task 06 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task06-excel-io-contract.md`；改变 `load_excel` 签名、`.xlsx` 单 sheet 范围、`read_options` 白名单、optional dependency、错误类型或稳定消息前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 06 不修改 CLI、workflow 或 reporting。
- Task 07 实现、测试和 API 文档必须遵守已接受的 `docs/decisions/task07-analysis-contract.md`；改变 analysis 函数签名、结果 dataclass 字段、输出表 schema、skipped reason codes/precedence、错误消息、排序或 non-target 范围前，必须先同步更新并评审该记录、`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。Task 07 不修改 CLI、workflow 或 reporting，不实现 target relationship、grouped analysis、visualization、feature engineering、modeling 或 evaluation。
- v0.1 默认以 `OSError` 表示文件读取失败，以 `ValueError` 表示无效参数、缺失列和非法列类型等用户输入错误；没有单独 SPEC 修改不得新增公共自定义异常体系。
- `analytics-workflow-builder` 最早可用于 Task 03 或 Task 04，不用于 Task 01 或 Task 02。
- `feature-engineering-builder` 不用于 Task 01、Task 02、Task 03 或 Task 04；仅在 `IMPLEMENTATION_PLAN.md` 进入 feature engineering Task 后使用。
- `visualization-system-builder` 不用于 Task 01，也不用于尚未进入 visualization Task 的工作。

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
- 所有建议包含来源列、公式、理由、风险、是否需要 fit 和稳定名称。
- 默认限制每类和总候选数；禁止无界笛卡尔组合。
- v0.1 仅可直接物化 ratio、difference、product 和基于显式 reference date 的确定性日期特征。
- 需要拟合的 suggestion 不得通过普通 DataFrame helper 在全量数据执行。
- 数据驱动分箱、group aggregate、target encoding、WOE 和监督分箱在 v0.1 只能建议，不提供 transform。
- 除零、无穷、未知类别、未见组和列名冲突必须有显式策略与测试。

## Visualization API 规则

- API 围绕“分布、缺失、相关、目标关系、模型评估”等分析任务，不镜像 matplotlib 原语。
- v0.1 统计分析型图表优先使用 seaborn；matplotlib 保留为底层 backend、Figure/Axes 对象契约和低级 fallback。
- 按 SPEC 冻结的 public contract 返回包含 `matplotlib.figure.Figure` 的具名结果或集合；不得自行改为裸 `Axes`，库代码不得调用 `show()`。
- 默认不写文件；保存由报告层或调用者负责。
- 不在函数中随意修改全局 matplotlib 或 seaborn style。高基数、过多列和大样本必须受预算约束并披露截断/抽样。
- v0.1 不建立多可视化后端系统，不引入 Plotly、Altair、Bokeh 或 dashboard。
- 绘图函数应消费已有分析结果；禁止为了绘图隐藏重算统计。
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

## Project Python environment

Use the project-local virtual environment for all verification commands.

Do not rely on a globally available `python` command.
Do not fall back to system `python3` unless explicitly instructed by the user.
Do not assume the virtual environment has already been activated.

From the repository root, use:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```

If `.venv/bin/python` is missing, pause and report that the project virtual
environment is unavailable. Do not silently use system Python.

For Task 01 clean-environment verification, use the same project-local
interpreter unless the task explicitly requires a separate clean environment.

If a command cannot be run because a dependency is missing from `.venv`, report
the missing dependency and the exact command that failed.

在 `pyproject.toml` 完成对应配置后，每个实现任务结束前运行：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

涉及打包、public API、CLI 或发布时，另运行：

```bash
.venv/bin/python -m build
```

Task 01 的干净环境验证范围仅包括：wheel/sdist 可以构建、安装后可以 `import sharper`、当前公共契约要求时可以读取 `sharper.__version__`，以及 pytest smoke tests 通过。`sharper --help` 仅从 Task 05 或其他明确的 CLI task 开始验证。若当前 Task 范围内的命令或入口尚未在 `pyproject.toml` 配置，不得声称相应检查通过；应明确报告未运行原因。

## 任务交付

交付说明应列出：行为变化、测试变化、执行过的命令及结果、未解决风险。不要用“覆盖率高”替代关键契约与 leakage 测试证据。
