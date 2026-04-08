# PREC 实验部分重设计指导（给项目工程 AI）

## 文档目的

本文件用于指导对当前 ECC / PREC 项目的实验部分进行全面、严谨、可复现的重设计。  
目标不是简单修补已有实验，而是重新建立一个：

1. 任务定义清晰
2. 决策时点一致
3. 无信息泄露
4. benchmark 对比公平
5. 能够直接服务最终报告 / 论文写作

的实验体系。

---

# 0. 总原则（必须遵守）

本项目当前最关键的问题，不是模型结构，而是实验口径混乱。  
因此，重设计实验必须优先满足以下原则：

## 原则 1：同一张表中的模型，必须共享同一决策时点
不能把：
- pre-call forecasting 模型
和
- post-call correction 模型
直接放在同一张 benchmark 主表里做公平比较。

## 原则 2：同一张表中的模型，必须共享同一信息集边界
如果某个模型可以看到 transcript / Q&A / proxy，而另一个只能看到 pre-call market，那么二者不能被表述为同任务下公平比较。

## 原则 3：严格禁止 post-call market outcome 进入输入特征
以下特征必须从所有主实验中删除：

- RV_post_60m
- post_call_60m_vw_rv
- post_call_60m_volume_sum
- 所有 post_call_*
- 所有 post_window_*

## 原则 4：严格禁止 within-call market features 进入主实验
以下特征必须从所有主实验中删除：

- within_call_rv
- within_call_vw_rv
- within_call_volume_sum

理由：
这些特征使用了 call 开始之后的市场反应，会污染任务定义。

## 原则 5：mu 和 z 必须保持功能分工
不允许：

- mu = market + controls
- z = ECC + market + controls

因为这会导致：

- 特征重复使用
- residual collapse
- z 直接预测 y

主实验中必须改回：

- mu = pre-call market + controls
- z = ECC-only

---

# 1. 统一任务定义（必须先明确）

本项目最终实验必须区分两种任务，但只能有一个作为主任务。

## Task A：Pre-call forecasting（参考任务）

### 决策时点
decision_time = call_start

### 可用信息
只能使用 call 开始前已知的信息：

- pre-call market variables
- structured controls
- analyst expectations
- schedule metadata

### 不能使用
- transcript
- Q&A
- audio
- A4 proxy
- within-call market
- post-call market

### 作用
这套任务只用于：
- 给出 task difficulty 的 reference
- 说明不用 ECC 时，强 baseline 能做到什么程度

### 结论
这不是 PREC 的主战场。

## Task B：Post-call correction（主任务）

### 决策时点
decision_time = call_end

### 可用信息
可以使用：

- pre-call market + controls
- ECC text / Q&A / audio（如果有）
- data-quality and observability indicators

### 不能使用
- 任何 post-call market outcome
- 任何 post-window outcome
- within-call market features
- call duration / end-time derived variables（主实验中禁用）

### 作用
这是 PREC 的主任务。

### 原因
PREC 的方法设计本身是：

- 先有 mu（market prior）
- 再有 z（ECC correction signal）
- 再根据 noisy proxies 决定 shrink / abstain

这天然是一个 post-call correction setting。

---

# 2. 最终实验结构（推荐采用）

最终实验部分建议分成 4 个小节：

- 5.1 Data, task, and split design
- 5.2 Pre-call reference benchmarks
- 5.3 Main post-call correction benchmarks
- 5.4 Selective prediction and ablation
- 5.5 Optional mechanism diagnostics（若时间允许）

---

# 3. 数据与切分规范

## 3.1 数据面板

主实验必须只使用：

- 同一份 clean panel
- 同一份 split
- 同一份 target construction

建议主面板：

- real / near-real timing reconstruction
- same panel for all post-call methods

## 3.2 切分方式

必须采用：

- strict chronological split
- no date overlap across train / val / test

允许同一 firm 在不同时间出现在不同 split，这是金融 panel 常见设定；但必须明确说明：

- 这是 time generalization，不是 ticker-held-out generalization

如有时间，可额外加：

- ticker-held-out robustness

但不是主实验必需。

## 3.3 目标变量

统一使用：

y = shock_minus_pre = RV_post_60m - RV_pre_60m

必须保证：

- target construction 与特征时间边界一致
- 任何输入 feature 不得直接包含 RV_post_60m 或其近似量

---

# 4. 干净特征规范（白名单 / 黑名单）

## 4.1 mu 层白名单（market prior）

### 唯一允许用于 mu 的特征

- pre_60m_rv
- pre_60m_vw_rv
- pre_60m_volume_sum
- scheduled_hour_et
- revenue_surprise_pct
- ebitda_surprise_pct
- eps_gaap_surprise_pct
- analyst_eps_norm_num_est
- analyst_eps_norm_std
- analyst_revenue_num_est
- analyst_revenue_std
- analyst_net_income_num_est
- analyst_net_income_std
- firm_size
- sector
- historical_volatility

### 备注
如果某些 surprise 变量是事后回填得到，而不是 call 前已知，则必须删除。

## 4.2 z 层白名单（ECC correction signal）

### 当前允许使用的 ECC-only 特征

- text_embedding_0 到 text_embedding_9
- qa_embedding_0 到 qa_embedding_9

### 如果后续加入干净音频特征，也允许
- audio_embedding_*
- aligned_audio_svd*
- prosody_summary_*

### 关键规则
z 中不得包含 mu 中的 market/control 特征。

## 4.3 gate / proxy 层白名单

当前 proxy 特征：

- transcript_coverage
- alignment_score
- audio_completeness

### 用途
这些只用于：

- sigma2 = g(proxy)
- alpha
- A

不应混入 mu 或 z 主预测器。

## 4.4 黑名单（所有主实验都禁止）

### 绝对禁止
- RV_post_60m
- post_call_60m_vw_rv
- post_call_60m_volume_sum
- 所有 post_call_*
- 所有 post_window_*

### 主实验中禁止
- within_call_rv
- within_call_vw_rv
- within_call_volume_sum
- call_duration_min
- call_duration_sec
- call_end_datetime

---

# 5. 需要运行的实验组（完整清单）

## Group A：Pre-call reference benchmarks

### 目的
给出不使用 ECC 时的 task difficulty 参考。

### 模型清单
1. market_precall_only
2. market_precall_plus_controls
3. xgboost_precall_controls
4. lightgbm_precall_controls
5. random_forest_precall_controls

### 特征
仅允许使用 mu 白名单特征。

### 输出指标
- R²
- MSE
- MAE
- nMAE
- Spearman

### 说明
这一组结果用于：

- 参考
- 对比 task difficulty

不与 post-call PREC 直接做公平横向比较。

## Group B：Post-call ECC-only baselines

### 目的
衡量原始 ECC 表征本身的预测能力。

### 模型清单
1. tfidf_elasticnet
2. compact_qa_baseline
3. finbert_pooled

如有音频再加：
4. audio_only_baseline
5. text_plus_audio_baseline

### 特征
只能使用 ECC-derived features。

### 输出指标
- R²
- MSE
- MAE
- nMAE
- Spearman

## Group C：PREC 主 family（最核心）

### 目的
验证你们的方法逻辑。

### 模型清单
1. prior_only
2. prior_plus_z_no_gate
3. prior_plus_alpha_z_gate_only
4. prec_selective

### 特征结构

#### prior_only
- mu only

#### prior_plus_z_no_gate
- mu + z

#### prior_plus_alpha_z_gate_only
- mu + alpha * z

#### prec_selective
- mu + A * alpha * z

### 关键规则
- mu = pre-call market + controls
- z = ECC-only
- proxy = proxy-only
- 不允许 hybrid ecc_plus_market_controls

### 输出指标
- Full-set R²
- Full-set MSE
- Full-set MAE
- nMAE
- Spearman
- Coverage
- Accepted-set R²
- Accepted-set MSE
- AURC
- Gain over prior（prior_mse - model_mse）

## Group D：Selective coverage sweep

### 目的
验证 precision–coverage tradeoff。

### 对象
仅对 prec_selective

### target coverage 点
- 0.20
- 0.40
- 0.60
- 0.80
- unconstrained

### 输出指标
- target coverage
- realized coverage
- full-set R²
- accepted-set R²
- accepted-set MSE
- AURC

### 结果用途
生成：
- coverage sweep 表
- risk-coverage curve

## Group E：可选机制实验（若时间允许）

### 目的
增强论文主线说服力。

### 建议实验
1. reliability bucket analysis
2. proxy permutation test
3. proxy noise injection

### 最低优先级
如果时间不够，可以不做 permutation / noise injection，但建议至少做 reliability bucket。

---

# 6. 主实验表格设计（最终写作输出）

## Table 1：Pre-call reference benchmarks

### 列
- Model
- R²
- MSE
- MAE
- nMAE
- Spearman

### 行
- market_precall_only
- market_precall_plus_controls
- xgboost_precall_controls
- lightgbm_precall_controls
- random_forest_precall_controls

## Table 2：Post-call main benchmark table

### 列
- Model
- Full-set R²
- MSE
- MAE
- nMAE
- Spearman
- Coverage
- Accepted-set R²
- Accepted-set MSE
- AURC
- Gain over prior

### 行
- prior_only
- tfidf_elasticnet
- compact_qa_baseline
- finbert_pooled
- prior_plus_z_no_gate
- prior_plus_alpha_z_gate_only
- prec_selective

## Table 3：Selective coverage sweep

### 列
- target coverage
- realized coverage
- full-set R²
- accepted-set R²
- accepted-set MSE
- AURC

### 行
- prec_selective 不同 coverage 点

---

# 7. 实验解释规范（必须遵守）

## 7.1 不允许这样写
- PREC 比所有 benchmark 都强
- PREC 优于 pre-call benchmark
- hybrid 版本是最强主结果

## 7.2 允许这样写
- pre-call benchmark 用于说明 task difficulty
- post-call correction 是主任务
- 在 post-call setting 中，blind correction harmful，gating helps，abstention improves accepted-set quality

## 7.3 对 hybrid 版本的处理
当前 ecc_plus_market_controls hybrid 结果只能：

- 放在附录
- 或作为 invalid / contaminated diagnostic run

不能作为主实验。

---

# 8. 工程实施顺序（必须按此执行）

## Phase 1：清理特征
1. 删除所有黑名单特征
2. 停用 ecc_plus_market_controls
3. 确认 mu 只用白名单 16 个
4. 确认 z 只用 ECC-only
5. 确认 proxy 只用 3 个 proxy

## Phase 2：重跑 pre-call benchmark
运行：
- market_precall_only
- market_precall_plus_controls
- xgboost_precall_controls
- lightgbm_precall_controls
- random_forest_precall_controls

## Phase 3：重跑 post-call ECC baselines
运行：
- tfidf_elasticnet
- compact_qa_baseline
- finbert_pooled

## Phase 4：重跑主 family
运行：
- prior_only
- prior_plus_z_no_gate
- prior_plus_alpha_z_gate_only
- prec_selective

## Phase 5：coverage sweep
对 prec_selective 跑多个 coverage 点，并导出：
- sweep 表
- risk-coverage curve
- accepted-set summary

---

# 9. 最终判断标准

重设计后的实验，应当满足：

## S1
主实验不再使用任何 within_call_*、post_call_*、call_duration_*

## S2
mu 与 z 功能分工明确，不再重复使用 market/control 特征

## S3
pre-call benchmark 与 post-call PREC 不再被混为公平对比

## S4
在 post-call 主任务中，能够清楚展示：
- prior_only
- mu + z
- mu + alpha z
- mu + A alpha z

之间的差异

## S5
最终实验可以直接支持论文主线：
- ECC correction 不能盲用
- reliability-aware gating 有用
- abstention 在 accepted subset 上提高预测质量

---

# 10. 给工程 AI 的最终一句话

请不要再继续优化高 R² 的 contaminated run。  
当前实验重设计的目标是：

- 恢复任务定义正确性
- 保证 benchmark 对比公平
- 用干净特征重建 PREC 的主实验逻辑

只有在这个基础上得到的结果，才可以作为最终报告 / 论文的主结果。
