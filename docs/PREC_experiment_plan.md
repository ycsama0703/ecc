# PREC 实验执行计划（Engineering Spec）

## Project
Proxy-Robust ECC Correction (PREC)

---

## 0. 文档目的

本文件是面向工程实现的实验执行计划。目标是让工程 AI 或工程同学能够直接按模块开始实现、训练、验证和测试 PREC 实验。

本计划默认主任务为：

- 目标变量：`shock_minus_pre`
- 主样本：`clean after-hours`
- 主方法：market prior + ECC correction signal + minimax gate + risk-controlled abstention

最终预测形式为：

\[
\hat y_i = \hat\mu_i + A_i \cdot \alpha_i^{\mathrm{mm}} \cdot z_i
\]

其中：

- `mu_hat[i] = \hat\mu_i`：market prior
- `z[i] = z_i`：ECC correction signal
- `alpha[i] = \alpha_i^{mm}`：minimax gate
- `A[i] = A_i`：abstention indicator

---

## 1. 实验总目标

实现并评估以下问题：

1. 强 market prior 能解释多少 `shock_minus_pre`
2. ECC correction signal 是否能解释 market prior 之外的残差
3. noisy observability proxies 是否能作为 latent reliability 的代理
4. minimax gate 是否能比直接使用 ECC correction 更稳健
5. risk-controlled abstention 是否能改善 risk-coverage trade-off

---

## 2. 数据输入与目标构造

### 2.1 事件级输入

每个事件 `i` 应包含以下信息：

#### A. 市场侧输入（必需）
- pre-call realized volatility
- pre-call return statistics
- pre-call volume statistics
- benchmark / market index data（如 SPY）
- structured controls（如 size, sector, liquidity, earnings-related controls）

#### B. ECC 输入（至少文本 + Q&A）
- transcript JSON / HTML
- Q&A 段落或问答对
- speaker / role 信息（如 analyst, management）
- 可选：audio representation

#### C. observability / proxy 输入
- alignment coverage
- transcript completeness
- timestamp quality
- metadata integrity
- ASR confidence（如果有）
- segment consistency / slice quality（如果有）

---

### 2.2 目标变量定义

对每个事件 `i`：

```text
RV_pre_60m  = pre-call 60-minute realized volatility
RV_post_60m = post-call 60-minute realized volatility

y_i = shock_minus_pre_i = RV_post_60m - RV_pre_60m
```

要求：

- 严格防止信息泄漏
- 所有输入特征必须只来自 call 前或 call 当下可观测信息
- 不允许使用 call 后文本、post-window summary 等泄漏信号

可选扩展：

- market-adjusted volatility
- 30m / 120m post window robustness

---

### 2.3 主样本定义

主实验样本：

- `clean after-hours`

鲁棒性样本：

- `off-hours`
- `regular hours`

默认先做主样本，跑通全流程后再做 robustness。

---

## 3. 数据切分

采用严格 chronological split：

```text
Train = earliest 60% to 70%
Val   = next 15% to 20%
Test  = last 15% to 20%
```

要求：

- 不允许时间重叠
- 同一事件只能出现在一个 split
- 所有标准化、特征选择、阈值选择都只能在 train / val 内完成
- test 只允许用于最终一次评估

可选额外泛化测试：

- ticker-held-out
- rolling temporal windows

---

## 4. 总体工程流程

按以下顺序执行：

```text
Step 1. build event table and clean panel
Step 2. build target y = shock_minus_pre
Step 3. train market prior model
Step 4. compute residual target r_tilde
Step 5. train ECC correction model
Step 6. compute correction error u
Step 7. build proxy features a
Step 8. fit proxy -> noise mapping
Step 9. compute minimax gate alpha
Step 10. tune abstention threshold kappa on val
Step 11. generate final test predictions
Step 12. run benchmarks, ablations, mechanism tests
Step 13. export tables and figures
```

---

## 5. Module 1: 数据面板与目标构造

### 5.1 输入

原始输入源：

- transcript JSON
- transcript HTML
- audio files（可选）
- alignment metadata / A4-like files
- market bars
- structured controls

### 5.2 输出

统一事件级 panel，例如：

```text
panel.csv / panel.parquet
```

每行一个事件，至少包含：

- event_id
- ticker
- event_date
- call_time
- split_label
- y = shock_minus_pre
- market features
- ECC raw references
- proxy raw features

### 5.3 质量控制

需要记录：

- 缺失原因
- 被过滤原因
- usable counts
- clean after-hours subset count

必须保留 QA 日志，便于论文写 data pipeline。

---

## 6. Module 2: Market Prior

### 6.1 目标

训练 market prior：

\[
\hat\mu_i = f_\phi(x_i)
\]

其中 `x_i` 为 pre-call market information + controls。

### 6.2 候选模型

优先顺序：

1. ElasticNet
2. LightGBM
3. XGBoost

建议至少实现两种：

- 一个线性可解释 baseline
- 一个树模型 baseline

在验证集上选择主 prior 模型。

### 6.3 训练目标

```text
loss_prior = MSE(y_i, mu_hat_i)
```

### 6.4 输出

保存：

```text
mu_hat_train.parquet
mu_hat_val.parquet
mu_hat_test.parquet
```

并构造残差目标：

```text
r_tilde[i] = y_i - mu_hat[i]
```

### 6.5 记录指标

在 val / test 上记录：

- R2
- MSE
- MAE
- Spearman

---

## 7. Module 3: ECC Correction Signal

### 7.1 目标

训练 ECC correction signal：

\[
z_i = g_\theta(e_i)
\]

这里 `e_i` 是 ECC-derived representation。

### 7.2 最低要求输入

必须实现：

- transcript pooled text embedding
- compact Q&A representation

可选实现：

- audio embedding
- role-aware representation
- MR-QA style representation

### 7.3 推荐实现路线

优先保证轻量可用：

#### 路线 A（最低可行）
- TF-IDF / bag-of-ngrams + ElasticNet
- FinBERT pooled + MLP / linear regressor
- compact Q&A pooled embedding + MLP / linear regressor

#### 路线 B（增强）
- text + audio dual tower
- MR-QA representation

建议先完成路线 A，再上路线 B。

### 7.4 训练目标

ECC 模块拟合的是残差，不是原始目标：

```text
loss_ecc = MSE(r_tilde[i], z[i])
```

### 7.5 输出

保存：

```text
z_train.parquet
z_val.parquet
z_test.parquet
```

并计算 correction error：

```text
u[i] = (r_tilde[i] - z[i])^2
```

保存：

```text
u_train.parquet
u_val.parquet
u_test.parquet
```

---

## 8. Module 4: Proxy Feature Builder

### 8.1 目标

构造 observability proxies `a_i`，这些不是 reliability 本身，而是 latent reliability 的 noisy proxies。

### 8.2 输入来源

从 A4-like metadata / repository signals 中抽取：

- alignment coverage
- transcript completeness
- timestamp integrity
- segment completeness
- file integrity / pass-fail flags
- audio availability / quality flag
- any other observability signals

### 8.3 特征规范化

需要统一 proxy 方向：

- proxy 越好，代表 reliability 越高
- 如果某列是“越大越差”，先取反或重标化

输出前执行：

- missing handling
- scaling / normalization
- sign consistency check

### 8.4 输出

```text
proxy_features_train.parquet
proxy_features_val.parquet
proxy_features_test.parquet
```

每列需附 metadata：

- feature name
- meaning
- direction (higher = better / worse)
- missing rule

---

## 9. Module 5: Proxy -> Noise Mapping

### 9.1 目标

从 proxy `a_i` 学习 noise level：

```text
a_i -> sigma_hat_sq[i]
```

使用 correction error `u[i]` 作为监督信号近似：

```text
sigma_hat_sq[i] ≈ E[u | a_i]
```

### 9.2 推荐方法

#### 低维情况
- Isotonic Regression
- 分桶 + 平滑

#### 高维情况
- Monotonic Neural Network
- monotone GAM / monotone GBM（如果工程更方便）

### 9.3 约束

必须体现单调性：

```text
proxy better -> estimated noise lower
proxy worse  -> estimated noise higher
```

### 9.4 输出

```text
sigma_hat_sq_train.parquet
sigma_hat_sq_val.parquet
sigma_hat_sq_test.parquet
```

以及模型文件：

```text
proxy_to_noise_model.pkl
```

---

## 10. Module 6: Identification Set and Worst-case Noise

### 10.1 目标

考虑 proxy uncertainty，不直接把 `sigma_hat_sq` 当点值，而是构造 worst-case noise：

\[
\bar\sigma_i^2
\]

### 10.2 推荐实现

至少实现一种：

#### 方案 A：bootstrap / subagging interval

对 proxy->noise mapping 重采样，得到：

```text
sigma_lower[i]
sigma_upper[i]
```

并令：

```text
sigma_bar_sq[i] = sigma_upper[i]
```

#### 方案 B：固定 conservative margin

```text
sigma_bar_sq[i] = sigma_hat_sq[i] + delta[i]
```

其中 `delta[i]` 可来自：

- validation residual quantile
- bootstrap std
- fixed fraction margin

### 10.3 输出

```text
sigma_bar_sq_train.parquet
sigma_bar_sq_val.parquet
sigma_bar_sq_test.parquet
```

---

## 11. Module 7: Minimax Gate

### 11.1 目标

计算 minimax gate：

\[
\alpha_i^{\mathrm{mm}} = \frac{\tau^2}{\tau^2 + \bar\sigma_i^2}
\]

### 11.2 tau^2 的估计

用 train set residual variance 估计：

```text
tau_sq = Var(r_tilde_train)
```

也可以验证集微调，但 test 不可参与。

### 11.3 输出

```text
alpha_train.parquet
alpha_val.parquet
alpha_test.parquet
```

### 11.4 sanity checks

需要检查：

- `alpha` 是否在 `[0,1]`
- proxy 越差时 `alpha` 是否整体下降
- `sigma_bar_sq` 增大时 `alpha` 是否单调下降

---

## 12. Module 8: Risk-Controlled Abstention

### 12.1 目标

当 worst-case conditional risk 太高时，不使用 ECC correction。

### 12.2 风险定义

实现上使用闭式风险：

```text
R[i] = tau_sq * sigma_bar_sq[i] / (tau_sq + sigma_bar_sq[i])
```

### 12.3 接受规则

```text
A[i] = 1 if R[i] <= kappa else 0
```

### 12.4 kappa 选择

在 validation set 上选择 `kappa`。

建议记录一条完整 risk-coverage curve，并从以下标准中选一个：

- 在 coverage >= target 的条件下最小化 risk
- 在 risk <= target 的条件下最大化 coverage
- 最小化 AURC

至少输出三个版本：

- conservative
- balanced
- aggressive

### 12.5 输出

```text
A_train.parquet
A_val.parquet
A_test.parquet
```

以及：

```text
kappa_config.json
```

---

## 13. Module 9: Final Prediction

### 13.1 最终预测

```text
y_hat[i] = mu_hat[i] + A[i] * alpha[i] * z[i]
```

### 13.2 输出

```text
pred_test_main.parquet
```

每行至少包含：

- event_id
- y_true
- mu_hat
- z
- sigma_hat_sq
- sigma_bar_sq
- alpha
- A
- y_hat
- split

---

## 14. Benchmark 实验（必须跑）

至少实现以下对照组：

### B1. Market only
- 输入：pre-call market variables
- 输出：`y_hat = mu_hat`

### B2. Market + controls
- 输入：market + structured controls

### B3. TF-IDF + ElasticNet
- transcript text baseline

### B4. FinBERT pooled
- transcript embedding baseline

### B5. compact Q&A baseline
- 只用 compact Q&A feature

### B6. text + audio baseline（若音频可用）
- MDRM style / dual tower simplified baseline

### B7. Ours: PREC
- `mu_hat + A * alpha * z`

建议统一输出到：

```text
benchmark_results.csv
```

---

## 15. Ablation（必须跑）

### A1. No-gate

```text
y_hat = mu_hat + z
```

### A2. Gate without abstention

```text
y_hat = mu_hat + alpha * z
```

### A3. Minimax gate + abstention

```text
y_hat = mu_hat + A * alpha * z
```

### A4. Observed gate vs minimax gate

比较：

```text
alpha_obs = tau_sq / (tau_sq + sigma_hat_sq)
alpha_mm  = tau_sq / (tau_sq + sigma_bar_sq)
```

### A5. Different proxy sets

比较：

- full proxy set
- A4 only
- completeness only
- timing only
- degraded proxy set

### A6. Different ECC feature sets

比较：

- text only
- compact Q&A only
- text + Q&A
- text + Q&A + audio

输出到：

```text
ablation_results.csv
```

---

## 16. Mechanism 实验（必须跑）

这一部分非常重要，用来验证 proxy 真的是 reliability proxy，而不是普通 feature。

### M1. 单调性诊断

验证：

```text
proxy better -> estimated noise lower
proxy better -> alpha higher
proxy better -> accepted correction more often
```

实现方式：

- 按 proxy score 分桶
- 画 bin-level `u`, `sigma_hat_sq`, `alpha`, acceptance rate

### M2. Permutation test

在合理分层内打乱 proxy：

- same time window / same ticker bucket / same split 内 shuffle
- 重新拟合 gate
- 比较性能下降幅度

如果 shuffle 后 gate 失效，支持“proxy is informative”。

### M3. Noise injection

主动恶化 proxy：

- 对 proxy 加高斯噪声
- 随机 mask 一部分 proxy 列
- 删除部分 observability columns

观察是否出现：

```text
sigma_bar_sq increases
alpha decreases
abstention increases
performance degrades gracefully
```

### M4. Reliability-bucket uplift

按 estimated reliability 分组，比较 ECC correction 带来的 uplift：

```text
( mu_hat + correction ) - mu_hat
```

验证 uplift 是否集中在高可靠样本。

输出到：

```text
mechanism_results.csv
```

---

## 17. Evaluation 指标

### 17.1 标准预测指标

在 val / test 上统一记录：

- R2
- MSE
- MAE
- Spearman correlation

### 17.2 选择性预测指标

必须记录：

- coverage
- accepted-set risk
- risk-coverage curve
- AURC

### 17.3 机制指标

- monotonic slope / rank correlation
- permutation drop
- noise injection sensitivity
- reliability-bucket uplift

---

## 18. 结果输出格式

### 18.1 主结果表

```text
| Model | R2 | MSE | MAE | Spearman | Coverage | AURC |
```

### 18.2 Ablation 表

```text
| Variant | ECC features | Gate type | Abstention | R2 | MSE | Coverage | AURC |
```

### 18.3 Mechanism 表

```text
| Test | Metric | Result |
```

---

## 19. 必须产出的图

至少输出以下图：

1. benchmark 主结果柱状图
2. risk-coverage curve
3. alpha vs proxy score plot
4. sigma_bar_sq vs proxy score plot
5. acceptance rate by reliability bucket
6. permutation test drop plot
7. noise injection sensitivity plot

文件输出目录建议：

```text
outputs/figures/
outputs/tables/
```

---

## 20. 推荐工程目录结构

```text
project/
  data/
    raw/
    interim/
    processed/
  features/
    build_targets.py
    build_market_features.py
    build_ecc_features.py
    build_proxy_features.py
  models/
    train_market_prior.py
    train_ecc_corrector.py
    fit_proxy_to_noise.py
    compute_minimax_gate.py
    fit_abstention.py
  eval/
    metrics.py
    selective_metrics.py
    mechanism_tests.py
  experiments/
    run_benchmarks.py
    run_ablation.py
    run_mechanism.py
    run_all.py
  outputs/
    tables/
    figures/
    predictions/
  configs/
    main.yaml
    benchmark.yaml
    ablation.yaml
```

---

## 21. 推荐执行顺序（严格）

### Phase 1. 跑通主线
1. build panel
2. build target
3. train market prior
4. train ECC corrector
5. fit proxy -> noise
6. compute minimax gate
7. tune kappa on val
8. evaluate on test

### Phase 2. 跑 benchmark
9. run all baselines
10. export main comparison table

### Phase 3. 跑 ablation
11. no-gate
12. observed gate
13. minimax gate
14. abstention variants

### Phase 4. 跑机制实验
15. monotonicity
16. permutation
17. noise injection
18. reliability-bucket uplift

### Phase 5. 导出论文素材
19. export final tables
20. export final figures
21. save predictions and configs for reproducibility

---

## 22. 成功标准

实验至少要验证以下结论中的大部分：

### S1. 预测层面
- PREC 相比强 market prior 有增量价值，或至少在 selective setting 下更优

### S2. 稳健性层面
- PREC 相比 no-gate 更稳
- minimax gate 比 observed gate 更保守且更稳定

### S3. 选择性层面
- risk-coverage curve 优于 no-gate / direct ECC correction
- abstention 能有效降低高风险样本错误

### S4. 机制层面
- proxy diagnostics 成立
- shuffle proxy 后性能下降
- proxy 注入噪声后，系统表现出“更保守而非崩溃”的反应

---

## 23. 最小可行交付物（MVP）

如果时间有限，先完成以下最小版本：

### MVP 必须项
- clean after-hours panel
- shock_minus_pre target
- market prior
- compact Q&A ECC corrector
- simple proxy feature set
- isotonic proxy->noise mapping
- minimax gate
- abstention threshold on val
- main benchmark table
- risk-coverage curve
- one monotonicity diagnostic

### MVP 暂缓项
- audio branch
- MR-QA fully reproduced
- ticker-held-out
- complex monotone neural net
- conformal extension

---

## 24. 给工程 AI 的直接执行指令

可直接按以下顺序开始实现：

```text
1. 先建立事件级 panel，并构造 shock_minus_pre。
2. 训练 market prior，输出 mu_hat 和 r_tilde。
3. 用 transcript / compact Q&A 训练 ECC residual corrector，输出 z 和 u。
4. 从 A4-like observability metadata 构造 proxy features，并统一方向。
5. 用 isotonic regression 先实现 proxy -> sigma_hat_sq。
6. 用 bootstrap 或 conservative margin 构造 sigma_bar_sq。
7. 计算 alpha_mm = tau_sq / (tau_sq + sigma_bar_sq)。
8. 在 validation 上搜索 kappa，得到 abstention rule。
9. 在 test 上输出 y_hat = mu_hat + A * alpha * z。
10. 跑 benchmark、ablation、risk-coverage、monotonicity、permutation、noise injection。
11. 导出 tables / figures / predictions。
```

---

## 25. 交付文件清单

工程实现完成后，至少应输出：

```text
processed_panel.parquet
benchmark_results.csv
ablation_results.csv
mechanism_results.csv
pred_test_main.parquet
risk_coverage_curve.csv
all_figures/*.png
all_configs/*.yaml
```

---

## 26. 备注

- 所有随机过程必须固定 seed。
- 所有实验配置必须保存。
- 所有 test 结果必须可复现。
- 所有表格与图应能直接服务论文第 5 章。

