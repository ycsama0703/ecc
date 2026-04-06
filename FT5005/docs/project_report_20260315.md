# 项目总报告 2026-03-15

## 0. 报告元信息

- 项目仓库: `/media/volume/dataset/xma8/work/icaif_ecc_news_attention`
- 兼容软链接: `/ocean/projects/cis250100p/xma8/icaif_ecc_news_attention`
- 当前已推送远端: `origin/main`
- 研究状态基线提交: `a8544c7`
- 当前报告状态: 截至 `2026-03-15` 的综合研究总结与下一阶段工作底稿

重要说明:
- `/home/exouser/ACM_ICAIF.txt` 仅是历史笔记，不是当前 authoritative 工作区。
- 本报告会明确区分:
  - 当前 authoritative 的最新结论
  - 历史上很强但尚未在 corrected panel 上完整重跑的结果

## 1. 项目一句话总结

本项目研究 earnings conference call (`ECC`) 的文本、`Q&A` 语义、时间信息和音频线索，是否能够在高频市场数据上解释或预测财报电话会议后的波动反应。

项目最初更像一个泛化的 multimodal ECC 预测问题，但经过一系列基线、目标重构、样本切分和完整性审计之后，当前最合理的研究对象已经收缩为:

- `off-hours` earnings calls
- `shock_minus_pre = post_call_60m_rv - pre_60m_rv`
- 在 `same-ticker prior` 之上的 residual prediction
- 用严格 benchmark ladder 判断 ECC 是否真的提供增量信息

换句话说，项目已经从“做一个更大的多模态模型”转向“用更干净的金融问题定义，检验电话会议是否提供稳健的增量波动冲击信号”。

## 2. 故事线演化

### 2.1 最早阶段

最早的思路偏向宽泛的 multimodal ECC volatility prediction:

- 对所有 calls 建模
- 用原始 `post_call_60m_rv` 作为主目标
- 期待文本、音频、时间对齐和序列结构共同带来提升

这个阶段的问题是:

- raw volatility level 被 firm identity 和历史波动水平强烈支配
- 如果不用强 prior baseline，容易高估模型效果
- regular-hours calls 的信息环境混杂

### 2.2 中期转折

研究在几个关键点发生转折:

1. 引入 `same-ticker expanding mean` 等 hard baseline 后，发现原始目标并不理想。
2. 目标重构实验表明，`shock_minus_pre` 明显优于 raw level。
3. regime 分析表明，`pre_market + after_hours` 比 `market_hours` 更干净、更合理。
4. prior-aware residual 设计比“把 prior 当作普通特征”更诚实。

于是项目故事线变为:

- 不是“预测所有 call 的原始波动”
- 而是“在 off-hours 更干净的信息释放环境中，预测 call 后相对 pre-call 的波动冲击”

### 2.3 当前阶段

当前阶段已经进一步收紧为:

- 先做完整性审计与 signal isolation
- 先回答“市场信息和 ECC 信息谁在主导”
- 再决定是否值得继续押注更复杂的架构和 SOTA 方法

这一步非常关键，因为它让项目从一个“可能漂亮但不稳”的 course project，变成了一个更诚实、更接近可投稿 pilot research 的形态。

## 3. 当前核心研究问题

当前项目实际上在回答五个问题:

1. `ECC` 特征能否在 `same-ticker prior` 之上解释增量性的 `post-call volatility shock`?
2. 这种信号是否主要出现在 `off-hours`，而不是 regular hours?
3. `Q&A` 语义是否真能提供稳定增量价值，还是只是 validation 上看起来有帮助?
4. 当前最强结果到底是 `ECC-driven`，还是 `market-driven`?
5. 如果复杂架构不能带来提升，下一步应该优先升级数据/标签，还是升级模型?

## 4. 数据资产与样本状态

### 4.1 原始数据包

当前 DJ30 package 主要包含:

- `A1.ECC_Text_Json_DJ30`: 结构化 transcript JSON
- `A2.ECC_Text_html_DJ30`: 带 scheduled time 的 transcript HTML
- `A3.ECC_Audio_DJ30`: mp3 音频
- `A4.ECC_Timestamp_DJ30`: 句级时间对齐 CSV
- `C1.Surprise_DJ30`: earnings surprise
- `C2.AnalystForecast_DJ30`: analyst estimates
- `D.Stock_5min_DJ30`: 5-minute intraday market data

### 4.2 当前观测覆盖

当前可确认覆盖为:

- `A1 = 2146`
- `A2 = 842`
- `A3 = 796`
- `A4 = 670`
- `D = 25` 个 ticker 文件

`D` 缺失 tickers:

- `CRM`
- `CVX`
- `PG`
- `TRV`
- `V`

### 4.3 当前建模面板

当前 joined modeling panel 为:

- `553` 个事件
- 使用 `A1 + A2 + A4 + C1 + C2 + D`
- `D` 包含 extended-hours bars，因此 off-hours 设计是可行的

### 4.4 关键数据约束

当前必须始终记住的约束包括:

- `A2` 的时间是 scheduled time，不是实际开会时间
- `A4` 是 noisy alignment，不是 gold timestamp
- `A2` 存在 HTML 完整性问题，当前已识别 `79` 行 `html_integrity_flag=fail`
- 当前尚无 benchmark intraday ETF 数据，所以 target 还不是 benchmark-adjusted
- 当前样本仍然是 DJ30 pilot，而不是广义大样本

## 5. 数据质量控制与迁移审计

### 5.1 质量控制

项目已经完成的 QC 工作包括:

- `A2` HTML integrity 诊断
- `A4` 行级与事件级 QC
- malformed scheduled time 的恢复
- transcript / audio / timestamp / analyst / market 的事件级链接

### 5.2 服务器迁移后的关键发现

服务器迁移之后最重要的新发现并不是文件丢失，而是一个基础字段错误:

- `scheduled_hour_et` 之前被截成整数小时
- `09:30` 会被错误地判成 `pre_market`
- 下游 regime 和 off-hours 结果因此受到污染

修复之后:

- old regime counts: `after_hours=204`, `pre_market=293`, `market_hours=56`
- corrected regime counts: `after_hours=204`, `pre_market=273`, `market_hours=76`
- `20` 个事件发生 regime 变更
- 当前确认这些都是 `GS 09:30` 事件

这使得一些历史最强结果必须被重新解释:

- 历史结果并未作废
- 但需要区分 “pre-fix evidence” 和 “corrected-panel evidence”

## 6. 当前 authoritative 数据切分

### 6.1 全样本主切分

- train `<= 2021`
- validation `= 2022`
- test `>= 2023`

对应 corrected full-sample split:

- `train=281`
- `val=90`
- `test=182`

### 6.2 corrected off-hours split

服务器迁移修复后的 authoritative off-hours split 为:

- `train=239`
- `val=78`
- `test=160`

### 6.3 clean sample split

排除 `html_integrity_flag=fail` 后:

- full sample clean: `244 / 74 / 156`
- off-hours clean: `212 / 62 / 137`

## 7. 端到端分析流程

当前完整 pipeline 可以概括为:

1. `build_event_manifest.py`
   - 扫描和记录原始文件资产
2. `run_initial_qc.py`
   - 建立 `A2` / `A4` 质量控制输出
3. `build_intraday_targets.py`
   - 构造 pre-call / within-call / post-call 目标
4. `build_modeling_panel.py`
   - 把 analyst、transcript、QC、target 合并成 event panel
5. `build_event_text_audio_features.py`
   - 构造 transcript、dialogue、timing、audio-proxy 特征
6. `build_real_audio_features.py`
   - 从 `A3` 生成真实音频统计特征
7. `build_qa_benchmark_features.py`
   - 从 `A1` 的 Q&A pair 生成 benchmark-inspired 特征
8. 一系列建模脚本
   - structured baseline
   - tf-idf baseline
   - dense multimodal baseline
   - identity baseline
   - prior residual baseline
   - target redesign
   - regime / subset / ablation / robustness / stress test
   - signal decomposition
   - qav2 checkpoint
   - hybrid architecture experiments

## 8. 方法设计与评估哲学

### 8.1 为什么必须做 prior-aware residual

当前项目最大的经验之一是:

- `same-ticker prior` 不是一个可有可无的 baseline
- 它必须成为研究设计的中心

因此当前更合理的预测框架是:

`prediction = same_ticker_prior + residual_model(event_features)`

它的优点是:

- 直接检验 ECC 是否提供 event-specific signal
- 避免把 identity 效应“藏进”大模型
- 更符合金融解释

### 8.2 为什么主目标从 raw level 改成 shock target

raw `post_call_60m_rv` 的问题在于:

- firm-level volatility scale 太强
- same-ticker 历史均值已经很强
- 文本和语义信息很难在这个目标上稳定战胜 prior

因此当前主目标改为:

- `shock_minus_pre = post_call_60m_rv - pre_60m_rv`

它的意义是:

- 把注意力从“绝对波动水平”转到“增量冲击”
- 更接近 call 作为信息事件的经济含义

### 8.3 为什么主样本是 off-hours

`off-hours` 更接近“电话会议本身是核心信息事件”的环境:

- 市场微结构更干净
- 同时期其他盘中噪声更少
- 更适合作为研究主场景

这也是当前 paper storyline 的关键一环。

## 9. 特征体系

当前主要特征块包括:

### 9.1 Structured

- pre-call / within-call market variables
- scheduled hour
- call duration
- earnings surprise
- analyst dispersion / coverage
- transcript / alignment 质量统计

### 9.2 Extra dense event features

- transcript length, Q&A share
- uncertainty / positive / negative / guidance 词率
- pair overlap
- multi-part question
- evasiveness proxy
- presenter vs Q&A 的差值特征
- `A4` timing density / span / gap 特征

### 9.3 `qna_lsa`

- `Q&A` 文本的 dense semantic compression

### 9.4 `qa_benchmark` / `qav2`

从 benchmark-inspired heuristic 延伸到更丰富的 v2:

- directness
- coverage
- delay
- hedge / subjectivity
- certainty / justification
- forward vs past framing
- external attribution vs internal action
- restatement / topic drift / numeric mismatch
- evasion score

### 9.5 Audio

当前音频路径包括:

- coarse audio proxy
- real audio chunk summary

但当前证据并不支持把 audio 写成主贡献。

## 10. 当前 authoritative 结果总结

这一部分只汇总当前 post-migration、corrected-panel 下最应该被当作 authoritative 的结论。

### 10.1 目标重构仍然成立

目标重构实验的总方向非常稳定:

- raw target 上 same-ticker prior 很强
- `log_post_over_pre` 与 `shock_minus_pre` 更可学
- `shock_minus_pre` 仍然是当前最强目标

### 10.2 市场侧基线非常强

在 corrected panel 上，`signal decomposition` 给出的信息非常关键:

- full sample `market_only` test `R^2 ≈ 0.904`
- full sample `market_plus_controls ≈ 0.909`
- off-hours `market_only ≈ 0.907`
- off-hours `market_plus_controls ≈ 0.909`
- clean full sample `market_plus_controls ≈ 0.911`
- clean off-hours `market_plus_controls ≈ 0.911`

这说明当前强结果的主导项仍然是 market-side information。

### 10.3 ECC-only 在 `qav2` 下有“部分恢复”，但不稳健

`qav1 -> qav2` 后，ECC-only 的确改善了:

- full sample all-html: `-0.022 -> 0.095`
- off-hours all-html: `-0.076 -> 0.139`

但 clean sample 上不稳:

- full clean: `-0.005`
- off-hours clean: `-0.085`

所以当前最诚实的表述是:

- 更强的 `Q&A` 语义不是没用
- 但还没有形成稳健的、可投稿级别的增量优势

### 10.4 `qav2` 没有提升当前 mixed headline

当前 corrected-panel 下:

- full-sample mixed best: `qav1 ≈ 0.890`, `qav2 ≈ 0.884`
- off-hours mixed best: `qav1 ≈ 0.897`, `qav2 ≈ 0.891`

也就是说:

- `qav2` 更像是一个 signal-recovery step
- 不是 headline breakthrough

### 10.5 架构升级没有打破当前上限

在 `run_hybrid_architecture_experiments.py` 中，尝试了:

- nonlinear market tree expert
- global blend
- regime-gated blend
- positive stack
- gated stack HGBR

结果如下:

- full all-html: `market_controls_ridge ≈ 0.909`, `regime_gated ≈ 0.908`
- full clean: `market_controls_ridge ≈ 0.911`, `regime_gated ≈ 0.910`
- off-hours all-html: `market_controls_ridge ≈ 0.909`, `regime_gated ≈ 0.906`

负面但重要的发现:

- `market_controls_hgbr` 明显不如 ridge
- positive stack 基本退化成对 `full_ridge` 的缩放
- gated HGBR stack 明显过拟合

这说明:

- 复杂架构本身不是当前瓶颈
- 真正瓶颈仍然是信号与 supervision 质量

## 11. 历史强结果与其当前地位

项目中还有一批非常有价值但需要谨慎对待的结果。

### 11.1 固定 off-hours shock setting 的强结果

历史上最强的一组结果来自固定 setting:

- `pre_market + after_hours`
- `shock_minus_pre`
- residual-on-prior

当时最强结果包括:

- `prior_only ≈ 0.196`
- `residual structured only ≈ 0.916`
- year-wise:
  - `2023 ≈ 0.804`
  - `2024 ≈ 0.933`
  - `2025 ≈ 0.938`
- regime-wise:
  - `after_hours ≈ 0.919`
  - `pre_market ≈ 0.727`
- unseen-ticker stress:
  - overall structured-only `R^2 ≈ 0.991`

### 11.2 为什么这些结果现在不能直接当最终 authoritative headline

原因不是这些结果不重要，而是:

- 它们产生于 migration timing fix 之前的 pipeline 阶段
- 当前 corrected off-hours split 已从 `248/82/167` 变为 `239/78/160`
- 因此这些 robustness / significance / stress-test 结果应被视为:
  - 高价值 pilot evidence
  - 但仍需在 corrected panel 上完整重跑

### 11.3 当前最稳妥的写法

当前最稳妥的写法不是:

- “我们已经最终证明 off-hours structured-only 稳健到投稿级”

而是:

- “我们已经拿到一批很强的 pilot robustness evidence，但在 timing fix 后仍需完成 corrected-panel rerun，才能作为 final headline 提交”

## 12. 目前已经做过的主要方法尝试

项目目前已经尝试过的路线非常多，且这些尝试本身已经构成有价值的研究积累:

1. structured ridge baseline
2. sparse tf-idf text baseline
3. dense semantic text baseline
4. audio-only / structured+audio
5. benchmark-inspired `Q&A`
6. prior as feature
7. prior-aware residual ridge
8. target redesign
9. regime-specific residual
10. off-hours subset
11. strict ablation and significance tests
12. leave-one-ticker-out and unseen-ticker stress
13. `qav2` heuristic upgrade
14. signal decomposition benchmark ladder
15. hybrid architecture / stacked / gated / tree experts

这些尝试带来的最大结论不是“哪个 fancy 模型最好”，而是:

- 正确的问题定义比复杂模型更重要
- 强基线和严格拆解比 feature sprawl 更重要
- 当前复杂模型没有替代 signal isolation

## 13. 当前可以安全宣称的结论

当前安全的结论包括:

1. 在当前 DJ30 pilot 上，target redesign 是成功的，`shock_minus_pre` 明显优于 raw level。
2. `same-ticker prior` 是必须控制的 hard baseline。
3. market-side features 在 corrected panel 上解释了当前大部分高 `R^2`。
4. richer `Q&A` features 可以恢复部分 ECC-only signal，但增量价值尚不稳健。
5. 架构升级本身没有突破当前上限。
6. 当前项目已经具备 strong pilot quality，但还不是最终 submission-ready 版本。

## 14. 当前不能安全宣称的结论

当前不应过度宣称:

1. “multimodal gain 已被稳健证明”
2. “audio 明确带来增量价值”
3. “ECC 在所有 regime 下都稳定优于 market-only”
4. “当前 strongest result 已在 corrected panel 上全部复现完毕”
5. “复杂架构已经带来方法贡献”

## 15. 核心风险与不足

### 15.1 样本与外部有效性

- 当前仍是 DJ30 pilot
- `D` 只有 25 个 ticker
- 高价值结论可能有大盘龙头样本偏置

### 15.2 方法贡献不足

- 当前最稳模型仍然偏简单
- 更复杂的 prior-gated / hybrid architectures 尚未形成 publishable gain

### 15.3 数据与标签层面不足

- `A4` noisy
- `A2` scheduled time 不是 actual time
- 缺 benchmark-adjusted target
- `Q&A` 目前仍以 heuristic labels 为主，而不是 transferred pair-level supervision

### 15.4 结果解释风险

- 需要持续区分 corrected-panel evidence 与历史 pre-fix evidence
- 需要避免把 market-driven result 写成 ECC-driven result

## 16. 研究前瞻与下一阶段路线图

### 16.1 第一优先级: corrected-panel rerun closure

最优先的工作是把一批关键历史强结果在 corrected panel 上完整重跑:

- fixed off-hours ablation
- significance tests
- year-wise robustness
- regime-wise robustness
- leave-one-ticker-out
- unseen-ticker stress

只有这一步完成，当前 strongest pilot storyline 才能闭环。

### 16.2 第二优先级: 升级 supervision，而不是继续堆架构

下一条最值得投入的研究线是:

- transferred pair-level `Q&A` labels
- 参考 `SubjECTive-QA`
- 参考 `EvasionBench`

原因是当前所有证据都指向:

- 模型复杂度不是最大短板
- `Q&A` supervision quality 才是更可能带来真实增量的地方

### 16.3 第三优先级: benchmark-adjusted target 与更强金融解释

如果能获得:

- `SPY`
- `DIA`
- sector ETF

则应尽快构造 benchmark-controlled target，以进一步增强金融论文叙事。

### 16.4 第四优先级: 更大样本与可复现 benchmark

如果 restricted larger sample 可得:

- 应优先做 SP500/SP1500 级扩展

如果 restricted larger sample 不可得:

- 应平行推进 SEC + open earnings-call corpora 的 public benchmark 路线

这会直接改善:

- external validity
- reproducibility
- publishability

### 16.5 第五优先级: 复杂架构只在信号闭环后继续

未来仍可继续探索:

- sequence modeling
- alignment-aware audio
- mixed-type event transformers
- richer MoE / gating

但前提必须是:

- corrected-panel headline 已闭环
- transferred `Q&A` supervision 已尝试
- market-vs-ECC incrementality 已更清楚

## 17. 当前建议的论文叙事

如果今天就要开始写论文，最稳的叙事应该是:

### 标题方向

`Off-Hours Earnings-Call Volatility Shock Prediction Under Noisy Timing`

### 核心主张

- 在 noisy scheduled-time 条件下
- 针对 off-hours earnings calls
- 用 prior-aware residual evaluation
- 可以研究电话会议事件对 post-call volatility shock 的增量解释

### 诚实且强的写法

- target redesign 是关键贡献之一
- benchmark ladder 是完整性贡献之一
- `Q&A` 语义是“部分恢复的增量线索”
- audio 和复杂架构目前是负结果或次要证据

## 18. 当前最值得继续执行的具体动作

如果按最有效率的顺序继续推进，建议是:

1. 重跑 corrected off-hours ablation / robustness / unseen-ticker 全套
2. 引入 transferred pair-level `Q&A` labels
3. 在同一 benchmark ladder 下复测
4. 若仍无法稳定超过 market-only，则转 public benchmark / larger-sample 路线

## 19. 相关关键文档导航

当前最关键的辅助文档包括:

- `docs/research_plan.md`
- `docs/data_inventory.md`
- `docs/progress_log.md`
- `docs/publishability_gap_checklist.md`
- `docs/server_migration_audit_20260314.md`
- `docs/qna_signal_checkpoint_20260314.md`
- `docs/hybrid_architecture_checkpoint_20260314.md`
- `docs/sota_novelty_flexible_roadmap_20260314.md`

## 20. 总结

截至 `2026-03-15`，这个项目最重要的变化不是“模型更复杂了”，而是研究已经变得更诚实、更聚焦，也更接近真正能写成 paper 的形态。

我们已经知道:

- 什么 target 是对的
- 什么 baseline 是必须的
- 当前高分主要来自哪里
- 哪些 fancy 方法并没有真正解决问题

我们也已经知道接下来最该做什么:

- 先把 corrected-panel 主线闭环
- 再升级 `Q&A` supervision
- 最后再决定是否值得进一步投入更复杂的 SOTA 架构

这意味着项目已经从“探索想法阶段”进入“围绕一个可信研究对象做收敛与定稿”的阶段。
