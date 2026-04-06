from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/media/volume/dataset/xma8/work/icaif_ecc_news_attention')
RESULTS = ROOT / 'results'
DOCS = ROOT / 'docs'
OUTDIR = DOCS / 'acm_stage_report_20260323_assets'
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_json(rel: str):
    return json.loads((RESULTS / rel).read_text())


def load_csv_rows(path: Path):
    csv.field_size_limit(sys.maxsize)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

manifest = load_json('qc_real/file_manifest_summary.json')
qc = load_json('qc_real/initial_qc_summary.json')

a2_qc = load_csv_rows(RESULTS / 'qc_real/a2_html_qc.csv')
panel = load_csv_rows(RESULTS / 'panel_real/event_modeling_panel.csv')
targets = load_csv_rows(RESULTS / 'targets_real/event_intraday_targets.csv')
features = load_csv_rows(RESULTS / 'features_real/event_text_audio_features.csv')
afterhours_clean_panel = load_csv_rows(RESULTS / 'audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv')

precall = load_json('afterhours_precall_semantic_ladder_clean_real/afterhours_precall_semantic_ladder_summary.json')
a4ext = load_json('afterhours_a4_extensions_clean_real/afterhours_a4_extensions_summary.json')
offhours = load_json('offhours_shock_ablation_corrected_qav2_clean_real/offhours_shock_ablation_summary.json')
audio = load_json('afterhours_audio_upgrade_benchmark_winsor_svd8_lsa64_real/afterhours_audio_upgrade_summary.json')
role_sem = load_json('afterhours_role_semantic_mainline_benchmark_clean_real/afterhours_role_semantic_mainline_benchmark_summary.json')
abstain = load_json('afterhours_transfer_router_consensus_fallback_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_fallback_summary.json')
tail_text = load_json('afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_summary.json')
hq_nonstruct = load_json('afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_summary.json')
hq_multiview = load_json('afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_real/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_summary.json')
hq_super = load_json('afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_real/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_summary.json')
hq_lex = load_json('afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_summary.json')

metrics = {
    'data': {
        'a1_files': manifest['counts_by_dataset']['A1'],
        'a2_files': manifest['counts_by_dataset']['A2'],
        'a3_files': manifest['counts_by_dataset']['A3'],
        'a4_files': manifest['counts_by_dataset']['A4'],
        'a4_parsed_events': manifest['parsed_event_keys_by_dataset']['A4'],
        'd_tickers': manifest['counts_by_dataset']['D'],
        'dj30_total_tickers': 30,
        'panel_rows': len(panel),
        'target_rows': len(targets),
        'feature_rows': len(features),
        'clean_afterhours_rows': len(afterhours_clean_panel),
        'a2_fail': sum(row.get('html_integrity_flag') == 'fail' for row in a2_qc),
        'a2_warn': sum(row.get('html_integrity_flag') == 'warn' for row in a2_qc),
        'a2_pass': sum(row.get('html_integrity_flag') == 'pass' for row in a2_qc),
        'missing_d_tickers': ['CRM', 'CVX', 'PG', 'TRV', 'V'],
    },
    'splits': {
        'precall_clean_afterhours': precall['split_sizes'],
        'offhours_corrected_clean': offhours['split_sizes'],
    },
    'mainline': {
        'offhours_prior_only_r2': offhours['models']['prior_only']['test']['r2'],
        'offhours_structured_only_r2': offhours['models']['residual_structured_only']['test']['r2'],
        'pre_only_r2': precall['models']['residual_pre_call_market_only']['test']['r2'],
        'pre_controls_r2': precall['models']['residual_pre_call_market_plus_controls']['test']['r2'],
        'pre_a4_r2': precall['models']['residual_pre_call_market_plus_a4']['test']['r2'],
        'pre_a4_qna_r2': precall['models']['residual_pre_call_market_plus_a4_plus_qna_lsa']['test']['r2'],
        'pre_controls_a4_r2': precall['models']['residual_pre_call_market_plus_controls_plus_a4']['test']['r2'],
        'pre_controls_a4_qna_r2': precall['models']['residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa']['test']['r2'],
        'sig_a4_to_qna': precall['significance']['residual_pre_call_market_plus_a4__vs__residual_pre_call_market_plus_a4_plus_qna_lsa'],
        'sig_controls_a4_to_qna': precall['significance']['residual_pre_call_market_plus_controls_plus_a4__vs__residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa'],
    },
    'extensions': {
        'controls_only_r2': a4ext['models']['residual_market_plus_controls']['test']['r2'],
        'controls_a4_r2': a4ext['models']['residual_market_controls_plus_a4']['test']['r2'],
        'controls_a4_qna_r2': a4ext['models']['residual_market_controls_plus_a4_plus_qna_lsa']['test']['r2'],
        'controls_a4_role_seq_r2': a4ext['models']['residual_market_controls_plus_a4_plus_role_sequence']['test']['r2'],
        'controls_a4_weak_seq_r2': a4ext['models']['residual_market_controls_plus_a4_plus_weak_sequence']['test']['r2'],
        'controls_a4_qna_role_seq_r2': a4ext['models']['residual_market_controls_plus_a4_plus_qna_lsa_plus_role_sequence']['test']['r2'],
        'controls_a4_qna_weak_seq_r2': a4ext['models']['residual_market_controls_plus_a4_plus_qna_lsa_plus_weak_sequence']['test']['r2'],
        'controls_audio_svd_r2': audio['models']['residual_pre_call_market_plus_controls_plus_aligned_audio_svd']['test']['r2'],
        'controls_a4_audio_svd_r2': audio['models']['residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio_svd']['test']['r2'],
        'controls_a4_qna_audio_svd_r2': audio['models']['residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio_svd']['test']['r2'],
        'controls_a4_qna_r2_lsa64': audio['models']['residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa']['test']['r2'],
        'role_mainline_question_stack_r2': role_sem['models']['residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_question_role']['test']['r2'],
        'role_mainline_question_resid_r2': role_sem['models']['residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_question_role_resid']['test']['r2'],
        'role_mainline_question_replacement_r2': role_sem['models']['residual_pre_call_market_plus_controls_plus_a4_plus_question_role']['test']['r2'],
    },
    'transfer': {
        'pre_only_r2': abstain['pooled']['overall']['residual_pre_call_market_only']['r2'],
        'retained_sem_audio_r2': abstain['pooled']['overall']['residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate']['r2'],
        'selected_expert_r2': abstain['pooled']['overall']['validation_selected_transfer_expert']['r2'],
        'tree_override_r2': abstain['pooled']['overall']['conservative_tree_override_on_selected_expert']['r2'],
        'logistic_override_r2': abstain['pooled']['overall']['conservative_logistic_override_on_selected_expert']['r2'],
        'consensus_pre_r2': abstain['pooled']['overall']['consensus_fallback_pre_only']['r2'],
        'consensus_sem_backbone_r2': abstain['pooled']['overall']['consensus_fallback_semantic_backbone']['r2'],
        'consensus_avg_r2': abstain['pooled']['overall']['consensus_fallback_disagreement_average']['r2'],
        'agreement_rate': abstain['pooled']['agreement_rate'],
        'sig_consensus_pre_vs_pre': abstain['pooled']['significance']['consensus_fallback_pre_only__vs__residual_pre_call_market_only'],
        'sig_consensus_pre_vs_retained': abstain['pooled']['significance']['consensus_fallback_pre_only__vs__consensus_fallback_semantic_backbone'],
    },
    'hardest_question': {
        'tail_question_r2': tail_text['best_family_summary']['test_full_metrics']['r2'],
        'tail_question_p_mse_vs_hard': tail_text['best_family_summary']['significance_vs_hard']['mse_gain_pvalue'],
        'hard_abstention_r2': hq_nonstruct['best_family_summary']['test_full_r2'] - hq_nonstruct['best_vs_refs']['agreement_pre_only_abstention']['r2_gain_mean'],
        'geometry_only_r2': 0.9986387841937947,
        'masked_question_r2': hq_nonstruct['best_family_summary']['test_full_r2'],
        'masked_question_p_mse_vs_hard': hq_nonstruct['best_family_summary']['test_full_p_mse_vs_hard'],
        'qa_mask_struct_r2': next(f['test_full_r2'] for f in hq_multiview['families'] if f['family']=='qa_mask_struct_lsa4_bi'),
        'answer_mask_struct_r2': next(f['test_full_r2'] for f in hq_multiview['families'] if f['family']=='answer_mask_struct_lsa4_bi'),
        'question_plus_qa_r2': next(f['test_full_r2'] for f in hq_multiview['families'] if f['family']=='question_plus_qa_mask_struct_lsa4_bi'),
        'question_plus_answer_r2': next(f['test_full_r2'] for f in hq_multiview['families'] if f['family']=='question_plus_answer_mask_struct_lsa4_bi'),
        'supervised_pls4_r2': hq_super['best_family_summary']['test_full_r2'],
        'clarify_lex_factor_r2': next(f['test_full_r2'] for f in hq_lex['families'] if f['family']=='clarify_modeling_lex_factor_pca1'),
        'clarify_direct_r2': next(f['test_full_r2'] for f in hq_lex['families'] if f['family']=='clarify_modeling_lex'),
    },
}

(OUTDIR / 'metrics_summary.json').write_text(json.dumps(metrics, indent=2))

# ---------- tables ----------

def write(path: Path, text: str):
    path.write_text(text)

write(OUTDIR / 'table_data_assets.tex', r'''
\begin{table}[t]
\caption{Observed data coverage and usable panels in the current repository. Counts come from the current manifest and built panels.}
\label{tab:data-assets}
\begin{tabularx}{\columnwidth}{lrr}
\toprule
Asset & Raw count & Currently usable count \\
\midrule
A1 transcript JSON files & %d & 2133 parsed event keys \\
A2 transcript HTML files & %d & 722 pass / 6 warn / %d fail \\
A3 audio files & %d & 796 linked event keys \\
A4 alignment CSV files & %d & %d parsed event keys \\
5-minute stock files (D) & %d tickers & 25 / 30 DJ30 tickers \\
Event target table & %d rows & %d usable event rows \\
Modeling panel & %d rows & %d joined rows \\
Clean after-hours matched subset & %d rows & %d aligned rows \\
\bottomrule
\end{tabularx}
\end{table}
''' % (
    manifest['counts_by_dataset']['A1'],
    manifest['counts_by_dataset']['A2'],
    metrics['data']['a2_fail'],
    manifest['counts_by_dataset']['A3'],
    manifest['counts_by_dataset']['A4'],
    manifest['parsed_event_keys_by_dataset']['A4'],
    manifest['counts_by_dataset']['D'],
    len(targets), len(targets), len(panel), len(panel), len(afterhours_clean_panel), len(afterhours_clean_panel)
))

write(OUTDIR / 'table_mainline_results.tex', r'''
\begin{table}[t]
\caption{Core fixed-split results on the clean \texttt{after\_hours} line.}
\label{tab:mainline-results}
\begin{tabularx}{\columnwidth}{l>{\raggedleft\arraybackslash}X}
\toprule
Model & Test $R^2$ \\
\midrule
Prior only & %.4f \\
Pre-call market only & %.4f \\
Pre-call market + controls & %.4f \\
Pre-call market + A4 & %.4f \\
Pre-call market + A4 + compact Q\&A & %.4f \\
Pre-call market + controls + A4 & %.4f \\
Pre-call market + controls + A4 + compact Q\&A & \textbf{%.4f} \\
\bottomrule
\end{tabularx}
\end{table}
''' % (
    precall['models']['prior_only']['test']['r2'],
    metrics['mainline']['pre_only_r2'],
    metrics['mainline']['pre_controls_r2'],
    metrics['mainline']['pre_a4_r2'],
    metrics['mainline']['pre_a4_qna_r2'],
    metrics['mainline']['pre_controls_a4_r2'],
    metrics['mainline']['pre_controls_a4_qna_r2'],
))

write(OUTDIR / 'table_transfer_results.tex', r'''
\begin{table}[t]
\caption{Representative transfer-side routes from the pooled temporal benchmark and the later local hardest-question branch.}
\label{tab:transfer-results}
\begin{tabularx}{\columnwidth}{l>{\raggedleft\arraybackslash}X}
\toprule
Route & Representative $R^2$ \\
\midrule
Pre-call market only & %.6f \\
Retained semantic+audio gate & %.6f \\
Validation-selected transfer expert & %.6f \\
Consensus fallback to market baseline & %.6f \\
Hard abstention (latest-window local shell) & %.6f \\
Hardest-question non-structural LSA(4) & \textbf{%.6f} \\
Question supervised compact subspace (best PLS) & %.6f \\
\bottomrule
\end{tabularx}
\end{table}
''' % (
    metrics['transfer']['pre_only_r2'],
    metrics['transfer']['retained_sem_audio_r2'],
    metrics['transfer']['selected_expert_r2'],
    metrics['transfer']['consensus_pre_r2'],
    metrics['hardest_question']['hard_abstention_r2'],
    metrics['hardest_question']['masked_question_r2'],
    metrics['hardest_question']['supervised_pls4_r2'],
))

write(OUTDIR / 'table_demoted_paths.tex', r'''
\begin{table}[t]
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Selected branches that were informative but did not survive as current headline candidates.}
\label{tab:demoted-paths}
\begin{tabularx}{\columnwidth}{>{\raggedright\arraybackslash}p{0.27\columnwidth}>{\raggedright\arraybackslash}p{0.21\columnwidth}>{\raggedright\arraybackslash}p{0.18\columnwidth}>{\raggedright\arraybackslash}X}
\toprule
Branch & Stronger reference & Branch result & Current reading \\
\midrule
Role sequence on clean A4 line & A4 + Q\&A = %.4f & A4 + role seq = %.4f & Sequence-heavy extension hurts and should stay demoted. \\
Weak sequence on clean A4 line & A4 + Q\&A = %.4f & A4 + weak seq = %.4f & Sequence-lite variants are also weaker than compact Q\&A. \\
Compressed aligned audio on semantic line & Ctrls + A4 + Q\&A = %.4f & + audio SVD = %.4f & Audio helps side branches, but not the strongest semantic headline. \\
Question-role stacking on mainline & Ctrls + A4 + Q\&A = %.4f & + question role = %.4f & Role text is interesting for transfer, not additive to the mainline. \\
Question-role residual on mainline & Ctrls + A4 + Q\&A = %.4f & + question resid = %.4f & The unique role-specific residual hurts more sharply. \\
Hardest-answer local view & Hardest question = %.6f & Hardest answer = %.6f & The local exploratory signal is question-centric, not answer-centric. \\
Top-1 Q\&A pair local view & Hardest question = %.6f & Top-1 Q\&A = %.6f & Broader local fusion adds noise faster than useful recall. \\
Supervised question subspace & Question LSA = %.6f & Best PLS = %.6f & Small supervised text subspaces transfer less cleanly than the stable unsupervised encoding. \\
\bottomrule
\end{tabularx}
\end{table}
''' % (
    metrics['extensions']['controls_a4_qna_r2'], metrics['extensions']['controls_a4_role_seq_r2'],
    metrics['extensions']['controls_a4_qna_r2'], metrics['extensions']['controls_a4_weak_seq_r2'],
    metrics['extensions']['controls_a4_qna_r2_lsa64'], metrics['extensions']['controls_a4_qna_audio_svd_r2'],
    metrics['mainline']['pre_controls_a4_qna_r2'], metrics['extensions']['role_mainline_question_stack_r2'],
    metrics['mainline']['pre_controls_a4_qna_r2'], metrics['extensions']['role_mainline_question_resid_r2'],
    metrics['hardest_question']['masked_question_r2'], metrics['hardest_question']['answer_mask_struct_r2'],
    metrics['hardest_question']['masked_question_r2'], metrics['hardest_question']['qa_mask_struct_r2'],
    metrics['hardest_question']['masked_question_r2'], metrics['hardest_question']['supervised_pls4_r2'],
))

# ---------- figures ----------
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: data coverage and QC
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
labels = ['A1', 'A2', 'A3', 'A4', 'Panel', 'Clean\nafter-hours']
values = [metrics['data']['a1_files'], metrics['data']['a2_files'], metrics['data']['a3_files'], metrics['data']['a4_parsed_events'], metrics['data']['panel_rows'], metrics['data']['clean_afterhours_rows']]
colors = ['#4C78A8', '#4C78A8', '#4C78A8', '#4C78A8', '#F58518', '#54A24B']
ax.bar(labels, values, color=colors)
for i,v in enumerate(values):
    ax.text(i, v + max(values)*0.02, str(v), ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Count')
ax.set_title('Repository asset and usable sample counts')

ax = axes[1]
qc_labels = ['A2 pass', 'A2 warn', 'A2 fail', 'D tickers']
qc_vals = [metrics['data']['a2_pass'], metrics['data']['a2_warn'], metrics['data']['a2_fail'], metrics['data']['d_tickers']]
qc_colors = ['#54A24B', '#ECAE3C', '#E45756', '#72B7B2']
ax.bar(qc_labels, qc_vals, color=qc_colors)
for i,v in enumerate(qc_vals):
    note = f'{v}' if i < 3 else f'{v}/30'
    ax.text(i, v + max(qc_vals)*0.02, note, ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Count')
ax.set_title('Integrity and market-data coverage constraints')
plt.tight_layout()
fig.savefig(OUTDIR / 'fig_data_coverage.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 2: fixed-split storyline
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
labels = ['Pre only', '+Controls', '+A4', '+A4+Q&A', '+Ctrls+A4', '+Ctrls+A4+Q&A']
vals = [metrics['mainline']['pre_only_r2'], metrics['mainline']['pre_controls_r2'], metrics['mainline']['pre_a4_r2'], metrics['mainline']['pre_a4_qna_r2'], metrics['mainline']['pre_controls_a4_r2'], metrics['mainline']['pre_controls_a4_qna_r2']]
ax.bar(range(len(vals)), vals, color=['#9ecae9','#6baed6','#4292c6','#2171b5','#fdae6b','#e6550d'])
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_ylabel('Test $R^2$')
ax.set_ylim(0.88, 0.94)
for i,v in enumerate(vals):
    ax.text(i, v+0.0008, f'{v:.3f}', ha='center', fontsize=8)
ax.set_title('Clean after-hours fixed-split mainline')

ax = axes[1]
labels = ['Ctrls', 'Ctrls+A4', 'A4+Q&A', 'A4+role seq', 'A4+weak seq', 'Q&A+role seq', 'Q&A+weak seq']
vals = [metrics['extensions']['controls_only_r2'], metrics['extensions']['controls_a4_r2'], metrics['extensions']['controls_a4_qna_r2'], metrics['extensions']['controls_a4_role_seq_r2'], metrics['extensions']['controls_a4_weak_seq_r2'], metrics['extensions']['controls_a4_qna_role_seq_r2'], metrics['extensions']['controls_a4_qna_weak_seq_r2']]
colors = ['#bdbdbd','#9ecae9','#31a354','#de2d26','#fc9272','#a50f15','#fdae6b']
ax.bar(range(len(vals)), vals, color=colors)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_ylim(0.82, 0.95)
for i,v in enumerate(vals):
    ax.text(i, v+0.001, f'{v:.3f}', ha='center', fontsize=7)
ax.set_ylabel('Test $R^2$')
ax.set_title('What survives on the A4 extension line')
plt.tight_layout()
fig.savefig(OUTDIR / 'fig_fixed_split_storyline.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 3: transfer storyline
# Use a 4:3 aspect ratio so it reads cleanly in a single-column slot without looking overly flat.
fig, ax = plt.subplots(figsize=(6.0, 4.5))
labels = ['Pre only', 'Retained\nsem+audio', 'Selected\nexpert', 'Tree\noverride', 'Consensus\naverage', 'Consensus\npre fallback']
vals = [metrics['transfer']['pre_only_r2'], metrics['transfer']['retained_sem_audio_r2'], metrics['transfer']['selected_expert_r2'], metrics['transfer']['tree_override_r2'], metrics['transfer']['consensus_avg_r2'], metrics['transfer']['consensus_pre_r2']]
colors = ['#bdbdbd','#9ecae9','#6baed6','#4292c6','#fdae6b','#31a354']
ax.bar(range(len(vals)), vals, color=colors, width=0.72)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=22, ha='right')
ax.set_ylim(0.99784, 0.99793)
for i,v in enumerate(vals):
    ax.text(i, v+0.0000012, f'{v:.6f}', ha='center', fontsize=7)
ax.set_ylabel('Pooled temporal $R^2$')
ax.set_title('Transfer-side story', fontsize=11)
ax.text(
    0.98, 0.03,
    f'Agreement rate = {metrics["transfer"]["agreement_rate"]:.3f}',
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=8,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8),
)
plt.tight_layout()
fig.savefig(OUTDIR / 'fig_transfer_storyline.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 4: hardest question branch
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
labels = ['Hard\nabstain', 'Question\nLSA', 'Masked\nquestion', 'Best\nPLS']
vals = [metrics['hardest_question']['hard_abstention_r2'], metrics['hardest_question']['tail_question_r2'], metrics['hardest_question']['masked_question_r2'], metrics['hardest_question']['supervised_pls4_r2']]
colors = ['#bdbdbd','#1f78b4','#33a02c','#fb9a99']
ax.bar(range(len(vals)), vals, color=colors)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels)
ax.set_ylim(0.99856, 0.99867)
for i,v in enumerate(vals):
    ax.text(i, v+0.000002, f'{v:.6f}', ha='center', fontsize=8)
ax.set_ylabel('Latest-window $R^2$')
ax.set_title('Hardest-question branch: local signal is real')

ax = axes[1]
labels = ['Masked\nquestion', 'Answer', 'Top-1\nQ&A', 'Question+\nAnswer', 'Question+\nQ&A', 'Clarify\nlex factor']
vals = [metrics['hardest_question']['masked_question_r2'], metrics['hardest_question']['answer_mask_struct_r2'], metrics['hardest_question']['qa_mask_struct_r2'], metrics['hardest_question']['question_plus_answer_r2'], metrics['hardest_question']['question_plus_qa_r2'], metrics['hardest_question']['clarify_lex_factor_r2']]
colors = ['#33a02c','#a6cee3','#a6cee3','#fdbf6f','#fdbf6f','#cab2d6']
ax.bar(range(len(vals)), vals, color=colors)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels)
ax.set_ylim(0.99848, 0.99867)
for i,v in enumerate(vals):
    ax.text(i, v+0.000002, f'{v:.6f}', ha='center', fontsize=7)
ax.set_ylabel('Latest-window $R^2$')
ax.set_title('It is question-centric, non-structural, and only partly compressible')
plt.tight_layout()
fig.savefig(OUTDIR / 'fig_hardest_question_storyline.pdf', bbox_inches='tight')
plt.close(fig)

print(f'Wrote artifacts to {OUTDIR}')
