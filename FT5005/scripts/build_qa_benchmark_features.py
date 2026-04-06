#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from dj30_qc_utils import build_event_path_lookup, iter_files, load_csv_rows, load_json, normalize_event_key_text, normalize_text, token_f1, write_csv, write_json


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "will", "with", "you", "your",
}

HEDGE_MARKERS = {
    "may", "might", "could", "would", "likely", "approximately", "around", "roughly",
    "perhaps", "maybe", "believe", "think", "assume", "potentially", "uncertain",
}

SUBJECTIVE_MARKERS = {
    "think", "believe", "feel", "hope", "view", "seem", "appears", "appears", "probably",
    "possibly", "confident", "comfortable",
}

CERTAINTY_MARKERS = {
    "will", "definitely", "certainly", "clearly", "confident", "committed", "expect",
    "strongly", "firmly", "disciplined", "remain", "continue",
}

JUSTIFICATION_MARKERS = {
    "because", "since", "therefore", "thus", "driven", "reflecting", "driving", "given",
    "due", "result", "resulting", "impact", "impacted", "reason",
}

FORWARD_MARKERS = {
    "will", "expect", "expects", "outlook", "guidance", "guide", "guiding", "target",
    "next", "forward", "ahead", "future", "plan", "plans", "planning",
}

PAST_MARKERS = {
    "was", "were", "had", "historically", "previously", "prior", "last", "earlier",
    "saw", "saw", "delivered", "reported",
}

EXTERNAL_ATTRIBUTION_MARKERS = {
    "macro", "macroeconomic", "consumer", "customers", "customer", "industry", "market",
    "markets", "competitive", "competition", "fx", "currency", "currencies", "rate",
    "rates", "tariff", "tariffs", "regulatory", "regulation", "weather", "seasonality",
    "supply", "chain", "geopolitical", "economy", "environment",
}

INTERNAL_ACTION_MARKERS = {
    "execute", "execution", "pricing", "product", "products", "strategy", "strategic",
    "inventory", "capacity", "efficiency", "cost", "costs", "margin", "margins",
    "hiring", "headcount", "investment", "investments", "roadmap", "operations",
}

NONRESPONSE_PHRASES = [
    "too early to say",
    "not going to give",
    "not going to guide",
    "not prepared to comment",
    "do not want to get ahead",
    "take that offline",
    "hard to say",
    "cannot quantify",
    "not something we can disclose",
    "we're not in a position to",
    "we are not in a position to",
    "not going to speculate",
    "won't speculate",
    "cannot comment",
    "can't comment",
    "not providing guidance",
    "not going to provide guidance",
    "we do not disclose",
    "we don't disclose",
    "let me take that offline",
]

DEFLECTION_PHRASES = [
    "as we said",
    "as we've said",
    "as we have said",
    "as i said",
    "as i mentioned",
    "as we mentioned",
    "as we've mentioned",
    "as we have mentioned",
    "as noted earlier",
    "to your point",
    "at a high level",
    "broadly speaking",
    "stepping back",
    "step back",
    "from a broader perspective",
]

NUMERIC_QUESTION_PHRASES = [
    "how much",
    "how many",
    "what percent",
    "what percentage",
    "how big",
    "quantify",
    "give us a sense of",
]

WH_WORDS = {"why", "how", "when", "where", "what", "which"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark-inspired Q&A weak-label features from A1 transcripts."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/qa_benchmark_features_real")
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS]


def build_idf(docs: list[list[str]]) -> dict[str, float]:
    df = Counter()
    total_docs = len(docs)
    for doc in docs:
        for token in set(doc):
            df[token] += 1
    return {token: math.log((total_docs + 1.0) / (count + 1.0)) + 1.0 for token, count in df.items()}


def specificity_score(tokens: list[str], idf_map: dict[str, float]) -> float:
    if not tokens:
        return 0.0
    return float(sum(idf_map.get(token, 0.0) for token in tokens) / len(tokens))


def phrase_hit(text: str, phrases: list[str]) -> bool:
    normalized = normalize_text(text).lower()
    return any(phrase in normalized for phrase in phrases)


def count_terms(tokens: list[str], vocabulary: set[str]) -> int:
    return sum(token in vocabulary for token in tokens)


def safe_rate(count: int, total: int) -> float:
    return 1000.0 * count / max(total, 1)


def earliest_overlap_position(question_tokens: list[str], answer_tokens: list[str]) -> float:
    if not question_tokens or not answer_tokens:
        return 1.0
    question_set = {token for token in question_tokens if token not in STOPWORDS}
    for idx, token in enumerate(answer_tokens):
        if token in question_set:
            return float(idx / max(len(answer_tokens), 1))
    return 1.0


def direct_answer_flag(question_text: str, answer_text: str, question_tokens: list[str], answer_tokens: list[str]) -> float:
    answer_head = " ".join(answer_tokens[:8])
    if re.match(r"^(yes|no|absolutely|definitely|certainly)\b", answer_head):
        return 1.0
    if answer_tokens and (answer_tokens[0].isdigit() or re.match(r"^\d", answer_tokens[0])):
        return 1.0
    if token_f1(question_text, " ".join(answer_tokens[:25])) >= 0.2:
        return 1.0
    if earliest_overlap_position(question_tokens, answer_tokens) <= 0.2:
        return 1.0
    return 0.0


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(statistics.mean(values)), float(statistics.median(values))


def share(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1)) if values else 0.0


def has_numeric_cue(question_text: str, question_tokens: list[str]) -> bool:
    if any(token.isdigit() or bool(re.match(r"^\d", token)) for token in question_tokens):
        return True
    return phrase_hit(question_text, NUMERIC_QUESTION_PHRASES)


def opening_restatement_flag(answer_text: str) -> float:
    normalized = normalize_text(answer_text).lower()
    opening = " ".join(normalized.split()[:8])
    return 1.0 if any(phrase in opening for phrase in DEFLECTION_PHRASES) else 0.0


def topic_drift_flag(coverage_value: float, overlap_pos: float) -> float:
    return 1.0 if coverage_value < 0.08 and overlap_pos > 0.6 else 0.0


def short_evasive_flag(
    direct_flag: float,
    coverage_value: float,
    question_tokens: list[str],
    answer_tokens: list[str],
) -> float:
    answer_ratio = len(answer_tokens) / max(len(question_tokens), 1)
    return 1.0 if direct_flag < 0.5 and coverage_value < 0.12 and answer_ratio < 0.7 else 0.0


def evasion_score(
    direct_flag: float,
    nonresponse_flag: float,
    deflection_flag: float,
    delay_share: float,
    coverage_value: float,
    hedge_rate: float,
    numeric_mismatch_flag: float,
    short_flag: float,
) -> float:
    normalized_hedge = min(hedge_rate / 40.0, 1.0)
    coverage_gap = max(0.0, 1.0 - min(coverage_value, 1.0))
    score = (
        0.22 * (1.0 - direct_flag)
        + 0.18 * nonresponse_flag
        + 0.12 * deflection_flag
        + 0.14 * min(max(delay_share, 0.0), 1.0)
        + 0.14 * coverage_gap
        + 0.10 * normalized_hedge
        + 0.05 * numeric_mismatch_flag
        + 0.05 * short_flag
    )
    return float(score)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    event_keys = {
        normalize_event_key_text(row.get("event_key", ""))
        for row in load_csv_rows(args.panel_csv.resolve())
        if normalize_event_key_text(row.get("event_key", ""))
    }

    a1_lookup, _ = build_event_path_lookup(iter_files(args.a1_dir.resolve(), [".json"]))

    raw_events = {}
    docs = []
    for event_key in sorted(event_keys):
        path = a1_lookup.get(event_key)
        if path is None:
            continue
        payload = load_json(path)
        components = payload.get("components", [])
        qa_pairs = []
        pending_question = None
        for component in components:
            text = normalize_text(component.get("text", ""))
            ctype = normalize_text(component.get("componenttypename", ""))
            if not text:
                continue
            if ctype == "Question":
                pending_question = text
            elif ctype == "Answer" and pending_question is not None:
                qa_pairs.append((pending_question, text))
                docs.append(content_tokens(pending_question))
                docs.append(content_tokens(text))
                pending_question = None
        raw_events[event_key] = qa_pairs

    idf_map = build_idf(docs)

    rows = []
    for event_key in sorted(event_keys):
        qa_pairs = raw_events.get(event_key, [])
        q_specificity = []
        a_specificity = []
        specificity_gap = []
        direct_flags = []
        nonresponse_flags = []
        early_overlap = []
        coverage = []
        hedge_rates = []
        subjective_rates = []
        numeric_rates = []
        delay_shares = []
        question_complexity = []
        certainty_rates = []
        justification_rates = []
        forward_rates = []
        past_rates = []
        external_attr_rates = []
        internal_action_rates = []
        restatement_flags = []
        topic_drift_flags = []
        numeric_mismatch_flags = []
        short_evasive_flags = []
        evasion_scores = []
        answer_to_question_ratios = []
        direct_early_combo = []

        for question_text, answer_text in qa_pairs:
            q_tokens = content_tokens(question_text)
            a_tokens = content_tokens(answer_text)
            answer_all_tokens = tokenize(answer_text)

            q_spec = specificity_score(q_tokens, idf_map)
            a_spec = specificity_score(a_tokens, idf_map)
            overlap_pos = earliest_overlap_position(q_tokens, answer_all_tokens)
            q_complex = len(question_text.split()) + 5 * question_text.count("?") + 3 * sum(token in WH_WORDS for token in tokenize(question_text))

            q_specificity.append(q_spec)
            a_specificity.append(a_spec)
            specificity_gap.append(a_spec - q_spec)
            coverage_value = token_f1(question_text, answer_text)
            coverage.append(coverage_value)
            early_overlap.append(1.0 - overlap_pos)
            delay_shares.append(overlap_pos)
            direct_flag = direct_answer_flag(question_text, answer_text, q_tokens, answer_all_tokens)
            nonresponse_flag = 1.0 if phrase_hit(answer_text, NONRESPONSE_PHRASES) else 0.0
            hedge_rate = safe_rate(count_terms(answer_all_tokens, HEDGE_MARKERS), len(answer_all_tokens))
            subjective_rate = safe_rate(count_terms(answer_all_tokens, SUBJECTIVE_MARKERS), len(answer_all_tokens))
            numeric_token_count = sum(token.isdigit() or bool(re.match(r"^\d", token)) for token in answer_all_tokens)
            numeric_rate = safe_rate(numeric_token_count, len(answer_all_tokens))
            certainty_rate = safe_rate(count_terms(answer_all_tokens, CERTAINTY_MARKERS), len(answer_all_tokens))
            justification_rate = safe_rate(count_terms(answer_all_tokens, JUSTIFICATION_MARKERS), len(answer_all_tokens))
            forward_rate = safe_rate(count_terms(answer_all_tokens, FORWARD_MARKERS), len(answer_all_tokens))
            past_rate = safe_rate(count_terms(answer_all_tokens, PAST_MARKERS), len(answer_all_tokens))
            external_attr_rate = safe_rate(count_terms(answer_all_tokens, EXTERNAL_ATTRIBUTION_MARKERS), len(answer_all_tokens))
            internal_action_rate = safe_rate(count_terms(answer_all_tokens, INTERNAL_ACTION_MARKERS), len(answer_all_tokens))
            restatement_flag = opening_restatement_flag(answer_text)
            numeric_mismatch_flag = 1.0 if has_numeric_cue(question_text, tokenize(question_text)) and numeric_token_count == 0 else 0.0
            drift_flag = topic_drift_flag(coverage_value, overlap_pos)
            short_flag = short_evasive_flag(direct_flag, coverage_value, q_tokens, a_tokens)
            evasion = evasion_score(
                direct_flag=direct_flag,
                nonresponse_flag=nonresponse_flag,
                deflection_flag=restatement_flag,
                delay_share=overlap_pos,
                coverage_value=coverage_value,
                hedge_rate=hedge_rate,
                numeric_mismatch_flag=numeric_mismatch_flag,
                short_flag=short_flag,
            )

            direct_flags.append(direct_flag)
            nonresponse_flags.append(nonresponse_flag)
            hedge_rates.append(hedge_rate)
            subjective_rates.append(subjective_rate)
            numeric_rates.append(numeric_rate)
            certainty_rates.append(certainty_rate)
            justification_rates.append(justification_rate)
            forward_rates.append(forward_rate)
            past_rates.append(past_rate)
            external_attr_rates.append(external_attr_rate)
            internal_action_rates.append(internal_action_rate)
            restatement_flags.append(restatement_flag)
            topic_drift_flags.append(drift_flag)
            numeric_mismatch_flags.append(numeric_mismatch_flag)
            short_evasive_flags.append(short_flag)
            evasion_scores.append(evasion)
            answer_to_question_ratios.append(len(answer_all_tokens) / max(len(tokenize(question_text)), 1))
            direct_early_combo.append(direct_flag * (1.0 - overlap_pos))
            question_complexity.append(float(q_complex))

        q_mean, q_med = summarize(q_specificity)
        a_mean, a_med = summarize(a_specificity)
        gap_mean, gap_med = summarize(specificity_gap)
        cov_mean, cov_med = summarize(coverage)
        early_mean, early_med = summarize(early_overlap)
        hedge_mean, hedge_med = summarize(hedge_rates)
        subj_mean, subj_med = summarize(subjective_rates)
        num_mean, num_med = summarize(numeric_rates)
        delay_mean, delay_med = summarize(delay_shares)
        qcomp_mean, qcomp_med = summarize(question_complexity)
        certainty_mean, certainty_med = summarize(certainty_rates)
        justification_mean, justification_med = summarize(justification_rates)
        forward_mean, forward_med = summarize(forward_rates)
        past_mean, past_med = summarize(past_rates)
        external_mean, external_med = summarize(external_attr_rates)
        internal_mean, internal_med = summarize(internal_action_rates)
        answer_ratio_mean, answer_ratio_med = summarize(answer_to_question_ratios)
        evasion_mean, evasion_med = summarize(evasion_scores)
        direct_early_mean, direct_early_med = summarize(direct_early_combo)

        row = {
            "event_key": event_key,
            "qa_bench_pair_count": str(len(qa_pairs)),
            "qa_bench_question_specificity_mean": f"{q_mean:.6f}",
            "qa_bench_question_specificity_median": f"{q_med:.6f}",
            "qa_bench_answer_specificity_mean": f"{a_mean:.6f}",
            "qa_bench_answer_specificity_median": f"{a_med:.6f}",
            "qa_bench_specificity_gap_mean": f"{gap_mean:.6f}",
            "qa_bench_specificity_gap_median": f"{gap_med:.6f}",
            "qa_bench_coverage_mean": f"{cov_mean:.6f}",
            "qa_bench_coverage_median": f"{cov_med:.6f}",
            "qa_bench_early_overlap_mean": f"{early_mean:.6f}",
            "qa_bench_early_overlap_median": f"{early_med:.6f}",
            "qa_bench_direct_answer_share": f"{(sum(direct_flags) / max(len(direct_flags), 1)):.6f}",
            "qa_bench_nonresponse_share": f"{(sum(nonresponse_flags) / max(len(nonresponse_flags), 1)):.6f}",
            "qa_bench_hedge_rate_mean": f"{hedge_mean:.6f}",
            "qa_bench_hedge_rate_median": f"{hedge_med:.6f}",
            "qa_bench_subjective_rate_mean": f"{subj_mean:.6f}",
            "qa_bench_subjective_rate_median": f"{subj_med:.6f}",
            "qa_bench_numeric_rate_mean": f"{num_mean:.6f}",
            "qa_bench_numeric_rate_median": f"{num_med:.6f}",
            "qa_bench_delay_share_mean": f"{delay_mean:.6f}",
            "qa_bench_delay_share_median": f"{delay_med:.6f}",
            "qa_bench_late_overlap_share": f"{(sum(value > 0.5 for value in delay_shares) / max(len(delay_shares), 1)):.6f}",
            "qa_bench_question_complexity_mean": f"{qcomp_mean:.6f}",
            "qa_bench_question_complexity_median": f"{qcomp_med:.6f}",
            "qa_bench_certainty_rate_mean": f"{certainty_mean:.6f}",
            "qa_bench_certainty_rate_median": f"{certainty_med:.6f}",
            "qa_bench_justification_rate_mean": f"{justification_mean:.6f}",
            "qa_bench_justification_rate_median": f"{justification_med:.6f}",
            "qa_bench_forward_rate_mean": f"{forward_mean:.6f}",
            "qa_bench_forward_rate_median": f"{forward_med:.6f}",
            "qa_bench_past_rate_mean": f"{past_mean:.6f}",
            "qa_bench_past_rate_median": f"{past_med:.6f}",
            "qa_bench_external_attr_rate_mean": f"{external_mean:.6f}",
            "qa_bench_external_attr_rate_median": f"{external_med:.6f}",
            "qa_bench_internal_action_rate_mean": f"{internal_mean:.6f}",
            "qa_bench_internal_action_rate_median": f"{internal_med:.6f}",
            "qa_bench_restatement_share": f"{share(restatement_flags):.6f}",
            "qa_bench_topic_drift_share": f"{share(topic_drift_flags):.6f}",
            "qa_bench_numeric_mismatch_share": f"{share(numeric_mismatch_flags):.6f}",
            "qa_bench_short_evasive_share": f"{share(short_evasive_flags):.6f}",
            "qa_bench_answer_to_question_ratio_mean": f"{answer_ratio_mean:.6f}",
            "qa_bench_answer_to_question_ratio_median": f"{answer_ratio_med:.6f}",
            "qa_bench_direct_early_score_mean": f"{direct_early_mean:.6f}",
            "qa_bench_direct_early_score_median": f"{direct_early_med:.6f}",
            "qa_bench_evasion_score_mean": f"{evasion_mean:.6f}",
            "qa_bench_evasion_score_median": f"{evasion_med:.6f}",
            "qa_bench_high_evasion_share": f"{share([1.0 if value >= 0.5 else 0.0 for value in evasion_scores]):.6f}",
        }
        rows.append(row)

    summary = {
        "num_events": len(rows),
        "num_events_with_pairs": sum(1 for row in rows if float(row["qa_bench_pair_count"]) > 0),
        "avg_pair_count": float(statistics.mean(float(row["qa_bench_pair_count"]) for row in rows)) if rows else 0.0,
    }

    write_csv(output_dir / "qa_benchmark_features.csv", rows)
    write_json(output_dir / "qa_benchmark_features_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
