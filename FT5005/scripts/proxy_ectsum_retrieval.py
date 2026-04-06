#!/usr/bin/env python3

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import requests


DATASET_URL = "https://datasets-server.huggingface.co/rows"


def fetch_rows(split: str, offset: int, length: int):
    params = {
        "dataset": "nyamuda/ECTSum",
        "config": "default",
        "split": split,
        "offset": offset,
        "length": length,
    }
    response = requests.get(DATASET_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return [item["row"] for item in payload["rows"]]


def normalize(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_segments(doc: str):
    lines = [normalize(line) for line in doc.splitlines()]
    lines = [line for line in lines if len(line.split()) >= 8]
    if len(lines) >= 4:
        return lines
    sentences = re.split(r"(?<=[.!?])\s+", normalize(doc))
    return [s for s in sentences if len(s.split()) >= 8]


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def build_tfidf_vectors(texts):
    tokenized = [tokenize(text) for text in texts]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    num_docs = len(tokenized)
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        length = sum(tf.values()) or 1
        vec = {}
        for term, count in tf.items():
            idf = math.log((1 + num_docs) / (1 + doc_freq[term])) + 1.0
            vec[term] = (count / length) * idf
        vectors.append(vec)
    return vectors


def cosine_dict(vec_a, vec_b):
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rouge1_like(candidate: str, reference: str):
    cand_counts = Counter(tokenize(candidate))
    ref_counts = Counter(tokenize(reference))
    overlap = sum(min(cand_counts[t], ref_counts[t]) for t in cand_counts)
    cand_total = sum(cand_counts.values()) or 1
    ref_total = sum(ref_counts.values()) or 1
    precision = overlap / cand_total
    recall = overlap / ref_total
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def retrieve_topk(segments, query, top_k):
    vectors = build_tfidf_vectors(segments + [query])
    segment_vectors = vectors[:-1]
    query_vector = vectors[-1]
    scores = [cosine_dict(segment_vec, query_vector) for segment_vec in segment_vectors]
    ranked = sorted(
        [{"segment": seg, "score": float(score)} for seg, score in zip(segments, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_k], ranked


def summarize_segments(items):
    return " ".join(item["segment"] for item in items)


def random_topk(segments, top_k, seed):
    rng = random.Random(seed)
    picks = segments[:]
    rng.shuffle(picks)
    return picks[:top_k]


def run_proxy(rows, top_k):
    aggregate = {
        "oracle_tfidf_f1": [],
        "lead_f1": [],
        "random_f1": [],
        "max_segment_similarity": [],
    }
    samples = []
    for idx, row in enumerate(rows):
        doc = normalize(row["doc"])
        summary = normalize(row["summaries"])
        segments = split_segments(doc)
        if len(segments) < top_k:
            continue

        topk, ranked = retrieve_topk(segments, summary, top_k)
        oracle_text = summarize_segments(topk)
        lead_text = " ".join(segments[:top_k])
        random_scores = []
        for seed in range(5):
            random_text = " ".join(random_topk(segments, top_k, seed + idx * 17))
            random_scores.append(rouge1_like(random_text, summary)["f1"])

        oracle_metric = rouge1_like(oracle_text, summary)
        lead_metric = rouge1_like(lead_text, summary)
        aggregate["oracle_tfidf_f1"].append(oracle_metric["f1"])
        aggregate["lead_f1"].append(lead_metric["f1"])
        aggregate["random_f1"].append(sum(random_scores) / len(random_scores))
        aggregate["max_segment_similarity"].append(ranked[0]["score"])

        if len(samples) < 3:
            samples.append(
                {
                    "summary_preview": summary[:400],
                    "top_segments": topk,
                    "lead_segments": segments[:top_k],
                    "metrics": {
                        "oracle": oracle_metric,
                        "lead": lead_metric,
                        "random_f1_mean": sum(random_scores) / len(random_scores),
                    },
                }
            )

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "num_examples": len(aggregate["oracle_tfidf_f1"]),
        "top_k": top_k,
        "mean_oracle_tfidf_f1": mean(aggregate["oracle_tfidf_f1"]),
        "mean_lead_f1": mean(aggregate["lead_f1"]),
        "mean_random_f1": mean(aggregate["random_f1"]),
        "mean_top_segment_similarity": mean(aggregate["max_segment_similarity"]),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="results/proxy_ectsum_retrieval.json",
    )
    args = parser.parse_args()

    rows = fetch_rows(args.split, args.offset, args.length)
    result = run_proxy(rows, args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
