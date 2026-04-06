# Proxy Feasibility Note

## What This Is

This is a lightweight public-data proxy, not the real paper experiment.

Because the teacher's ECC and news datasets are not yet in hand, the proxy uses the public `nyamuda/ECTSum` benchmark to test whether downstream text can be aligned back to salient segments of long earnings-call transcripts.

This is directly relevant to the proposed paper because the first technical hurdle is segment retrieval and weak supervision, not full end-to-end market prediction.

## Setup

- Dataset: `nyamuda/ECTSum`
- Split: `test`
- Examples used: `40`
- Retrieval target: benchmark summary text as a proxy for downstream news-like salience
- Segmenter: newline-based transcript segments with sentence fallback
- Retriever: pure-Python TF-IDF cosine ranking
- Extractive budget: top `5` retrieved segments

Command used:

```bash
python scripts/proxy_ectsum_retrieval.py --length 40 --top-k 5
```

## Result

Saved artifact:
- `results/proxy_ectsum_retrieval.json`

Aggregate metrics:
- Mean TF-IDF retrieval F1: `0.5597`
- Mean lead baseline F1: `0.3443`
- Mean random baseline F1: `0.2745`
- Mean top-segment similarity: `0.8660`

## Interpretation

Three things matter here:

1. Even a very simple retrieval model can align downstream text back to specific transcript segments much better than lead or random selection.
2. That means the weak-supervision idea is technically plausible before any fancy LLM tuning.
3. In the real project, replacing benchmark summaries with next-day news and adding press-release subtraction should produce a much more finance-specific signal.

## Important Caveat

This proxy is not a forecasting benchmark and should never be presented as one.

It is an alignment sanity check only:
- query text is the downstream summary
- the task is to recover salient transcript segments
- no market target is being predicted here

That is still useful because the proposed paper depends on this alignment stage.
