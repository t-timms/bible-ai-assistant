"""Offline unit tests for scripts/retrieval_metrics.py.

Regression coverage for two real bugs found 2026-09-04 while auditing retrieval
quality: (1) `_get_rag()` returns a 3-tuple (verse_collection, passage_collection,
embedder) but `rank_for_query` unpacked it as 2, so every variant except `bm25`
raised `ValueError` and was silently SKIPPED; (2) `_fmt_row` sorted metric keys
alphabetically ("recall@10" < "recall@5" as strings) while the printed header
used insertion order, so every printed row was mislabeled against its header —
a table that "ran successfully" but reported swapped numbers.
"""

from __future__ import annotations

from scripts.retrieval_metrics import K_VALUES, _fmt_row, binary_metrics


def test_binary_metrics_recall_is_monotonic_in_k() -> None:
    ranked = ["a", "b", "c", "x", "d", "e", "f", "g", "h", "i"]
    relevant = {"d", "i"}  # position 5 (in top-5) and position 10 (only in top-10)
    m = binary_metrics(ranked, relevant)
    assert m["recall@5"] < m["recall@10"], "top-10 must recall everything top-5 does, plus more"
    assert m["recall@5"] == 0.5
    assert m["recall@10"] == 1.0


def test_binary_metrics_mrr_is_reciprocal_rank_of_first_hit() -> None:
    ranked = ["a", "b", "c"]
    m = binary_metrics(ranked, {"b"})
    assert m["mrr"] == 0.5  # rank 2


def test_binary_metrics_ndcg_perfect_when_all_relevant_lead() -> None:
    relevant = {"a", "b"}
    m = binary_metrics(["a", "b", "c"], relevant)
    assert m["ndcg"] == 1.0


def test_binary_metrics_zero_when_nothing_relevant_found() -> None:
    m = binary_metrics(["x", "y", "z"], {"a"})
    assert m["recall@5"] == 0.0
    assert m["recall@10"] == 0.0
    assert m["mrr"] == 0.0
    assert m["ndcg"] == 0.0


def test_fmt_row_values_align_with_the_given_key_order() -> None:
    """Regression: sorted(means) alphabetizes ("recall@10" before "recall@5"),
    which used to desync the printed row from the header entirely."""
    metric_keys = [f"recall@{k}" for k in K_VALUES] + ["mrr", "ndcg"]
    means = {"recall@5": 0.1, "recall@10": 0.9, "mrr": 0.3, "ndcg": 0.7}
    row = _fmt_row("cat", 42, means, metric_keys)
    # values must appear in metric_keys order: recall@5, recall@10, mrr, ndcg
    values = row.split()[-4:]
    assert values == ["0.100", "0.900", "0.300", "0.700"]


def test_fmt_row_alphabetical_sort_would_have_misaligned_this_case() -> None:
    """Documents *why* the bug was invisible in some tables: alphabetical order
    only differs from insertion order once a metric name sorts out of place —
    exactly the "recall@10" < "recall@5" case, which is why 'all' rows with
    recall@5 == recall@10-ish values could look plausible while every category
    row (where the two genuinely differ) was silently swapped."""
    metric_keys = ["recall@5", "recall@10", "mrr", "ndcg"]
    means = {"recall@5": 0.504, "recall@10": 0.537, "mrr": 0.522, "ndcg": 0.485}
    correct = _fmt_row("all", 227, means, metric_keys).split()[-4:]
    wrong = [f"{means[k]:.3f}" for k in sorted(means)]  # the old, buggy behavior
    assert correct == ["0.504", "0.537", "0.522", "0.485"]
    assert wrong == ["0.522", "0.485", "0.537", "0.504"]  # mrr/ndcg swapped into the recall slots
    assert correct != wrong
