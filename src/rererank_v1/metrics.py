from __future__ import annotations

import numpy as np


def calculate_mrr(rank_list):
    if not rank_list:
        return 0.0
    return float(np.mean([1.0 / r if r > 0 else 0.0 for r in rank_list]))


def calculate_ndcg(rank_list, k=5):
    if not rank_list:
        return 0.0

    dcg = 0.0
    for rank in rank_list:
        if rank <= k:
            dcg += 1.0 / np.log2(rank + 1)

    idcg = 0.0
    num_relevant = len(rank_list)
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def calculate_precision(rank_list, k=3):
    if not rank_list:
        return 0.0
    relevant_in_top_k = sum(1 for r in rank_list if r <= k)
    return float(relevant_in_top_k / k)

