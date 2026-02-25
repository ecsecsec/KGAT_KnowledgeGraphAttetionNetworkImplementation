
import random
import numpy as np
import torch
import math

def set_seed(seed: int = 2023):
    """
    Set random seed for Python, NumPy and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred_items, ground_truth):
    """
    Compute Recall@K and NDCG@K given predicted items and ground-truth items.

    Args:
        pred_items: iterable of predicted item IDs (length = K)
        ground_truth: iterable or set of ground-truth item IDs

    Returns:
        (recall, ndcg)
    """
    gt_set = set(ground_truth)
    if len(gt_set) == 0:
        return 0.0, 0.0

    pred_list = list(pred_items)
    K = len(pred_list)

    hits = [1 if item in gt_set else 0 for item in pred_list]

    # Recall@K
    recall = sum(hits) / len(gt_set)

    # NDCG@K
    dcg = 0.0
    for i, h in enumerate(hits):
        if h:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(gt_set), K)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits)) if ideal_hits > 0 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg
