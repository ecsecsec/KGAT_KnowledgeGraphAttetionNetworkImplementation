import os
import numpy as np
import torch

from src.config import get_args
from src.dataset import KGATDataset
from src.lightgcn import LightGCN
from src.utils import set_seed, compute_metrics
from train_lightgcn import build_ui_adj_matrix


def test_lightgcn():
    args = get_args()
    set_seed(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- START LIGHTGCN TESTING ---")

    # 1. Load Dataset
    dataset_generator = KGATDataset(args)
    n_users = dataset_generator.n_users
    n_items = dataset_generator.n_items

    # 2. Build adjacency user–item (giống lúc train)
    adj = build_ui_adj_matrix(
        n_users, n_items, dataset_generator.train_data, device
    )

    # 3. Load model
    n_layers = len(eval(args.layer_size))
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embed_dim=args.embed_dim,
        n_layers=n_layers,
        adj=adj,
    ).to(device)

    assert os.path.exists(args.model_path), \
        f"Model path not found: {args.model_path}"
    print(f"Loading weights from: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4. Lấy toàn bộ embedding user & item
    with torch.no_grad():
        user_embs, item_embs = model.get_all_embeddings()
        user_embs = user_embs.to(device)
        item_embs = item_embs.to(device)

    test_users = list(dataset_generator.test_data.keys())
    avg_recall = []
    avg_ndcg = []

    test_batch_size = 100

    for i in range(0, len(test_users), test_batch_size):
        batch_u = test_users[i: i + test_batch_size]
        batch_u_tensor = torch.LongTensor(batch_u).to(device)

        # score: (batch_size, n_items)
        batch_u_emb = user_embs[batch_u_tensor]  # (B, d)
        scores = torch.matmul(batch_u_emb, item_embs.t())  # (B, n_items)
        scores = scores.cpu().numpy()

        # Mask train items
        for idx, u in enumerate(batch_u):
            train_items_global = dataset_generator.train_data[u]
            train_items_rel = [
                t - n_users for t in train_items_global
                if t >= n_users and (t - n_users) < n_items
            ]
            scores[idx, train_items_rel] = -1e9  # rất nhỏ

        # Top-K items
        K = args.topk
        # argpartition để lấy top-K nhanh
        idx_part = np.argpartition(-scores, K, axis=1)[:, :K]
        topk_scores = scores[np.arange(len(batch_u))[:, None], idx_part]
        order = np.argsort(-topk_scores, axis=1)
        sorted_topk = idx_part[np.arange(len(batch_u))[:, None], order]

        # Tính metric
        for idx, u in enumerate(batch_u):
            # Ground truth: danh sách item global trong test
            gt_items_global = dataset_generator.test_data[u]
            gt_items_global = [
                t for t in gt_items_global if t >= n_users
            ]

            pred_items_rel = sorted_topk[idx]
            pred_items_global = pred_items_rel + n_users

            r, n = compute_metrics(pred_items_global, gt_items_global)
            avg_recall.append(r)
            avg_ndcg.append(n)

        if (i + test_batch_size) % 1000 == 0:
            print(f"Processed {i + test_batch_size} users...")

    print(
        f"LIGHTGCN RESULT | Recall@{args.topk}: "
        f"{np.mean(avg_recall):.4f}, "
        f"NDCG@{args.topk}: {np.mean(avg_ndcg):.4f}"
    )


if __name__ == "__main__":
    test_lightgcn()
