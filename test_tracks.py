
import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.config import get_args
from src.dataset import KGATDataset
from src.model import KGAT
from src.utils import set_seed


TOPK_TRACKS = 20
TOPN_ARTISTS = 100          # số artist dùng để sinh candidate
TOPM_TRACKS_PER_ARTIST = 50 # max track/artist lấy từ catalog


def load_track_catalog(structured_path: str):
    """
    Đọc:
      - track_train.csv      -> seen_tracks[u_idx]: track_id user đã nghe trong TRAIN
      - track_test.csv       -> gt_tracks[u_idx]  : track_id ground-truth (test)
      - artist_tracks_catalog.csv -> artist_id -> list(track_id, track_name, pop)
    """
    track_train_path = os.path.join(structured_path, "track_train.csv")
    track_test_path = os.path.join(structured_path, "track_test.csv")
    catalog_path = os.path.join(structured_path, "artist_tracks_catalog.csv")

    if not os.path.exists(track_train_path) or not os.path.exists(track_test_path):
        raise FileNotFoundError("Missing track_train.csv or track_test.csv. "
                                "Hãy chạy preprocess/lastfm_track_pipeline.py trước.")

    if not os.path.exists(catalog_path):
        raise FileNotFoundError("Missing artist_tracks_catalog.csv. "
                                "Hãy chạy preprocess/lastfm_track_pipeline.py trước.")

    track_train = pd.read_csv(track_train_path)
    track_test = pd.read_csv(track_test_path)
    catalog = pd.read_csv(catalog_path)

    seen_tracks = defaultdict(set)
    for _, row in track_train.iterrows():
        u = int(row["u_idx"])
        tid = str(row["track_id"])
        seen_tracks[u].add(tid)

    gt_tracks = defaultdict(set)
    for _, row in track_test.iterrows():
        u = int(row["u_idx"])
        tid = str(row["track_id"])
        gt_tracks[u].add(tid)

    artist_catalog = defaultdict(list)
    for _, row in catalog.iterrows():
        aid = str(row["artist_id"])
        tid = str(row["track_id"])
        tname = str(row["track_name"])
        pop = float(row["track_pop"])
        artist_catalog[aid].append((tid, tname, pop))

    # sort tracks per artist by popularity desc
    for aid in artist_catalog:
        artist_catalog[aid].sort(key=lambda x: -x[2])

    return seen_tracks, gt_tracks, artist_catalog


def recall_ndcg_at_k(pred_ids, gt_set, k):
    pred = list(pred_ids)[:k]
    if not gt_set:
        return 0.0, 0.0

    hits = [1 if t in gt_set else 0 for t in pred]
    recall = sum(hits) / len(gt_set)

    dcg = 0.0
    for i, h in enumerate(hits):
        if h:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits)) if ideal_hits > 0 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg


def test_tracks():
    args = get_args()
    set_seed(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- START TRACK-LEVEL TESTING ---")

    # 1. Load Dataset (artist-level) & Build Graph
    dataset = KGATDataset(args)
    adj_indices, adj_relations = dataset.get_adj_matrix()
    adj_indices = adj_indices.to(device)
    adj_relations = adj_relations.to(device)

    n_users = dataset.n_users
    n_items = dataset.n_items

    # 2. Tạo Model & load checkpoint
    model = KGAT(args, dataset.n_users, dataset.n_entities, dataset.n_relations).to(device)

    if not args.model_path:
        args.model_path = os.path.join(args.save_dir, 'model_final.pth')

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {args.model_path}")

    print(f"Loading weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load track-level catalog & splits
    structured_path = os.path.join(args.data_dir, args.dataset)
    seen_tracks, gt_tracks, artist_catalog = load_track_catalog(structured_path)

    # 4. Map i_idx -> artist_id (để tra catalog)
    items_info_path = os.path.join(structured_path, "items_info.csv")
    if not os.path.exists(items_info_path):
        raise FileNotFoundError("Missing items_info.csv trong structured folder.")
    items_info = pd.read_csv(items_info_path)
    iidx_to_artist_id = dict(zip(items_info["i_idx"].astype(int), items_info["artist_id"].astype(str)))

    # 5. Forward model 1 lần để lấy embedding user & item
    with torch.no_grad():
        print("Forwarding model...")
        final_embs = model(adj_indices, adj_relations)
        user_embs = final_embs[:n_users]
        item_embs = final_embs[n_users : n_users + n_items]

    all_users = np.arange(n_users, dtype=np.int64)
    batch_size = getattr(args, "test_batch_size", 128)

    avg_recall, avg_ndcg, n_eval = 0.0, 0.0, 0

    # 6. Lặp qua user batch, tính scores artist, sinh track candidates, đánh giá
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_u = all_users[start:end]
        batch_u_tensor = torch.LongTensor(batch_u).to(device)

        with torch.no_grad():
            batch_u_emb = user_embs[batch_u_tensor]                   # [B, d]
            scores = torch.matmul(batch_u_emb, item_embs.t())         # [B, n_items]
            scores_np = scores.cpu().numpy()

        for idx, u in enumerate(batch_u):
            user_scores = scores_np[idx]  # [n_items]

            # Bước 1: KGAT gợi ý artist (cả đã nghe & mới) -> TopN theo score
            if TOPN_ARTISTS >= n_items:
                topN_idx = np.argsort(-user_scores)
            else:
                part = np.argpartition(-user_scores, TOPN_ARTISTS)[:TOPN_ARTISTS]
                topN_idx = part[np.argsort(-user_scores[part])]

            # Bước 2–4: từ artist -> track, lọc track đã nghe, tính điểm
            cand_tracks = {}  # track_id -> (score, track_name)
            seen_set = seen_tracks.get(int(u), set())
            for a_iidx in topN_idx:
                aid = iidx_to_artist_id.get(int(a_iidx))
                if aid is None:
                    continue
                a_score = float(user_scores[a_iidx])
                tracks = artist_catalog.get(aid, [])
                if not tracks:
                    continue

                for tid, tname, pop in tracks[:TOPM_TRACKS_PER_ARTIST]:
                    # Bước 3: loại hẳn track user đã nghe trong TRAIN
                    if tid in seen_set:
                        continue
                    # Bước 4: score_track = score_artist * log(1 + pop)
                    score_t = a_score * math.log1p(pop)
                    if tid not in cand_tracks or score_t > cand_tracks[tid][0]:
                        cand_tracks[tid] = (score_t, tname)

            if not cand_tracks:
                continue

            ranked = sorted(cand_tracks.items(), key=lambda x: -x[1][0])
            reco_ids = [tid for tid, _ in ranked]

            gt_set = gt_tracks.get(int(u), set())
            if not gt_set:
                continue

            r, n = recall_ndcg_at_k(reco_ids, gt_set, TOPK_TRACKS)
            avg_recall += r
            avg_ndcg += n
            n_eval += 1

        if (start + batch_size) % 1000 == 0:
            print(f"Processed {start + batch_size} users...")

    if n_eval == 0:
        print("No users with test tracks to evaluate.")
    else:
        print(f"TRACK RESULT | Recall@{TOPK_TRACKS}: {avg_recall / n_eval:.4f}, "
              f"NDCG@{TOPK_TRACKS}: {avg_ndcg / n_eval:.4f}")


if __name__ == "__main__":
    test_tracks()
