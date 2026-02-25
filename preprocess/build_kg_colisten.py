import os
import pandas as pd
from collections import Counter


def build_colisten_kg(
    structured_dir: str,
    interactions_file: str = "interactions_union.csv",
    per_user_topk: int = 50,      # mỗi user: lấy tối đa K artist nghe nhiều nhất
    per_item_topk: int = 20,      # mỗi artist: giữ tối đa N neighbors
    min_edge_users: int = 2,      # cạnh phải được >= số user cùng nghe
    relation_id: int = 0,         # id của quan hệ "co_listen"
    out_name: str = "kg_triples.csv",  # ghi đè luôn file KG hiện tại
):
    """
    Xây KG co-listen cho LastFM từ interactions_union.csv (artist-level).

    structured_dir: folder chứa lastfm_structured
    interactions_file: thường là "interactions_union.csv"
    per_user_topk: lấy tối đa bao nhiêu artist/user để tạo cặp (co-listen)
    per_item_topk: giữ top neighbors/artist
    min_edge_users: cặp artist phải có ít nhất số user này cùng nghe
    """

    path = os.path.join(structured_dir, interactions_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    print(f"[co-listen] loading interactions from {path}")
    inter = pd.read_csv(path)

    # chỉ dùng train (type == 1)
    if "type" in inter.columns:
        inter = inter[inter["type"] == 1].copy()

    # đảm bảo kiểu số
    inter["u_idx"] = inter["u_idx"].astype(int)
    inter["i_idx"] = inter["i_idx"].astype(int)

    # sort theo user, listen_count giảm dần để lấy topK / user
    inter = inter.sort_values(["u_idx", "listen_count"], ascending=[True, False])

    # lấy topK artist/user
    topk = inter.groupby("u_idx").head(per_user_topk)
    print(f"[co-listen] interactions after top-{per_user_topk} per user: {len(topk):,}")

    # đếm co-listen pair
    edge_counter = Counter()

    for u, grp in topk.groupby("u_idx"):
        items = grp["i_idx"].tolist()
        if len(items) < 2:
            continue
        # loại trùng, sort để đảm bảo (i,j) với i<j
        items = sorted(set(items))
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                edge_counter[(items[a], items[b])] += 1

    print(f"[co-listen] raw pairs: {len(edge_counter):,}")

    # lọc cạnh theo số user cùng nghe
    edges = []
    for (i, j), c in edge_counter.items():
        if c >= min_edge_users:
            edges.append((i, j, c))

    if not edges:
        raise ValueError(
            "No edges left after filtering. "
            "Try giảm min_edge_users hoặc tăng per_user_topk."
        )

    edges_df = pd.DataFrame(edges, columns=["i", "j", "cnt_users"])
    print(f"[co-listen] edges after min_edge_users={min_edge_users}: {len(edges_df):,}")

    # tạo cạnh vô hướng -> 2 chiều (i->j, j->i)
    dir_edges = pd.concat([
        edges_df.rename(columns={"i": "h", "j": "t"}),
        edges_df.rename(columns={"j": "h", "i": "t"}),
    ], ignore_index=True)

    # sort theo h, cnt_users giảm dần
    dir_edges = dir_edges.sort_values(["h", "cnt_users"], ascending=[True, False])

    # giữ top neighbors cho mỗi artist
    dir_edges = dir_edges.groupby("h").head(per_item_topk).reset_index(drop=True)

    # build kg_triples: h, r, t
    kg = pd.DataFrame({
        "h": dir_edges["h"].astype(int),
        "r": relation_id,
        "t": dir_edges["t"].astype(int),
    })

    out_path = os.path.join(structured_dir, out_name)
    kg.to_csv(out_path, index=False)

    print("=== CO-LISTEN KG BUILT ===")
    print(f"Saved KG to: {out_path}")
    print(f"Directed edges: {len(kg):,}")
    print(f"Unique heads (artists with neighbors): {kg['h'].nunique():,}")


if __name__ == "__main__":
    # sửa lại đường dẫn cho đúng
    build_colisten_kg(
        structured_dir=r"D:\KGAT\Data\lastfm_structured",
        interactions_file="interactions_union.csv",
        per_user_topk=50,
        per_item_topk=20,
        min_edge_users=2,
        relation_id=0,
        out_name="kg_triples.csv",  # ghi đè KG cũ
    )
