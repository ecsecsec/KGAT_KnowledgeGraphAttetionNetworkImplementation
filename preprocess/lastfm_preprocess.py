import os
import pandas as pd
import numpy as np
import gzip

RAW_LOG_CANDIDATES = [
    "lastfm_union_clean.tsv",
    "lastfm_union_clean.tsv.gz",
]


def _read_lastfm_log_robust(path: str) -> pd.DataFrame:
    """Read LastFM 1K listening log robustly.

    The canonical format is 6 tab-separated fields:
        user_id, timestamp, artist_id, artist_name, track_id, track_name

    In practice, artist_name/track_name may contain stray tab characters, which
    breaks pandas' C parser (variable number of fields). We parse line-by-line
    and recover fields using a mix of split/rsplit so that *any* extra tabs stay
    inside the name fields.
    """
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # Split first 3 fixed fields, keep the rest as a blob.
            first = line.split("\t", 3)
            if len(first) < 4:
                continue
            user_id, timestamp, artist_id, rest = first
            # Now split from the right to reliably extract track_id & track_name.
            tail = rest.rsplit("\t", 2)
            if len(tail) < 3:
                continue
            artist_name, track_id, track_name = tail
            rows.append((user_id, timestamp, artist_id, artist_name, track_id, track_name))
    return pd.DataFrame(
        rows,
        columns=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
    )

def _find_log_file(raw_dir: str) -> str:
    for fn in RAW_LOG_CANDIDATES:
        p = os.path.join(raw_dir, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find LastFM log file in {raw_dir}. Tried: {RAW_LOG_CANDIDATES}")

def preprocess_lastfm(raw_dir: str, out_dir: str,
                      test_size: int = 5,
                      min_global_count: int = 1,
                      min_user_interactions: int = 1,
                      min_train_listen: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    log_path = _find_log_file(raw_dir)
    profile_path = os.path.join(raw_dir, "userid-profile.tsv")

    # Load raw logs robustly (handles stray tabs inside names)
    df = _read_lastfm_log_robust(log_path)
    # Ensure dtypes
    df["user_id"] = df["user_id"].astype(str)
    df["artist_id"] = df["artist_id"].astype(str)
    df["track_id"] = df["track_id"].astype(str)
    df = df.dropna(subset=["user_id","timestamp","artist_id","track_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Clean names (display only)
    df["artist_name"] = df["artist_name"].astype(str).str.strip()
    df["track_name"]  = df["track_name"].astype(str).str.strip()

    # Aggregate interactions by stable IDs
    inter = (df.groupby(["user_id","track_id","artist_id"], as_index=False)
               .agg(listen_count=("timestamp","size"),
                    first_time=("timestamp","min"),
                    last_time=("timestamp","max")))
    # attach one representative name for info
    info = (df.sort_values("timestamp")
              .drop_duplicates(subset=["track_id"])
              [["track_id","artist_id","track_name","artist_name"]])
    inter = inter.merge(info, on=["track_id","artist_id"], how="left")

    # Global item pruning (by total listens)
    if min_global_count > 1:
        item_sum = inter.groupby("track_id")["listen_count"].sum()
        keep = item_sum[item_sum >= min_global_count].index
        inter = inter[inter["track_id"].isin(keep)]

    # User pruning (by number of interactions)
    if min_user_interactions > 1:
        user_cnt = inter.groupby("user_id")["track_id"].nunique()
        keep_u = user_cnt[user_cnt >= min_user_interactions].index
        inter = inter[inter["user_id"].isin(keep_u)]

    # Build user map
    users = sorted(inter["user_id"].unique().tolist())
    user_map = {u:i for i,u in enumerate(users)}
    users_df = pd.DataFrame({"u_idx": [user_map[u] for u in users], "user_id": users})

    # merge profile (optional)
    if os.path.exists(profile_path):
        prof = pd.read_csv(profile_path, sep="\t")
        if "userid" in prof.columns and "user_id" not in prof.columns:
            prof = prof.rename(columns={"userid":"user_id"})
        users_df = users_df.merge(prof, on="user_id", how="left")
    users_df.to_csv(os.path.join(out_dir, "users_profile.csv"), index=False)

    # Build item map by track_id (stable)
    tracks = inter[["track_id","artist_id","track_name","artist_name"]].drop_duplicates(subset=["track_id"]).copy()
    tracks = tracks.reset_index(drop=True)
    tracks["i_idx"] = np.arange(len(tracks), dtype=int)

    # Artist map by artist_id
    artists = tracks[["artist_id","artist_name"]].drop_duplicates(subset=["artist_id"]).reset_index(drop=True)
    artists["a_idx"] = np.arange(len(artists), dtype=int)

    tracks = tracks.merge(artists, on=["artist_id","artist_name"], how="left")
    items_info = tracks[["i_idx","track_id","track_name","a_idx","artist_id","artist_name"]]
    items_info.to_csv(os.path.join(out_dir, "items_info.csv"), index=False)

    # Map interactions to indices
    inter["u_idx"] = inter["user_id"].map(user_map)
    item_map = dict(zip(tracks["track_id"], tracks["i_idx"]))
    inter["i_idx"] = inter["track_id"].map(item_map)

    # Split train/test by last_time (most recent)
    inter = inter.sort_values(["u_idx","last_time"], ascending=[True, False])
    test = inter.groupby("u_idx").head(test_size).copy()
    test["type"] = 0
    train = inter.drop(test.index).copy()
    train["type"] = 1

    # Train prune by listen_count
    if min_train_listen > 1:
        train = train[train["listen_count"] >= min_train_listen]

    union = pd.concat([train, test], ignore_index=True)
    union = union[["u_idx","i_idx","listen_count","first_time","last_time","type"]]
    union.to_csv(os.path.join(out_dir, "interactions_union.csv"), index=False)

    # Build KG triples: item -> artist (relation 0), local entity ids
    # local item id = i_idx ; local artist entity id = n_items + a_idx
    n_items = len(tracks)
    kg = tracks[["i_idx","a_idx"]].drop_duplicates()
    kg_df = pd.DataFrame({
        "h": kg["i_idx"].astype(int),
        "r": 0,
        "t": (n_items + kg["a_idx"].astype(int))
    })
    kg_df.to_csv(os.path.join(out_dir, "kg_triples.csv"), index=False)

    print("=== LASTFM PREPROCESS DONE ===")
    print("Users:", len(users_df), "Items:", n_items, "Artists:", len(artists),
          "Entities:", n_items + len(artists))
    print("Interactions:", len(union), "Train:", int((union['type']==1).sum()), "Test:", int((union['type']==0).sum()))
