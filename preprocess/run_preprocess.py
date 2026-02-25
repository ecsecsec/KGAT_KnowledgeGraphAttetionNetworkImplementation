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

    
    # =========================
    # Build KG triples
    # Relations:
    #   r=0: track -> artist
    #   r=1: artist -> related_artist (co-listen similarity)
    #   r=2: user  -> context (gender/age/country from userid-profile.tsv when available)
    #
    # Entity indexing (local):
    #   track entities:   0 .. n_items-1
    #   artist entities:  n_items .. n_items+n_artists-1
    #   user entities:    n_items+n_artists .. n_items+n_artists+n_users-1
    #   context entities: n_items+n_artists+n_users .. ...
    # =========================

    n_items = len(tracks)
    n_artists = len(artists)
    n_users = len(users)

    # r=0: track -> artist
    kg0 = tracks[["i_idx","a_idx"]].drop_duplicates()
    kg0_df = pd.DataFrame({
        "h": kg0["i_idx"].astype(int),
        "r": 0,
        "t": (n_items + kg0["a_idx"].astype(int)),
    })

    # r=1: artist -> related_artist (co-listen among users, from train interactions)
    # Build artist-level interactions per user from TRAIN ONLY
    per_user_topk = 50
    per_artist_topk = 20
    min_edge_users = 2

    # reconstruct artist index for each interaction: a_idx via tracks mapping
    track_to_aidx = dict(zip(tracks["i_idx"].astype(int), tracks["a_idx"].astype(int)))
    union_full = pd.concat([train, test], ignore_index=False)  # train/test still have u_idx,i_idx,listen_count
    # Only train
    inter_train = train[["u_idx","i_idx","listen_count"]].copy()
    inter_train["a_idx"] = inter_train["i_idx"].map(track_to_aidx)

    # Sum listen_count per (user, artist)
    ua = (inter_train.groupby(["u_idx","a_idx"], as_index=False)
                   .agg(cnt=("listen_count","sum")))
    # Take top-K artists per user
    ua = ua.sort_values(["u_idx","cnt"], ascending=[True, False])
    ua_top = ua.groupby("u_idx").head(per_user_topk)

    from collections import Counter
    edge_counter = Counter()
    for u, grp in ua_top.groupby("u_idx"):
        a_list = sorted(set(grp["a_idx"].astype(int).tolist()))
        if len(a_list) < 2:
            continue
        for i in range(len(a_list)):
            ai = a_list[i]
            for j in range(i+1, len(a_list)):
                aj = a_list[j]
                edge_counter[(ai, aj)] += 1

    edges = [(ai, aj, c) for (ai, aj), c in edge_counter.items() if c >= min_edge_users]
    if edges:
        edges_df = pd.DataFrame(edges, columns=["a_i","a_j","cnt_users"])
        # undirected -> directed
        dir_edges = pd.concat([
            edges_df.rename(columns={"a_i":"a_src","a_j":"a_dst"}),
            edges_df.rename(columns={"a_j":"a_src","a_i":"a_dst"}),
        ], ignore_index=True)
        dir_edges = dir_edges.sort_values(["a_src","cnt_users"], ascending=[True, False])
        dir_edges = dir_edges.groupby("a_src").head(per_artist_topk).reset_index(drop=True)

        kg1_df = pd.DataFrame({
            "h": (n_items + dir_edges["a_src"].astype(int)),
            "r": 1,
            "t": (n_items + dir_edges["a_dst"].astype(int)),
        })
    else:
        kg1_df = pd.DataFrame(columns=["h","r","t"], dtype=int)

    # r=2: user -> context
    # Contexts are taken from userid-profile.tsv when available (gender/age/country).
    # Missing values are skipped.
    ctx_rows = []
    if os.path.exists(profile_path):
        prof = users_df.copy()
        # normalize columns
        # Expected columns from original dataset: gender, age, country, signup
        # Some files may use different cases.
        colmap = {c.lower(): c for c in prof.columns}
        gender_col = colmap.get("gender")
        age_col    = colmap.get("age")
        country_col= colmap.get("country")

        def _add_ctx(u_local: int, key: str, val):
            if pd.isna(val):
                return
            sval = str(val).strip()
            if not sval or sval.lower() == "nan":
                return
            ctx_rows.append((u_local, f"{key}:{sval}"))

        for _, row in prof.iterrows():
            u_local = int(row["u_idx"])
            if gender_col:
                _add_ctx(u_local, "gender", row[gender_col])
            if age_col:
                _add_ctx(u_local, "age", row[age_col])
            if country_col:
                _add_ctx(u_local, "country", row[country_col])

    if ctx_rows:
        ctx_df = pd.DataFrame(ctx_rows, columns=["u_idx","context"])
        ctx_df = ctx_df.drop_duplicates()
        contexts = sorted(ctx_df["context"].unique().tolist())
        ctx_map = {c:i for i,c in enumerate(contexts)}
        user_offset = n_items + n_artists
        ctx_offset = user_offset + n_users

        kg2_df = pd.DataFrame({
            "h": (user_offset + ctx_df["u_idx"].astype(int)),
            "r": 2,
            "t": (ctx_offset + ctx_df["context"].map(ctx_map).astype(int)),
        })

        # debug vocab
        pd.DataFrame({"ctx_idx": [ctx_map[c] for c in contexts], "context": contexts}).to_csv(
            os.path.join(out_dir, "entity_contexts.csv"), index=False
        )
    else:
        kg2_df = pd.DataFrame(columns=["h","r","t"], dtype=int)
        user_offset = n_items + n_artists
        ctx_offset = user_offset + n_users

    # Save vocab/debug maps
    artists_vocab = artists.copy()
    artists_vocab["ent_id"] = n_items + artists_vocab["a_idx"].astype(int)
    artists_vocab[["ent_id","a_idx","artist_id","artist_name"]].to_csv(
        os.path.join(out_dir, "entity_artists.csv"), index=False
    )
    users_vocab = users_df[["u_idx","user_id"]].copy()
    users_vocab["ent_id"] = (n_items + n_artists) + users_vocab["u_idx"].astype(int)
    users_vocab.to_csv(os.path.join(out_dir, "entity_users.csv"), index=False)

    # Concatenate all triples
    kg_df = pd.concat([kg0_df, kg1_df, kg2_df], ignore_index=True)
    kg_df = kg_df.astype({"h": int, "r": int, "t": int})
    kg_df.to_csv(os.path.join(out_dir, "kg_triples.csv"), index=False)

    print("=== LASTFM PREPROCESS DONE ===")
    print("Users:", len(users_df), "Items:", n_items, "Artists:", n_artists,
          "Contexts:", (0 if not ctx_rows else len(contexts)))
    print("Entities:", ctx_offset + (0 if not ctx_rows else len(contexts)))
    print("KG triples:", len(kg_df), "| r0(track-artist):", len(kg0_df),
          "| r1(artist-related):", len(kg1_df), "| r2(user-context):", len(kg2_df))
    print("Interactions:", len(union), "Train:", int((union['type']==1).sum()), "Test:", int((union['type']==0).sum()))
