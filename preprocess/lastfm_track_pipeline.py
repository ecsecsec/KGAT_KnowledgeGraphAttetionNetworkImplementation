
import os
import gzip
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


RAW_LOG_CANDIDATES = [
    "lastfm_union_clean.tsv",
    "lastfm_union_clean.tsv.gz",
]

PROFILE_CANDIDATES = ["userid-profile.tsv"]


def _open_text(path: str):
    """Open text file or gzip transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _parse_ts(ts: str):
    """Parse ISO-like timestamp; returns datetime or None."""
    ts = (ts or "").strip()
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _find_first_existing(raw_dir: str, candidates):
    for fn in candidates:
        p = os.path.join(raw_dir, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find any of {candidates} in {raw_dir}")


def iter_lastfm_rows(path: str):
    """
    Yield cleaned rows: (user_id, dt, artist_id, artist_name, track_id, track_name)

    Line format (nominal):
      user_id, timestamp, artist_id, artist_name, track_id, track_name (tab-separated)

    We parse robustly to handle extra tabs in names, and drop rows if:
      - cannot split into required pieces
      - missing user_id / artist_id / track_id
      - invalid timestamp
    """
    with _open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            # Split first 3 fixed fields, keep rest in one blob.
            first = line.split("\t", 3)
            if len(first) < 4:
                continue
            user_id, timestamp, artist_id, rest = first

            # Now split from right into 3: artist_name, track_id, track_name
            tail = rest.rsplit("\t", 2)
            if len(tail) < 3:
                continue
            artist_name, track_id, track_name = tail

            user_id = (user_id or "").strip()
            artist_id = (artist_id or "").strip()
            track_id = (track_id or "").strip()
            timestamp = (timestamp or "").strip()
            artist_name = (artist_name or "").strip()
            track_name = (track_name or "").strip()

            if not user_id or not artist_id or not track_id:
                continue

            dt = _parse_ts(timestamp)
            if dt is None:
                continue

            yield user_id, dt, artist_id, artist_name, track_id, track_name


def build_track_agg(raw_path: str, progress_every: int = 500_000) -> pd.DataFrame:
    """
    Aggregate raw listening events into track-level interactions:

      key:   (user_id, artist_id, track_id)
      value: listen_count, first_time, last_time, artist_name, track_name
    """
    rows = []
    cnt = 0
    for user_id, dt, artist_id, artist_name, track_id, track_name in iter_lastfm_rows(raw_path):
        rows.append((user_id, artist_id, track_id, artist_name, track_name, dt))
        cnt += 1
        if progress_every and cnt % progress_every == 0:
            print(f"[track_agg] scanned {cnt:,} events...", flush=True)

    df = pd.DataFrame(
        rows,
        columns=["user_id", "artist_id", "track_id", "artist_name", "track_name", "timestamp"],
    )

    agg = (
        df.groupby(["user_id", "artist_id", "track_id"], as_index=False)
          .agg(
              listen_count=("timestamp", "size"),
              first_time=("timestamp", "min"),
              last_time=("timestamp", "max"),
              artist_name=("artist_name", "first"),
              track_name=("track_name", "first"),
          )
    )
    print(f"[track_agg] aggregated rows: {len(agg):,}")
    return agg


def split_track_train_test(agg: pd.DataFrame, test_k: int = 10):
    """
    For each user, sort by last_time (descending) and take top-k rows as test,
    remaining as train.
    """
    agg = agg.sort_values(["user_id", "last_time"], ascending=[True, False])

    test = agg.groupby("user_id").head(test_k).copy()
    test["type"] = 0

    train = agg.drop(test.index).copy()
    train["type"] = 1

    print(f"[split] train interactions: {len(train):,}, test interactions: {len(test):,}")
    return train, test


def build_user_artist_maps(train: pd.DataFrame, test: pd.DataFrame, raw_dir: str, out_dir: str):
    """
    From track_train & track_test, build:

      - users_profile.csv
      - items_info.csv (artists as items)
      - interactions_union.csv (artist-level for KGAT)
      - track_train.csv / track_test.csv (track-level splits)
      - artist_tracks_catalog.csv (per-artist track popularity from TRAIN ONLY)
      - kg_triples.csv (simple self-loops, can be replaced by co-listen KG)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- user map ---
    users = sorted(set(train["user_id"]) | set(test["user_id"]))
    user_map = {u: i for i, u in enumerate(users)}
    users_df = pd.DataFrame({"u_idx": [user_map[u] for u in users], "user_id": users})

    # optional: merge profile
    profile_path = None
    for fn in PROFILE_CANDIDATES:
        p = os.path.join(raw_dir, fn)
        if os.path.exists(p):
            profile_path = p
            break

    if profile_path is not None:
        prof = pd.read_csv(profile_path, sep="\t")
        if "user_id" not in prof.columns:
            if "userid" in prof.columns:
                prof = prof.rename(columns={"userid": "user_id"})
            elif "#id" in prof.columns:
                prof = prof.rename(columns={"#id": "user_id"})
        if "user_id" in prof.columns:
            prof["user_id"] = prof["user_id"].astype(str)
            users_df = users_df.merge(prof, on="user_id", how="left")

    users_df.to_csv(os.path.join(out_dir, "users_profile.csv"), index=False)

    # --- artist map ---
    artists = (
        pd.concat(
            [train[["artist_id", "artist_name"]],
             test[["artist_id", "artist_name"]]],
            ignore_index=True,
        )
        .drop_duplicates(subset=["artist_id"])
        .reset_index(drop=True)
    )
    artists["i_idx"] = np.arange(len(artists), dtype=int)
    items_info = artists[["i_idx", "artist_id", "artist_name"]]
    items_info.to_csv(os.path.join(out_dir, "items_info.csv"), index=False)

    artist_map = dict(zip(artists["artist_id"], artists["i_idx"]))

    # --- save track_train / track_test with u_idx ---
    for df, name in ((train, "track_train.csv"), (test, "track_test.csv")):
        df = df.copy()
        df["u_idx"] = df["user_id"].map(user_map).astype(int)
        df.to_csv(os.path.join(out_dir, name), index=False)

    # --- artist-level interactions for KGAT (union) ---
    # train side: sum listen_count per (user, artist)
    artist_train = (
        train.groupby(["user_id", "artist_id"], as_index=False)
             .agg(
                 listen_count=("listen_count", "sum"),
                 first_time=("first_time", "min"),
                 last_time=("last_time", "max"),
             )
    )
    artist_train["u_idx"] = artist_train["user_id"].map(user_map).astype(int)
    artist_train["i_idx"] = artist_train["artist_id"].map(artist_map).astype(int)
    artist_train["type"] = 1

    # test side (optional, used for artist-level eval if needed)
    artist_test = (
        test.groupby(["user_id", "artist_id"], as_index=False)
            .agg(
                listen_count=("listen_count", "sum"),
                first_time=("first_time", "min"),
                last_time=("last_time", "max"),
            )
    )
    artist_test["u_idx"] = artist_test["user_id"].map(user_map).astype(int)
    artist_test["i_idx"] = artist_test["artist_id"].map(artist_map).astype(int)
    artist_test["type"] = 0

    union = pd.concat([artist_train, artist_test], ignore_index=True)
    union = union[["u_idx", "i_idx", "listen_count", "first_time", "last_time", "type"]]
    union.to_csv(os.path.join(out_dir, "interactions_union.csv"), index=False)

    # --- artist_tracks_catalog (TRAIN ONLY) ---
    track_train = pd.read_csv(os.path.join(out_dir, "track_train.csv"))
    cat = (
        track_train.groupby(["artist_id", "track_id", "track_name"], as_index=False)
                   .agg(track_pop=("listen_count", "sum"))
    )
    cat.to_csv(os.path.join(out_dir, "artist_tracks_catalog.csv"), index=False)

    # --- KG: simple self-loops (can be replaced later) ---
    n_items = len(artists)
    kg_df = pd.DataFrame({
        "h": np.arange(n_items, dtype=int),
        "r": 0,
        "t": np.arange(n_items, dtype=int),
    })
    kg_df.to_csv(os.path.join(out_dir, "kg_triples.csv"), index=False)

    print("=== LASTFM TRACK PIPELINE DONE ===")
    print(f"Users: {len(users_df)} | Artists(Items): {n_items}")
    print(f"Track train: {len(train):,} | Track test: {len(test):,}")
    print(f"Artist interactions (union): {len(union):,}")


def preprocess_lastfm_track(raw_dir: str, out_dir: str, test_k: int = 10):
    raw_path = _find_first_existing(raw_dir, RAW_LOG_CANDIDATES)
    print(f"[pipeline] using log: {raw_path}")
    agg = build_track_agg(raw_path)
    train, test = split_track_train_test(agg, test_k=test_k)
    build_user_artist_maps(train, test, raw_dir, out_dir)


if __name__ == "__main__":
    # Example direct run; in practice, you can call preprocess_lastfm_track
    preprocess_lastfm_track(
        raw_dir="./Data/lastfm-dataset-1K",
        out_dir="./Data/lastfm_structured",
        test_k=10,
    )
