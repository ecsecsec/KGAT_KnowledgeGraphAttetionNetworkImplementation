import os
import re
import pandas as pd
import numpy as np


def parse_year_from_title(title: str):
    """Try to extract (YYYY) from movie title."""
    if not isinstance(title, str):
        return np.nan
    m = re.search(r"\((\d{4})\)", title)
    if m:
        return int(m.group(1))
    return np.nan


def preprocess_ml1m(
    raw_dir: str,
    out_dir: str,
    min_rating: int = 4,
    test_size: int = 1,
):
    """
    Preprocess MovieLens-1M for KGAT-style pipeline.

    Outputs (out_dir):
      - users_profile.csv
      - items_info.csv
      - interactions_union.csv
      - kg_triples.csv   (movie-genre + user-age + user-gender)

    KG relations:
      r=0: movie -> genre
      r=1: user  -> age
      r=2: user  -> gender

    Entity indexing (to avoid ID collisions inside kg_triples):
      movies:   [0 .. n_items-1]
      genres:   [n_items .. n_items+n_genres-1]
      ages:     [n_items+n_genres .. +n_ages-1]
      genders:  [... +n_genders-1]
      users:    [... +n_users-1]
    """
    os.makedirs(out_dir, exist_ok=True)

    ratings_path = os.path.join(raw_dir, "ratings.dat")
    users_path = os.path.join(raw_dir, "users.dat")
    movies_path = os.path.join(raw_dir, "movies.dat")

    # -----------------------------
    # 1. Load ratings.dat
    # -----------------------------
    print("[ml1m_preprocess] Loading ratings.dat ...")
    df_r = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    # Convert timestamp to datetime
    df_r["timestamp"] = pd.to_datetime(df_r["timestamp"], unit="s")

    # Implicit feedback filter
    if min_rating is not None:
        df_r = df_r[df_r["rating"] >= min_rating].copy()

    # -----------------------------
    # 2. Map user_id, movie_id -> contiguous index
    # -----------------------------
    print("[ml1m_preprocess] Building user/movie index ...")
    unique_users = sorted(df_r["user_id"].unique().tolist())
    unique_movies = sorted(df_r["movie_id"].unique().tolist())

    user_map = {u: idx for idx, u in enumerate(unique_users)}
    item_map = {m: idx for idx, m in enumerate(unique_movies)}

    df_r["u_idx"] = df_r["user_id"].map(user_map)
    df_r["i_idx"] = df_r["movie_id"].map(item_map)

    # -----------------------------
    # 3. Split train/test by timestamp per user
    # -----------------------------
    print("[ml1m_preprocess] Splitting train/test by timestamp ...")
    df_r = df_r.sort_values(["u_idx", "timestamp"])

    def assign_train_test(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= test_size:
            group["type"] = 1
        else:
            group["type"] = 1
            group.loc[group.index[-test_size:], "type"] = 0
        return group

    df_r = df_r.groupby("u_idx", group_keys=False).apply(assign_train_test)

    # -----------------------------
    # 4. Load users.dat -> users_profile.csv
    # -----------------------------
    print("[ml1m_preprocess] Loading users.dat ...")
    df_u = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
    )

    df_u = df_u[df_u["user_id"].isin(user_map.keys())].copy()
    df_u["u_idx"] = df_u["user_id"].map(user_map)

    users_profile = df_u[
        ["u_idx", "user_id", "gender", "age", "occupation", "zip"]
    ].sort_values("u_idx")

    users_profile.to_csv(os.path.join(out_dir, "users_profile.csv"), index=False)

    # -----------------------------
    # 5. Load movies.dat -> items_info.csv
    # -----------------------------
    print("[ml1m_preprocess] Loading movies.dat ...")
    df_m = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["movie_id", "title", "genres"],
    )

    df_m = df_m[df_m["movie_id"].isin(item_map.keys())].copy()
    df_m["i_idx"] = df_m["movie_id"].map(item_map)
    df_m["year"] = df_m["title"].apply(parse_year_from_title)

    # keep original genres string (pipe-separated)
    df_m["genres"] = df_m["genres"].fillna("(no genres listed)")

    items_info = df_m[["i_idx", "movie_id", "title", "year", "genres"]].sort_values("i_idx")
    items_info.to_csv(os.path.join(out_dir, "items_info.csv"), index=False)

    # -----------------------------
    # 6. Write interactions_union.csv
    # -----------------------------
    print("[ml1m_preprocess] Writing interactions_union.csv ...")
    interactions_union = df_r[["u_idx", "i_idx", "rating", "timestamp", "type"]].copy()
    interactions_union.to_csv(os.path.join(out_dir, "interactions_union.csv"), index=False)

    # -----------------------------
    # 7. Build KG triples: movie-genre + user-age + user-gender
    # -----------------------------
    print("[ml1m_preprocess] Building KG triples (movie-genre, user-age, user-gender) ...")

    n_items = int(items_info["i_idx"].max()) + 1 if len(items_info) else 0
    n_users = int(users_profile["u_idx"].max()) + 1 if len(users_profile) else 0

    # --- movie -> genre (r=0), expand pipe-separated genres to individual genre entities
    m_gen = items_info[["i_idx", "genres"]].copy()
    m_gen["genre"] = m_gen["genres"].str.split("\\|")
    m_gen = m_gen.explode("genre")
    m_gen["genre"] = m_gen["genre"].fillna("(no genres listed)").astype(str).str.strip()

    unique_genres = sorted(m_gen["genre"].unique().tolist())
    genre_map = {g: idx for idx, g in enumerate(unique_genres)}
    n_genres = len(unique_genres)

    kg_movie_genre = pd.DataFrame({
        "h": m_gen["i_idx"].astype(int),
        "r": 0,
        "t": (n_items + m_gen["genre"].map(genre_map).astype(int)),
    }).drop_duplicates()

    # --- user -> age (r=1)
    unique_ages = sorted(users_profile["age"].dropna().unique().tolist())
    age_map = {a: idx for idx, a in enumerate(unique_ages)}
    n_ages = len(unique_ages)
    age_offset = n_items + n_genres

    kg_user_age = pd.DataFrame({
        "h": (n_items + n_genres + n_ages + 2 + users_profile["u_idx"].astype(int)),  # placeholder, will fix after genders computed
        "r": 1,
        "t": (age_offset + users_profile["age"].map(age_map).astype(int)),
    })

    # --- user -> gender (r=2)
    unique_genders = sorted(users_profile["gender"].dropna().unique().tolist())
    gender_map = {g: idx for idx, g in enumerate(unique_genders)}
    n_genders = len(unique_genders)
    gender_offset = n_items + n_genres + n_ages

    # now compute user_offset after genders known
    user_offset = n_items + n_genres + n_ages + n_genders

    kg_user_age["h"] = (user_offset + users_profile["u_idx"].astype(int))
    kg_user_gender = pd.DataFrame({
        "h": (user_offset + users_profile["u_idx"].astype(int)),
        "r": 2,
        "t": (gender_offset + users_profile["gender"].map(gender_map).astype(int)),
    })

    kg_df = pd.concat([kg_movie_genre, kg_user_age, kg_user_gender], ignore_index=True)
    kg_df = kg_df.astype({"h": int, "r": int, "t": int})

    kg_df.to_csv(os.path.join(out_dir, "kg_triples.csv"), index=False)

    # (optional) vocab/debug
    pd.DataFrame({"genre": unique_genres, "genre_idx": list(range(n_genres)),
                  "entity_id": [n_items + i for i in range(n_genres)]}).to_csv(
        os.path.join(out_dir, "entity_genres.csv"), index=False
    )
    pd.DataFrame({"age": unique_ages, "age_idx": list(range(n_ages)),
                  "entity_id": [age_offset + i for i in range(n_ages)]}).to_csv(
        os.path.join(out_dir, "entity_ages.csv"), index=False
    )
    pd.DataFrame({"gender": unique_genders, "gender_idx": list(range(n_genders)),
                  "entity_id": [gender_offset + i for i in range(n_genders)]}).to_csv(
        os.path.join(out_dir, "entity_genders.csv"), index=False
    )
    pd.DataFrame({"u_idx": list(range(n_users)),
                  "entity_id": [user_offset + i for i in range(n_users)]}).to_csv(
        os.path.join(out_dir, "entity_users.csv"), index=False
    )

    total_entities = user_offset + n_users

    print("=== ML1M PREPROCESS DONE ===")
    print(
        f"Users: {len(users_profile)} | "
        f"Items: {len(items_info)} | "
        f"Genres: {n_genres} | Ages: {n_ages} | Genders: {n_genders} | "
        f"Entities (incl. users): {total_entities}"
    )
    print(
        f"Interactions: {len(interactions_union)} | "
        f"Train: {int((interactions_union['type']==1).sum())} | "
        f"Test: {int((interactions_union['type']==0).sum())}"
    )
    print(f"KG Triples: {len(kg_df)} (movie-genre + user-age + user-gender)")


if __name__ == "__main__":
    raw_dir = r"D:\KGAT\Data\MovieLens1M"
    out_dir = r"D:\KGAT\Data\ml1m_structured"

    preprocess_ml1m(
        raw_dir=raw_dir,
        out_dir=out_dir,
        min_rating=4,
        test_size=1,
    )
