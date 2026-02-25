import os
import gzip
from collections import Counter
from datetime import datetime

RAW_LOG_CANDIDATES = [
    "userid-timestamp-artid-artname-traid-traname.tsv",
    "userid-timestamp-artid-artname-traid-traname.tsv.gz",
]

def find_log(raw_dir: str) -> str:
    for fn in RAW_LOG_CANDIDATES:
        p = os.path.join(raw_dir, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find log file in {raw_dir}. Tried {RAW_LOG_CANDIDATES}")

def parse_timestamp(ts: str):
    # Examples:
    # 2009-05-04T23:08:57Z
    # 2009-05-04 23:08:57+00:00
    ts = ts.strip()
    if not ts:
        return None
    try:
        # Handle ...Z
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def clean_lastfm_tsv(
    raw_dir: str,
    out_dir: str,
    out_name: str = "lastfm_union_clean.tsv",
    progress_every: int = 500_000,
):
    os.makedirs(out_dir, exist_ok=True)
    in_path = find_log(raw_dir)
    out_path = os.path.join(out_dir, out_name)

    opener = gzip.open if in_path.endswith(".gz") else open

    stats = Counter()
    kept = 0

    with opener(in_path, "rt", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "wt", encoding="utf-8", newline="") as fout:

        # header
        fout.write("user_id\ttimestamp\tartist_id\tartist_name\ttrack_id\ttrack_name\n")

        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line:
                stats["empty_line"] += 1
                continue

            # Robust parse:
            # Split first 3 fixed fields, keep the rest blob
            first = line.split("\t", 3)
            if len(first) < 4:
                stats["too_few_fields"] += 1
                continue

            user_id, timestamp, artist_id, rest = first

            # Extract last 2 fields: track_id, track_name
            tail = rest.rsplit("\t", 2)
            if len(tail) < 3:
                stats["cannot_rsplit_tail"] += 1
                continue

            artist_name, track_id, track_name = tail

            # Strip
            user_id = user_id.strip()
            timestamp = timestamp.strip()
            artist_id = artist_id.strip()
            track_id = track_id.strip()
            artist_name = artist_name.strip()
            track_name = track_name.strip()

            # Required non-empty
            if not user_id:
                stats["missing_user_id"] += 1
                continue
            if not artist_id:
                stats["missing_artist_id"] += 1
                continue
            if not track_id:
                stats["missing_track_id"] += 1
                continue

            dt = parse_timestamp(timestamp)
            if dt is None:
                stats["bad_timestamp"] += 1
                continue

            # Optional: also require non-empty names (uncomment if you want stricter)
            # if not artist_name:
            #     stats["missing_artist_name"] += 1
            #     continue
            # if not track_name:
            #     stats["missing_track_name"] += 1
            #     continue

            # Write cleaned line (keep original timestamp string to preserve format)
            fout.write(f"{user_id}\t{timestamp}\t{artist_id}\t{artist_name}\t{track_id}\t{track_name}\n")
            kept += 1
            stats["kept"] += 1

            if progress_every and (line_no % progress_every == 0):
                print(f"[clean] processed {line_no:,} lines | kept {kept:,}", flush=True)

    # Print summary
    print("=== CLEAN DONE ===")
    print("Input:", in_path)
    print("Output:", out_path)
    print("Kept rows:", kept)
    print("Dropped rows breakdown:")
    for k, v in stats.most_common():
        if k != "kept":
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    RAW_DIR = r"D:\KGAT\Data\lastfm-dataset-1K"
    OUT_DIR = r"D:\KGAT\Data\lastfm_clean_strict_tsv"
    clean_lastfm_tsv(RAW_DIR, OUT_DIR)
