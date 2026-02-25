import os

# Đường dẫn
DATA_FILE = r'./Data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
PROFILE_FILE = r'./Data/lastfm-dataset-1K/userid-profile.tsv'
OUTPUT_DIR = r'./Data/lastfm_structured'

# File trung gian
AGGREGATED_FILE = os.path.join(OUTPUT_DIR, 'interactions_aggregated.csv')

# Tham số lọc (Chỉ áp dụng cho file interactions_union.csv)
TEST_SIZE = 5
MIN_TRAIN_LISTEN = 3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)