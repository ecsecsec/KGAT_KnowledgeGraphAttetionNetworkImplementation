KGAT Rebuild Package (Complete) - LastFM-1K + MovieLens1M
=======================================================

Mục tiêu
- Fix triệt để lỗi "nổ item" của LastFM do dùng track_name làm key -> chuyển sang track_id/artist_id.
- Chuẩn hoá output để dataset loader đọc 1 format chung cho cả LastFM và MovieLens1M.
- Thêm first_time + last_time cho interactions, split test theo last_time.
- Xuất KG triples ra file kg_triples.csv (local entity ids), dataset.py sẽ auto-load.

Cấu trúc output (mỗi dataset nằm trong: Data/<dataset_name>/)
- users_profile.csv: u_idx, user_id, (optional profile cols)
- items_info.csv: i_idx, item_id, (optional metadata...), (optional a_idx/artist_id/...)
- interactions_union.csv: u_idx, i_idx, listen_count/weight, first_time, last_time, type (1=train,0=test)
- kg_triples.csv: h, r, t  (local entity ids; items: 0..n_items-1; other entities tiếp nối từ n_items)

Chạy LastFM (sanity - chưa prune nặng)
1) Preprocess:
   python preprocess/run_preprocess.py --dataset lastfm --raw_dir <THU_MUC_RAW_LASTFM> --out_dir ./Data/lastfm_structured --test_size 5

2) Train/Test:
   python train.py --dataset lastfm_structured --data_dir ./Data/ ...

MovieLens1M
1) Preprocess:
   python preprocess/run_preprocess.py --dataset ml1m --raw_dir <THU_MUC_RAW_ML1M> --out_dir ./Data/ml1m_structured --test_size 1

Ghi chú
- Bạn có thể bật prune sau khi sanity ok:
  --min_global_count, --min_user_interactions, --min_train_listen
- Dataset loader có thêm 2 tham số để kiểm soát prune graph:
  --max_user_neighbors (mặc định 0 = không prune)
  --max_item_degree (mặc định 0 = không prune)
