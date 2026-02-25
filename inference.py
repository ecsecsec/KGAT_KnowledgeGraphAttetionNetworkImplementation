import os
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from src.model import KGAT
from src.dataset import KGATDataset

def predict(args):
    # Ép sử dụng CPU
    device = torch.device('cpu') 
    print(f"--- ĐANG CHẠY GỢI Ý TRÊN: {device} ---")

    print("-> Đang khởi tạo Dataset (Load Graph Structure)...")
    dataset = KGATDataset(args)
    
    item_info_path = os.path.join(args.data_dir, args.dataset, 'items_info.csv')
    user_info_path = os.path.join(args.data_dir, args.dataset, 'users_profile.csv')
    
    if not os.path.exists(item_info_path):
        print("LỖI: Không tìm thấy file items_info.csv")
        return
        
    df_items = pd.read_csv(item_info_path)
    df_users = pd.read_csv(user_info_path)
    
    idx_to_track = dict(zip(df_items['i_idx'], df_items['track_name']))
    idx_to_artist = dict(zip(df_items['i_idx'], df_items['artist_name']))
    
    try:
        df_users['user_id'] = df_users['user_id'].astype(str)
        target_user_row = df_users[df_users['user_id'] == str(args.target_user_id)]
        
        if target_user_row.empty:
            print(f"LỖI: Không tìm thấy User ID: '{args.target_user_id}'")
            return
        target_u_idx = target_user_row['u_idx'].values[0]
        print(f"-> Đã tìm thấy User: {args.target_user_id} (Internal ID: {target_u_idx})")
    except Exception as e:
        print(f"Lỗi truy xuất user: {e}")
        return

    # --- XỬ LÝ THAM SỐ ---
    if isinstance(args.mess_dropout, str):
        parsed_dropout = eval(args.mess_dropout)
        if isinstance(parsed_dropout, list):
            args.mess_dropout = parsed_dropout[0]
        else:
            args.mess_dropout = float(parsed_dropout)
            
    # Giữ nguyên layer_size dạng string cho model
    pass

    print("-> Đang khởi tạo và load trọng số mô hình...")
    model = KGAT(args, dataset.n_users, dataset.n_entities, dataset.n_relations)
    model.to(device)
    
    model_path = args.pretrain_r if args.pretrain_r else f'./checkpoints/model_epoch_{args.epochs}.pth'
        
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy file model tại: {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"\nLỖI LOAD MODEL: Kích thước không khớp! ({e})")
        print("Gợi ý: Kiểm tra --embed_dim hoặc --relation_dim (thử thêm --relation_dim 64).")
        return
        
    model.eval()

    print("-> Đang tính toán điểm số (Scoring)...")
    with torch.no_grad():
        # Lấy dữ liệu Graph
        adj, relations = dataset.get_adj_matrix()
        adj = adj.to(device)
        relations = relations.to(device)
        
        # [SỬA LỖI QUAN TRỌNG TẠI ĐÂY]
        # Model trả về 1 Tensor duy nhất chứa TẤT CẢ Embedding (User + Item + Entity)
        all_embeddings = model(adj, relations) 
        
        # Kiểm tra nếu nó trả về tuple (phòng hờ), thì lấy phần tử đầu
        if isinstance(all_embeddings, tuple):
            all_embeddings = all_embeddings[0]
            
        # Cắt (Slice) embedding dựa trên ID
        # ID của User: từ 0 đến n_users - 1
        # ID của Item: từ n_users đến n_users + n_items - 1
        
        u_emb = all_embeddings[target_u_idx]  # Lấy vector của User mục tiêu
        
        # Lấy toàn bộ vector của Items (bắt đầu từ n_users)
        start_item_idx = dataset.n_users
        end_item_idx = dataset.n_users + dataset.n_items
        i_embs = all_embeddings[start_item_idx : end_item_idx]
        
        # Tính điểm
        scores = torch.matmul(u_emb, i_embs.t()) 
        
        # Lọc bài đã nghe
        watched_items = []
        if target_u_idx in dataset.train_data:
            watched_items.extend([i - dataset.n_users for i in dataset.train_data[target_u_idx]])
        if target_u_idx in dataset.test_data:
            watched_items.extend([i - dataset.n_users for i in dataset.test_data[target_u_idx]])
            
        if watched_items:
            watched_items = list(set([i for i in watched_items if 0 <= i < dataset.n_items]))
            scores[watched_items] = -np.inf
        
        k = 10
        _, top_indices = torch.topk(scores, k)
        top_indices = top_indices.cpu().numpy()
        
    print("\n" + "="*70)
    print(f"TOP {k} GỢI Ý CHO USER: {args.target_user_id}")
    print("="*70)
    print(f"{'RANK':<5} | {'ARTIST':<30} | {'SONG NAME'}")
    print("-" * 70)
    
    for rank, i_idx in enumerate(top_indices):
        artist = str(idx_to_artist.get(i_idx, 'Unknown'))
        track = str(idx_to_track.get(i_idx, 'Unknown'))
        
        artist_display = (artist[:28] + '..') if len(artist) > 28 else artist
        print(f"#{rank+1:<4} | {artist_display:<30} | {track}")
    print("="*70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KGAT Inference")
    parser.add_argument('--dataset', type=str, default='lastfm_structured')
    parser.add_argument('--data_dir', type=str, default='./Data/')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--relation_dim', type=int, default=32)
    parser.add_argument('--layer_size', type=str, default='[32, 16, 16]')
    parser.add_argument('--aggregator_type', type=str, default='bi')
    parser.add_argument('--target_user_id', type=str, required=True)
    parser.add_argument('--pretrain_r', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mess_dropout', type=str, default='[0.1, 0.1, 0.1]')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2_weight', type=float, default=1e-7)

    args = parser.parse_args()
    predict(args)