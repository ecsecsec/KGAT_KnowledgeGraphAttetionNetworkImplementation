import torch
import numpy as np
import os
from src.config import get_args
from src.dataset import KGATDataset
from src.model import KGAT
from src.utils import compute_metrics, set_seed

def test():
    args = get_args()
    set_seed(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("--- START TESTING (FIXED ID OFFSET) ---")
    
    # 1. Load Data
    dataset_generator = KGATDataset(args)
    
    adj_indices, adj_relations = dataset_generator.get_adj_matrix()
    adj_indices = adj_indices.to(device)
    adj_relations = adj_relations.to(device)
    
    # 2. Init Model
    model = KGAT(args, dataset_generator.n_users, dataset_generator.n_entities, dataset_generator.n_relations).to(device)
    
    # 3. Load Weights
    if not args.model_path:
        args.model_path = os.path.join(args.save_dir, 'model_final.pth')
        
    if os.path.exists(args.model_path):
        print(f"Loading weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Error: Không tìm thấy file model tại {args.model_path}")
        return

    model.eval()
    
    # Lấy thông số kích thước
    n_users = dataset_generator.n_users
    n_items = dataset_generator.n_items
    
    with torch.no_grad():
        # A. Lấy Embedding
        print("Forwarding model...")
        final_embs = model(adj_indices, adj_relations)
        
        # Tách User và Item
        # User: 0 -> n_users
        user_embs = final_embs[:n_users]
        # Item: n_users -> n_users + n_items
        item_embs = final_embs[n_users : n_users + n_items]
        
        test_users = list(dataset_generator.test_data.keys())
        avg_recall = []
        avg_ndcg = []
        
        test_batch_size = 100 
        
        for i in range(0, len(test_users), test_batch_size):
            batch_u = test_users[i : i + test_batch_size]
            batch_u_tensor = torch.LongTensor(batch_u).to(device)
            
            # Tính điểm: (Batch x N_Items)
            # score[0] tương ứng với Item đầu tiên trong item_embs (tức là Global ID = n_users)
            batch_u_emb = user_embs[batch_u_tensor]
            scores = torch.matmul(batch_u_emb, item_embs.t())
            
            # Chuyển về CPU để xử lý mask
            scores = scores.cpu().numpy() 
            
            # B. Masking (SỬA LỖI QUAN TRỌNG)
            for idx, u in enumerate(batch_u):
                train_items_global = dataset_generator.train_data[u]
                
                # train_items_global đang là ID thật (vd: 1000).
                # scores đang index từ 0 (tương ứng 1000).
                # Cần trừ đi n_users để map về index của scores
                train_items_relative = [t - n_users for t in train_items_global if t >= n_users]
                
                # Gán điểm thấp cho bài đã train
                scores[idx][train_items_relative] = -np.inf
            
            # C. Top-K
            # top_k_items này trả về Relative Index (0, 1, 2...)
            top_k_indices = np.argpartition(scores, -args.topk, axis=1)[:, -args.topk:]
            
            # Sort lại cho chính xác thứ tự
            sorted_top_k = []
            for idx in range(len(batch_u)):
                u_scores = scores[idx]
                top_items = top_k_indices[idx]
                top_items = top_items[np.argsort(u_scores[top_items])[::-1]]
                sorted_top_k.append(top_items)
            
            # D. So khớp (SỬA LỖI QUAN TRỌNG)
            for idx, u in enumerate(batch_u):
                ground_truth = dataset_generator.test_data[u] # ID Global (vd: 1005)
                
                # pred_items đang là Relative (vd: 5). Cần cộng n_users để thành Global (1005)
                pred_items_relative = sorted_top_k[idx]
                pred_items_global = pred_items_relative + n_users
                
                r, n = compute_metrics(pred_items_global, ground_truth)
                avg_recall.append(r)
                avg_ndcg.append(n)
                
            if (i + test_batch_size) % 1000 == 0:
                print(f"Processed {i + test_batch_size} users...")
                
        print(f"RESULT | Recall@{args.topk}: {np.mean(avg_recall):.4f}, NDCG@{args.topk}: {np.mean(avg_ndcg):.4f}")

if __name__ == '__main__':
    test()