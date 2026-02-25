import argparse

def get_args():
    parser = argparse.ArgumentParser(description="KGAT Implementation")

    # -- Dataset & Paths --
    parser.add_argument('--dataset', type=str, default='amazon-book', help='Tên dataset folder (vd: lastfm_structured, ml1m_structured)')
    parser.add_argument('--data_dir', type=str, default='./Data/', help='Thư mục chứa data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='Thư mục lưu model')
    parser.add_argument('--model_path', type=str, default='', help='Đường dẫn file model để test')

    # -- Model Architecture --
    parser.add_argument('--embed_dim', type=int, default=64, help='Kích thước embedding (d)')
    parser.add_argument('--relation_dim', type=int, default=64, help='Kích thước embedding quan hệ (k)')
    parser.add_argument('--layer_size', nargs='?', default='[64, 32, 16]', help='Output size các lớp GNN')
    parser.add_argument('--aggregator_type', type=str, default='bi', help='Loại: bi (Bi-Interaction), gcn, graphsage')
    parser.add_argument('--mess_dropout', type=float, default=0.1, help='Tỷ lệ dropout khi truyền tin')

    # -- Graph pruning (loader-level, để giảm RAM nếu cần) --
    parser.add_argument('--max_user_neighbors', type=int, default=0,
                        help='Giới hạn số item mỗi user dùng để build graph (0 = không prune)')
    parser.add_argument('--max_item_degree', type=int, default=0,
                        help='Giới hạn số user mỗi item nhận tin khi build graph (0 = không prune)')

    # -- Training --
    parser.add_argument('--epochs', type=int, default=100, help='Số lượng epoch')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size cho CF và KG')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--l2_weight', type=float, default=1e-5, help='L2 Regularization')
    parser.add_argument('--kg_l2_weight', type=float, default=1e-5, help='L2 Regularization cho KG')

    # -- Evaluation --
    parser.add_argument('--topk', type=int, default=20, help='Top K cho metric Recall/NDCG')

    return parser.parse_args()
