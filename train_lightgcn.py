import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from src.config import get_args
from src.dataset import KGATDataset, CFDataset
from src.lightgcn import LightGCN
from src.utils import set_seed


def build_ui_adj_matrix(n_users, n_items, train_data, device):
    """
    Xây adjacency bipartite user–item:
      A = [[0, R],
           [R^T, 0]]
    Sau đó chuẩn hoá đối xứng: D^{-1/2} A D^{-1/2}
    """
    print("   -> Building user–item adjacency for LightGCN...")
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)

    for u, items in train_data.items():
        for g_i in items:
            # g_i là Global ID của item: n_users .. n_users + n_items - 1
            i = g_i - n_users
            if 0 <= i < n_items:
                R[u, i] = 1.0

    R = R.tocsr()

    N = n_users + n_items
    adj_mat = sp.dok_matrix((N, N), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.tocsr()

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj_mat.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = D_inv_sqrt @ adj_mat @ D_inv_sqrt
    norm_adj = norm_adj.tocoo()

    indices = torch.from_numpy(
        np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
    )
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    sparse_adj = torch.sparse.FloatTensor(indices, values, shape)

    return sparse_adj.to(device)


def train_lightgcn():
    args = get_args()
    set_seed(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading Dataset...")
    dataset_generator = KGATDataset(args)

    n_users = dataset_generator.n_users
    n_items = dataset_generator.n_items

    # Xây adjacency user–item
    adj = build_ui_adj_matrix(n_users, n_items,
                              dataset_generator.train_data,
                              device)

    # Số layer LightGCN = độ dài layer_size (ví dụ "[64,32,16]" -> 3 layer)
    n_layers = len(eval(args.layer_size))

    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embed_dim=args.embed_dim,
        n_layers=n_layers,
        adj=adj,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cf_loader = DataLoader(
        CFDataset(
            dataset_generator.train_data,
            user_count=n_users,
            item_count=n_items
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_bpr = 0.0
        total_reg = 0.0

        for users, pos_i, neg_i in cf_loader:
            users = users.to(device)
            pos_i = pos_i.to(device)
            neg_i = neg_i.to(device)

            optimizer.zero_grad()
            loss, bpr, reg = model.bpr_loss(
                users, pos_i, neg_i, l2_reg=args.l2_weight
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bpr += bpr
            total_reg += reg

        avg_loss = total_loss / len(cf_loader)
        avg_bpr = total_bpr / len(cf_loader)
        avg_reg = total_reg / len(cf_loader)

        print(f"Epoch {epoch+1}: "
              f"Loss={avg_loss:.4f}, "
              f"BPR={avg_bpr:.4f}, Reg={avg_reg:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(
                args.save_dir, f'lightgcn_epoch_{epoch+1}.pth'
            )
            torch.save(model.state_dict(), save_path)

    final_path = os.path.join(args.save_dir, 'lightgcn_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Training Finished. Saved to {final_path}")


if __name__ == '__main__':
    train_lightgcn()
