import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN cơ bản cho bipartite user–item graph:
      - Node ID: 0..n_users-1 -> Users
                  n_users..n_users+n_items-1 -> Items
      - adj: torch.sparse.FloatTensor (N x N) normalized adjacency
    """
    def __init__(self, n_users, n_items, embed_dim, n_layers, adj):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.adj = adj  # sparse normalized adjacency (to(device) trước)

        # Embedding riêng cho user & item (không chia sẻ như KGAT)
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def get_all_embeddings(self):
        """
        Trả về:
          user_all: (n_users, d)
          item_all: (n_items, d)
        Sau khi đã propagate K layer và average.
        """
        # Nút 0..n_users-1 là user, n_users..n_users+n_items-1 là item
        all_emb = torch.cat([self.user_emb.weight,
                             self.item_emb.weight], dim=0)  # (N, d)

        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)  # (N, K+1, d)
        final_emb = torch.mean(embs, dim=1)  # (N, d)

        user_all, item_all = torch.split(
            final_emb, [self.n_users, self.n_items], dim=0
        )
        return user_all, item_all

    def bpr_loss(self, users, pos_items_global, neg_items_global, l2_reg=0.0):
        """
        users: (B,) user index [0..n_users-1]
        pos_items_global, neg_items_global: (B,) global ID của item
          (>= n_users, do KGATDataset đã cộng offset)
        """
        user_embs, item_embs = self.get_all_embeddings()

        # Map global ID -> local item index
        pos_local = pos_items_global - self.n_users
        neg_local = neg_items_global - self.n_users

        u_e = user_embs[users]
        p_e = item_embs[pos_local]
        n_e = item_embs[neg_local]

        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)

        bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization nhẹ
        reg = (u_e.pow(2).sum() +
               p_e.pow(2).sum() +
               n_e.pow(2).sum()) / users.shape[0]

        loss = bpr + l2_reg * reg
        return loss, bpr.item(), reg.item()
