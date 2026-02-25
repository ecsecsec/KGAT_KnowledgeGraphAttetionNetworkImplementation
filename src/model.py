import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, type='bi'):
        super(Aggregator, self).__init__()
        self.type = type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        
        if type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)
        elif type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)
        elif type == 'bi':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.type != 'bi':
            nn.init.xavier_uniform_(self.W.weight)
        else:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, entity_embed, neighbor_embed):
        if self.type == 'gcn':
            out = self.W(entity_embed + neighbor_embed)
            return self.act(out)
        elif self.type == 'graphsage':
            out = torch.cat([entity_embed, neighbor_embed], dim=1)
            out = self.W(out)
            return self.act(out)
        elif self.type == 'bi':
            sum_embed = self.W1(entity_embed + neighbor_embed)
            prod_embed = self.W2(entity_embed * neighbor_embed)
            return self.act(sum_embed + prod_embed)

class KGAT(nn.Module):
    def __init__(self, args, n_users, n_entities, n_relations):
        super(KGAT, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations + 1
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        
        # --- SỬA LỖI Ở ĐÂY ---
        # Embedding phải chứa cả User và Entity để tránh lỗi Index Out of Bounds
        # ID 0 -> n_users-1: User
        # ID n_users -> n_users + n_entities - 1: Entity
        self.entity_embed = nn.Embedding(self.n_users + self.n_entities, self.embed_dim)
        
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.W_R)

        # Aggregation Layers
        self.layer_dims = [self.embed_dim] + eval(args.layer_size)
        self.aggregator_layers = nn.ModuleList()
        self.relation_projs = nn.ModuleList()
        
        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i+1]
            self.aggregator_layers.append(Aggregator(in_dim, out_dim, args.mess_dropout, args.aggregator_type))
            self.relation_projs.append(nn.Linear(self.relation_dim, in_dim))

    def calc_kg_loss(self, h, r, t, neg_t):
        r_embed = self.relation_embed(r)
        W_r = self.W_R[r]
        
        h_embed = self.entity_embed(h)
        t_embed = self.entity_embed(t)
        neg_t_embed = self.entity_embed(neg_t)
        
        h_proj = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        t_proj = torch.bmm(t_embed.unsqueeze(1), W_r).squeeze(1)
        neg_t_proj = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)
        
        pos_score = torch.sum(torch.pow(h_proj + r_embed - t_proj, 2), dim=1)
        neg_score = torch.sum(torch.pow(h_proj + r_embed - neg_t_proj, 2), dim=1)
        
        kg_loss = F.softplus(pos_score - neg_score).mean()
        return kg_loss

    def forward(self, edge_index, edge_type):
        current_embed = self.entity_embed.weight
        all_embeds = [current_embed]
        
        heads = edge_index[0]
        tails = edge_index[1]
        
        for i, aggregator in enumerate(self.aggregator_layers):
            h_emb = current_embed[heads]
            t_emb = current_embed[tails]
            
            r_emb = self.relation_embed(edge_type)
            r_emb = self.relation_projs[i](r_emb)
            
            score = torch.sum(h_emb * torch.tanh(h_emb + r_emb), dim=1)
            
            neighbor_embed = torch.zeros_like(current_embed)
            weighted_tails = t_emb * score.unsqueeze(1)
            neighbor_embed.index_add_(0, heads, weighted_tails)
            
            current_embed = aggregator(current_embed, neighbor_embed)
            current_embed = F.normalize(current_embed, p=2, dim=1)
            all_embeds.append(current_embed)
        
        return torch.cat(all_embeds, dim=1)

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, final_embeds):
        u_embed = final_embeds[user_ids]
        i_pos_embed = final_embeds[item_pos_ids]
        i_neg_embed = final_embeds[item_neg_ids]
        
        pos_score = torch.sum(u_embed * i_pos_embed, dim=1)
        neg_score = torch.sum(u_embed * i_neg_embed, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
        return loss