import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from src.config import get_args
from src.dataset import KGATDataset, CFDataset, KGTripletDataset
from src.model import KGAT
from src.utils import set_seed

def train():
    args = get_args()
    set_seed(2023)
    device = torch.device('cpu')
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset_generator = KGATDataset(args)
    adj_indices, adj_relations = dataset_generator.get_adj_matrix()
    adj_indices = adj_indices.to(device)
    adj_relations = adj_relations.to(device)
    
    # 2. Tạo Model
    model = KGAT(args, dataset_generator.n_users, dataset_generator.n_entities, dataset_generator.n_relations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # 3. Training Loop
    print("Start Training...")
    
    # Tính toán range ID cho việc sampling
    # Entity ID bắt đầu từ n_users
    entity_start_id = dataset_generator.n_users
    entity_end_id = dataset_generator.n_users + dataset_generator.n_entities
    # Item ID cũng bắt đầu từ n_users (vì Item là Entity đầu tiên)
    item_end_id = dataset_generator.n_users + dataset_generator.n_items

    for epoch in range(args.epochs):
        model.train()
        
        # -- Phase 1: Train KG (TransR) --
        kg_loader = DataLoader(
            KGTripletDataset(
                dataset_generator.kg_data, 
                entity_start=entity_start_id, 
                entity_end=entity_end_id
            ), 
            batch_size=args.batch_size, shuffle=True
        )
        
        total_kg_loss = 0
        for h, r, t, neg_t in kg_loader:
            h, r, t, neg_t = h.to(device), r.to(device), t.to(device), neg_t.to(device)
            
            optimizer.zero_grad()
            kg_loss = model.calc_kg_loss(h, r, t, neg_t)
            
            l2_loss = 0
            for name, param in model.named_parameters():
                if 'relation' in name or 'W_R' in name:
                    l2_loss += torch.norm(param) ** 2
            loss = kg_loss + args.kg_l2_weight * l2_loss
            
            loss.backward()
            optimizer.step()
            total_kg_loss += kg_loss.item()

        # -- Phase 2: Train CF (Recommendation) --
        cf_loader = DataLoader(
            CFDataset(
                dataset_generator.train_data, 
                user_count=dataset_generator.n_users, # Start ID của Item
                item_count=dataset_generator.n_items  # Số lượng Item
            ),
            batch_size=args.batch_size, shuffle=True
        )
        
        total_cf_loss = 0
        for u, i_pos, i_neg in cf_loader:
            u, i_pos, i_neg = u.to(device), i_pos.to(device), i_neg.to(device)
            
            optimizer.zero_grad()
            final_embs = model(adj_indices, adj_relations)
            
            cf_loss = model.calc_cf_loss(u, i_pos, i_neg, final_embs)
            
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.norm(param) ** 2
                
            loss = cf_loss + args.l2_weight * l2_loss
            loss.backward()
            optimizer.step()
            total_cf_loss += cf_loss.item()
            
        print(f"Epoch {epoch+1}: KG Loss = {total_kg_loss:.4f}, CF Loss = {total_cf_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pth'))
    print("Training Finished.")

if __name__ == '__main__':
    train()