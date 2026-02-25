import os
import numpy as np
import pandas as pd
import collections
import torch
from torch.utils.data import Dataset

class KGATDataset(Dataset):
    """Dataset loader dùng chung cho LastFM-structured và ML1M-structured.

    Quy ước ID:
    - User node: 0 .. n_users-1
    - Entity node: n_users .. n_users + n_entities - 1
    - Item là subset của entity: local item id i_idx (0..n_items-1) -> global id = n_users + i_idx
    - KG triples file (kg_triples.csv) dùng local entity ids: 0..n_entities-1
      -> global = n_users + local_id
    """
    def __init__(self, args):
        self.path = os.path.join(args.data_dir, args.dataset)
        self.batch_size = args.batch_size

        self.max_user_neighbors = getattr(args, "max_user_neighbors", 0)
        self.max_item_degree = getattr(args, "max_item_degree", 0)

        print(f"--- LOADING DATASET (MASTER DATA MODE) ---")

        interact_path = os.path.join(self.path, 'interactions_union.csv')
        item_info_path = os.path.join(self.path, 'items_info.csv')
        user_profile_path = os.path.join(self.path, 'users_profile.csv')
        kg_triples_path = os.path.join(self.path, 'kg_triples.csv')

        if not os.path.exists(interact_path):
            raise FileNotFoundError(f"Missing {interact_path}")
        if not os.path.exists(item_info_path):
            raise FileNotFoundError(f"Missing {item_info_path}")

        df_interact = pd.read_csv(interact_path)
        df_items = pd.read_csv(item_info_path)

        # n_users
        if os.path.exists(user_profile_path):
            df_users = pd.read_csv(user_profile_path)
            self.n_users = len(df_users)
        else:
            # fallback
            self.n_users = int(df_interact['u_idx'].max()) + 1 if len(df_interact) else 0

        # n_items
        self.n_items = len(df_items)

        # kg triples
        print("   -> Generating/Loading KG Triples...")
        if os.path.exists(kg_triples_path):
            kg_df = pd.read_csv(kg_triples_path)
            if not set(['h','r','t']).issubset(kg_df.columns):
                raise ValueError("kg_triples.csv must have columns: h,r,t (local entity ids)")
            self.kg_data = kg_df[['h','r','t']].astype(np.int32).values
            self.n_relations = int(kg_df['r'].max()) + 1 if len(kg_df) else 0
            # n_entities derived from kg local ids and n_items (items always exist)
            if len(kg_df):
                self.n_entities = int(max(kg_df['h'].max(), kg_df['t'].max(), self.n_items-1)) + 1
            else:
                self.n_entities = self.n_items
        else:
            # fallback minimal KG: if a_idx exists, create item->artist relation 0
            kg_triples = []
            if 'a_idx' in df_items.columns:
                n_artists = int(df_items['a_idx'].max()) + 1
                self.n_entities = self.n_items + n_artists
                self.n_relations = 1
                src = df_items[['i_idx','a_idx']].drop_duplicates()
                for _, row in src.iterrows():
                    h = int(row['i_idx'])
                    t = self.n_items + int(row['a_idx'])
                    kg_triples.append([h, 0, t])
            else:
                self.n_entities = self.n_items
                self.n_relations = 0
            self.kg_data = np.array(kg_triples, dtype=np.int32)

        print(f"Stats (Master Data): {self.n_users} Users")
        print(f"Stats (Master Data): {self.n_items} Items, {self.n_entities} Entities, {self.n_relations} Relations")

        # Train/Test interactions
        print("   -> Processing Train/Test Interactions...")
        train_df = df_interact[df_interact['type'].astype(str) == '1']
        test_df  = df_interact[df_interact['type'].astype(str) == '0']

        # convert to dict: user -> list(global_item_id)
        self.train_data = self.dataframe_to_dict_with_offset(train_df, offset=self.n_users)
        self.test_data  = self.dataframe_to_dict_with_offset(test_df,  offset=self.n_users)

        # remove users that have test but no train (stabilize eval)
        bad_users = [u for u in self.test_data.keys() if len(self.train_data.get(u, [])) == 0]
        for u in bad_users:
            self.test_data.pop(u, None)
        if bad_users:
            print(f"   -> Filtered {len(bad_users)} users with test>0 but train=0")

        self.train_users = list(self.train_data.keys())

        # build graph
        self.graph_data = self.build_graph()

    def dataframe_to_dict_with_offset(self, df_data, offset):
        user_item = collections.defaultdict(list)
        if len(df_data) == 0:
            return user_item

        grouped = df_data.groupby('u_idx')['i_idx'].apply(list)

        for u, items in grouped.items():
            items = list(items)

            # optional prune: limit neighbors per user for graph building
            if self.max_user_neighbors and len(items) > self.max_user_neighbors:
                # pick most recent if last_time exists; else random
                if 'last_time' in df_data.columns:
                    # take top by last_time for this user
                    sub = df_data[df_data['u_idx'] == u].sort_values('last_time', ascending=False)
                    items = sub['i_idx'].tolist()[:self.max_user_neighbors]
                else:
                    items = list(np.random.choice(items, self.max_user_neighbors, replace=False))

            final_items = [int(i) + offset for i in items]
            user_item[int(u)] = final_items

        return user_item

    def build_graph(self):
        print("   -> Building Graph Edges ...")
        interact_rel = self.n_relations  # interaction relation id is after KG relations
        graph_edges = []

        # KG edges (undirected)
        for h_local, r, t_local in self.kg_data:
            h = int(h_local) + self.n_users
            t = int(t_local) + self.n_users
            graph_edges.append([h, int(r), t])
            graph_edges.append([t, int(r), h])

        # User-Item edges
        item_to_users = collections.defaultdict(list)

        for u, items in self.train_data.items():
            for i in items:
                graph_edges.append([u, interact_rel, i])
                item_to_users[i].append(u)

        # Reverse edges: Item -> User (optional prune degree)
        for i, users in item_to_users.items():
            users = list(users)
            if self.max_item_degree and len(users) > self.max_item_degree:
                users = list(np.random.choice(users, self.max_item_degree, replace=False))
            for u in users:
                graph_edges.append([i, interact_rel, u])

        return np.array(graph_edges, dtype=np.int64)

    def get_adj_matrix(self):
        edges = self.graph_data
        indices = torch.LongTensor(edges[:, [0, 2]].T)
        relations = torch.LongTensor(edges[:, 1])
        return indices, relations

class CFDataset(Dataset):
    def __init__(self, train_data, user_count, item_count):
        self.train_data = train_data
        self.users = list(train_data.keys())
        self.item_start = user_count
        self.item_end = user_count + item_count

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = self.train_data[u]
        if len(pos_items) == 0:
            return u, 0, 0

        pos_i = np.random.choice(pos_items)
        neg_i = np.random.randint(self.item_start, self.item_end)
        while neg_i in pos_items:
            neg_i = np.random.randint(self.item_start, self.item_end)
        return u, pos_i, neg_i

class KGTripletDataset(Dataset):
    def __init__(self, kg_data, entity_start, entity_end):
        self.kg_data = kg_data
        self.ent_start = entity_start
        self.ent_end = entity_end

    def __len__(self): return len(self.kg_data)

    def __getitem__(self, idx):
        h, r, t = self.kg_data[idx]
        neg_t = np.random.randint(self.ent_start, self.ent_end)
        return h, r, t, neg_t
