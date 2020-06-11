# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphRec(nn.Module):
    def __init__(self, num_users, num_items, num_ratings, history_u, history_i, history_ur,\
                                     history_ir, embed_dim, social_neighbor, cuda='cpu'):
        super(GraphRec, self).__init__()

        self.embed_dim = embed_dim
        u2e = nn.Embedding(num_users, self.embed_dim)
        i2e = nn.Embedding(num_items, self.embed_dim)
        r2e = nn.Embedding(num_ratings, self.embed_dim)
        self.enc_u = UI_Aggregator(i2e, r2e, u2e, embed_dim, history_u, history_ur, cuda, user=True)
        self.enc_i = UI_Aggregator(i2e, r2e, u2e, embed_dim, history_i, history_ir, cuda, user=False)
        self.enc_social = Social_Aggregator(None, u2e, embed_dim, social_neighbor, cuda)

        self.w_u = nn.Linear(2*self.embed_dim, self.embed_dim)

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ir1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ir2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ui1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_ui2 = nn.Linear(self.embed_dim, 16)
        self.w_ui3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_i):
        item_space = self.enc_u(nodes_u)
        social_space = self.enc_social(nodes_u)
        user_latent_feature = torch.cat([item_space, social_space], dim=1)
        user_latent_feature = F.relu(self.w_u(user_latent_feature))
        item_latent_feature = self.enc_i(nodes_i)

        user_latent_feature = F.relu(self.bn1(self.w_ur1(user_latent_feature)))
        user_latent_feature = F.dropout(user_latent_feature, training=self.training)
        user_latent_feature = self.w_ur2(user_latent_feature)
        item_latent_feature = F.relu(self.bn2(self.w_ir1(item_latent_feature)))
        item_latent_feature = F.dropout(item_latent_feature, training=self.training)
        item_latent_feature = self.w_ir2(item_latent_feature)

        latent_feature = torch.cat((user_latent_feature, item_latent_feature), 1)
        latent_feature = F.relu(self.bn3(self.w_ui1(latent_feature)))
        latent_feature = F.dropout(latent_feature, training=self.training)
        latent_feature = F.relu(self.bn4(self.w_ui2(latent_feature)))
        latent_feature = F.dropout(latent_feature, training=self.training)
        scores = self.w_ui3(latent_feature)

        return scores.squeeze()

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)

        return self.criterion(scores, ratings)


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)
        return att


class UI_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """
    def __init__(self, i2e, r2e, u2e, embed_dim, history_ui, history_r, cuda="cpu", user=True):
        super(UI_Aggregator, self).__init__()
        # history_ui: {user1:[item1, ...], ...} or {item1:[user1, ...], ...}
        # history_r: {user1:[rating1, ...], ...} or {item1:[rating1, ...], ...} corresponding with history_ui
        self.user = user
        self.i2e = i2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.history_ui = history_ui
        self.history_r = history_r
        self.w_r1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.linear1 = nn.Linear(2*self.embed_dim, self.embed_dim) 

    def forward(self, nodes):
        # nodes: [2, 1, 3, 0] user or item index
        ui_history = []
        r_history = []
        for node in nodes:
            ui_history.append(self.history_ui[int(node)])
            r_history.append(self.history_r[int(node)])

        num_len = len(ui_history)

        embed_matrix = torch.empty(num_len, self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(num_len):
            history = ui_history[i]
            num_histroy_ui = len(history)
            tmp_label = r_history[i]

            if self.user == True:
                # user component
                e_ui = self.i2e.weight[history]
                ui_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_ui = self.u2e.weight[history]
                ui_rep = self.i2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_ui, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, ui_rep, num_histroy_ui)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        neigh_feats = embed_matrix

        if self.user == True:
            self_feats = self.u2e.weight[nodes]
        else:
            self_feats = self.i2e.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined_feats = F.relu(self.linear1(combined))

        return combined_feats


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """
    def __init__(self, features, u2e, embed_dim, social_neighbor, cuda="cpu"):
        super(Social_Aggregator, self).__init__()
        # social_neighbor: {2:[2,..], } neighbors
        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.social_neighbor = social_neighbor
        self.att = Attention(self.embed_dim)
        self.linear1 = nn.Linear(2*self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        # nodes: [2, 1, 3, 0] user index
        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_neighbor[int(node)])
        num_len = len(nodes)
        embed_matrix = torch.empty(num_len, self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(num_len):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            e_u = self.u2e.weight[list(tmp_adj)] # fast: user embedding 
            #slow: item-space user latent factor (item aggregation)
            # feature_neighbors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            # e_u = torch.t(feature_neighbors)

            u_rep = self.u2e.weight[nodes[i]]

            att_w = self.att(e_u, u_rep, num_neighs)
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        neigh_feats = embed_matrix

        # self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        # self_feats = self_feats.t()
        self_feats = self.u2e.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined_feats = F.relu(self.linear1(combined))

        return combined_feats
