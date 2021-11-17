import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import dgl.function as fn


class DAGNNConv(nn.Module):
    def __init__(self, G, feature_attn_size, k):
        super(DAGNNConv, self).__init__()

        self.s = nn.Parameter(torch.FloatTensor(feature_attn_size, 1))
        self.k = k

        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):

        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feats = graph.ndata['h']
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self, G, feature_attn_size, bias=True, activation=None, dropout=0, slope=0.2):
        super(MLPLayer, self).__init__()

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.G = G
        self.slope = slope

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, G):

        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)

        feats = self.G.ndata.pop('z')

        return feats

class AttentionLayer(nn.Module):
    def __init__(self, G, feature_attn_size, bias=True, activation=None, dropout=0, slope=0.2):
        super(AttentionLayer, self).__init__()

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.G = G
        self.slope = slope

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = alpha * nodes.mailbox['z']
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, G):

        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)

        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)

        feats = self.G.ndata.pop('h')

        return feats

class ADPMDA(nn.Module):
    def __init__(self, G, k, feature_attn_size, num_diseases,num_mirnas, d_sim_dim,m_sim_dim,
                        out_dim,dropout,slope,activation=F.relu, bias=True):
        super(ADPMDA, self).__init__()

        self.G = G
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas

        self.mlp = nn.ModuleList()

        self.mlp.append(AttentionLayer(G, feature_attn_size, bias=bias,
                                 activation=activation, dropout=dropout, slope=slope))
        self.mlp.append(AttentionLayer(G, feature_attn_size, bias=bias,
                                 activation=None, dropout=dropout, slope=slope))                 

        self.dagnn = DAGNNConv(G, feature_attn_size, k=k)

        self.m_fc = nn.Linear(feature_attn_size + m_sim_dim, out_dim)
        self.d_fc = nn.Linear(feature_attn_size + d_sim_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.h_fc = nn.Linear(out_dim, out_dim)
        self.predict = nn.Linear(out_dim * 2, 1)

    def forward(self, G,  diseases, mirnas):
        for layer in self.mlp:
            feats = layer(G)
        feats = self.dagnn(G, feats)

        h_d = torch.cat((feats[:self.num_diseases], self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        h_m = torch.cat((feats[self.num_diseases:], self.G.ndata['m_sim'][self.num_diseases:]), dim=1)

        h_m = self.dropout(F.elu(self.m_fc(h_m)))
        h_d = self.dropout(F.elu(self.d_fc(h_d)))
        h = torch.cat((h_d, h_m), dim=0)

        h_diseases = h[diseases]
        h_mirnas = h[mirnas]

        h_concat = torch.cat((h_diseases, h_mirnas), 1)
        predict_score = torch.sigmoid(self.predict(h_concat))

        return predict_score
