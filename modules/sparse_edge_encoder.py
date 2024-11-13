
import torch, random
import torch.nn as nn
import torch.nn.functional as F
from itertools import product


class SparseEdgeEncoder(nn.Module):

    def __init__(self, args):
        super(SparseEdgeEncoder, self).__init__()
        self.module = []
        # self.hook = self.module.register_forward_hook(self.hook_fn)
        self.nfeat = args.dim_feat
        self.nhid = 16
        self.nlayers = 3
        self.gcls = args.gcls
        self.n_graphs = args.fake_num
        self.dev = args.gpu

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(self.nfeat * 2, self.nhid))

        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for i in range(self.nlayers - 2):
            self.layers.append(nn.Linear(self.nhid, self.nhid))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.layers.append(nn.Linear(self.nhid, 1))

        # generate node features
        # num_graphs = len(adj_list)
        # assert self.n_graphs < num_graphs
        # random_indices = torch.randperm(num_graphs)[:self.n_graphs]
        # self.sampled_adjs = [adj_list[idx].to(f'cuda:{self.dev}') for idx in random_indices]
        # self.number_list = [adj.shape[0] for adj in self.sampled_adjs]

        self.number_list = random.choices(range(10, 30), k=self.n_graphs)

        self.node_features = nn.ParameterList([
            nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(n, self.nfeat))) for n in self.number_list
        ])

        # self.node_features = nn.ParameterList([
        #     nn.Parameter(torch.rand(n, self.nfeat), requires_grad=False) for n in self.number_list
        # ])
        # sample_labels = target_labels[random_indices].to(f'cuda:{self.dev}')
        sample_labels = torch.tensor([random.randint(0, args.gcls - 1) for _ in range(self.n_graphs)])
        self.register_buffer("labels", sample_labels)

        # self.reset_parameters()

    def forward(self):

        labels = self.labels.tolist()
        feas = []
        # adjs = self.sampled_adjs
        adjs = []
        for feat in self.node_features:
            # feat = torch.softmax(feat, 0)
            feas += [feat.to(f'cuda:{self.dev}')]
            nnodes = feat.shape[0]
            
            ####### generate A from X 
            edge_index = torch.tensor(list(product(range(nnodes), range(nnodes)))).T.to(feat.device)

            # adj = feat @ feat.t()
            # wx
            edge_embed = torch.cat([feat[edge_index[0]], feat[edge_index[1]]], axis=1)
            for ix, layer in enumerate(self.layers):
                edge_embed = layer(edge_embed)
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)

            # edge_prob = torch.sigmoid(edge_embed)
            adj = edge_embed.reshape(nnodes, nnodes)
            adj = torch.sigmoid(adj)
            adj = (adj + adj.T) / 2  # 111
            new_adj = ((adj * (adj > 0.9).float()) - adj).detach() + adj
            # 对邻接矩阵每一行用gumbel_softmax, 并使其保持对称性和离散性
            # adj_gumbel = self.sym_gum_soft(adj)

            adj_expanded = torch.stack([1 - new_adj, new_adj], dim=-1)

            adj_gumbel = F.gumbel_softmax(adj_expanded, tau=0.1, hard=True)
            adj_gumbel = adj_gumbel[..., 1]
            adj_gumbel.fill_diagonal_(fill_value=0)
            adjs += [adj_gumbel]
            #######

            ####### generate X and random A
            # adj = torch.randint(0, 2, (nnodes, nnodes))
            # adj = (adj + adj.T) / 2
            # adj.fill_diagonal_(fill_value=0)
            # adjs += [adj.to(f'cuda:{self.dev}')]

            ########

        return labels, adjs, feas


    def sym_gum_soft(self, adj):
        adj_gumbel = F.gumbel_softmax(adj, tau=0.2, hard=True)
        for i in range(2):
            adj_gumbel += F.gumbel_softmax(adj, tau=0.2, hard=True)
        new_adj_symmetric = (adj_gumbel + adj_gumbel.T) / 2
        # 使用 Straight-Through Estimator (STE) 方法
        new_adj_binary = ((new_adj_symmetric > 0.4).float() - new_adj_symmetric).detach() + new_adj_symmetric

        return new_adj_binary