
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from pretrain.GIN.readout import SumPooling, AvgPooling, MaxPooling


class GCN(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, graph_pooling_type, dev):
        super(GCN, self).__init__()
        # allow_zero_in_degree = True
        self.num_layers = num_layers
        self.gcnlayers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dev = dev

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.gcnlayers.append(GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True))
            else:
                self.gcnlayers.append(
                    GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
                )
            self.norms.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))

        self.linears_prediction = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h, edge_weight=None, training=False):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):

            x = h
            h = h.to(f"cuda:{self.dev}")
            g = g.to(f"cuda:{self.dev}")
            h = self.gcnlayers[i](g, h, edge_weight=edge_weight)
            
            h = self.norms[i](h)

            if i != 0:
                h = F.relu(h) + x
            else:
                h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        pooled_h = self.pool(g, hidden_rep[-1])
        if training:
            score_over_layer += self.drop(self.linears_prediction(pooled_h))
        else:
            score_over_layer += self.linears_prediction(pooled_h)

        return score_over_layer


def GCN5_64(args):
    return GCN(num_layers=5, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu)


def GCN5_32(args):
    return GCN(num_layers=5, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu)


def GCN3_64(args):
    return GCN(num_layers=3, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu)


def GCN3_32(args):
    return GCN(num_layers=3, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu)


def GCN2_64(args):
    return GCN(num_layers=2, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu)


def GCN2_32(args):
    return GCN(num_layers=2, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu)