import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch import GATConv
from pretrain.GIN.gin_all import ApplyNodeFunc, MLP, GINConv
from pretrain.GIN.readout import SumPooling, AvgPooling, MaxPooling
from modules.mask_layer import MaskGraphConv, MaskLinear


class MaskGCN(nn.Module):
    """
    复制预训练模型参数，输出output + mask_loss
    """
    def __init__(self, num_layers, input_dim, hidden_dim,
                 output_dim, final_dropout=0.5, graph_pooling_type='sum', dev=0, mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
        super(MaskGCN, self).__init__()
        # allow_zero_in_degree = True
        self.num_layers = num_layers
        self.gcnlayers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dev = dev

        for layer in range(self.num_layers - 1):
            # if layer == 0:
            #     self.gcnlayers.append(
            #         MaskGraphConv(input_dim, hidden_dim, allow_zero_in_degree=True, mask_init=mask_init,
            #                       mask_scale=mask_scale,
            #                       threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask, dev=self.dev))
            # else:
            #     self.gcnlayers.append(
            #         MaskGraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True, mask_init=mask_init,
            #                       mask_scale=mask_scale,
            #                       threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask, dev=self.dev)
            #     )

            if layer == 0:
                self.gcnlayers.append(GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True))
            else:
                self.gcnlayers.append(
                    GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
                )

            self.norms.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))

        self.linears_prediction = MaskLinear(hidden_dim, output_dim, mask_init=mask_init, mask_scale=mask_scale,
                                             threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask,
                                             dev=self.dev)

        # self.linears_prediction = nn.Linear(hidden_dim, output_dim)

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
            # h = h.to(f"cuda:{self.dev}")
            # g = g.to(f"cuda:{self.dev}")

            h = self.gcnlayers[i](g, h, edge_weight=edge_weight)
            if torch.isnan(h).any():
                print('before norm, h is nan')
            h = self.norms[i](h)
            if torch.isnan(h).any():
                print('after norm, h is nan')
            if i != 0:
                h = F.relu(h) + x
            else:
                h = F.relu(h)
            hidden_rep.append(h)
            # if torch.isnan(h).any():
            #     print('after relu, h is nan')

        score_over_layer = 0

        pooled_h = self.pool(g, hidden_rep[-1])
        if training:
            score_over_layer += self.drop(self.linears_prediction(pooled_h))
        else:
            score_over_layer += self.linears_prediction(pooled_h)
        return score_over_layer


class MaskGAT(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, graph_pooling_type, dev=0, num_heads=8, mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
        super(MaskGAT, self).__init__()
        self.num_layers = num_layers
        self.gatlayers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dev = dev
        self.num_heads = num_heads

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.gatlayers.append(GATConv(input_dim, hidden_dim, num_heads=self.num_heads, allow_zero_in_degree=True))
            else:
                self.gatlayers.append(GATConv(hidden_dim * self.num_heads, hidden_dim, num_heads=self.num_heads, allow_zero_in_degree=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim * self.num_heads, eps=1e-3))

        # self.linears_prediction = nn.Linear(hidden_dim * self.num_heads, output_dim)
        self.linears_prediction = MaskLinear(hidden_dim * self.num_heads, output_dim, mask_init=mask_init, mask_scale=mask_scale,
                                             threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask,
                                             dev=self.dev)
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
            h = self.gatlayers[i](g, h, edge_weight=edge_weight)
            # Flatten the output to (n, num_heads * hidden_dim)
            h = h.view(h.size(0), -1)
            h = self.norms[i](h)

            if i != 0:
                h = F.relu(h) + x  # Skip connection
            else:
                h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # Pooling the final layer's hidden state
        pooled_h = self.pool(g, hidden_rep[-1])
        if training:
            score_over_layer += self.drop(self.linears_prediction(pooled_h))
        else:
            score_over_layer += self.linears_prediction(pooled_h)

        return score_over_layer
    

class MaskGIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, dev=0, mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
        super(MaskGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.dev = dev
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps)
            )

        # self.linears_prediction = nn.Linear(hidden_dim, output_dim)
        self.linears_prediction = MaskLinear(hidden_dim, output_dim, mask_init=mask_init, mask_scale=mask_scale,
                                             threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask,
                                             dev=self.dev)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h, training=True):
        hidden_rep = [h]
        split_list = g.batch_num_nodes

        for i in range(self.num_layers - 1):
            x = h
            h = self.ginlayers[i](g, split_list, h)

            if i != 0:
                h = h + x
            hidden_rep.append(h)

        score_over_layer = 0
        pooled_h = self.pool(g, hidden_rep[-1])  # 批次内每张图节点特征相加
        score_over_layer = score_over_layer + self.drop(self.linears_prediction(pooled_h))

        return score_over_layer


def MaskGCN5_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGCN(num_layers=5, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGCN5_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGCN(num_layers=5, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGCN3_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGCN(num_layers=3, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGCN3_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGCN(num_layers=3, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGCN2_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGCN(num_layers=2, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGCN2_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGCN(num_layers=2, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


##### GIN ###########

def MaskGIN5_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGIN(num_layers=5, num_mlp_layers=2, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGIN5_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGIN(num_layers=5, num_mlp_layers=2, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGIN3_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGIN(num_layers=3, num_mlp_layers=2, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGIN3_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGIN(num_layers=3, num_mlp_layers=2, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGIN2_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGIN(num_layers=2, num_mlp_layers=2, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


def MaskGIN2_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGIN(num_layers=2, num_mlp_layers=2, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, learn_eps=False, graph_pooling_type='sum',
               neighbor_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)


#### GAT ###########

def MaskGAT5_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGAT(num_layers=5, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)

def MaskGAT5_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGAT(num_layers=5, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)

def MaskGAT3_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGAT(num_layers=3, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)

def MaskGAT3_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGAT(num_layers=3, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)

def MaskGAT2_64(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGAT(num_layers=2, input_dim=args.dim_feat, hidden_dim=64,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)

def MaskGAT2_32(args, mask_init='1s', mask_scale=1,
                 threshold_fn='binarizer', threshold=0.5, apply_mask=True):
    return MaskGAT(num_layers=2, input_dim=args.dim_feat, hidden_dim=32,
               output_dim=args.gcls, final_dropout=0.5, graph_pooling_type='sum', dev=args.gpu, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, apply_mask=apply_mask)
