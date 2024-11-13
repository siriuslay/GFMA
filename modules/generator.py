import torch
import torch.nn as nn
from modules.sparse_edge_encoder import SparseEdgeEncoder
# from Model.moe import MoE
from dgl.dataloading import GraphDataLoader
import dgl
from data_loader.Gene_dataset import MergeDataset, FKDataset
from utils import evaluate_ptm, evaluate_moe
import torch.nn.functional as F
import torch.linalg as linalg


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss.
    Adopted from "Dreaming to distill: Data-free knowledge transfer via deepinversion"
    '''

    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        nch = input[0].shape[1]

        mean = input[0].mean([0])

        var = input[0].permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class GRE_Single(nn.Module):
    def __init__(self, args):
        super(GRE_Single, self).__init__()
        self.conf = args
        self.ptm_list = None
        self.experts_num = None
        self.generator = SparseEdgeEncoder(self.conf)
        self.cross_ent = nn.CrossEntropyLoss()

    def build_generator(self, ptm_list):
        self.ptm_list = ptm_list
        self.experts_num = len(ptm_list)

    def forward(self, pretrained_model_list):

        self.build_generator(pretrained_model_list)

        labels, adjs, feas = self.generator()

        labels = torch.tensor(labels).to(feas[0].device)

        graphs, features, edge_weight = self.transform_data(adjs, feas)

        bn_loss_list, ori_ce_list, output_list, conf_loss_list = self.fake_loss(graphs, features, edge_weight, labels)

        data_loader = self.pack_data(labels, adjs, feas)

        # ori_ce_loss = gates.unsqueeze(dim=-1) * torch.stack(ori_ce_list, dim=1)
        # ori_ce_loss = torch.stack(ori_ce_list, dim=-1)
        # ori_ce_loss = ori_ce_loss.sum(dim=-1) / self.conf.num_experts
        # bn_loss = gates.unsqueeze(dim=-1) * torch.stack(bn_loss_list, dim=1)

        return data_loader, ori_ce_list, bn_loss_list, output_list, labels, conf_loss_list

    def fake_loss(self, graphs, features, edge_weight, labels):

        # create hook
        loss_r_feature_layers = []
        for ptm in self.ptm_list:
            ptm_loss_r_feature_layers = []
            for module in ptm.modules():
                if isinstance(module, nn.BatchNorm1d):
                    ptm_loss_r_feature_layers.append(DeepInversionFeatureHook(module))  # list中元素是层数-1个hook
            loss_r_feature_layers.append(ptm_loss_r_feature_layers)

        loss_bn = []
        ori_ce_list = []
        out_list = []
        conf_loss_list = []
        for ptm_id, ptm in enumerate(self.ptm_list):

            output = ptm(graphs, features, edge_weight)

            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers[ptm_id]])

            loss_bn += [loss_distr]
            out_list += [output]
            ori_ce_list += [self.cross_ent(output, labels)]

            softmax_output = F.softmax(output, dim=1)  # 确保输出是 softmax 形式
            log_softmax_output = F.log_softmax(output)  # 防止 log(0) 错误
            entropy = -torch.sum(softmax_output * log_softmax_output, dim=1)
            conf_loss_list += [torch.mean(entropy)]

        # if features_list[0].shape[1] != 1:
        #     # features = torch.softmax(features, 1)
        #     # b = features * torch.log(features)
        #     # fea_loss = -1.0 * b.sum() / len(features)
        #     fea_loss = 0
        # else:
        #     fea_loss = 0

        return loss_bn, ori_ce_list, out_list, conf_loss_list

    def transform_data(self, adjs, feas):

        dgl_graphs = []
        adj_loss = 0
        # 遍历邻接矩阵列表和节点特征列表
        for adj, fea in zip(adjs, feas):
            # 获取边的起始节点和终止节点索引
            src, dst = adj.nonzero(as_tuple=True)

            # 创建DGLGraph
            g = dgl.graph((src, dst), num_nodes=adj.shape[0])
            g.edata['weight'] = adj[src, dst]

            # 将节点特征赋值给图
            # print(fea.shape)
            # assert fea.shape[0] == num_nodes, "Feature dimension mismatch with the number of nodes"
            g.ndata['feat'] = fea

            # 添加到列表中
            dgl_graphs.append(g)

            # adj = torch.softmax(adj, dim=-1)  # 正则化邻接矩阵
            # adj_loss += torch.sum(adj * torch.log(adj + 1e-10)) / adj.size(0)  # 熵作为损失，熵越小损失越小
            # adj_loss += self.fiedler_value_loss(adj)

        graphs = dgl.batch(dgl_graphs)
        features = graphs.ndata['feat']
        edge_weight = graphs.edata['weight']

        return graphs, features, edge_weight

    def pack_data(self, labels, adjs, feas):

        dataset = FKDataset(adjs, feas, labels)
        data_loader = GraphDataLoader(dataset, batch_size=self.conf.batch_size, shuffle=True)

        return data_loader

    def mixup(self, adjs_list, feas_list, labels_list):
        adjs = []
        feas = []
        for i in range(len(adjs_list)):
            adjs += adjs_list[i]
            feas += feas_list[i]
        labels = torch.cat(labels_list)

        return labels, adjs, feas

    def reparameter_loss(self, mu, logvar):
        '''
        graph feature to a Normal Gaussian distribution N (0, 1)
        :param mu:
        :param logvar: std
        :return: reg_loss of features
        '''
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def kl_divergence_bernoulli(self, edge_prob, q=0.1):
        '''

        :param edge_prob: weights
        :param q: hyperparameter
        :return: reg_loss of edges
        '''
        return torch.sum(edge_prob * torch.log(edge_prob / q) + (1 - edge_prob) * torch.log((1 - edge_prob) / (1 - q)))

    def fiedler_value_loss(self, adj):

        laplacian = self.normalized_laplacian(adj)

        # 计算拉普拉斯矩阵的特征值
        eigvals = linalg.eigvalsh(laplacian)  # 使用 eigvalsh 确保返回的特征值是有序的

        # 次小特征值（Fiedler值）是排序后的第 2 个特征值
        fiedler_value = eigvals[1]
        fied_loss = 2 - fiedler_value
        # 我们可以对 Fiedler 值施加一个损失，例如最小化它接近 0 的情况
        return fied_loss

    def normalized_laplacian(self, adj):
        # 计算 D^(-1/2)
        d_inv_sqrt = torch.diag(torch.pow(adj.sum(dim=1), -0.5))

        # 计算归一化拉普拉斯矩阵 L_norm = I - D^(-1/2) * A * D^(-1/2)
        identity = torch.eye(adj.size(0), device=adj.device)
        laplacian_norm = identity - d_inv_sqrt @ adj @ d_inv_sqrt

        return laplacian_norm


class GRE_Shared(nn.Module):
    def __init__(self, args):
        super(GRE_Shared, self).__init__()
        self.conf = args
        self.ptm_list = None
        self.experts_num = None
        self.generator = SparseEdgeEncoder(self.conf)
        self.cross_ent = nn.CrossEntropyLoss()

    def build_generator(self, ptm_list):
        self.ptm_list = ptm_list
        self.experts_num = len(ptm_list)

    def forward(self, pretrained_model_list):

        self.build_generator(pretrained_model_list)

        labels, adjs, feas = self.generator()

        labels = torch.tensor(labels).to(feas[0].device)

        graphs, features, edge_weight = self.transform_data(adjs, feas)

        bn_loss_list, ori_ce_list, output_list, conf_loss_list = self.fake_loss(graphs, features, edge_weight, labels)

        data_loader = self.pack_data(labels, adjs, feas)

        # ori_ce_loss = gates.unsqueeze(dim=-1) * torch.stack(ori_ce_list, dim=1)
        # ori_ce_loss = torch.stack(ori_ce_list, dim=-1)
        # ori_ce_loss = ori_ce_loss.sum(dim=-1) / self.conf.num_experts
        # bn_loss = gates.unsqueeze(dim=-1) * torch.stack(bn_loss_list, dim=1)

        return data_loader, ori_ce_list, bn_loss_list, output_list, labels, conf_loss_list

    def fake_loss(self, graphs, features, edge_weight, labels):

        # create hook
        loss_r_feature_layers = []
        for ptm in self.ptm_list:
            ptm_loss_r_feature_layers = []
            for module in ptm.modules():
                if isinstance(module, nn.BatchNorm1d):
                    ptm_loss_r_feature_layers.append(DeepInversionFeatureHook(module))  # list中元素是层数-1个hook
            loss_r_feature_layers.append(ptm_loss_r_feature_layers)

        loss_bn = []
        ori_ce_list = []
        out_list = []
        conf_loss_list = []
        for ptm_id, ptm in enumerate(self.ptm_list):

            output = ptm(graphs, features, edge_weight)

            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers[ptm_id]])

            loss_bn += [loss_distr]
            out_list += [output]
            ori_ce_list += [self.cross_ent(output, labels)]

            softmax_output = F.softmax(output, dim=1)  # 确保输出是 softmax 形式
            log_softmax_output = F.log_softmax(output)  # 防止 log(0) 错误
            entropy = -torch.sum(softmax_output * log_softmax_output, dim=1)
            conf_loss_list += [torch.mean(entropy)]

        # if features_list[0].shape[1] != 1:
        #     # features = torch.softmax(features, 1)
        #     # b = features * torch.log(features)
        #     # fea_loss = -1.0 * b.sum() / len(features)
        #     fea_loss = 0
        # else:
        #     fea_loss = 0

        return loss_bn, ori_ce_list, out_list, conf_loss_list

    def transform_data(self, adjs, feas):

        dgl_graphs = []
        adj_loss = 0
        # 遍历邻接矩阵列表和节点特征列表
        for adj, fea in zip(adjs, feas):
            # 获取边的起始节点和终止节点索引
            src, dst = adj.nonzero(as_tuple=True)

            # 创建DGLGraph
            g = dgl.graph((src, dst), num_nodes=adj.shape[0])
            g.edata['weight'] = adj[src, dst]

            # 将节点特征赋值给图
            # print(fea.shape)
            # assert fea.shape[0] == num_nodes, "Feature dimension mismatch with the number of nodes"
            g.ndata['feat'] = fea

            # 添加到列表中
            dgl_graphs.append(g)

            # adj = torch.softmax(adj, dim=-1)  # 正则化邻接矩阵
            # adj_loss += torch.sum(adj * torch.log(adj + 1e-10)) / adj.size(0)  # 熵作为损失，熵越小损失越小
            # adj_loss += self.fiedler_value_loss(adj)

        graphs = dgl.batch(dgl_graphs)
        features = graphs.ndata['feat']
        edge_weight = graphs.edata['weight']

        return graphs, features, edge_weight

    def pack_data(self, labels, adjs, feas):

        dataset = FKDataset(adjs, feas, labels)
        data_loader = GraphDataLoader(dataset, batch_size=self.conf.batch_size, shuffle=True)

        return data_loader

    def mixup(self, adjs_list, feas_list, labels_list):
        adjs = []
        feas = []
        for i in range(len(adjs_list)):
            adjs += adjs_list[i]
            feas += feas_list[i]
        labels = torch.cat(labels_list)

        return labels, adjs, feas

    def reparameter_loss(self, mu, logvar):
        '''
        graph feature to a Normal Gaussian distribution N (0, 1)
        :param mu:
        :param logvar: std
        :return: reg_loss of features
        '''
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def kl_divergence_bernoulli(self, edge_prob, q=0.1):
        '''

        :param edge_prob: weights
        :param q: hyperparameter
        :return: reg_loss of edges
        '''
        return torch.sum(edge_prob * torch.log(edge_prob / q) + (1 - edge_prob) * torch.log((1 - edge_prob) / (1 - q)))

    def fiedler_value_loss(self, adj):

        laplacian = self.normalized_laplacian(adj)

        # 计算拉普拉斯矩阵的特征值
        eigvals = linalg.eigvalsh(laplacian)  # 使用 eigvalsh 确保返回的特征值是有序的

        # 次小特征值（Fiedler值）是排序后的第 2 个特征值
        fiedler_value = eigvals[1]
        fied_loss = 2 - fiedler_value
        # 我们可以对 Fiedler 值施加一个损失，例如最小化它接近 0 的情况
        return fied_loss

    def normalized_laplacian(self, adj):
        # 计算 D^(-1/2)
        d_inv_sqrt = torch.diag(torch.pow(adj.sum(dim=1), -0.5))

        # 计算归一化拉普拉斯矩阵 L_norm = I - D^(-1/2) * A * D^(-1/2)
        identity = torch.eye(adj.size(0), device=adj.device)
        laplacian_norm = identity - d_inv_sqrt @ adj @ d_inv_sqrt

        return laplacian_norm


class GRE_Multi(nn.Module):
    def __init__(self, args):
        super(GRE_Multi, self).__init__()
        self.conf = args
        self.ptm_list = None
        self.experts_num = None
        self.generator = torch.nn.ModuleList()
        self.cross_ent = nn.CrossEntropyLoss()

    def build_generator(self, ptm_list):
        self.ptm_list = ptm_list
        self.experts_num = len(ptm_list)
        for ptm in self.ptm_list:
            self.generator.append(SparseEdgeEncoder(self.conf))

    def forward(self, pretrained_model_list):
        self.build_generator(pretrained_model_list)
        adjs_list = []
        feas_list = []
        labels_list = []
        graphs_list = []
        features_list = []
        edges_list = []
        for i in range(self.experts_num):
            labels_i, adjs_i, feas_i = self.generator[i]()
            labels_i = torch.tensor(labels_i).to(feas_i[0].device)
            labels_list.append(labels_i)
            adjs_list.append(adjs_i)
            feas_list.append(feas_i)

            graphs_i, features_i, edge_weight_i, _ = self.transform_data(adjs_i, feas_i)
            graphs_list.append(graphs_i)
            features_list.append(features_i)
            edges_list.append(edge_weight_i)

        bn_loss_list, ori_ce_list = self.fake_loss(graphs_list, features_list, edges_list, labels_list)

        labels, adjs, feas = self.mixup(adjs_list, feas_list, labels_list)

        graphs, features, edge_weight, adj_loss = self.transform_data(adjs, feas)

        data_loader = self.pack_data(labels, adjs, feas)

        if train_graphs is not None:
            output, gates, expert_out_list, gate_loss, mask_loss = self.moe(train_graphs, train_feat)
        else:
            output, gates, expert_out_list, gate_loss, mask_loss = self.moe(graphs, features, edge_weight=edge_weight)
        # ori_ce_loss = gates.unsqueeze(dim=-1) * torch.stack(ori_ce_list, dim=1)
        ori_ce_loss = torch.stack(ori_ce_list, dim=-1)
        ori_ce_loss = ori_ce_loss.sum(dim=-1) / self.conf.num_experts
        # bn_loss = gates.unsqueeze(dim=-1) * torch.stack(bn_loss_list, dim=1)

        return output, gates, labels, data_loader, expert_out_list, ori_ce_loss, bn_loss_list, gate_loss, adj_loss, mask_loss

    def fake_loss(self, graphs_list, features_list, edges_list, labels_list):

        # create hook
        loss_r_feature_layers = []
        for ptm in self.ptm_list:
            ptm_loss_r_feature_layers = []
            for module in ptm.modules():
                if isinstance(module, nn.BatchNorm1d):
                    ptm_loss_r_feature_layers.append(DeepInversionFeatureHook(module))  # list中元素是层数-1个hook
            loss_r_feature_layers.append(ptm_loss_r_feature_layers)

        loss_bn = []
        ori_ce = []
        for ptm_id, ptm in enumerate(self.ptm_list):

            output = ptm(graphs_list[ptm_id], features_list[ptm_id], edges_list[ptm_id])

            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers[ptm_id]])

            loss_bn += [loss_distr]
            ori_ce += [self.cross_ent(output, labels_list[ptm_id])]

        # if features_list[0].shape[1] != 1:
        #     # features = torch.softmax(features, 1)
        #     # b = features * torch.log(features)
        #     # fea_loss = -1.0 * b.sum() / len(features)
        #     fea_loss = 0
        # else:
        #     fea_loss = 0

        return loss_bn, ori_ce

    def transform_data(self, adjs, feas):

        dgl_graphs = []
        adj_loss = 0
        # 遍历邻接矩阵列表和节点特征列表
        for adj, fea in zip(adjs, feas):
            # 获取边的起始节点和终止节点索引
            src, dst = adj.nonzero(as_tuple=True)

            # 创建DGLGraph
            g = dgl.graph((src, dst), num_nodes=adj.shape[0])
            g.edata['weight'] = adj[src, dst]

            # 将节点特征赋值给图
            # print(fea.shape)
            # assert fea.shape[0] == num_nodes, "Feature dimension mismatch with the number of nodes"
            g.ndata['feat'] = fea

            # 添加到列表中
            dgl_graphs.append(g)

            adj = torch.softmax(adj, dim=-1)  # 正则化邻接矩阵
            adj_loss += -torch.sum(adj * torch.log(adj + 1e-10)) / adj.size(0)  # 负熵作为损失，熵越大损失越小

        graphs = dgl.batch(dgl_graphs)
        features = graphs.ndata['feat']
        edge_weight = graphs.edata['weight']

        return graphs, features, edge_weight, adj_loss

    def pack_data(self, labels, adjs, feas):

        dataset = FKDataset(adjs, feas, labels)
        data_loader = GraphDataLoader(dataset, batch_size=self.conf.batch_size, shuffle=True)

        return data_loader

    def mixup(self, adjs_list, feas_list, labels_list):
        adjs = []
        feas = []
        for i in range(len(adjs_list)):
            adjs += adjs_list[i]
            feas += feas_list[i]
        labels = torch.cat(labels_list)

        return labels, adjs, feas

