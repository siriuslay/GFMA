import dgl
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import argparse
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from data_loader.GIN_data_downloader import GINDataset
from data_loader.GIN_data_downloader import GraphDataLoader, collate
from modules.mome import MoME
from pretrain.GCN import GCN_dict
from pretrain.GIN import GIN_dict
from pretrain.GAT import GAT_dict
from modules.KL_loss import DistillKL
from modules.scheduler import LinearSchedule
from utils import evaluate_ptm, test_ptm, test_moe, evaluate_moe, RegLoss, merge_batches
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Mixture of pretrained GNNs for SFDA')

parser.add_argument("--epoch", type=int, default=100, help="number of training iteration")

parser.add_argument("--gpu", type=int, default=0, help="gpu")

parser.add_argument("--seed", type=int, default=1, help='random seed')  # just for real test loader and path

parser.add_argument("--training", type=bool, default=True, help='train or eval')

parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")

parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight for L2 Loss')

parser.add_argument('--choose_model', type=str, default='best',
                    choices=['last', 'best'], help='test the last / best trained model')

# path
parser.add_argument('--model_name', type=str, default='moe', help='')

parser.add_argument("--model_arch", type=str, default='GCN2_32',
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32',
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32',
                                 'GAT5_64', 'GAT5_32', 'GAT3_64', 'GAT3_32', 'GAT2_64', 'GAT2_32'], help='graph models')

parser.add_argument("--base_model", type=str, default='GCN', choices=['GIN', 'GCN', 'GAT'], help='graph models')

parser.add_argument('--path_t', type=str, default='saved_models/pretrained_models', help='teacher path')

# dataset
parser.add_argument('--dataset', type=str, default='PTC', choices=['MUTAG', 'PTC', 'COLLAB', 'REDDITBINARY', 'NCI1'],
                    help='name of dataset (default: MUTAG)')

parser.add_argument('--data_dir', type=str, default='./dataset', help='data path')

parser.add_argument("--self_loop", action='store_false', help='add self_loop to graph data')

parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')

parser.add_argument('--split_name', type=str, default='mean_degree_sort', choices=['rand', 'mean_degree_sort'],
                    help='rand split with dataseed')

parser.add_argument("--split", type=int, default=4, help="number of splits")

parser.add_argument("--dim_feat", type=int, default=19,
                    help="number of node feature dim:{'IMDBBINARY': 1, 'MUTAG': 7, 'COLLAB': 1, 'PROTEINS': 3, 'PTC': 19}")

parser.add_argument("--gcls", type=int, default=2,
                    help="number of graph classes:{'IMDBBINARY': 2, 'MUTAG': 2, 'COLLAB': 3, 'PROTEINS':2, 'PTC': 2}")

parser.add_argument('--batch_size', type=int, default=100000,
                    help='batch size for training and validation (default: 32)')

parser.add_argument('--test_data', type=str, default='real',
                    choices=['real', 'fake'], help='choose type of dataset')

parser.add_argument("--test_index", type=int, default=2, help="use the x-th data as testing data")

parser.add_argument('--train_data', type=str, default='real',
                    choices=['real', 'fake'], help='choose type of dataset')

parser.add_argument("--train_index", type=int, default=-1, help="use the x-th data as testing data")

args = parser.parse_args()

os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir

# args.path_list = ['train0.pth', 'train1.pth', 'train2.pth', 'train3.pth']
if args.dataset == 'MUTAG':
    args.path_list = ['best_train_0_seed6_acc93.pth',
                      'best_train_1_seed6_acc93.pth',
                      'best_train_2_seed6_acc73.pth',
                      'best_train_3_seed6_acc93.pth']
elif args.dataset == 'PTC':
    args.path_list = ['best_train_0_seed0_acc84.pth',
                      'best_train_1_seed8_acc84.pth',
                      'best_train_2_seed5_acc80.pth',
                      'best_train_3_seed7_acc80.pth']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

def task_data(args):
    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    print(dataset.dim_nfeats)

    train_loader, valid_loader, test_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name=args.split_name, split=args.split, train_index=-1,
        test_index=args.test_index).train_valid_test_loader()

    return dataset, train_loader, valid_loader, test_loader


def load_ptm(args):
    #  step 1: prepare model
    model_list = []
    for path in args.path_list:
        path = os.path.join(args.path_t, args.dataset, args.base_model, path)
        tmodel = torch.load(path)['model_type']
        modelt = torch.load(path)['model_arch']
        if tmodel == 'GIN':
            model = GIN_dict[modelt](args)
        elif tmodel == 'GCN':
            model = GCN_dict[modelt](args)
        elif tmodel == 'GAT':
            model = GAT_dict[modelt](args)
        else:
            raise 'Not supporting such model!'
        state_dict = torch.load(path)['model']
        # for key in state_dict:
        #     state_dict[key] = state_dict[key].detach()
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        # model.eval()
        if args.gpu >= 0:
            model = model.to(f'cuda:{args.gpu}')
        model_list.append(model)

    return model_list


methods = ['MUTAG', 'PTC', 'PROTEINS', 'IMDBBINARY', 'REDDITBINARY']
gcls = [2, 2, 2, 3, 2]
n_dim = [7, 19, 3, 1, 1]
n_methods = len(methods)
features_list = []
labels_list = []
special_feature_list = []
for i in range(n_methods):
    args.dataset = methods[i]
    args.gcls = gcls[i]
    args.dim_feat = n_dim[i]
    # prepare real data
    dataset, train_loader, valid_loader, test_loader = task_data(args)

    # prepare fake data
    # save_file = 'SAVE/MUTAG_mome/gen_moe_best_GCN_seed0_ne1.pth'
    # fake_data_loader = torch.load(save_file)['fake_data']

    # prepare pretrained model
    # print("loading PTMs")
    # net_list = load_ptm(args)

    for graphs, labels in train_loader:

        node_density = []
        original_graphs = dgl.unbatch(graphs)
        for graph in original_graphs:
            node_density.append(graph.num_edges() / (graph.num_nodes() * (graph.num_nodes() - 1)))
        node_density = torch.tensor(node_density)

        features = graphs.ndata['attr'].to('cuda:0')

        batch_indices = graphs.batch_num_nodes()
        start_idx = 0
        features_graphs = torch.zeros(graphs.batch_size, features.shape[1]).to(features.device)
        for idx, num_nodes in enumerate(batch_indices):
            end_idx = start_idx + num_nodes
            features_graphs[idx] = torch.sum(features[start_idx:end_idx], dim=0).clone()
            start_idx = end_idx

        # features = graphs.ndata['attr'].to('cuda:0')
        #
        # graphs = graphs.to('cuda:0')

        # n_output = []
        # #
        # for net in net_list:
        #     n_output.append(net(graphs, features))
        #
        # cat_n_out = torch.cat(n_output, dim=-1)
        # labels = labels.to(output.device)


    # features_graphs

    # Create a random feature matrix (e.g., 100 samples with 50 features)
    # features = torch.rand((100, 50))

    attributes_tensor = node_density
    # Convert the PyTorch tensor to a NumPy array for PCA/t-SNE processing
    attributes_np = attributes_tensor.numpy()

    scaler = MinMaxScaler()
    attributes_np = scaler.fit_transform(attributes_np.reshape(-1, 1)).flatten()

    features_np = features_graphs.detach().cpu().numpy()

    # features_np = cat_n_out.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # n_samples_per_method =features_np.shape[0]
    # # t-SNE for 2D visualization (more suitable for non-linear structures)
    # tsne = TSNE(n_components=1, random_state=42)
    # features_tsne = tsne.fit_transform(features_np).flatten()

    features_list.append(features_np)
    labels_list.append(labels_np)
    special_feature_list.append(attributes_np)

##########################
# 柱状散点图
# 初始化数据：假设有3个不同的方法，每个方法生成不同数量的样本
# 模拟每个方法得到的不同数量样本的特征，标签和特殊属性
np.random.seed(0)

# 存储整合后的数据
all_methods = []
all_tsne = []
all_special_features = []
all_labels = []
# 处理每个方法的数据
for i, (features, labels, special_feature) in enumerate(zip(features_list, labels_list, special_feature_list)):
    # 1. 对特征进行t-SNE降维（降为1维，用作横坐标的一部分）
    tsne = TSNE(n_components=1, random_state=0)
    tsne_results = tsne.fit_transform(features).flatten()

    # 2. 对特殊属性进行归一化处理
    scaler = MinMaxScaler()
    special_feature_normalized = scaler.fit_transform(special_feature.reshape(-1, 1)).flatten()

    # 3. 合并数据
    all_methods.extend([methods[i]] * len(labels))  # 记录当前方法名
    all_tsne.extend(tsne_results)  # 记录t-SNE结果
    all_special_features.extend(special_feature_normalized)  # 记录归一化的特殊属性
    all_labels.extend(labels)  # 记录标签

# 创建最终的DataFrame
data = pd.DataFrame({
    'Dataset': all_methods,
    't-SNE': all_tsne,
    'SpecialFeature': all_special_features,
    'Label': all_labels
})

# 使用 Seaborn 生成柱状散点图，横轴是不同方法，纵轴是归一化后的特殊属性
plt.figure(figsize=(10, 6))

# 通过 hue 来区分标签，jitter 使得散点图中的点更均匀分布
sns.stripplot(x='Dataset', y='SpecialFeature', data=data, hue='Label', jitter=True, palette="Set1", size=4)

# 设置标题和标签
plt.title('Samples Visualization Across Different Datasets')
plt.xlabel('Datasets')
plt.ylabel('Normalized Density')

# 显示图例
plt.legend(title='Label')

plt.savefig('visual/results/plot.svg', format='svg')
# 显示图表
plt.show()



#############################
# # 设置点的大小（使用属性值的一个线性变换，例如将范围归一化到某个合理的大小范围）
# sizes = (attributes_np - attributes_np.min()) / (attributes_np.max() - attributes_np.min()) * 100 + 10  # 将大小映射到[10, 110]之间
#
# # 定义每个标签的颜色
# colors = ['red', 'blue']  # 为每个类别分配不同的颜色 'green',
#
# # 绘制降维后的数据，按标签分颜色，并根据属性值调整点的大小或颜色
# plt.figure(figsize=(8, 6))
# for label in np.unique(labels_np):
#     scatter = plt.scatter(
#         features_tsne[labels_np == label, 0],
#         features_tsne[labels_np == label, 1],
#         s=sizes[labels_np == label],
#         color=colors[label],
#         cmap='viridis',
#         label=f"Class {label}",
#         alpha=0.7,
#         edgecolor='k'
#     )
#
# for size in [10, 50, 100]:
#     plt.scatter([], [], s=size, color='gray', alpha=0.5,
#                 label=f'density ~ {np.round((size - 10) / 100 * (attributes_np.max() - attributes_np.min()) + attributes_np.min(), 2)}')
#
# plt.title(f't-SNE Projection of the {args.test_index + 1}-th part of {args.dataset}')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.legend()
# plt.show()


