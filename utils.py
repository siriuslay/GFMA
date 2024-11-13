import torch, os
import dgl
from pretrain.GCN import GCN_dict
from pretrain.GIN import GIN_dict
from pretrain.GAT import GAT_dict

def evaluate_ptm(model, dataloader, args):
    model.eval()
    total = 0
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            graphs = graphs.to(f'cuda:{args.gpu}')
            feat = graphs.ndata['attr'].to(f'cuda:{args.gpu}')
            labels = labels.to(f'cuda:{args.gpu}')
            total += len(labels)
            outputs = model(graphs, feat)
            # outputs, embs = model(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            # loss = loss_fcn(outputs, labels)

            # total_loss += loss * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    # model.train()
    return loss, acc


def test_ptm(data_loader, model, args):
    _, test_acc = evaluate_ptm(model, data_loader, args)
    print('Test acc {:.4f}'.format(float(test_acc)))

    return test_acc


def evaluate_moe(model, dataloader, args):
    model.eval()
    total = 0
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            graphs = graphs.to(f'cuda:{args.gpu}')
            feat = graphs.ndata['attr'].to(f'cuda:{args.gpu}')
            labels = labels.to(f'cuda:{args.gpu}')
            total += len(labels)
            outputs, gates, expert_out_list, gate_loss = model(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            # loss = loss_fcn(outputs, labels)
            #
            # total_loss += loss * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    model.train()
    return loss, acc


def test_moe(data_loader, model, cross_ent, args):
    _, test_acc = evaluate_moe(model, data_loader, cross_ent, args)
    print('Test acc {:.4f}'.format(float(test_acc)))

    return test_acc


def compute_laplacian(graph):
    # 获取度矩阵 D
    degree = graph.in_degrees().float()
    D = torch.diag(degree)

    # 获取邻接矩阵 A
    A = graph.adjacency_matrix().to_dense()

    # 计算拉普拉斯矩阵 L = D - A
    L = D - A
    return L


def laplacian_loss(graph):
    L = compute_laplacian(graph)
    eigvals = torch.linalg.eigvalsh(L)
    return eigvals.sum()  # 希望特征值的和较小，表示图的连通性较好


def RegLoss(param, k):
    assert k in [1, 2]
    param = param.view(-1)
    reg_loss = torch.norm(param, k)
    return reg_loss


def merge_batches(batch_graphs_list, batch_labels_list):
    # 合并图
    all_graphs = []
    for batch_graphs in batch_graphs_list:
        graphs = dgl.unbatch(batch_graphs)
        all_graphs += graphs

    # merged_graph = dgl.batch(all_graphs)

    # 合并标签
    all_labels = [l for batch_labels in batch_labels_list for l in batch_labels]
    merged_labels = torch.stack(all_labels)

    return all_graphs, merged_labels


def load_ptm(args, select=True):
    #  step 1: prepare model
    model_list = []
    model_arch_list = []
    
    if select:
        for path in args.path_list:
            # print(args.path_t, args.dataset, str(args.split), path)
            path = os.path.join(args.path_t, args.dataset, str(args.split), path)
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
            model_arch_list.append(modelt)

    else:
        for path in args.path_list:
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
            model_arch_list.append(modelt)

    return model_list, model_arch_list