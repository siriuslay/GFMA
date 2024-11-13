import argparse, time, random, os
import statistics

from data_loader.GIN_data_downloader import GINDataset

from data_loader.GIN_data_downloader import GraphDataLoader, collate, Split_GraphDataLoader, Continual_GraphDataLoader
from modules.scheduler import LinearSchedule
from pretrain.GCN import GCN_dict
from pretrain.GIN import GIN_dict
from pretrain.GAT import GAT_dict
import torch
import numpy as np
import torch.nn as nn
import json


torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, loss_fcn):
    model.eval()
    total = 0
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            graphs = graphs.to('cuda')
            feat = graphs.ndata['attr'].cuda()
            labels = labels.cuda()
            total += len(labels)
            outputs = model(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            loss = loss_fcn(outputs, labels)

            total_loss += loss * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    return loss, acc

def task_data(args):

    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    print(dataset.dim_nfeats)

    train_loader, valid_loader, test_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name=args.split_name, split=args.split, train_index=args.train_index,
        test_index=args.test_index).train_valid_test_loader()

    return dataset, train_loader, valid_loader, test_loader


def split_task_data(args):

    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    print(dataset.dim_nfeats)

    if continual == 1:
        train_loader_list, valid_loader_list, test_loader= Continual_GraphDataLoader(
            dataset, batch_size=args.batch_size, device=args.gpu,
            collate_fn=collate, seed=args.seed, shuffle=True,
            split_name=args.split_name, split=args.split, train_index=args.train_index,
            test_index=-1).train_valid_loader_list()
    else:
        train_loader_list, valid_loader_list, test_loader= Split_GraphDataLoader(
            dataset, batch_size=args.batch_size, device=args.gpu,
            collate_fn=collate, seed=args.seed, shuffle=True,
            split_name=args.split_name, split=args.split, train_index=args.train_index,
            test_index=-1).train_valid_loader_list()

    return dataset, train_loader_list, valid_loader_list, test_loader


def task_model(args, path_model=None):

    assert args.base_model in ['GIN', 'GCN', 'GAT']

    if args.base_model == 'GIN':
        model = GIN_dict[args.model_arch](args)
    elif args.base_model == 'GCN':
        model = GCN_dict[args.model_arch](args)
    elif args.base_model == 'GAT':
        model = GAT_dict[args.model_arch](args)
    else:
        raise ('Not supporting such model!')

    if path_model != None:
        model.load_state_dict(torch.load(path_model)['model'])

    cross_ent = nn.CrossEntropyLoss()

    if args.gpu >= 0:
        model = model.to(f'cuda:{args.gpu}')
        cross_ent = cross_ent.to(f'cuda:{args.gpu}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, cross_ent, optimizer


def train(args, train_loader, valid_loader, model, cross_ent, optimizer, test_loader):

    scheduler = LinearSchedule(optimizer, args.epoch)

    dur = []
    best_acc = 0
    ptm_dir = '{}/{}/{}part'.format(args.path_t, args.dataset, args.split)
    if not os.path.isdir(ptm_dir):
        os.makedirs(ptm_dir)

    for epoch in range(1, args.epoch + 1):

        model.train()

        t0 = time.time()

        for graphs, labels in train_loader:

            features = graphs.ndata['attr'].to(f'cuda:{args.gpu}')

            graphs = graphs.to(f'cuda:{args.gpu}')

            outputs = model(graphs, features, training=args.training)

            labels = labels.to(outputs.device)

            optimizer.zero_grad()

            loss_div = cross_ent(outputs, labels)

            loss = loss_div

            loss.backward()

            optimizer.step()

        dur.append(time.time() - t0)

        _, valid_acc = evaluate(model, valid_loader, cross_ent)
        _, train_acc = evaluate(model, train_loader, cross_ent)
        _, test_acc = evaluate(model, test_loader, cross_ent)

        print('Average Epoch Time {:.4f}'.format(float(sum(dur) / len(dur))))
        print('Epoch: %d' % epoch)
        print('Train acc {:.4f}'.format(float(train_acc)))
        print('Valid acc {:.4f}'.format(float(valid_acc)))
        print('Traing_loss {:.4f}'.format(float(loss.item())))

        if valid_acc > best_acc:
            best_acc = valid_acc
            state = {
                'model_type': args.base_model,
                'model_arch': args.model_arch,
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': valid_acc,
            }
            save_file = os.path.join(ptm_dir, '{}2-32_best_train_{}_seed{}_acc{}_test{}.pth'.format(args.base_model, args.train_index, args.seed, int(100*best_acc), int(100*test_acc)))
            print('saving the best model!')
            torch.save(state, save_file)

        scheduler.step()

    save_file_last = os.path.join(ptm_dir, '2-32_last_train_{}_seed{}_acc{}_test{}.pth'.format(args.train_index, args.seed, int(100*best_acc), int(100*test_acc)))
    state = {
        'model_type': args.base_model,
        'model_arch': args.model_arch,
        'epoch': epoch,
        'model': model.state_dict(),
        'accuracy': valid_acc,
    }

    print('last_acc: %f' % valid_acc)
    print('best_acc: %f' % best_acc)

    if args.choose_model == 'last':
        torch.save(state, save_file_last)
        save_file = save_file_last

    return valid_acc, best_acc, save_file


def test(test_loader, model, cross_ent):
    # test trained model
    _, test_acc = evaluate(model, test_loader, cross_ent)
    print('Test acc {:.4f}'.format(float(test_acc)))

    return test_acc


def main(args):

    assert args.train_data in ['real', 'fake']
    assert args.test_data in ['real', 'fake']

    # prepare real data
    args.train_index = 0
    args.test_index = -1
    dataset, train_loader_list, valid_loader_list, test_loader = split_task_data(args)

    model, cross_ent, optimizer = task_model(args)
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable ptm parameters:{trainable_param}")

    v_ac, b_ac, pre_save_file = train(args, train_loader_list[0], valid_loader_list[0], model, cross_ent, optimizer, test_loader)

    pretrained_model, cross_ent, optimizer = task_model(args, path_model=pre_save_file)

    print(f'0-th fine-tuned test acc:')
    test_b_ac = test(test_loader, pretrained_model, cross_ent)
    
    trainable_param = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    print(f"number of trainable ptm parameters:{trainable_param}")

    original_params = {name: param.clone() for name, param in pretrained_model.named_parameters()}

    all_diffs = []

    for index in range(1, args.split):
        args.train_index = index
        print(f"{index}-th train data:")

        v_ac, b_ac, pre_save_file = train(args, train_loader_list[index], valid_loader_list[index], pretrained_model, cross_ent, optimizer, test_loader)

        pretrained_model, cross_ent, optimizer = task_model(args, path_model=pre_save_file)

        print(f'{index}-th fine-tuned test acc:')
        test_b_ac = test(test_loader, pretrained_model, cross_ent)

        trainable_param = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
        print(f"number of trainable ptm parameters:{trainable_param}")

        finetuned_params = {name: param.clone() for name, param in pretrained_model.named_parameters()}

        param_diffs = {}
        for name, param in pretrained_model.named_parameters():
            param_diff = torch.abs(finetuned_params[name] - original_params[name])
            param_diffs[name] = param_diff.cpu().detach().numpy().tolist()

        all_diffs.append(param_diffs)

        original_params = {name: param.clone() for name, param in pretrained_model.named_parameters()}
    
    with open(f"visual/results/continual_{args.dataset}_param_diffs_split_{args.split}.json", "w") as f:
        json.dump(all_diffs, f)

    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Mixture of pretrained GNNs for SFDA')

    parser.add_argument("--epoch", type=int, default=100, help="number of training iteration")

    parser.add_argument('--batch_size', type=int, default=1000000,
                        help='batch size for training and validation (default: 32)')

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--seed", type=int, default=0, help='random seed')  # just for real test loader and path

    parser.add_argument("--training", type=bool, default=True, help='train or eval')

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight for L2 Loss')

    parser.add_argument('--choose_model', type=str, default='best',
                        choices=['last', 'best'], help='test the last / best trained model')

    # path
    parser.add_argument("--model_arch", type=str, default='GCN2_32',
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32',
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32',
                                 'GAT5_64', 'GAT5_32', 'GAT3_64', 'GAT3_32', 'GAT2_64', 'GAT2_32'], help='graph models')

    parser.add_argument("--base_model", type=str, default='GCN', choices=['GIN', 'GCN', 'GAT'], help='graph models')

    # dataset
    parser.add_argument('--dataset', type=str, default='NCI1', choices=['REDDITBINARY', 'PROTEINS', 'MUTAG', 'PTC', 'COLLAB', 'IMDBBINARY', 'NCI1', 'REDDITMULTI5K'],
                        help='name of dataset (default: MUTAG)')

    parser.add_argument('--data_dir', type=str, default='./dataset', help='data path')

    parser.add_argument("--self_loop", action='store_true', help='add self_loop to graph data')

    parser.add_argument("--dim_feat", type=int, default=37,
                        help="number of node feature dim:{'IMDBBINARY': 1, 'MUTAG': 7, 'COLLAB': 1, 'PTC': 19, 'PROTEINS': 3, 'REDDITBINARY': 1, 'REDDITMULTI5K': 1, 'NCI1': 37}")

    parser.add_argument("--gcls", type=int, default=2,
                        help="number of graph classes:{'IMDBBINARY': 2, 'MUTAG': 2, 'COLLAB': 3, 'PTC': 2, 'PROTEINS': 2, 'REDDITBINARY': 2, 'REDDITMULTI5K': 5, 'NCI1': 2}")

    parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')

    parser.add_argument('--split_name', type=str, default='mean_degree_sort', choices=['rand', 'mean_degree_sort'],
                        help='rand split with dataseed')

    parser.add_argument("--split", type=int, default=10, help="number of splits")

    parser.add_argument('--test_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--test_index", type=int, default=0, help="use the x-th data as testing data")

    parser.add_argument('--train_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--train_index", type=int, default=0, help="use the x-th data as testing data")

    args = parser.parse_args()

    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir
    # args.path_list = ['2-32_best_train_0_seed9_acc91.pth',
    #                   '2-32_best_train_1_seed8_acc86.pth']

    continual = 1
    main(args)

    # m_v = []
    # for args.train_index in [0, 1]: # , 1, 2, 3
    #     for args.test_index in [2]:
    #         acc = []
    #         for args.seed in range(10):      # [7]: 
    #             set_seed(args.seed)
    #             print('seed: %d' % args.seed)
    #             acc.append(main(args) * 100)
    #         raw = [acc, f'{round(np.mean(acc), 2)}±{round(np.std(acc, ddof=0), 2)}']
    #         with open('pretrain GIN2-32 outs.txt', 'a') as file:
    #             print(f'train/test: {args.train_index} / {args.test_index}', file=file)
    #             print(raw, file=file)
    #         m_v.append([args.train_index, args.test_index, f'{np.mean(acc)}±{np.std(acc, ddof=0)}'])
    # print(m_v)