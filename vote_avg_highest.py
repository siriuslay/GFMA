import argparse, time, random, os
import statistics
from data_loader.GIN_data_downloader import GINDataset, GraphDataLoader, collate
from modules.scheduler import LinearSchedule
from pretrain.GCN import GCN_dict
from pretrain.GIN import GIN_dict
from pretrain.GAT import GAT_dict
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from utils import load_ptm

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
    all_preds = []
    all_labels = []
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
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return loss, acc, precision, recall, f1, cm


def task_data(args):

    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    print(dataset.dim_nfeats)

    train_loader, valid_loader, test_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name=args.split_name, split=args.split, train_index=args.train_index,
        test_index=args.test_index).train_valid_test_loader()

    return dataset, train_loader, valid_loader, test_loader


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

        _, valid_acc, precision, recall, f1, cm = evaluate(model, valid_loader, cross_ent)
        _, train_acc, precision, recall, f1, cm = evaluate(model, train_loader, cross_ent)
        _, test_acc, precision, recall, f1, cm = evaluate(model, test_loader, cross_ent)

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

    save_file_last = os.path.join(ptm_dir, '{}2-32_last_train_{}_seed{}_acc{}_test{}.pth'.format(args.base_model, args.train_index, args.seed, int(100*best_acc), int(100*test_acc)))
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
    _, test_acc, precision, recall, f1, cm = evaluate(model, test_loader, cross_ent)
    print('Test acc {:.4f}'.format(float(test_acc)))

    return test_acc, precision, recall, f1, cm

def test_ptm_list(test_loader, net_list):
    for model in net_list:
        model.eval()
    
    total = 0
    total_correct = 0
    avg_correct_count = 0
    model_correct_counts = [0] * len(net_list)
    model_totals = [0] * len(net_list)

    max_preds = []
    avg_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            graphs, labels = data
            graphs = graphs.to('cuda')
            feat = graphs.ndata['attr'].cuda()
            labels = labels.cuda()

            total += len(labels)

            all_outputs = []

            for i, model in enumerate(net_list):
                outputs = model(graphs, feat)
                all_outputs.append(outputs)

                _, predicted = torch.max(outputs.data, 1)
                model_correct_counts[i] += (predicted == labels.data).sum().item()
                model_totals[i] += len(labels)

            stacked_outputs = torch.stack(all_outputs)

            max_outputs, _ = torch.max(stacked_outputs, dim=0)

            _, predicted = torch.max(max_outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()


            avg_outputs = torch.mean(stacked_outputs, dim=0)

            _, avg_predicted = torch.max(avg_outputs.data, 1)
            avg_correct_count += (avg_predicted == labels.data).sum().item()

            max_preds.extend(predicted.cpu().numpy())
            avg_preds.extend(avg_predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    max_acc = 1.0 * total_correct / total

    avg_accuracy = avg_correct_count / total

    acc_list = [model_correct_counts[i] / model_totals[i] for i in range(len(net_list))]

    max_precision, max_recall, max_f1, _ = precision_recall_fscore_support(all_labels, max_preds, average='weighted', zero_division=0)
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(all_labels, avg_preds, average='weighted', zero_division=0)

    return max_acc, acc_list, avg_accuracy, max_precision, max_recall, max_f1, avg_precision, avg_recall, avg_f1


def main(args):

    assert args.train_data in ['real', 'fake']
    assert args.test_data in ['real', 'fake']

    args.path_list = []
    # pretrain
    for args.base_model in ['GCN', 'GAT', 'GIN']:
        args.model_arch = f'{args.base_model}2_32'
        for args.train_index in [0, 1]:
            # prepare real data
            dataset, train_loader, valid_loader, test_loader = task_data(args)
            model, cross_ent, optimizer = task_model(args)
            trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"number of trainable ptm parameters:{trainable_param}")
            v_ac, b_ac, save_file = train(args, train_loader, valid_loader, model, cross_ent, optimizer, test_loader)
            print(v_ac, b_ac)
            args.path_list.append(save_file)

    print(args.path_list)
    net_list, net_arch = load_ptm(args, select=False)

    max_acc, acc_list, avg_accuracy, max_precision, max_recall, max_f1, avg_precision, avg_recall, avg_f1 = test_ptm_list(test_loader, net_list)

    return max_acc, acc_list, avg_accuracy, max_precision, max_recall, max_f1, avg_precision, avg_recall, avg_f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Mixture of pretrained GNNs for SFDA')

    parser.add_argument("--epoch", type=int, default=300, help="number of training iteration")

    parser.add_argument('--batch_size', type=int, default=1000000,
                        help='batch size for training and validation (default: 32)')

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--seed", type=int, default=6, help='random seed')  # just for real test loader and path

    parser.add_argument("--training", type=bool, default=True, help='train or eval')

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight for L2 Loss')

    parser.add_argument('--choose_model', type=str, default='best',
                        choices=['last', 'best'], help='test the last / best trained model')

    # path
    parser.add_argument("--model_arch", type=str, default='GCN2_32',
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32',
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32'], help='graph models')

    parser.add_argument("--base_model", type=str, default='GCN', choices=['GIN', 'GCN'], help='graph models')

    parser.add_argument('--path_t', type=str, default='saved_models/pretrained_models', help='ptm path')

    # dataset
    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['REDDITBINARY', 'PROTEINS', 'MUTAG', 'PTC', 'COLLAB', 'IMDBBINARY', 'NCI1', 'REDDITMULTI5K'],
                        help='name of dataset (default: MUTAG)')

    parser.add_argument('--data_dir', type=str, default='./dataset', help='data path')

    parser.add_argument("--self_loop", action='store_true', help='add self_loop to graph data')

    parser.add_argument("--dim_feat", type=int, default=7,
                        help="number of node feature dim:{'IMDBBINARY': 1, 'MUTAG': 7, 'COLLAB': 1, 'PTC': 19, 'PROTEINS': 3, 'REDDITBINARY': 1}")

    parser.add_argument("--gcls", type=int, default=2,
                        help="number of graph classes:{'IMDBBINARY': 2, 'MUTAG': 2, 'COLLAB': 3, 'PTC': 2, 'PROTEINS': 2, 'REDDITBINARY': 2}")

    parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')

    parser.add_argument('--split_name', type=str, default='mean_degree_sort', choices=['rand', 'mean_degree_sort'],
                        help='rand split with dataseed')

    parser.add_argument("--split", type=int, default=3, help="number of splits")

    parser.add_argument('--test_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--test_index", type=int, default=2, help="use the x-th data as testing data")

    parser.add_argument('--train_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--train_index", type=int, default=0, help="use the x-th data as testing data")

    args = parser.parse_args()

    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir
    # args.path_list = ['2-32_best_train_0_seed4_acc92_test32.pth',
    #                   '2-32_best_train_1_seed0_acc92_test39.pth']
    # main(args)

    m_v = []
    # for args.train_index in [0]: # , 1, 2, 3
    for args.test_index in [2]:
        maxacc = []
        avgacc = []
        maxpre = []
        avgpre = []
        maxrec = []
        avgrec = []
        maxf1 = []
        avgf1 = []
        for args.seed in range(10):      # [7]: 
            set_seed(args.seed)
            print('seed: %d' % args.seed)
            max_acc, acc_list, avg_accuracy, max_precision, max_recall, max_f1, avg_precision, avg_recall, avg_f1 = main(args)
            maxacc.append(max_acc * 100)
            avgacc.append(avg_accuracy * 100)
            maxpre.append(max_precision * 100)
            avgpre.append(avg_precision * 100)
            maxrec.append(max_recall * 100)
            avgrec.append(avg_recall * 100)
            maxf1.append(max_f1 * 100)
            avgf1.append(avg_f1 * 100)
        max_raw1 = ['max-acc:', maxacc, f'{round(np.mean(maxacc), 2)}±{round(np.std(maxacc, ddof=0), 2)}']
        avg_raw1 = ['avg-acc:', avgacc, f'{round(np.mean(avgacc), 2)}±{round(np.std(avgacc, ddof=0), 2)}']
        max_raw2 = ['max-pre:', maxpre, f'{round(np.mean(maxpre), 2)}±{round(np.std(maxpre, ddof=0), 2)}']
        avg_raw2 = ['avg-pre:', avgpre, f'{round(np.mean(avgpre), 2)}±{round(np.std(avgpre, ddof=0), 2)}']
        max_raw3 = ['max-rec:', maxrec, f'{round(np.mean(maxrec), 2)}±{round(np.std(maxrec, ddof=0), 2)}']
        avg_raw3 = ['avg-rec:', avgrec, f'{round(np.mean(avgrec), 2)}±{round(np.std(avgrec, ddof=0), 2)}']
        max_raw4 = ['max-f1:', maxf1, f'{round(np.mean(maxf1), 2)}±{round(np.std(maxf1, ddof=0), 2)}']
        avg_raw4 = ['avg-f1:', avgf1, f'{round(np.mean(avgf1), 2)}±{round(np.std(avgf1, ddof=0), 2)}']
        with open('results/train outs.txt', 'a') as file:
            print(f'\n{args.dataset}: mean-prob_acc', file=file)
            print(avg_raw1, file=file)
            print(avg_raw2, file=file)
            print(avg_raw3, file=file)
            print(avg_raw4, file=file)
            print(f'\n{args.dataset}: max-prob_acc', file=file)
            print(max_raw1, file=file)
            print(max_raw2, file=file)
            print(max_raw3, file=file)
            print(max_raw4, file=file)