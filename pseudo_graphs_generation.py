import argparse, dgl
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from data_loader.GIN_data_downloader import GINDataset
from data_loader.GIN_data_downloader import GraphDataLoader, collate
from pretrain.GCN import GCN_dict
from pretrain.GIN import GIN_dict
from pretrain.GAT import GAT_dict
from modules.KL_loss import DistillKL
from modules.scheduler import LinearSchedule
from utils import evaluate_ptm, test_ptm, test_moe, evaluate_moe, RegLoss
from torch.utils.tensorboard import SummaryWriter
from modules.generator import GRE_Shared, GRE_Single
from visual.graph_visualization import visualize


torch.autograd.set_detect_anomaly(True)


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def task_data(args):
    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    print(dataset.dim_nfeats)

    train_loader, valid_loader, test_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name=args.split_name, split=args.split, train_index=args.train_index,
        test_index=args.test_index).train_valid_test_loader()

    return dataset, train_loader, valid_loader, test_loader


def load_ptm(args):
    #  step 1: prepare model
    model_list = []
    for path in args.path_list[args.test_index]:
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

    return model_list


def train(args, model, ptm_list, optimizer):
    # log_dir = "~/autodl-tmp/code/WWW/tensorboard/runs/gen_moe_test/exp_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = "runs/gen_moe_test/"
    # writer = SummaryWriter(log_dir=log_dir)

    scheduler = LinearSchedule(optimizer, args.epoch)
    dur = []
    mini_cost = 1e9
    model_name = 'pseudo_graphs/{}'.format(args.dataset)
    if not os.path.isdir(model_name):
        os.makedirs(model_name)

    for epoch in range(1, args.epoch + 1):
        model.train()
        t0 = time.time()

        data_loader, ori_ce_list, bn_loss_list, output_list, labels, conf_loss_list = model(ptm_list)

        # if epoch % 10 == 1:
        #     visualize(data_loader, f'{args.dataset}/fake/ind_expert_{args.id}/{args.epoch}_{args.conn_p}/process/{epoch}', 3, 3)
        optimizer.zero_grad()

        acc_list = []
        for i, out in enumerate(output_list):
            _, predicted = torch.max(out.data, 1)
            total_correct_i = (predicted == labels.data).sum().item()
            acc_i = total_correct_i / len(labels)
            acc_list.append(acc_i)

        ori_ce_loss = sum(ori_ce_list) / len(ori_ce_list)
        bn_loss = sum(bn_loss_list) / len(bn_loss_list)
        conf_loss = sum(conf_loss_list) / len(conf_loss_list)

        # gen_loss = args.conn_p * adj_loss + args.gen_p * ori_ce_loss + args.bn_p * bn_loss + args.confi_p * conf_loss
        gen_loss = args.gen_p * ori_ce_loss + args.bn_p * bn_loss + args.confi_p * conf_loss

        loss = gen_loss

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        dur.append(time.time() - t0)

        # for idx in range(len(ptm_list)):
        #     writer.add_scalar(f'ACC/ptm{idx}_acc', acc_list[idx], epoch)

        # writer.add_scalar('Loss/train', loss.item(), epoch)
        # writer.add_scalar('Loss/gen', gen_loss.item(), epoch)
        # # writer.add_scalar('Loss/adj_loss', adj_loss.item(), epoch)
        # writer.add_scalar('Loss/ori_ce_loss', ori_ce_loss.item(), epoch)
        # writer.add_scalar('Loss/bn_loss', bn_loss.item(), epoch)
        # writer.add_scalar('Loss/conf_loss', conf_loss.item(), epoch)

        loss_target = loss.item()
        if (epoch % 50 == 0) & (epoch != 0):
            print('Epoch: %d' % epoch)
            print('Average Epoch Time {:.4f}'.format(float(sum(dur) / len(dur))))
            print('Training_loss {:.4f}'.format(float(loss_target)))
            print(f'PTM training acc = {acc_list}')
            # save_file = os.path.join(model_name,
            #                          'test{}_mid_graph_sparse0.9_best_{}_seed{}_e{}_epo{}-{}_acc{}.pth'.
            #                          format(args.test_index, args.model_arch, args.seed, args.id, epoch, args.epoch,
            #                                 acc_list[0]))
            # torch.save(state, save_file)

        if mini_cost > loss_target:
            mini_cost = loss_target
            state = {
                'model_name': args.model_name,
                'model_arch': args.model_arch,
                'epoch': epoch,
                'generator': model.state_dict(),
                'loss': loss_target,
                'fake_data': data_loader,
            }

            save_file = os.path.join(model_name, 'fakenum{}_{}parts_test{}_process_gen_graph_sparse0.9_adj_best_{}_seed{}_e{}_epo{}.pth'.
                                     format(args.fake_num, args.split, args.test_index, args.model_arch, args.seed, args.id, args.epoch))
            # print('saving the best model!')
            torch.save(state, save_file)
        scheduler.step()

    # writer.close()

    save_file_last = os.path.join(model_name, 'process_gen_graph_sparse0.9_adj_test{}_last_{}_seed{}_e{}_epo{}.pth'.
                                  format(args.test_index, args.model_arch, args.seed, args.id, args.epoch))
    state = {
        'model_name': args.model_name,
        'model_arch': args.model_arch,
        'epoch': epoch,
        'generator': model.state_dict(),
        'loss': loss_target,
        'fake_data': data_loader,
    }
    if args.choose_model == 'last':
        torch.save(state, save_file_last)
        mini_cost = loss_target
        save_file = save_file_last

    return mini_cost, save_file


def test(test_loader, model, cross_ent):
    # test trained model
    _, test_acc = evaluate(model, test_loader, cross_ent)
    print('Test acc {:.4f}'.format(float(test_acc)))

    return test_acc


def main(args):
    # prepare real data
    # dataset, train_loader, valid_loader, test_loader = task_data(args)
    # visualize(test_loader, f'{args.dataset}/real_0919/{args.test_index}', 2, 3)

    # prepare pretrained model & Generator

    print("loading PTMs")
    net_list = load_ptm(args)
    #
    for i, net in enumerate(net_list):
        print(f'{i}-th PTM: ')
        trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"number of trainable PTM {i} parameters:{trainable_param}")

    # adj_list = []
    # for batched_graph, real_labels in test_loader:
    #     # 获取每个批次的图
    #     for g in dgl.unbatch(batched_graph):
    #         # 获取邻接矩阵 (稠密形式)
    #         adj_matrix = g.adj().to_dense()  # 邻接矩阵的密集格式
    #         adj_list.append(adj_matrix)
    if args.id == -1:
        model = GRE_Shared(args)
    else:
        model = GRE_Single(args)
    if args.gpu >= 0:
        model = model.to(f'cuda:{args.gpu}')

    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable mask parameters:{trainable_param}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('************************')
    print('Generating start')

    # train
    if args.id == -1:
        mini_loss, save_file = train(args, model, net_list, optimizer)
    else:
        mini_loss, save_file = train(args, model, [net_list[args.id]], optimizer)
    print(f'mini_loss: {mini_loss}')

    print('Generating finished')
    print('************************')

    # visualize
    # save_file = f'SAVE/{args.dataset}_{args.model_name}/gen_graph_{args.choose_model}_{args.base_model}' + \
    #             f'_seed{args.seed}_e{args.id}_epo{args.epoch}.pth'
    fake_data_loader = torch.load(save_file)['fake_data']

    # visualize(fake_data_loader, f'{args.dataset}/fake/test{args.test_index}/GCN2-32_ind_expert_{args.id}/{args.epoch}_{args.conn_p}', 2, 3)

    # test
    cross_ent = nn.CrossEntropyLoss()

    if args.gpu >= 0:
        cross_ent = cross_ent.to(f'cuda:{args.gpu}')

    for i, net in enumerate(net_list):
        # print(f'{i}-th PTM: ')
        # trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # print(f"number of trainable PTM {i} parameters:{trainable_param}")
        test_ac = test(fake_data_loader, net, cross_ent)
        print(f'{i}-th PTM on fake data, test_ac={test_ac}')
        # with open('results/generation outs.txt', 'a') as file:
        #     print(f'index_expert: {args.id}, epoch: {args.epoch}, seed: {args.seed}', file=file)
        #     print(test_ac, file=file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Mixture of pretrained GNNs for GRE')

    parser.add_argument("--epoch", type=int, default=200, help="number of training iteration")

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--seed", type=int, default=1001, help='random seed')  # just for real test loader and path

    parser.add_argument("--training", type=bool, default=True, help='train or eval')

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight for L2 Loss')

    parser.add_argument('--choose_model', type=str, default='best',
                        choices=['last', 'best'], help='test the last / best trained model')

    # path
    parser.add_argument('--model_name', type=str, default='mome', help='')

    parser.add_argument("--model_arch", type=str, default='GCN2_32',
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32',
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32',
                                 'GAT5_64', 'GAT5_32', 'GAT3_64', 'GAT3_32', 'GAT2_64', 'GAT2_32'], help='graph models')

    parser.add_argument("--base_model", type=str, default='GCN', choices=['GIN', 'GCN', 'GAT'], help='graph models')

    parser.add_argument('--path_t', type=str, default='saved_models/pretrained_models', help='teacher path')

    # dataset
    parser.add_argument('--dataset', type=str, default='PTC', choices=['REDDITBINARY', 'PROTEINS', 'MUTAG', 'PTC', 'COLLAB', 'IMDBBINARY', 'NCI1'],
                        help='name of dataset (default: MUTAG)')

    parser.add_argument('--data_dir', type=str, default='./dataset', help='data path')

    parser.add_argument("--self_loop", action='store_true', help='add self_loop to graph data')

    parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')

    parser.add_argument('--split_name', type=str, default='mean_degree_sort', choices=['rand', 'mean_degree_sort'],
                        help='rand split with dataseed')

    parser.add_argument("--split", type=int, default=3, help="number of splits")

    parser.add_argument("--dim_feat", type=int, default=19,
                        help="number of node feature dim:{'IMDBBINARY': 1, 'MUTAG': 7, 'COLLAB': 1, 'PTC': 19, 'PROTEINS': 3, 'REDDITBINARY': 1, 'NCI1': 37}")

    parser.add_argument("--gcls", type=int, default=2,
                        help="number of graph classes:{'IMDBBINARY': 2, 'MUTAG': 2, 'COLLAB': 3, 'PTC': 2, 'PROTEINS': 2, 'REDDITBINARY': 2, 'NCI1': 2}")

    parser.add_argument('--batch_size', type=int, default=3000000,
                        help='batch size for training and validation (default: 32)')

    parser.add_argument('--fake_num', type=int, default=50,
                        help='number of fake graphs, defualt 131 on MUTAG, 188*0.7=131.6')
    # parser.add_argument('--total_num', type=int, default=1000,
    #                     help='number of fake graphs, defualt 131 on MUTAG, 188*0.7=131.6')

    # regularization
    # parser.add_argument("--fea_p", type=float, default=1, help="lambda for fea_loss")
    parser.add_argument("--conn_p", type=float, default=1, help="lambda for adj_loss")

    parser.add_argument("--gen_p", type=float, default=10, help="lambda for ori_ce")
    parser.add_argument("--bn_p", type=float, default=1, help="lambda for bn_loss")
    parser.add_argument("--confi_p", type=float, default=1, help="lambda for ori_ce")

    parser.add_argument("--gate_p", type=float, default=1, help="lambda for gate_loss")
    parser.add_argument("--mask_p", type=float, default=1, help="lambda for mask_loss")
    parser.add_argument("--kl_p", type=float, default=1, help="lambda for kl_loss")


    parser.add_argument("--kd_T", type=float, default=2, help="temperature for KL")

    parser.add_argument("--threshold", type=float, default=0.5, help="mask threshold")

    parser.add_argument("--apply_mask", action='store_false', help='apply mask')

    parser.add_argument("--num_experts", type=int, default=2, help="select number of experts")

    parser.add_argument("--mask_scale", type=float, default=1, help="fill in the initial mask")

    parser.add_argument('--test_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--test_index", type=int, default=0, help="use the x-th data as testing data")

    parser.add_argument('--train_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--train_index", type=int, default=0, help="use the x-th data as testing data")

    args = parser.parse_args()

    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir

    for args.split in [3]:
        if args.dataset == 'MUTAG':
            args.epoch = 100
            if args.split == 3:
                args.fake_num = 50
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed4_acc92_test32.pth'],
                                    ['GCN2-32_best_train_1_seed0_acc92_test39.pth']]
                elif args.base_model =='GIN':
                    args.path_list = [['GIN2-32_best_train_0_seed1_acc100_test96.pth'],
                                    ['GIN2-32_best_train_1_seed4_acc92_test96.pth']]
                elif args.base_model =='GAT':
                    args.path_list = [['GAT2-32_best_train_0_seed1_acc100_test96.pth'],
                                    ['GAT2-32_best_train_1_seed9_acc100_test90.pth']]
                    
        elif args.dataset == 'NCI1':
            args.epoch = 200
            if args.split == 16:
                args.fake_num = 50
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed6_acc92_test50.pth'],
                                    ['GCN2-32_best_train_1_seed6_acc76_test54.pth'],
                                    ['GCN2-32_best_train_2_seed9_acc84_test52.pth'],
                                    ['GCN2-32_best_train_3_seed7_acc80_test55.pth'],
                                    ['GCN2-32_best_train_4_seed3_acc73_test57.pth'],
                                    ['GCN2-32_best_train_5_seed9_acc73_test63.pth'],
                                    ['GCN2-32_best_train_6_seed3_acc71_test73.pth'],
                                    ['GCN2-32_best_train_7_seed8_acc61_test69.pth'],
                                    ['GCN2-32_best_train_8_seed6_acc73_test65.pth'],
                                    ['GCN2-32_best_train_9_seed7_acc59_test61.pth'],
                                    ['GCN2-32_best_train_10_seed9_acc73_test74.pth'],
                                    ['GCN2-32_best_train_11_seed6_acc63_test68.pth'],
                                    ['GCN2-32_best_train_12_seed2_acc71_test68.pth'],
                                    ['GCN2-32_best_train_13_seed6_acc73_test75.pth'],
                                    ['GCN2-32_best_train_14_seed3_acc84_test78.pth']]
            elif args.split == 12:
                args.fake_num = 50
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed4_acc85_test44.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc76_test53.pth'],
                                    ['GCN2-32_best_train_2_seed6_acc79_test58.pth'],
                                    ['GCN2-32_best_train_3_seed4_acc78_test61.pth'],
                                    ['GCN2-32_best_train_4_seed4_acc78_test64.pth'],
                                    ['GCN2-32_best_train_5_seed4_acc71_test68.pth'],
                                    ['GCN2-32_best_train_6_seed4_acc60_test64.pth'],
                                    ['GCN2-32_best_train_7_seed4_acc60_test66.pth'],
                                    ['GCN2-32_best_train_8_seed4_acc72_test66.pth'],
                                    ['GCN2-32_best_train_9_seed3_acc66_test75.pth'],
                                    ['GCN2-32_best_train_10_seed3_acc79_test72.pth']]

            elif args.split == 10:
                args.fake_num = 100
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed7_acc89_test43.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc80_test61.pth'],
                                    ['GCN2-32_best_train_2_seed8_acc77_test63.pth'],
                                    ['GCN2-32_best_train_3_seed8_acc66_test66.pth'],
                                    ['GCN2-32_best_train_4_seed5_acc67_test65.pth'],
                                    ['GCN2-32_best_train_5_seed4_acc80_test71.pth'],
                                    ['GCN2-32_best_train_6_seed0_acc65_test78.pth'],
                                    ['GCN2-32_best_train_7_seed7_acc68_test76.pth'],
                                    ['GCN2-32_best_train_8_seed0_acc81_test75.pth']]

            elif args.split == 8:
                args.fake_num = 150
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed5_acc80_test54.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc70_test62.pth'],
                                    ['GCN2-32_best_train_2_seed5_acc58_test68.pth'],
                                    ['GCN2-32_best_train_3_seed8_acc71_test63.pth'],
                                    ['GCN2-32_best_train_4_seed5_acc66_test65.pth'],
                                    ['GCN2-32_best_train_5_seed5_acc62_test64.pth'],
                                    ['GCN2-32_best_train_6_seed1_acc71_test72.pth']]

            elif args.split == 6:
                args.fake_num = 250
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed9_acc82_test62.pth'],
                                    ['GCN2-32_best_train_1_seed9_acc71_test60.pth'],
                                    ['GCN2-32_best_train_2_seed1_acc67_test62.pth'],
                                    ['GCN2-32_best_train_3_seed0_acc72_test68.pth'],
                                    ['GCN2-32_best_train_4_seed3_acc64_test72.pth']]

            elif args.split == 4:
                args.fake_num = 250
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed3_acc79_test50.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc71_test55.pth'],
                                    ['GCN2-32_best_train_2_seed3_acc69_test67.pth']]
                
            elif args.split == 3:
                args.fake_num = 250
                if args.base_model == 'GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed7_acc82_test61.pth'],
                                    ['GCN2-32_best_train_1_seed9_acc68_test65.pth']]
                elif args.base_model == 'GIN':
                    args.path_list = [['GIN2-32_best_train_0_seed7_acc82_test82.pth'],
                                    ['GIN2-32_best_train_1_seed1_acc78_test76.pth']]
                elif args.base_model == 'GAT':
                    args.path_list = [['GAT2-32_best_train_0_seed6_acc78_test78.pth'],
                                    ['GAT2-32_best_train_1_seed2_acc74_test73.pth']]
            
        elif args.dataset == 'REDDITBINARY':
            args.epoch = 200
            if args.split == 16:
                args.fake_num = 50
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed5_acc92_test88.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc88_test76.pth'],
                                    ['GCN2-32_best_train_2_seed6_acc72_test89.pth'],
                                    ['GCN2-32_best_train_3_seed8_acc88_test70.pth'],
                                    ['GCN2-32_best_train_4_seed3_acc88_test84.pth'],
                                    ['GCN2-32_best_train_5_seed2_acc80_test79.pth'],
                                    ['GCN2-32_best_train_6_seed6_acc80_test92.pth'],
                                    ['GCN2-32_best_train_7_seed5_acc76_test87.pth'],
                                    ['GCN2-32_best_train_8_seed6_acc80_test88.pth'],
                                    ['GCN2-32_best_train_9_seed1_acc84_test89.pth'],
                                    ['GCN2-32_best_train_10_seed2_acc80_test92.pth'],
                                    ['GCN2-32_best_train_11_seed1_acc84_test87.pth'],
                                    ['GCN2-32_best_train_12_seed2_acc100_test81.pth'],
                                    ['GCN2-32_best_train_13_seed2_acc92_test83.pth'],
                                    ['GCN2-32_best_train_14_seed2_acc96_test88.pth']]
            elif args.split == 12:
                args.fake_num = 80
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed4_acc91_test91.pth'],
                                    ['GCN2-32_best_train_1_seed9_acc76_test84.pth'],
                                    ['GCN2-32_best_train_2_seed4_acc79_test81.pth'],
                                    ['GCN2-32_best_train_3_seed8_acc85_test81.pth'],
                                    ['GCN2-32_best_train_4_seed3_acc79_test85.pth'],
                                    ['GCN2-32_best_train_5_seed1_acc76_test81.pth'],
                                    ['GCN2-32_best_train_6_seed9_acc85_test90.pth'],
                                    ['GCN2-32_best_train_7_seed9_acc88_test87.pth'],
                                    ['GCN2-32_best_train_8_seed9_acc85_test91.pth'],
                                    ['GCN2-32_best_train_9_seed4_acc97_test88.pth'],
                                    ['GCN2-32_best_train_10_seed2_acc94_test86.pth']]

            elif args.split == 10:
                args.fake_num = 100
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed0_acc95_test84.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc73_test83.pth'],
                                    ['GCN2-32_best_train_2_seed4_acc78_test82.pth'],
                                    ['GCN2-32_best_train_3_seed4_acc87_test82.pth'],
                                    ['GCN2-32_best_train_4_seed3_acc80_test83.pth'],
                                    ['GCN2-32_best_train_5_seed5_acc85_test89.pth'],
                                    ['GCN2-32_best_train_6_seed1_acc87_test91.pth'],
                                    ['GCN2-32_best_train_7_seed1_acc87_test91.pth'],
                                    ['GCN2-32_best_train_8_seed1_acc90_test89.pth']]

            elif args.split == 8:
                args.fake_num = 125
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed7_acc82_test88.pth'],
                                    ['GCN2-32_best_train_1_seed3_acc70_test81.pth'],
                                    ['GCN2-32_best_train_2_seed2_acc86_test40.pth'],
                                    ['GCN2-32_best_train_3_seed4_acc66_test89.pth'],
                                    ['GCN2-32_best_train_4_seed9_acc82_test90.pth'],
                                    ['GCN2-32_best_train_5_seed4_acc90_test91.pth'],
                                    ['GCN2-32_best_train_6_seed0_acc94_test87.pth']]

            elif args.split == 6:
                args.fake_num = 160
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed3_acc85_test85.pth'],
                                    ['GCN2-32_best_train_1_seed5_acc79_test26.pth'],
                                    ['GCN2-32_best_train_2_seed1_acc68_test79.pth'],
                                    ['GCN2-32_best_train_3_seed3_acc80_test91.pth'],
                                    ['GCN2-32_best_train_4_seed8_acc91_test88.pth']]

            elif args.split == 4:
                args.fake_num = 250
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed9_acc72_test32.pth'],
                                    ['GCN2-32_best_train_1_seed1_acc69_test36.pth'],
                                    ['GCN2-32_best_train_2_seed1_acc90_test91.pth']]
                
            elif args.split == 3:
                args.fake_num = 300
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed4_acc82_test49.pth'],
                                    ['GCN2-32_best_train_1_seed6_acc69_test88.pth']]
                elif args.base_model =='GIN':
                    args.path_list = [['GIN2-32_best_train_0_seed9_acc73_test78.pth'],
                                    ['GIN2-32_best_train_1_seed6_acc79_test91.pth']]
                elif args.base_model =='GAT':
                    args.path_list = [['GAT2-32_best_train_0_seed7_acc73_test76.pth'],
                                    ['GAT2-32_best_train_1_seed2_acc77_test79.pth']]
                    
        elif args.dataset == 'PTC':
            args.epoch = 100
            if args.split == 3:
                args.fake_num = 50
                if args.base_model =='GCN':
                    args.path_list = [['GCN2-32_best_train_0_seed2_acc78_test53.pth'],
                                    ['GCN2-32_best_train_1_seed7_acc73_test54.pth']]
                elif args.base_model =='GIN':
                    args.path_list = [['GIN2-32_best_train_0_seed0_acc82_test75.pth'],
                                    ['GIN2-32_best_train_1_seed3_acc65_test83.pth']]
                elif args.base_model =='GAT':
                    args.path_list = [['GAT2-32_best_train_0_seed8_acc73_test66.pth'],
                                    ['GAT2-32_best_train_1_seed8_acc69_test75.pth']]
                    
        # for args.fake_num in [100, 200, 300, 400]:
        # for args.base_model in ['GAT', 'GIN']:
        args.model_arch = f'{args.base_model}2_32'
        for args.test_index in range(args.split - 1):
            for args.id in [-1]:
                set_seed(args.seed)
                print('seed: %d' % args.seed)
                main(args)
