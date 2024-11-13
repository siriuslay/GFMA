import argparse
import numpy as np
import os, sys
import random
import time
import torch
import torch.nn as nn
from data_loader.GIN_data_downloader import GINDataset
from data_loader.Gene_dataset import MergeDataset
from data_loader.GIN_data_downloader import GraphDataLoader, collate
from modules.mome import MoME
from modules.KL_loss import DistillKL
from modules.scheduler import LinearSchedule
from utils import evaluate_ptm, test_ptm, test_moe, evaluate_moe, RegLoss, merge_batches, load_ptm
from torch.utils.tensorboard import SummaryWriter
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

project_root = os.path.abspath(os.path.dirname(__file__))

# 遍历所有子文件夹
for root, dirs, files in os.walk(project_root):
    sys.path.append(root)

# writer = SummaryWriter('runs/gen_moe_test')

torch.autograd.set_detect_anomaly(True)

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
            graphs = graphs.to(model.mean.device)
            feat = graphs.ndata['attr'].to(model.mean.device)
            labels = labels.to(model.mean.device)
            total += len(labels)
            outputs, gates, expert_out_list, gate_loss, mask_loss = model(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            loss = loss_fcn(outputs, labels)

            total_loss += loss * len(labels)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return loss, acc, gates, precision, recall, f1, cm


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


def task_model(args, net_list=None, path_model=None, net_arch_list=None):
    if path_model is not None:
        model = MoME(args, pretrained_model_list=net_list, ptm_arch_list=net_arch_list)
        model.load_state_dict(torch.load(path_model)['model'])
    else:
        model = MoME(args, pretrained_model_list=net_list, ptm_arch_list=net_arch_list)
    # step 2: prepare loss
    cross_ent = nn.CrossEntropyLoss()
    kl_div = DistillKL(args.kd_T)

    if args.gpu >= 0:
        model = model.to(f'cuda:{args.gpu}')
        cross_ent = cross_ent.to(f'cuda:{args.gpu}')
        kl_div = kl_div.to(f'cuda:{args.gpu}')

    return model, cross_ent, kl_div


def train(args, model, train_loader, valid_loader, cross_ent, kl_div, optimizer_m, optimizer_w, test_loader):
    # log_dir = "~/autodl-tmp/code/WWW/tensorboard/runs/gen_moe_test/exp_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = "runs/train_mome/"
    # writer = SummaryWriter(log_dir=log_dir)

    if args.apply_mask:
        scheduler1 = LinearSchedule(optimizer_m, args.epoch)
    scheduler2 = LinearSchedule(optimizer_w, args.epoch)
    dur = []
    best_acc = 0
    mini_loss = 1e9
    best_test_acc = 0
    model_name = 'saved_models/trained_models/{}_{}'.format(args.dataset, args.model_name)
    if not os.path.isdir(model_name):
        os.makedirs(model_name)

    for epoch in range(1, args.epoch + 1):
        model.train()
        t0 = time.time()

        for graphs, labels in train_loader:

            features = graphs.ndata['attr'].to(f'cuda:{args.gpu}')

            graphs = graphs.to(f'cuda:{args.gpu}')

            output, gates, me_out_list, gate_loss, mask_loss = model(graphs, features)

            labels = labels.to(output.device)

        # show the acc
        _, predicted = torch.max(output.data, 1)

        total_correct = (predicted == labels.data).sum().item()

        train_acc = total_correct / len(labels)

        if args.apply_mask:
            optimizer_m.zero_grad()
        optimizer_w.zero_grad()

        kl_loss_list = []
        ce_loss_list = []
        for i, out in enumerate(me_out_list):
            kl_loss_list.append(kl_div(output, out))
            ce_loss_list.append(cross_ent(out, labels))
            # _, predicted = torch.max(out.data, 1)
            # total_correct_i = (predicted == labels.data).sum().item()
            # acc_i = total_correct_i / len(labels)
            # print(f'{i}-th mPTM training acc = {acc_i}')

        main_loss = cross_ent(output, labels)
        
        # ori_ce_loss = sum(ori_ce_loss_list) / len(ori_ce_loss_list)
        kl_loss = sum(kl_loss_list) / len(kl_loss_list)
        ce_loss = sum(ce_loss_list) / len(ce_loss_list)

        moe_loss = main_loss + args.gate_p * gate_loss + args.ce_p * ce_loss + args.mask_p * mask_loss
        # moe_loss = main_loss + args.gate_p * gate_loss + args.mask_p * (ce_loss + mask_loss)

        loss = moe_loss

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        if args.apply_mask:
            optimizer_m.step()
        optimizer_w.step()

        dur.append(time.time() - t0)

        # writer.add_scalar('Loss/train', loss.item(), epoch)
        # writer.add_scalar('Loss/moe', moe_loss.item(), epoch)
        # writer.add_scalar('Loss/main_ce', main_loss.item(), epoch)
        # writer.add_scalar('Loss/gate_loss', gate_loss.item(), epoch)
        # writer.add_scalar('Loss/ce_loss', ce_loss.item(), epoch)
        # writer.add_scalar('Loss/mask_loss', mask_loss.item(), epoch)

        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)
        #     if param.grad is not None:
        #         writer.add_histogram(f'{name}.grad', param.grad, epoch)

        model.eval()  # 切换子模块到评估模式
        total = 0
        total_correct = 0
        ##############
        # expert_total_correct = [0, 0, 0, 0]
        #################
        with torch.no_grad():  # 在验证时不计算梯度
            for data in valid_loader:
                graphs, labels = data
                graphs = graphs.to(f'cuda:{args.gpu}')
                feat = graphs.ndata['attr'].to(f'cuda:{args.gpu}')
                labels = labels.to(f'cuda:{args.gpu}')
                total += len(labels)
                outputs, gates, test_expert_out_list, gate_loss, mask_loss = model(graphs, feat)
        #################
                # for test_i, test_out in enumerate(test_expert_out_list):
                #     _, test_expert_predicted = torch.max(test_out.data, 1)
                #     expert_total_correct[test_i] += (test_expert_predicted == labels.data).sum().item()
        #################
                _, predicted = torch.max(outputs.data, 1)

                total_correct += (predicted == labels.data).sum().item()

            fake_valid_acc = 1.0 * total_correct / total
        #################
            # for i in range(len(expert_total_correct)):
            #     expert_test_acc = 1.0 * expert_total_correct[i] / total
            #     print(f'{i}-th mPTM valid acc = {expert_test_acc}')
        #################

        # writer.add_scalar('ACC/valid_acc', fake_valid_acc, epoch)

        ################### real test
        total = 0
        total_correct = 0
        # expert_total_correct = [0, 0, 0, 0]
        with torch.no_grad():  # 在验证时不计算梯度
            for data in test_loader:
                graphs, labels = data
                graphs = graphs.to(f'cuda:{args.gpu}')
                feat = graphs.ndata['attr'].to(f'cuda:{args.gpu}')
                labels = labels.to(f'cuda:{args.gpu}')
                total += len(labels)
                outputs, gates, test_expert_out_list, gate_loss, mask_loss = model(graphs, feat)
                # for test_i, test_out in enumerate(test_expert_out_list):
                #     _, test_expert_predicted = torch.max(test_out.data, 1)
                #     expert_total_correct[test_i] += (test_expert_predicted == labels.data).sum().item()
                _, predicted = torch.max(outputs.data, 1)

                total_correct += (predicted == labels.data).sum().item()

            real_test_acc = 1.0 * total_correct / total
            #################
            # for i in range(len(expert_total_correct)):
            #     expert_test_acc = 1.0 * expert_total_correct[i] / total
            #     print(f'{i}-th mPTM test acc = {expert_test_acc}')
        #################

        if real_test_acc >= best_test_acc:
            best_test_acc = real_test_acc
            best_epoch = epoch
        # writer.add_scalar('ACC/test_acc', real_test_acc, epoch)

        # 可以选择在验证后恢复子模块的 train 模式
        model.train()

        print(f'Epoch {epoch} complete. fake_valid_acc = {fake_valid_acc}')
        print(f'Epoch {epoch} complete. real_test_acc = {real_test_acc}')

        if (epoch % 5 == 0) & (epoch != 0):
            print(f'{epoch} epoch: training acc = {train_acc}')
            print('Valid acc {:.4f}'.format(float(fake_valid_acc)))
            print('Test acc {:.4f}'.format(float(real_test_acc)))
            print('Average Epoch Time {:.4f}'.format(float(sum(dur) / len(dur))))
            print('Training_loss {:.4f}'.format(float(loss.item())))

        # if fake_valid_acc > best_acc:
            # best_acc = fake_valid_acc
        if mini_loss >= loss.item():
            mini_loss = loss.item()
            state = {
                'model_name': args.model_name,
                'model_arch': args.model_arch,
                'epoch': epoch,
                'model': model.state_dict(),
                # 'masks_list': masks_list
            }

            save_file = os.path.join(model_name, 'GFMA_best_allexperts{}_seed{}_ne{}.pth'.
                                     format(args.total_model, args.seed, args.num_experts))
            print('saving the best model!')
            torch.save(state, save_file)
        if args.apply_mask:
            scheduler1.step()
        scheduler2.step()

    # writer.close()

    save_file_last = os.path.join(model_name, 'GFMA_last_allexperts{}_seed{}_ne{}.pth'.
                                  format(args.total_model, args.seed, args.num_experts))
    state = {
        'model_name': args.model_name,
        'model_arch': args.model_arch,
        'epoch': epoch,
        'model': model.state_dict(),
        # 'masks_list': masks_list
    }

    if args.choose_model == 'last':
        torch.save(state, save_file_last)
        best_acc = fake_valid_acc
        save_file = save_file_last

    return best_acc, save_file, best_test_acc, best_epoch


def test(data_loader, model, cross_ent):
    _, test_acc, gates, precision, recall, f1, cm = evaluate(model, data_loader, cross_ent)
    print('gate:')
    print(gates[0:5])
    print(f'test_acc: {test_acc}')

    return test_acc, precision, recall, f1, cm


def main(args):

    # prepare real data
    dataset, train_loader, valid_loader, test_loader = task_data(args)

    # prepare pretrained model
    print("loading PTMs")
    net_list, net_arch_list = load_ptm(args)

    # prepare fake data
    all_graphs = []
    all_labels = []
    for net_arch in net_arch_list:
        for id in range(args.split - 1):
            save_file = f'pseudo_graphs/{args.dataset}/fakenum{args.fake_num}_{args.split}parts_test{id}_process_gen_graph_sparse0.9_adj_best_{net_arch}_seed1001_e-1_epo100.pth'
            # save_file = f'pseudo_graphs/{args.dataset}/fakenum{args.fake_num}_{args.split}parts_test{id}_process_gen_graph_sparse0.9_adj_best_GCN2_32_seed1001_e-1_epo200.pth'
            fake_data_loader = torch.load(save_file)['fake_data']
            for graphs, labels in fake_data_loader:
                all_graphs.append(graphs)
                all_labels.append(labels)

    all_graphs, merged_labels = merge_batches(all_graphs, all_labels)
    merged_dataset = MergeDataset(all_graphs, merged_labels)
    merged_train_data_loader, merged_valid_data_loader, merged_test_data_loader = GraphDataLoader(
        merged_dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name=args.split_name, split=args.split, train_index=-1,
        test_index=args.split - 1).train_valid_test_loader()


    # test PTMs & make mask_ptm
    print("making masked MOE")

    for i, net in enumerate(net_list):
        print(f'{i}-th PTM: ')
        test_ptm_ac = test_ptm(test_loader, net, args)

        trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"number of trainable PTM {i} parameters:{trainable_param}")

    # prepare model
    model, cross_ent, kl_div = task_model(args, net_list=net_list, net_arch_list=net_arch_list)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)

    print(f"Parameters to be updated: {enabled}")

    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable mask parameters:{trainable_param}")

    params_outside = [param for name, param in model.named_parameters() if (name == 'w_noise' or name == 'w_gate')]

    if args.apply_mask:
        optimizer_m = torch.optim.AdamW(model.mask_experts.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_m = 0
    optimizer_w = torch.optim.AdamW(params_outside, lr=0.01, weight_decay=args.weight_decay)

    print('************************')
    print('Training start')

    # train
    best_acc, save_file, best_test_acc, best_epoch = train(args, model, merged_train_data_loader, merged_valid_data_loader, cross_ent, kl_div, optimizer_m, optimizer_w, test_loader)

    # best_acc, save_file, best_test_acc, best_epoch = train(args, model, train_loader, valid_loader, cross_ent, kl_div, optimizer_m, optimizer_w, test_loader)

    print(f'best_acc: {best_acc}')


    print('Training finished')
    print('************************')

    # save_file = 'SAVE/{}_{}'.format(args.dataset, args.model_name) + '/mome_best_{}_seed{}_ne{}.pth'\
    #     .format(args.base_model, args.seed, args.num_experts)

    # test
    args.training = False
    model, cross_ent, kl_div = task_model(args, net_list=net_list, path_model=save_file, net_arch_list=net_arch_list)

    test_real_ac, precision, recall, f1, cm = test(test_loader, model, cross_ent)
    # test_fake_ac = test(merged_test_data_loader, model, cross_ent)
    print(f'real: {test_real_ac}')
    # print(f'fake: {test_fake_ac}')

    return test_real_ac, best_test_acc, best_epoch, precision, recall, f1, cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Mixture of pretrained GNNs for SFDA')

    parser.add_argument("--epoch", type=int, default=100, help="number of training iteration")

    parser.add_argument("--total_model", type=int, default=6, help="number of ptm")

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--seed", type=int, default=0, help='random seed')  # just for real test loader and path

    parser.add_argument("--training", type=bool, default=True, help='train or eval')

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")

    parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight for L2 Loss')

    parser.add_argument('--choose_model', type=str, default='best',
                        choices=['last', 'best'], help='test the last / best trained model')

    # path
    parser.add_argument('--model_name', type=str, default='GFMA', help='')

    parser.add_argument("--model_arch", type=str, default='GCN2_32',
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32',
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32',
                                 'GAT5_64', 'GAT5_32', 'GAT3_64', 'GAT3_32', 'GAT2_64', 'GAT2_32'], help='graph models')

    parser.add_argument("--base_model", type=str, default='GCN', choices=['GIN', 'GCN', 'GAT'], help='graph models')

    parser.add_argument('--path_t', type=str, default='saved_models/pretrained_models', help='ptm path')

    # dataset
    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['REDDITBINARY', 'PROTEINS', 'MUTAG', 'PTC', 'COLLAB', 'IMDBBINARY', 'NCI1'],
                        help='name of dataset (default: MUTAG)')

    parser.add_argument('--data_dir', type=str, default='./dataset', help='data path')

    parser.add_argument("--self_loop", action='store_false', help='add self_loop to graph data')

    parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')

    parser.add_argument('--split_name', type=str, default='mean_degree_sort', choices=['rand', 'mean_degree_sort'],
                        help='rand split with dataseed')

    parser.add_argument("--split", type=int, default=16, help="number of splits")

    parser.add_argument("--dim_feat", type=int, default=7,
                        help="number of node feature dim:{'IMDBBINARY': 1, 'MUTAG': 7, 'COLLAB': 1, 'PTC': 19, 'PROTEINS': 3, 'REDDITBINARY': 1, 'NCI1': 37}")

    parser.add_argument("--gcls", type=int, default=2,
                        help="number of graph classes:{'IMDBBINARY': 2, 'MUTAG': 2, 'COLLAB': 3, 'PTC': 2, 'PROTEINS': 2, 'REDDITBINARY': 2, 'NCI1': 2}")

    parser.add_argument('--batch_size', type=int, default=300000,
                        help='batch size for training and validation (default: 32)')

    parser.add_argument('--fake_num', type=int, default=50,
                        help='number of fake graphs, defualt 131 on MUTAG, 188*0.7=131.6')
    # parser.add_argument('--total_num', type=int, default=750,
    #                     help='number of fake graphs, defualt 131 on MUTAG, 188*0.7=131.6')

    # regularization
    # parser.add_argument("--fea_p", type=float, default=1, help="lambda for fea_loss")
    parser.add_argument("--gate_p", type=float, default=1, help="lambda for gate_loss")
    parser.add_argument("--mask_p", type=float, default=1e-2, help="lambda for mask_loss")
    parser.add_argument("--ce_p", type=float, default=1, help="lambda for kl_loss")

    parser.add_argument("--conn_p", type=float, default=0, help="lambda for adj_loss")
    
    parser.add_argument("--bn_p", type=float, default=0, help="lambda for bn_loss")
    
    parser.add_argument("--gen_p", type=float, default=0, help="lambda for ori_ce")
    parser.add_argument("--kd_T", type=float, default=1, help="temperature for KL")

    parser.add_argument("--threshold", type=float, default=0.5, help="mask threshold")

    parser.add_argument("--apply_mask", action='store_false', help='apply mask')

    parser.add_argument("--num_experts", type=int, default=1, help="select number of experts")

    parser.add_argument("--mask_scale", type=float, default=1, help="fill in the initial mask")

    parser.add_argument('--test_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--test_index", type=int, default=2, help="use the x-th data as testing data")

    parser.add_argument('--train_data', type=str, default='real',
                        choices=['real', 'fake'], help='choose type of dataset')

    parser.add_argument("--train_index", type=int, default=-1, help="use the x-th data as testing data")

    args = parser.parse_args()

    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir
    
    for args.split in [3]:      # , 10, 8, 6, 4
        if args.dataset == 'MUTAG':
            args.epoch = 50
            if args.split == 3:
                args.fake_num = 50
                args.path_list = ['GCN2-32_best_train_0_seed4_acc92_test32.pth',
                                'GCN2-32_best_train_1_seed0_acc92_test39.pth',
                                'GIN2-32_best_train_0_seed1_acc100_test96.pth',
                                'GIN2-32_best_train_1_seed4_acc92_test96.pth',
                                'GAT2-32_best_train_0_seed1_acc100_test96.pth',
                                'GAT2-32_best_train_1_seed9_acc100_test90.pth']
                    
        elif args.dataset == 'NCI1':
            args.epoch = 20
            if args.split == 16:
                args.fake_num = 50
                args.path_list = ['GCN2-32_best_train_0_seed6_acc92_test50.pth',
                                'GCN2-32_best_train_1_seed6_acc76_test54.pth',
                                'GCN2-32_best_train_2_seed9_acc84_test52.pth',
                                'GCN2-32_best_train_3_seed7_acc80_test55.pth',
                                'GCN2-32_best_train_4_seed3_acc73_test57.pth',
                                'GCN2-32_best_train_5_seed9_acc73_test63.pth',
                                'GCN2-32_best_train_6_seed3_acc71_test73.pth',
                                'GCN2-32_best_train_7_seed8_acc61_test69.pth',
                                'GCN2-32_best_train_8_seed6_acc73_test65.pth',
                                'GCN2-32_best_train_9_seed7_acc59_test61.pth',
                                'GCN2-32_best_train_10_seed9_acc73_test74.pth',
                                'GCN2-32_best_train_11_seed6_acc63_test68.pth',
                                'GCN2-32_best_train_12_seed2_acc71_test68.pth',
                                'GCN2-32_best_train_13_seed6_acc73_test75.pth',
                                'GCN2-32_best_train_14_seed3_acc84_test78.pth']
            elif args.split == 12:
                args.fake_num = 50
                args.path_list = ['GCN2-32_best_train_0_seed4_acc85_test44.pth',
                                'GCN2-32_best_train_1_seed3_acc76_test53.pth',
                                'GCN2-32_best_train_2_seed6_acc79_test58.pth',
                                'GCN2-32_best_train_3_seed4_acc78_test61.pth',
                                'GCN2-32_best_train_4_seed4_acc78_test64.pth',
                                'GCN2-32_best_train_5_seed4_acc71_test68.pth',
                                'GCN2-32_best_train_6_seed4_acc60_test64.pth',
                                'GCN2-32_best_train_7_seed4_acc60_test66.pth',
                                'GCN2-32_best_train_8_seed4_acc72_test66.pth',
                                'GCN2-32_best_train_9_seed3_acc66_test75.pth',
                                'GCN2-32_best_train_10_seed3_acc79_test72.pth']

            elif args.split == 10:
                args.fake_num = 100
                args.path_list = ['GCN2-32_best_train_0_seed7_acc89_test43.pth',
                                'GCN2-32_best_train_1_seed3_acc80_test61.pth',
                                'GCN2-32_best_train_2_seed8_acc77_test63.pth',
                                'GCN2-32_best_train_3_seed8_acc66_test66.pth',
                                'GCN2-32_best_train_4_seed5_acc67_test65.pth',
                                'GCN2-32_best_train_5_seed4_acc80_test71.pth',
                                'GCN2-32_best_train_6_seed0_acc65_test78.pth',
                                'GCN2-32_best_train_7_seed7_acc68_test76.pth',
                                'GCN2-32_best_train_8_seed0_acc81_test75.pth']

            elif args.split == 8:
                args.fake_num = 150
                args.path_list = ['GCN2-32_best_train_0_seed5_acc80_test54.pth',
                                'GCN2-32_best_train_1_seed3_acc70_test62.pth',
                                'GCN2-32_best_train_2_seed5_acc58_test68.pth',
                                'GCN2-32_best_train_3_seed8_acc71_test63.pth',
                                'GCN2-32_best_train_4_seed5_acc66_test65.pth',
                                'GCN2-32_best_train_5_seed5_acc62_test64.pth',
                                'GCN2-32_best_train_6_seed1_acc71_test72.pth']

            elif args.split == 6:
                args.fake_num = 250
                args.path_list = ['GCN2-32_best_train_0_seed9_acc82_test62.pth',
                                'GCN2-32_best_train_1_seed9_acc71_test60.pth',
                                'GCN2-32_best_train_2_seed1_acc67_test62.pth',
                                'GCN2-32_best_train_3_seed0_acc72_test68.pth',
                                'GCN2-32_best_train_4_seed3_acc64_test72.pth']

            elif args.split == 4:
                args.fake_num = 250
                args.path_list = ['GCN2-32_best_train_0_seed3_acc79_test50.pth',
                                'GCN2-32_best_train_1_seed3_acc71_test55.pth',
                                'GCN2-32_best_train_2_seed3_acc69_test67.pth']
                
            elif args.split == 3:
                args.fake_num = 250
                args.path_list = ['GCN2-32_best_train_0_seed7_acc82_test61.pth',
                                'GCN2-32_best_train_1_seed9_acc68_test65.pth',
                                'GIN2-32_best_train_0_seed7_acc82_test82.pth',
                                'GIN2-32_best_train_1_seed1_acc78_test76.pth',
                                'GAT2-32_best_train_0_seed6_acc78_test78.pth',
                                'GAT2-32_best_train_1_seed2_acc74_test73.pth']
            
        elif args.dataset == 'REDDITBINARY':
            args.epoch = 20
            if args.split == 16:
                args.fake_num = 50
                args.path_list = ['GCN2-32_best_train_0_seed5_acc92_test88.pth',
                                'GCN2-32_best_train_1_seed3_acc88_test76.pth',
                                'GCN2-32_best_train_2_seed6_acc72_test89.pth',
                                'GCN2-32_best_train_3_seed8_acc88_test70.pth',
                                'GCN2-32_best_train_4_seed3_acc88_test84.pth',
                                'GCN2-32_best_train_5_seed2_acc80_test79.pth',
                                'GCN2-32_best_train_6_seed6_acc80_test92.pth',
                                'GCN2-32_best_train_7_seed5_acc76_test87.pth',
                                'GCN2-32_best_train_8_seed6_acc80_test88.pth',
                                'GCN2-32_best_train_9_seed1_acc84_test89.pth',
                                'GCN2-32_best_train_10_seed2_acc80_test92.pth',
                                'GCN2-32_best_train_11_seed1_acc84_test87.pth',
                                'GCN2-32_best_train_12_seed2_acc100_test81.pth',
                                'GCN2-32_best_train_13_seed2_acc92_test83.pth',
                                'GCN2-32_best_train_14_seed2_acc96_test88.pth']
            elif args.split == 12:
                args.fake_num = 80
                args.path_list = ['GCN2-32_best_train_0_seed4_acc91_test91.pth',
                                'GCN2-32_best_train_1_seed9_acc76_test84.pth',
                                'GCN2-32_best_train_2_seed4_acc79_test81.pth',
                                'GCN2-32_best_train_3_seed8_acc85_test81.pth',
                                'GCN2-32_best_train_4_seed3_acc79_test85.pth',
                                'GCN2-32_best_train_5_seed1_acc76_test81.pth',
                                'GCN2-32_best_train_6_seed9_acc85_test90.pth',
                                'GCN2-32_best_train_7_seed9_acc88_test87.pth',
                                'GCN2-32_best_train_8_seed9_acc85_test91.pth',
                                'GCN2-32_best_train_9_seed4_acc97_test88.pth',
                                'GCN2-32_best_train_10_seed2_acc94_test86.pth']

            elif args.split == 10:
                args.fake_num = 100
                args.path_list = ['GCN2-32_best_train_0_seed0_acc95_test84.pth',
                                'GCN2-32_best_train_1_seed3_acc73_test83.pth',
                                'GCN2-32_best_train_2_seed4_acc78_test82.pth',
                                'GCN2-32_best_train_3_seed4_acc87_test82.pth',
                                'GCN2-32_best_train_4_seed3_acc80_test83.pth',
                                'GCN2-32_best_train_5_seed5_acc85_test89.pth',
                                'GCN2-32_best_train_6_seed1_acc87_test91.pth',
                                'GCN2-32_best_train_7_seed1_acc87_test91.pth',
                                'GCN2-32_best_train_8_seed1_acc90_test89.pth']

            elif args.split == 8:
                args.fake_num = 125
                args.path_list = ['GCN2-32_best_train_0_seed7_acc82_test88.pth',
                                'GCN2-32_best_train_1_seed3_acc70_test81.pth',
                                'GCN2-32_best_train_2_seed2_acc86_test40.pth',
                                'GCN2-32_best_train_3_seed4_acc66_test89.pth',
                                'GCN2-32_best_train_4_seed9_acc82_test90.pth',
                                'GCN2-32_best_train_5_seed4_acc90_test91.pth',
                                'GCN2-32_best_train_6_seed0_acc94_test87.pth']

            elif args.split == 6:
                args.fake_num = 160
                args.path_list = ['GCN2-32_best_train_0_seed3_acc85_test85.pth',
                                'GCN2-32_best_train_1_seed5_acc79_test26.pth',
                                'GCN2-32_best_train_2_seed1_acc68_test79.pth',
                                'GCN2-32_best_train_3_seed3_acc80_test91.pth',
                                'GCN2-32_best_train_4_seed8_acc91_test88.pth']

            elif args.split == 4:
                args.fake_num = 250
                args.path_list = ['GCN2-32_best_train_0_seed9_acc72_test32.pth',
                                'GCN2-32_best_train_1_seed1_acc69_test36.pth',
                                'GCN2-32_best_train_2_seed1_acc90_test91.pth']
                
            elif args.split == 3:
                args.fake_num = 300
                args.path_list = ['GCN2-32_best_train_0_seed4_acc82_test49.pth',
                                'GCN2-32_best_train_1_seed6_acc69_test88.pth',
                                'GIN2-32_best_train_0_seed9_acc73_test78.pth',
                                'GIN2-32_best_train_1_seed6_acc79_test91.pth',
                                'GAT2-32_best_train_0_seed7_acc73_test76.pth',
                                'GAT2-32_best_train_1_seed2_acc77_test79.pth']
                    
        elif args.dataset == 'PTC':
            args.epoch = 100
            if args.split == 3:
                args.fake_num = 50
                args.path_list = ['GCN2-32_best_train_0_seed2_acc78_test53.pth',
                                'GCN2-32_best_train_1_seed7_acc73_test54.pth',
                                'GIN2-32_best_train_0_seed0_acc82_test75.pth',
                                'GIN2-32_best_train_1_seed3_acc65_test83.pth',
                                'GAT2-32_best_train_0_seed8_acc73_test66.pth',
                                'GAT2-32_best_train_1_seed8_acc69_test75.pth']
                    


        args.test_index = args.split - 1
        # args.epoch = 100
        args.mask_loss = 100
        # args.lr = 0.001
        with open('results/train outs.txt', 'a') as file:
            print(f'\n*********************\n{args.dataset}: GFMA_acc', file=file)
            print(f'split:{args.split}', file=file)
            print(f'fake_num:{args.fake_num}', file=file)
        for args.epoch in [20, 50, 100]:
            for args.lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                for args.mask_p in [0.01, 0.1, 1, 10, 100]:   # [0.01, 0.1, 1, 10, 100]:
                    for args.gate_p in [0.01, 0.1, 1, 10, 100]:   # [0.01, 0.1, 1, 10, 100]:
                        for args.ce_p in [args.gate_p]:  # [0.01, 0.1, 1, 10, 100]:
                            for args.num_experts in range(1, args.total_model+1):
                                acc, precision, recall, f1score, confm = [], [], [], [], []
                                best_acc = []
                                for args.seed in range(10):
                                    set_seed(args.seed)
                                    print('seed: %d' % args.seed)
                                    ac, best_test_acc, best_epoch, pre, rec, f1, cm = main(args)
                                    best_acc.append([round(best_test_acc * 100, 2), best_epoch])
                                    acc.append(ac * 100)
                                    precision.append(pre * 100)
                                    recall.append(rec * 100)
                                    f1score.append(f1 * 100)
                                    confm.append(cm * 100)
                                raw1 = ['acc:', acc, f'{round(np.mean(acc), 2)}±{round(np.std(acc, ddof=0), 2)}'] 
                                raw2 = ['precision:', precision, f'{round(np.mean(precision), 2)}±{round(np.std(precision, ddof=0), 2)}']
                                raw3 = ['recall:', recall, f'{round(np.mean(recall), 2)}±{round(np.std(recall, ddof=0), 2)}'] 
                                raw4 = ['f1score:', f1score, f'{round(np.mean(f1score), 2)}±{round(np.std(f1score, ddof=0), 2)}']
                                with open('results/train outs.txt', 'a') as file:
                                    # print(f'\n{args.dataset}: Inverse_X_acc', file=file)
                                    print(f'\nbest_acc:{best_acc}', file=file)
                                    print(f'apply_mask:{args.apply_mask}, lr:{args.lr}, mask_p:{args.mask_p}, epoch:{args.epoch}, mask_loss:{args.mask_loss}, num_experts: {args.num_experts}, test_index: {args.test_index}', file=file)
                                    print(raw1, file=file)
                                    print(raw2, file=file)
                                    print(raw3, file=file)
                                    print(raw4, file=file)


            # print(f'apply_mask:{args.apply_mask}, lr:{args.lr}, mask_p:{args.mask_p}, epoch:{args.epoch}, mask_loss:{args.mask_loss}, test_index: {args.test_index}', file=file)