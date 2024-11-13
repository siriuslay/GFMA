import torch


save_file = f'pseudo_graphs/PTC/test0_process_gen_graph_sparse0.9_adj_best_GCN2_32_seed1001_e-1_epo100.pth'
# save_file = f'pseudo_graphs/{args.dataset}/fakenum{args.fake_num}_{args.split}parts_test{id}_process_gen_graph_sparse0.9_adj_best_GCN2_32_seed1001_e-1_epo200.pth'
fake_data_loader = torch.load(save_file)['fake_data']
path_model1 = 'saved_models/pretrained_models/MUTAG/3part/GAT2-32_best_train_0_seed0_acc92_test26.pth'
path_model2 = 'saved_models/pretrained_models/MUTAG/3part/GCN2-32_best_train_0_seed9_acc92_test26.pth'
path_model3 = 'saved_models/pretrained_models/MUTAG/3part/GIN2-32_best_train_0_seed8_acc92_test26.pth'
gat = torch.load(path_model1)['model']
gcn = torch.load(path_model2)['model']
gin = torch.load(path_model3)['model']
print(gat)

# for args.split in [3]:      # , 10, 8, 6, 4
#         if args.dataset == 'MUTAG':
#             args.epoch = 100
#             if args.split == 3:
#                 args.fake_num = 50
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed4_acc92_test32.pth',
#                                     'GCN2-32_best_train_1_seed0_acc92_test39.pth']
#                 elif args.base_model =='GIN':
#                     args.path_list = ['GIN2-32_best_train_0_seed1_acc100_test96.pth',
#                                     'GIN2-32_best_train_1_seed4_acc92_test96.pth']
#                 elif args.base_model =='GAT':
#                     args.path_list = ['GAT2-32_best_train_0_seed1_acc100_test96.pth',
#                                     'GAT2-32_best_train_1_seed9_acc100_test90.pth']
                    
#         elif args.dataset == 'NCI1':
#             args.epoch = 200
#             if args.split == 16:
#                 args.fake_num = 50
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed6_acc92_test50.pth',
#                                     'GCN2-32_best_train_1_seed6_acc76_test54.pth',
#                                     'GCN2-32_best_train_2_seed9_acc84_test52.pth',
#                                     'GCN2-32_best_train_3_seed7_acc80_test55.pth',
#                                     'GCN2-32_best_train_4_seed3_acc73_test57.pth',
#                                     'GCN2-32_best_train_5_seed9_acc73_test63.pth',
#                                     'GCN2-32_best_train_6_seed3_acc71_test73.pth',
#                                     'GCN2-32_best_train_7_seed8_acc61_test69.pth',
#                                     'GCN2-32_best_train_8_seed6_acc73_test65.pth',
#                                     'GCN2-32_best_train_9_seed7_acc59_test61.pth',
#                                     'GCN2-32_best_train_10_seed9_acc73_test74.pth',
#                                     'GCN2-32_best_train_11_seed6_acc63_test68.pth',
#                                     'GCN2-32_best_train_12_seed2_acc71_test68.pth',
#                                     'GCN2-32_best_train_13_seed6_acc73_test75.pth',
#                                     'GCN2-32_best_train_14_seed3_acc84_test78.pth']
#             elif args.split == 12:
#                 args.fake_num = 50
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed4_acc85_test44.pth',
#                                     'GCN2-32_best_train_1_seed3_acc76_test53.pth',
#                                     'GCN2-32_best_train_2_seed6_acc79_test58.pth',
#                                     'GCN2-32_best_train_3_seed4_acc78_test61.pth',
#                                     'GCN2-32_best_train_4_seed4_acc78_test64.pth',
#                                     'GCN2-32_best_train_5_seed4_acc71_test68.pth',
#                                     'GCN2-32_best_train_6_seed4_acc60_test64.pth',
#                                     'GCN2-32_best_train_7_seed4_acc60_test66.pth',
#                                     'GCN2-32_best_train_8_seed4_acc72_test66.pth',
#                                     'GCN2-32_best_train_9_seed3_acc66_test75.pth',
#                                     'GCN2-32_best_train_10_seed3_acc79_test72.pth']

#             elif args.split == 10:
#                 args.fake_num = 100
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed7_acc89_test43.pth',
#                                     'GCN2-32_best_train_1_seed3_acc80_test61.pth',
#                                     'GCN2-32_best_train_2_seed8_acc77_test63.pth',
#                                     'GCN2-32_best_train_3_seed8_acc66_test66.pth',
#                                     'GCN2-32_best_train_4_seed5_acc67_test65.pth',
#                                     'GCN2-32_best_train_5_seed4_acc80_test71.pth',
#                                     'GCN2-32_best_train_6_seed0_acc65_test78.pth',
#                                     'GCN2-32_best_train_7_seed7_acc68_test76.pth',
#                                     'GCN2-32_best_train_8_seed0_acc81_test75.pth']

#             elif args.split == 8:
#                 args.fake_num = 150
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed5_acc80_test54.pth',
#                                     'GCN2-32_best_train_1_seed3_acc70_test62.pth',
#                                     'GCN2-32_best_train_2_seed5_acc58_test68.pth',
#                                     'GCN2-32_best_train_3_seed8_acc71_test63.pth',
#                                     'GCN2-32_best_train_4_seed5_acc66_test65.pth',
#                                     'GCN2-32_best_train_5_seed5_acc62_test64.pth',
#                                     'GCN2-32_best_train_6_seed1_acc71_test72.pth']

#             elif args.split == 6:
#                 args.fake_num = 250
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed9_acc82_test62.pth',
#                                     'GCN2-32_best_train_1_seed9_acc71_test60.pth',
#                                     'GCN2-32_best_train_2_seed1_acc67_test62.pth',
#                                     'GCN2-32_best_train_3_seed0_acc72_test68.pth',
#                                     'GCN2-32_best_train_4_seed3_acc64_test72.pth']

#             elif args.split == 4:
#                 args.fake_num = 250
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed3_acc79_test50.pth',
#                                     'GCN2-32_best_train_1_seed3_acc71_test55.pth',
#                                     'GCN2-32_best_train_2_seed3_acc69_test67.pth']
                
#             elif args.split == 3:
#                 args.fake_num = 250
#                 if args.base_model == 'GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed7_acc82_test61.pth',
#                                     'GCN2-32_best_train_1_seed9_acc68_test65.pth']
#                 elif args.base_model == 'GIN':
#                     args.path_list = ['GIN2-32_best_train_0_seed7_acc82_test82.pth',
#                                     'GIN2-32_best_train_1_seed1_acc78_test76.pth']
#                 elif args.base_model == 'GAT':
#                     args.path_list = ['GAT2-32_best_train_0_seed6_acc78_test78.pth',
#                                     'GAT2-32_best_train_1_seed2_acc74_test73.pth']
            
#         elif args.dataset == 'REDDITBINARY':
#             args.epoch = 200
#             if args.split == 16:
#                 args.fake_num = 50
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed5_acc92_test88.pth',
#                                     'GCN2-32_best_train_1_seed3_acc88_test76.pth',
#                                     'GCN2-32_best_train_2_seed6_acc72_test89.pth',
#                                     'GCN2-32_best_train_3_seed8_acc88_test70.pth',
#                                     'GCN2-32_best_train_4_seed3_acc88_test84.pth',
#                                     'GCN2-32_best_train_5_seed2_acc80_test79.pth',
#                                     'GCN2-32_best_train_6_seed6_acc80_test92.pth',
#                                     'GCN2-32_best_train_7_seed5_acc76_test87.pth',
#                                     'GCN2-32_best_train_8_seed6_acc80_test88.pth',
#                                     'GCN2-32_best_train_9_seed1_acc84_test89.pth',
#                                     'GCN2-32_best_train_10_seed2_acc80_test92.pth',
#                                     'GCN2-32_best_train_11_seed1_acc84_test87.pth',
#                                     'GCN2-32_best_train_12_seed2_acc100_test81.pth',
#                                     'GCN2-32_best_train_13_seed2_acc92_test83.pth',
#                                     'GCN2-32_best_train_14_seed2_acc96_test88.pth']
#             elif args.split == 12:
#                 args.fake_num = 80
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed4_acc91_test91.pth',
#                                     'GCN2-32_best_train_1_seed9_acc76_test84.pth',
#                                     'GCN2-32_best_train_2_seed4_acc79_test81.pth',
#                                     'GCN2-32_best_train_3_seed8_acc85_test81.pth',
#                                     'GCN2-32_best_train_4_seed3_acc79_test85.pth',
#                                     'GCN2-32_best_train_5_seed1_acc76_test81.pth',
#                                     'GCN2-32_best_train_6_seed9_acc85_test90.pth',
#                                     'GCN2-32_best_train_7_seed9_acc88_test87.pth',
#                                     'GCN2-32_best_train_8_seed9_acc85_test91.pth',
#                                     'GCN2-32_best_train_9_seed4_acc97_test88.pth',
#                                     'GCN2-32_best_train_10_seed2_acc94_test86.pth']

#             elif args.split == 10:
#                 args.fake_num = 100
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed0_acc95_test84.pth',
#                                     'GCN2-32_best_train_1_seed3_acc73_test83.pth',
#                                     'GCN2-32_best_train_2_seed4_acc78_test82.pth',
#                                     'GCN2-32_best_train_3_seed4_acc87_test82.pth',
#                                     'GCN2-32_best_train_4_seed3_acc80_test83.pth',
#                                     'GCN2-32_best_train_5_seed5_acc85_test89.pth',
#                                     'GCN2-32_best_train_6_seed1_acc87_test91.pth',
#                                     'GCN2-32_best_train_7_seed1_acc87_test91.pth',
#                                     'GCN2-32_best_train_8_seed1_acc90_test89.pth']

#             elif args.split == 8:
#                 args.fake_num = 125
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed7_acc82_test88.pth',
#                                     'GCN2-32_best_train_1_seed3_acc70_test81.pth',
#                                     'GCN2-32_best_train_2_seed2_acc86_test40.pth',
#                                     'GCN2-32_best_train_3_seed4_acc66_test89.pth',
#                                     'GCN2-32_best_train_4_seed9_acc82_test90.pth',
#                                     'GCN2-32_best_train_5_seed4_acc90_test91.pth',
#                                     'GCN2-32_best_train_6_seed0_acc94_test87.pth']

#             elif args.split == 6:
#                 args.fake_num = 160
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed3_acc85_test85.pth',
#                                     'GCN2-32_best_train_1_seed5_acc79_test26.pth',
#                                     'GCN2-32_best_train_2_seed1_acc68_test79.pth',
#                                     'GCN2-32_best_train_3_seed3_acc80_test91.pth',
#                                     'GCN2-32_best_train_4_seed8_acc91_test88.pth']

#             elif args.split == 4:
#                 args.fake_num = 250
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed9_acc72_test32.pth',
#                                     'GCN2-32_best_train_1_seed1_acc69_test36.pth',
#                                     'GCN2-32_best_train_2_seed1_acc90_test91.pth']
                
#             elif args.split == 3:
#                 args.fake_num = 300
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed4_acc82_test49.pth',
#                                     'GCN2-32_best_train_1_seed6_acc69_test88.pth']
#                 elif args.base_model =='GIN':
#                     args.path_list = ['GIN2-32_best_train_0_seed9_acc73_test78.pth',
#                                     'GIN2-32_best_train_1_seed6_acc79_test91.pth']
#                 elif args.base_model =='GAT':
#                     args.path_list = ['GAT2-32_best_train_0_seed7_acc73_test76.pth',
#                                     'GAT2-32_best_train_1_seed2_acc77_test79.pth']
                    
#         elif args.dataset == 'PTC':
#             args.epoch = 100
#             if args.split == 3:
#                 args.fake_num = 50
#                 if args.base_model =='GCN':
#                     args.path_list = ['GCN2-32_best_train_0_seed2_acc78_test53.pth',
#                                     'GCN2-32_best_train_1_seed7_acc73_test54.pth']
#                 elif args.base_model =='GIN':
#                     args.path_list = ['GIN2-32_best_train_0_seed0_acc82_test75.pth',
#                                     'GIN2-32_best_train_1_seed3_acc65_test83.pth']
#                 elif args.base_model =='GAT':
#                     args.path_list = ['GAT2-32_best_train_0_seed8_acc73_test66.pth',
#                                     'GAT2-32_best_train_1_seed8_acc69_test75.pth']
                    