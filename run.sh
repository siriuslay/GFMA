# python pretrain.py --dataset 'MUTAG' --dim_feat 7
# python pretrain.py --dataset 'PTC' --dim_feat 19
# python pretrain.py --dataset 'REDDITBINARY' --dim_feat 1
# python pretrain.py --dataset 'NCI1' --dim_feat 37


# python pseudo_graphs_generation.py --dataset 'PTC' --dim_feat 19 --base_model 'GAT'
# python pseudo_graphs_generation.py --dataset 'PTC' --dim_feat 19 --base_model 'GIN'
# python pseudo_graphs_generation.py --dataset 'REDDITBINARY' --dim_feat 1 --base_model 'GCN'


python train_model_fusion.py --dataset 'MUTAG' --dim_feat 7
python train_model_fusion.py --dataset 'PTC' --dim_feat 19
python train_model_fusion.py --dataset 'REDDITBINARY' --dim_feat 1
python train_model_fusion.py --dataset 'NCI1' --dim_feat 37