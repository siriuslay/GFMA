
import os
import numpy as np
import torch, dgl, math
from dgl import backend as F

from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
from dgl import DGLGraph

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import StratifiedKFold


_url = 'https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip.'


class GINDataset(object):
    """Datasets for Graph Isomorphism Network (GIN)
    Adapted from https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip.

    The dataset contains the compact format of popular graph kernel datasets, which includes:
    MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K

    This datset class processes all data sets listed above. For more graph kernel datasets,
    see :class:`TUDataset`
    """

    def __init__(self, name, self_loop, degree_as_nlabel=False):
        """Initialize the dataset."""

        self.name = name  # MUTAG
        self.ds_name = 'nig'
        self.extract_dir = self._download()
        self.file = self._file_path()

        self.self_loop = self_loop

        self.graphs = []
        self.labels = []
        self.num_edges = []
        self.num_nodes = []

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False
        self.verbosity = False

        # calc all values
        self._load()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    def _download(self):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(
            download_dir, "{}.zip".format(self.ds_name))
        # TODO move to dgl host _get_dgl_url
        # download(_url, path=zip_file_path)
        # extract_dir = os.path.join(
        #     download_dir, "{}".format(self.ds_name))  # /dataset/nig/dataset/MUTAG/xxx.txt
        # extract_archive(zip_file_path, extract_dir)
        return download_dir  # /dataset

    def _file_path(self):
        return os.path.join(self.extract_dir, self.name, "{}.txt".format(self.name))  # /dataset/MUTAG/xxx.txt

    def _load(self):
        """ Loads input dataset from dataset/NAME/NAME.txt file

        """

        print('loading data...')
        with open(self.file, 'r') as f:
            # line_1 == N, total number of graphs
            self.N = int(f.readline().strip())

            for i in range(self.N):
                if (i + 1) % 10 == 0 and self.verbosity is True:
                    print('processing graph {}...'.format(i + 1))

                grow = f.readline().strip().split()
                # line_2 == [n_nodes, l] is equal to
                # [node number of a graph, class label of a graph]
                n_nodes, glabel = [int(w) for w in grow]

                # relabel graphs
                if glabel not in self.glabel_dict:
                    mapped = len(self.glabel_dict)
                    self.glabel_dict[glabel] = mapped

                self.labels.append(self.glabel_dict[glabel])

                g = DGLGraph()
                g.add_nodes(n_nodes)

                nlabels = []  # node labels
                nattrs = []  # node attributes if it has
                m_edges = 0

                for j in range(n_nodes):
                    nrow = f.readline().strip().split()

                    # handle edges and attributes(if has)
                    tmp = int(nrow[1]) + 2  # tmp == 2 + #edges
                    if tmp == len(nrow):
                        # no node attributes
                        nrow = [int(w) for w in nrow]
                        nattr = None
                    elif tmp > len(nrow):
                        nrow = [int(w) for w in nrow[:tmp]]
                        nattr = [float(w) for w in nrow[tmp:]]
                        nattrs.append(nattr)
                    else:
                        raise Exception('edge number is incorrect!')

                    # relabel nodes if it has labels
                    # if it doesn't have node labels, then every nrow[0]==0
                    if not nrow[0] in self.nlabel_dict:
                        mapped = len(self.nlabel_dict)
                        self.nlabel_dict[nrow[0]] = mapped

                    # nlabels.append(self.nlabel_dict[nrow[0]])
                    nlabels.append(nrow[0])

                    m_edges += nrow[1]
                    g.add_edges(j, nrow[2:])

                    # add self loop
                    if self.self_loop:
                        m_edges += 1
                        g.add_edges(j, j)

                    if (j + 1) % 10 == 0 and self.verbosity is True:
                        print(
                            'processing node {} of graph {}...'.format(
                                j + 1, i + 1))
                        print('this node has {} edgs.'.format(
                            nrow[1]))

                self.num_edges.append(m_edges)
                self.num_nodes.append(n_nodes)
                if nattrs != []:
                    nattrs = np.stack(nattrs)
                    g.ndata['attr'] = nattrs
                    self.nattrs_flag = True
                else:
                    nattrs = None

                g.ndata['label'] = torch.tensor(nlabels, device='cpu')

                if len(self.nlabel_dict) > 1:
                    self.nlabels_flag = True

                assert g.number_of_nodes() == n_nodes

                # update statistics of graphs
                self.n += n_nodes
                self.m += m_edges

                self.graphs.append(g)

        # if no attr
        if not self.nattrs_flag:
            print('there are no node features in this dataset!')
            label2idx = {}
            # generate node attr by node degree
            if self.degree_as_nlabel:
                print('generate node features by node degree...')
                nlabel_set = set([])
                for g in self.graphs:
                    g.ndata['label'] = g.in_degrees()
                    # extracting unique node labels
                    nlabel_set = nlabel_set.union(set(g.in_degrees().numpy().tolist()))

                nlabel_set = list(nlabel_set)
                # in case the labels/degrees are not continuous number
                self.ndegree_dict = {
                    nlabel_set[i]: i
                    for i in range(len(nlabel_set))
                }
                label2idx = self.ndegree_dict
            # generate node attr by node label
            else:
                print('generate node features by node label...')

                label2idx = self.nlabel_dict

            for g in self.graphs:
                device = g.device  # 获取图数据的设备
                # g.ndata['attr'] = np.zeros((g.number_of_nodes(), len(label2idx)))
                g.ndata['attr'] = torch.zeros((g.number_of_nodes(), len(label2idx)), device=device)

                # g.ndata['attr'] = np.zeros((
                #     g.number_of_nodes(), len(label2idx)))
                g.ndata['attr'][
                    range(len(g.ndata['label'])), [label2idx[F.as_scalar(nl)] for nl in g.ndata['label']]] = 1

        # after load, get the #classes and #dim
        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata['attr'][0])

        print('Done.')
        print(
            """
            -------- Data Statistics --------'
            #Graphs: %d
            #Graph Classes: %d
            #Nodes: %d
            #Node Classes: %d
            #Node Features Dim: %d
            #Edges: %d
            #Edge Classes: %d
            Avg. of #Nodes: %.2f
            Avg. of #Edges: %.2f
            Graph Relabeled: %s
            Node Relabeled: %s
            Degree Relabeled(If degree_as_nlabel=True): %s \n """ % (
                self.N, self.gclasses, self.n, self.nclasses,
                self.dim_nfeats, self.m, self.eclasses,
                self.n / self.N, self.m / self.N, self.glabel_dict,
                self.nlabel_dict, self.ndegree_dict))


def collate(samples):
    # 'samples (graph, label)'
    graphs, labels = map(list, zip(*samples))
    # batch_features_per_graph = []
    for g in graphs:
        # batch_features_per_graph.append(g.ndata['attr'])
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels


class GraphDataLoader():
    def __init__(self, dataset, batch_size, device,
                 collate_fn=collate, seed=0, shuffle=True,
                 split_name='rand', t_v_split_ratio=1, split=3, train_index=0, test_index=3):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if device >= 0 else {}

        labels = [l for _, l in dataset]

        if split_name == 'rand':
            train_idx, valid_idx, test_idx = self._split_rand(
                labels, t_v_split_ratio, seed, shuffle
            )
        elif split_name == 'mean_degree_sort':
            # 按图的节点平均度进行排序
            node_density = []
            for i in range(len(dataset)):
                # node_density.append(dataset.num_edges[i] / (dataset.num_nodes[i]))
                node_density.append(dataset.num_edges[i] / ((dataset.num_nodes[i]) * (dataset.num_nodes[i] - 1)))
            node_density = torch.tensor(node_density)
            node_density, data_idx = torch.sort(node_density, descending=False)
            splits = len(dataset) // split
            bias = 0  # splits // 2
            # 按度平均值划分 source & target
            index = []
            for idx in range(split - 1):
                index.append([idx * splits, (idx + 1) * splits])
            index.append([(split - 1) * splits, len(dataset)])
            # 分出train set
            train_source_index = data_idx[index[train_index][0] : index[train_index][1]]
            # 分出test set
            test_source_index = data_idx[index[test_index][0] : index[test_index][1]]

            if train_index == -1:
                train_source_index = data_idx[torch.randperm(data_idx.size(0))]
            if test_index == -1:
                test_source_index = data_idx[torch.randperm(data_idx.size(0))]
                
            train_source_index = train_source_index[torch.randperm(train_source_index.size(0))]
            test_source_index = test_source_index[torch.randperm(test_source_index.size(0))]

            tvt_split = int(math.floor(t_v_split_ratio * len(train_source_index)))
            t_v_split = int(math.floor((t_v_split_ratio - 0.2) * len(train_source_index)))

            # 划分train / valid / test : 5/2/3
            train_idx, valid_idx = train_source_index[:t_v_split], train_source_index[t_v_split:tvt_split]
            if train_index == test_index:
                if t_v_split_ratio == 1:
                    tvt_split = int(math.floor((t_v_split_ratio - 0.5) * len(train_source_index)))
                test_idx = train_source_index[tvt_split:]

            else:
                test_idx = test_source_index

            # if test_index == -1:
            #     test_idx = data_idx[torch.randperm(data_idx.size(0))]

            print(
                'train_set: valid_set: test_set = %d : %d : %d' % (len(train_idx), len(valid_idx), len(test_idx))
            )
            
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        self.train_loader = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

        self.valid_loader = DataLoader(
            dataset, sampler=valid_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

        self.test_loader = DataLoader(
            dataset, sampler=test_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

    def train_valid_test_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def _split_rand(self, labels, t_v_split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))

        np.random.seed(seed)
        np.random.shuffle(indices)

        split = int(math.floor(t_v_split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            'train_set: test_set = %d : %d' % (len(train_idx), len(valid_idx))
        )

        return train_idx, valid_idx, valid_idx
    



class Split_GraphDataLoader():
    def __init__(self, dataset, batch_size, device,
                 collate_fn=collate, seed=0, shuffle=True,
                 split_name='rand', t_v_split_ratio=0.7, split=3, train_index=0, test_index=-1):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if device >= 0 else {}

        labels = [l for _, l in dataset]

        self.train_loader = []

        self.valid_loader = []

        if split_name == 'rand':
            train_idx, valid_idx, test_idx = self._split_rand(
                labels, t_v_split_ratio, seed, shuffle
            )
        elif split_name == 'mean_degree_sort':
            # 按图的节点平均度进行排序
            node_density = []
            for i in range(len(dataset)):
                # node_density.append(dataset.num_edges[i] / (dataset.num_nodes[i]))
                node_density.append(dataset.num_edges[i] / ((dataset.num_nodes[i]) * (dataset.num_nodes[i] - 1)))
            node_density = torch.tensor(node_density)
            node_density, data_idx = torch.sort(node_density, descending=False)
            splits = len(dataset) // split
            
        else:
            raise NotImplementedError()
        
        test_idx = data_idx[torch.randperm(data_idx.size(0))]
        test_sampler = SubsetRandomSampler(test_idx)
        self.test_loader = DataLoader(
            dataset, sampler=test_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )
        print(
                'test_set = %d' % (len(test_idx))
            )
        
        for train_index in range(split):
            source_index = data_idx[splits * train_index: splits * (train_index + 1)]  # 取第[train_index]份
            source_index = source_index[torch.randperm(source_index.size(0))]
            t_v_split = int(math.floor(t_v_split_ratio * len(source_index)))
            train_idx, valid_idx = source_index[:t_v_split], source_index[t_v_split:]

            print(
                'train_set: valid_set = %d : %d' % (len(train_idx), len(valid_idx))
            )

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)


            self.train_loader.append(DataLoader(
                dataset, sampler=train_sampler,
                batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
            ))

            self.valid_loader.append(DataLoader(
                dataset, sampler=valid_sampler,
                batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
            ))


    def train_valid_loader_list(self):
        return self.train_loader, self.valid_loader, self.test_loader
    
    def train_valid_test_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def _split_rand(self, labels, t_v_split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))

        np.random.seed(seed)
        np.random.shuffle(indices)

        split = int(math.floor(t_v_split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            'train_set: test_set = %d : %d' % (len(train_idx), len(valid_idx))
        )

        return train_idx, valid_idx, valid_idx
    

class Continual_GraphDataLoader():
    def __init__(self, dataset, batch_size, device,
                 collate_fn=collate, seed=0, shuffle=True,
                 split_name='rand', t_v_split_ratio=0.7, split=3, train_index=0, test_index=-1):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if device >= 0 else {}

        labels = [l for _, l in dataset]

        self.train_loader = []

        self.valid_loader = []

        if split_name == 'rand':
            train_idx, valid_idx, test_idx = self._split_rand(
                labels, t_v_split_ratio, seed, shuffle
            )
        elif split_name == 'mean_degree_sort':
            # 按图的节点平均度进行排序
            node_density = []
            for i in range(len(dataset)):
                # node_density.append(dataset.num_edges[i] / (dataset.num_nodes[i]))
                node_density.append(dataset.num_edges[i] / ((dataset.num_nodes[i]) * (dataset.num_nodes[i] - 1)))
            node_density = torch.tensor(node_density)
            node_density, data_idx = torch.sort(node_density, descending=False)
            splits = len(dataset) // split
            
        else:
            raise NotImplementedError()
        
        test_idx = data_idx[torch.randperm(data_idx.size(0))]
        test_sampler = SubsetRandomSampler(test_idx)
        self.test_loader = DataLoader(
            dataset, sampler=test_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )
        print(
                'test_set = %d' % (len(test_idx))
            )
        
        for train_index in range(split):
            source_index = data_idx[: splits * (train_index + 1)]  # 取第[train_index]份
            source_index = source_index[torch.randperm(source_index.size(0))]
            t_v_split = int(math.floor(t_v_split_ratio * len(source_index)))
            train_idx, valid_idx = source_index[:t_v_split], source_index[t_v_split:]

            print(
                'train_set: valid_set = %d : %d' % (len(train_idx), len(valid_idx))
            )

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)


            self.train_loader.append(DataLoader(
                dataset, sampler=train_sampler,
                batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
            ))

            self.valid_loader.append(DataLoader(
                dataset, sampler=valid_sampler,
                batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
            ))


    def train_valid_loader_list(self):
        return self.train_loader, self.valid_loader, self.test_loader
    
    def train_valid_test_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def _split_rand(self, labels, t_v_split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))

        np.random.seed(seed)
        np.random.shuffle(indices)

        split = int(math.floor(t_v_split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            'train_set: test_set = %d : %d' % (len(train_idx), len(valid_idx))
        )

        return train_idx, valid_idx, valid_idx