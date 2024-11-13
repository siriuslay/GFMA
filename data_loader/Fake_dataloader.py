
import os
import numpy as np
import torch

from dgl import backend as F

from dgl import DGLGraph

_url = 'https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip'


class FAKEGINDataset(object):

    def __init__(self, name, fake_path, self_loop, fea_dim=7, g_cls=2, degree_as_nlabel=False):
        """Initialize the dataset."""

        self.name = name  # MUTAG
        # self.fake_path = fake_path
        # self.file = 'fake_'+name
        self.file = fake_path
        self.fea_dim = fea_dim
        self.g_cls = g_cls
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

                self.labels.append(glabel)

                g = DGLGraph()
                g.add_nodes(n_nodes)

                nlabels = []  # node labels
                nattrs = []  # node attributes if it has
                m_edges = 0

                for j in range(n_nodes):
                    nrow = f.readline().strip().split()

                    # handle edges and attributes(if has)
                    if int(nrow[1]) == 0:
                        nlabels.append(int(nrow[0]))
                        continue

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

                        print('current tmp and len(nrow):')
                        print(nrow)
                        raise Exception('edge number is incorrect!')

                    # relabel nodes if it has labels
                    # if it doesn't have node labels, then every nrow[0]==0

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
                g.ndata['label'] = torch.tensor(nlabels, device='cpu')

                assert g.number_of_nodes() == n_nodes

                # update statistics of graphs
                self.n += n_nodes
                self.m += m_edges

                self.graphs.append(g)

            for g in self.graphs:
                # print('g.number_of_nodes(): %d' %g.number_of_nodes())
                device = g.device  # 获取图数据的设备
                # g.ndata['attr'] = np.zeros((g.number_of_nodes(), len(label2idx)))
                # print(self.nlabel_dict, self.dim_nfeats)
                g.ndata['attr'] = torch.zeros((g.number_of_nodes(), self.fea_dim), device=device)
                # g.ndata['attr'] = np.zeros((g.number_of_nodes(), 7))
                g.ndata['attr'][range(len(g.ndata['label'])), [F.as_scalar(nl) for nl in g.ndata['label']]] = 1

        # after load, get the #classes and #dim
        self.gclasses = self.g_cls
        self.nclasses = self.fea_dim
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
