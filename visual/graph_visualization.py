import os.path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import dgl


def draw_graph_list(graph_list, label_list, row, col, f_path, iterations=100, layout='spring', is_single=False, k=1,
                    node_size=55, alpha=1, width=1.3, remove=False):

    G_list = [nx.to_networkx_graph(graph_list[i]) for i in range(len(graph_list))]

    # remove isolate nodes in graphs
    if remove:
        for gg in G_list:
            gg.remove_nodes_from(list(nx.isolates(gg)))

    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.axis("off")

        # turn off axis label
        plt.xticks([])
        plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=iterations)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(f'{layout} not recognized.')

        if is_single:
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0,
                                   )
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2)
            # nx.draw_networkx_nodes(G, pos, node_size=2.0, node_color='#336699', alpha=1, linewidths=1.0)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)
            # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0.2)
        nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)

        # Add graph-level information (e.g., number of nodes, number of edges, and graph label)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        graph_label = label_list[i].item() if isinstance(label_list[i], np.ndarray) else label_list[i]

        # Use plt.text to add this information on the graph plot
        plt.text(0.5, -0.1, f'Nodes: {num_nodes}, Edges: {num_edges}, Label: {graph_label}', horizontalalignment='center', verticalalignment='top',
                 transform=plt.gca().transAxes, fontsize=5, color='black')
        # plt.text(0.05, 0.90, f'Nodes: {num_nodes}', horizontalalignment='left', verticalalignment='center',
        #          transform=plt.gca().transAxes, fontsize=5, color='black')
        # plt.text(0.05, 0.85, f'Edges: {num_edges}', horizontalalignment='left', verticalalignment='center',
        #          transform=plt.gca().transAxes, fontsize=5, color='black')
        # plt.text(0.05, 0.80, f'Label: {graph_label}', horizontalalignment='left', verticalalignment='center',
        #          transform=plt.gca().transAxes, fontsize=5, color='black')

    plt.tight_layout()
    plt.savefig(f_path, dpi=1600)
    plt.close()


def visualize_graphs(graph_list, label_list, dir_path, vis_row, vis_col, remove=True):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    row = vis_row
    col = vis_col
    n_graph = row * col

    n_fig = int(np.ceil(len(graph_list) / n_graph))
    for i in range(n_fig):
        draw_graph_list(graph_list[i*n_graph:(i+1)*n_graph], label_list[i * n_graph:(i + 1) * n_graph], row, col,
                        f_path=os.path.join(dir_path, "sample"+str(i)+".png"), remove=remove)


def visualize(data_loader, data_name, vis_row, vis_col):
    graph_list = []
    label_list = []
    # 从 data_loader 中获取批次
    for batch, labels in data_loader:
        # 将 DGLGraph 转换为 NetworkX 图
        for dgl_graph in dgl.unbatch(batch):  # dgl.unbatch 将批次分解为单个图
            dgl_graph = dgl_graph.cpu()
            nx_graph = dgl.to_networkx(dgl_graph)
            graph_list.append(nx_graph)
            # label_list.append(labels)  # 添加图的标签

    # 现在你有了一个 graph_list，可以调用绘图函数
    visualize_graphs(graph_list, labels, dir_path=f'visual/fake_data/{data_name}', vis_row=vis_row, vis_col=vis_col)
