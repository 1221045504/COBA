import numpy as np
def load_data(train_filename):
    nodes = set()
    nodes_s = set()
    egs = []
    graph = [{}, {}]

    with open(train_filename) as infile:
        for line in infile.readlines():
            source_node, target_node = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)

            nodes.add(source_node)
            nodes.add(target_node)
            nodes_s.add(source_node)
            egs.append([source_node, target_node])

            if source_node not in graph[0]:
                graph[0][source_node] = []
            if target_node not in graph[1]:
                graph[1][target_node] = []

            graph[0][source_node].append(target_node)
            graph[1][target_node].append(source_node)

    n_node = len(nodes)

    return graph, list(nodes), egs