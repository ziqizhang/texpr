import networkx as nx


def add_graph_nodes(sequence, graph):
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
    return graph


def remove_unreachable_nodes(graph):
    remove=[]
    for node in graph.nodes():
        if graph.degree(node)==0:
            remove.append(node)

    for node in remove:
        graph.remove_node(node)
