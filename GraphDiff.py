# Documentation:
# https://github.com/ysig/GraKeL
# https://ysig.github.io/GraKeL/0.1a8/documentation/introduction.html

import SoilToGraph
from grakel.kernels import ShortestPath
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph

# Get graph from adjacency matrix and node labels
    # Graphs represented as adjacency matrices, with labels to check
    # both structural difference and semantic information encoded in the labels  
def getGraph(adj, labels):
    return Graph(initialization_object=adj, node_labels=labels)

def readFile(file_path):
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    # Initialize Graph Kernel
    sp_kernel = ShortestPath(normalize=True)
    
    # Initialize Weisfeiler-Lehman framework
    wl_kernel = WeisfeilerLehman(base_graph_kernel=VertexHistogram)

    example1 = readFile("example1.soil")
    example2 = readFile("example2.soil")

    adj1, labels1, edges1 = SoilToGraph.soilToGraph(example1)
    adj2, labels2, edges2 = SoilToGraph.soilToGraph(example2)

    print("Adj1: ")
    print(adj1)
    print("Labels1: ")
    print(labels1)
    print("Edges1: ")
    print(edges1)

    print("Adj2: ")
    print(adj2)
    print("Labels2: ")
    print(labels2)
    print("Edges2: ")
    print(edges2)

    graph1 = getGraph(adj1, labels1)
    graph2 = getGraph(adj2, labels2)

    # Compute the kernel (similarity) between the two graphs
    kernel = sp_kernel.fit_transform([graph1, graph2])
    print("Kernel: ")
    print(kernel)
