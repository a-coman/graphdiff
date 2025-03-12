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

def saveFile(content, filename):
    with open(filename, "w") as file:
        file.write(content)


def array_to_markdown_table(matrix):
    # Create the header row
    header = "|       | " + " | ".join(["gen{}".format(i+1) for i in range(len(matrix[0]))]) + " |"
    
    # Create the separator row
    separator = "|-------|" + "|".join(["---"] * len(matrix[0])) + "|"
    
    # Create the data rows
    data_rows = []
    for i, row in enumerate(matrix):
        data_row = ["**gen{}**".format(i+1)]  # Add row name
        for j, cell in enumerate(row):
            if j >= i:  # Include diagonal and upper triangle
                data_row.append("{:.6f}".format(cell))
            else:  # Leave lower triangle empty
                data_row.append("")
        data_row = "| " + " | ".join(data_row) + " |"
        data_rows.append(data_row)
    
    # Combine all rows into the final Markdown table
    markdown_table = "\n".join([header, separator] + data_rows)
    
    return markdown_table 


if __name__ == "__main__":
    # Initialize Graph Kernel
    sp_kernel = ShortestPath(normalize=True)
    
    # Initialize Weisfeiler-Lehman framework
    wl_kernel = WeisfeilerLehman(base_graph_kernel=VertexHistogram)

    numberGraphs = 2
    graphs = []
    #filepath = "/mnt/c/Users/Andrei/Desktop/Experiments-Bank-Bikes-01/Bikes/Simple-GPT4o--27-02-2025--10-09-19/"
    filepath = "./instances/"

    result = []
    result.append("# Adj, edge, label \n```\n")

    for i in range(1, numberGraphs + 1):
        file = readFile(filepath + "gen" + str(i) + ".soil")
        adj, labels, edges = SoilToGraph.soilToGraph(file)
        result.append("Adj" + str(i) + ": ")
        result.append(str(adj))
        result.append("\n\n")
        result.append("Labels" + str(i) + ": ")
        result.append(str(labels))
        result.append("\n\n")
        result.append("Edges" + str(i) + ": ")
        result.append(str(edges))
        graph = getGraph(adj, labels)
        graphs.append(graph)

    result.append("\n```\n")
    kernel = sp_kernel.fit_transform(graphs)
    result.append("# Kernel: \n```\n")
    result.append(str(kernel))
    result.append("\n```\n")
    result.append("# Kernel 2D table: \n")
    markdown = array_to_markdown_table(kernel)
    result.append(markdown)

    result = "".join(result)
    saveFile(result, "./Grakel.md")
    print(result)
