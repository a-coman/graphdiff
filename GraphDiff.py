"""
Get graph from adjacency matrix and node labels
    Graphs are represented as adjacency matrices with labels to check
    both structural difference and semantic information from labels

Documentation:
    https://github.com/ysig/GraKeL
    https://ysig.github.io/GraKeL/0.1a8/documentation/introduction.html
"""

import SoilToGraph
from grakel.kernels import ShortestPath
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph
import os

 
def getGraph(adj, labels):
    return Graph(initialization_object=adj, node_labels=labels)

def readFile(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def saveFile(content, filename):
    with open(filename, "w") as file:
        file.write(content)


def array_to_markdown_table(matrix, items):
    # Create the header row
    header = "|       | " + " | ".join([items[i].replace(".soil", "") + str((i+1)) for i in range(len(matrix[0]))]) + " |"
    
    # Create the separator row
    separator = "|-------|" + "|".join(["---"] * len(matrix[0])) + "|"
    
    # Create the data rows
    data_rows = []
    for i, row in enumerate(matrix):
        data_row = ["**" + items[i].replace(".soil", "") + str((i+1)) + "**"]  # Add row name
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

    graphs = []
    llm = "GPT_4O"
    #llm = "GEMINI_2_FLASH_LITE"
    type = "CoT"
    system = "addressbook"
    time = "25-03-2025--10-38-22"
    filepath = "/home/andrei/Repos/llm-instancer/src/main/resources/instances/" + type + "/" + system + "/" + llm + "/" + time + "/"
    #filepath = "./instances/"
    numberOfGen = len([item for item in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, item)) and item.startswith("gen")])

    result = []
    result.append("# Adj, edge, label \n```\n")
    processedItems = []

    for i in range(1, numberOfGen + 1):
        items = os.listdir(filepath + "gen" + str(i))
        for item in items:
            if not item.endswith(".soil"):
                continue # Skip non-SOIL files
            if type == "CoT" and (item.startswith("output") or item.startswith("temp")):
                continue # Skip output* and temp files for CoT
            
            processedItems.append(item)
            print("Processing file " + item + " of gen" + str(i))
            file = readFile(filepath + "gen" + str(i) + "/" + item)
            adj, labels, edges = SoilToGraph.soilToGraph(file)
            result.append("Adj" + str(i) + "-" + item + ": ")
            result.append(str(adj))
            result.append("\n\n")
            result.append("Labels" + str(i) + "-" + item + ": ")
            result.append(str(labels))
            result.append("\n\n")
            result.append("Edges" + str(i) + "-" + item + ": ")
            result.append(str(edges))
            result.append("\n\n")
            graph = getGraph(adj, labels)
            graphs.append(graph)

    result.append("```\n")
    kernel = sp_kernel.fit_transform(graphs)
    result.append("# Kernel: \n```\n")
    result.append(str(kernel))
    result.append("\n```\n")
    result.append("# Kernel 2D table: \n")
    markdown = array_to_markdown_table(kernel, processedItems)
    result.append(markdown)

    result = "".join(result)
    saveFile(result, filepath + "grakel.md")
    print(result)
    print("\n")
    print(filepath)
