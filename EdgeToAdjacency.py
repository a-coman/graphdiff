# Credits to: 
# https://youcademy.org/conversion-between-edge-list-adjacency-matrix/#:~:text=To%20convert%20from%20an%20edge,the%20edge%2C%20if%20applicable).

# Function to convert edge list to adjacency matrix
def convert_to_adjacency_matrix(edge_list, num_vertices):
  # Initialize the adjacency matrix with all zeros
  adj_mat = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
  
  # Iterate through all the edges
  for edge in edge_list:
    # Get the source and destination nodes
    source = edge[0]
    destination = edge[1]
    
    # Mark the edge between source and destination in adjacency matrix
    adj_mat[source][destination] = 1
    
    #  Mark the edge between destination and source in adjacency matrix
    # (since the graph is bidirectional)
    adj_mat[destination][source] = 1
  
  return adj_mat

if __name__ == "__main__":
    edge_list = [(0, 2), (1, 0), (2, 0)]
    num_vertices = 3
    adj_mat = convert_to_adjacency_matrix(edge_list, num_vertices)
    print("For the following Edge list")
    print(edge_list)
    print("Below is the Adjacency Matrix")
    for row in adj_mat:
        for edge_status in row:
            print(str(edge_status) + " "),
        print("")
