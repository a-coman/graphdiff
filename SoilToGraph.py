"""
Parse an instance in the .SOIL format and return its graph representation, i.e., adgency_matrix, node_labels, and edges.
adgency_matrix, node_labels, edges = SoilToGraph.soilToGraph(text)
"""

import re
import EdgeToAdjacency

class Pair(object):
    """Entity pair with incremental ID and type, e.g. 0 - Bank"""
    def __init__(self, id, type):
        self.id = id
        self.type = type

def soilToGraph(text):
    """Parse an instance in the .SOIL format and return its graph representation, i.e., adgency_matrix, node_labels, and edges."""

    entity_map = {}     # Maps entity names to their IDs and types
    node_labels = {}    # Stores node labels as {id: entity_type}
    edges = []          # Stores edges as [(edgei1, edgej1), (edgei2, edgej2), ...]
    entity_indice = 0   # Incremental id for entities

    regex_labels = r"!\s*new\s+(\w+)\s*\(\s*'([^']+)'\s*\)"
    pattern_labels = re.compile(regex_labels)

    regex_edges = r"!\s*insert\s*\(([^)]+)\)\s+into\s+\w+"
    pattern_edges = re.compile(regex_edges)

    for line in text.split("\n"):
        matcher_labels = pattern_labels.search(line)
        matcher_edges = pattern_edges.search(line)

        if matcher_labels:
            entity_type = matcher_labels.group(1)
            entity_name = matcher_labels.group(2)
            entity_map[entity_name] = Pair(entity_indice, entity_type)

            # Add to node_labels dictionary
            node_labels[entity_indice] = entity_type

            entity_indice += 1
            continue

        if matcher_edges:
            edges_str = matcher_edges.group(1)
            split_edges = edges_str.split(",")
            source = split_edges[0].strip()
            target = split_edges[1].strip()

            # Get IDs of source and target entities
            edgei = entity_map[source].id
            edgej = entity_map[target].id

            # Add to edges list
            edges.append((edgei, edgej))

    # Print
    for entity_name, pair in entity_map.items():
        print("Entity Name: {}, ID: {}, Type: {}".format(entity_name, pair.id, pair.type))
    
    # Results
    adgency_matrix = EdgeToAdjacency.convert_to_adjacency_matrix(edges, len(node_labels))
    return adgency_matrix, node_labels, edges


# Main for testing purposes
if __name__ == "__main__":
    text = """
    !new Bank('bank2')
    !bank2.country := 'India'
    !bank2.name := 'Bharat Bank'
    !bank2.bic := 'INBBINBB'

    !new Person('person2')
    !person2.age := 45
    !person2.firstName := 'Raj'
    !person2.lastName := 'Kumar'

    !new Account('account2')
    !account2.iban := 'IN0212450000001234567890'
    !account2.balance := 1000000000

    !new Account('account3')
    !account3.iban := 'IN0212450000009876543210'
    !account3.balance := 500000000

    !new Person('person3')
    !person3.age := 32
    !person3.firstName := 'Priya'
    !person3.lastName := 'Kapoor'

    !insert (person2, account2) into Ownership
    !insert (person2, account3) into Ownership
    !insert (person2, account2) into Use
    !insert (person3, account2) into Use
    !insert (person3, account3) into Use
    !insert (bank2, account2) into AccountOfBanks
    !insert (bank2, account3) into AccountOfBanks

    !new Bank('bank2')
    !bank2.country := 'India'
    !bank2.name := 'Bharat Bank'
    !bank2.bic := 'INBBINBB'
    """

    adgency_matrix, node_labels, edges = soilToGraph(text)
    
    print("Edges: "), 
    print(edges)
    print("Adjacency matrix: "), 
    print(adgency_matrix)
    print("Node labels: "), 
    print(node_labels)