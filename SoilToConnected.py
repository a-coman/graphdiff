# -*- coding: utf-8 -*-
"""
Helper function to parse an instance in the .SOIL format with disconnected intances (graphs)
and return separate files for each conneted instances (graphs), i.e., g1.soil, g2.soil, g3.soil, ...
"""

import re
from collections import defaultdict

# Input text
text = """
-- Bank 3 example
!new Bank('bank3')
!bank3.country := 'France'
!bank3.name := 'BNP Paribas'
!bank3.bic := 'BNPAFRPPXXX'

!new Account('account3')
!account3.iban := 'FR7630006000011234567890189'
!account3.balance := 1500

!new Person('person4')
!person4.age := 22
!person4.firstName := 'Sophie'
!person4.lastName := 'Dubois'

!new Person('person5')
!person5.age := 19
!person5.firstName := 'Lucas'
!person5.lastName := 'Moreau'

!insert (person4, account3) into Ownership
!insert (person4, account3) into Use
!insert (person5, account3) into Use
!insert (bank3, account3) into AccountOfBanks

-- Bank 4 example

!new Bank('bank4')
!bank4.country := 'Italy'
!bank4.name := 'UniCredit'
!bank4.bic := 'UNCRITMMXXX'

!new Account('account4')
!account4.iban := 'IT60X0542811101000000123456'
!account4.balance := 500

!new Person('person6')
!person6.age := 35
!person6.firstName := 'Giovanni'
!person6.lastName := 'Rossi'

!new Person('person7')
!person7.age := 28
!person7.firstName := 'Maria'
!person7.lastName := 'Bianchi'

!insert (person6, account4) into Ownership
!insert (person6, account4) into Use
!insert (person7, account4) into Use
!insert (bank4, account4) into AccountOfBanks
"""

# Parse the text into entities and relationships
entities = {}
relationships = []

# Regular expressions to match entities and relationships
entity_pattern = re.compile(r"!new (\w+)\(['\"](.*?)['\"]\)")
attribute_pattern = re.compile(r"!(\w+)\.(\w+) := (.*)")
relationship_pattern = re.compile(r"!insert\s*\((\w+),\s*(\w+)\)\s*into\s*(\w+)")

# Parse entities and attributes
for line in text.strip().split("\n"):
    entity_match = entity_pattern.match(line)
    if entity_match:
        entity_type, entity_id = entity_match.groups()
        entities[entity_id] = {"type": entity_type, "attributes": {}}
        continue

    attribute_match = attribute_pattern.match(line)
    if attribute_match:
        entity_id, attribute, value = attribute_match.groups()
        entities[entity_id]["attributes"][attribute] = value
        continue

    relationship_match = relationship_pattern.match(line)
    if relationship_match:
        source, target, relationship_type = relationship_match.groups()
        relationships.append((source, target, relationship_type))

# Build a graph using adjacency list
graph = defaultdict(list)
for source, target, _ in relationships:
    graph[source].append(target)
    graph[target].append(source)

# Find connected components (graphs)
def dfs(node, visited, component):
    if node not in visited:
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, visited, component)

visited = set()
graphs = []
for node in graph:
    if node not in visited:
        component = []
        dfs(node, visited, component)
        graphs.append(component)

# Save each graph to a separate file
for i, graph_nodes in enumerate(graphs, start=1):
    filename = "g{0}.soil".format(i)
    with open("graphs/" + filename, "w") as file:
        # Write entities
        for node in graph_nodes:
            entity = entities[node]
            file.write("!new {0}('{1}')\n".format(entity["type"], node))
            for attr, value in entity["attributes"].items():
                file.write("!{0}.{1} := {2}\n".format(node, attr, value))
        # Write relationships
        for source, target, rel_type in relationships:
            if source in graph_nodes and target in graph_nodes:
                file.write("!insert ({0}, {1}) into {2}\n".format(source, target, rel_type))
    print("Saved {0}".format(filename))

print("Graphs have been separated and saved to individual files.")