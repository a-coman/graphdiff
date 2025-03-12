# graphdiff
This repository provides tools for converting instances in USE (UML-based Specification Environment) [.SOIL] to graph representations and compute graph kernels for similarity analysis. It convert edge lists representation of graphs to adjacency matrices, create a graph representation from instances in the USE environment (.SOIL), and compute graph similarities using GraKeL.

## Background

**GraKel** uses to main structures to represent a graph: adjacency matrix and the list of edges. In our case for graphs that contain node labels, we are using an **adjacency matrix** and a **dictionary keyed by node indices** to their labels as its the easiest way to extract a graph representation from the USE instance.

**Example from** ***[1]***:

![alt text](https://ysig.github.io/GraKeL/0.1a8/_images/example_graph_labeled.svg)

```
adj = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
node_labels = {0: 'a', 1: 'b', 2: 'a'}
G = Graph(adj, node_labels=node_labels)
```

## Usage

#### ``EdgeToAdjacency.py`` 
Converts edge lists tuples to adjacency matrices ***[2]***.

#### ``SoilToGraph.py`` 
Converts instances in USE (UML-based Specification Environment) [.SOIL] to graph representations.

#### ``SoilToConnected.py`` 
Helper function to parse an instance in the .SOIL format with disconnected intances (graphs) and return separate files for each conneted instance (graph), i.e., g1.soil, g2.soil, g3.soil, ...

### Example
Given an instance as the following:

```
!new Person('person2')
!person2.age := 45
!person2.firstName := 'Raj'
!person2.lastName := 'Kumar'

!new Account('account2')
!account2.iban := 'IN0212450000001234567890'
!account2.balance := 1000000000

!insert (person2, account2) into Ownership
```
Based on RegEx, we can extract:
- The entityType i.e. Person, Account, etc. from the ``!new`` keyword [which will be used for **node labels**] 
- The entityName i.e. person2, account2, etc. from the ``!new`` keyword [which with the entityIndice (incremental) we can obtain the **edge tuple list** from the ``!insert`` keyword]

In the above example:
```
entiy_map -> {person2:['Person', 0], account2:['Account', 1]}
edges -> [(0, 1)]
adj_matrix -> [[0, 1], [1, 0]]
node_labels -> {0: 'Person', 1: 'Account'}
```

#### ``GraphDiff.py`` 
Parses a .SOIL file to String, gets its graph representation using SoilToGraph, and computes graph similarities using GraKeL.

## Installation
Python 2 is required for GrakeL library.

We recommend using [pyenv](https://github.com/pyenv/pyenv) to install Python 2.7.18 and create a virtual environment for GraphDiff.

```
pyenv install 2.7.18
pyenv virtualenv 2.7.18 graphdiff
pyenv activate graphdiff
pip install -r requirements.txt
```

## Attributions

***[1]*** **GraKeL Library**:
https://ysig.github.io/GraKeL/0.1a8/documentation/introduction.html
https://github.com/ysig/GraKeL

***[2]*** **Converting edge list to adjacency matrix**:
https://youcademy.org/conversion-between-edge-list-adjacency-matrix/#:~:text=To%20convert%20from%20an%20edge,the%20edge%2C%20if%20applicable