# Adj, edge, label 
```
Adj1-InstructionLearning.soil: [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

Labels1-InstructionLearning.soil: {0: 'Bank', 1: 'Bank', 2: 'Account', 3: 'Account', 4: 'Account', 5: 'Account', 6: 'Person', 7: 'Person', 8: 'Person', 9: 'Person', 10: 'Account', 11: 'Person'}

Edges1-InstructionLearning.soil: [(0, 2), (0, 3), (1, 4), (1, 5), (6, 2), (7, 2), (8, 3), (9, 4), (8, 4), (8, 5), (6, 3), (7, 3), (8, 2), (9, 5), (1, 10), (11, 10)]

Adj2-ChainOfThought.soil: [[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0], [1, 0, 1, 0, 0]]

Labels2-ChainOfThought.soil: {0: 'Bank', 1: 'Person', 2: 'Person', 3: 'Account', 4: 'Account'}

Edges2-ChainOfThought.soil: [(1, 3), (2, 3), (2, 4), (0, 3), (0, 4)]
```
# Kernel: 
```
[[1.        0.8203549]
 [0.8203549 1.       ]]
```
# Kernel 2D table: 
|       | InstructionLearning | ChainOfThought |
|-------|---|---|
| **InstructionLearning** | 1.000000 | 0.820355 |
| **ChainOfThought** |  | 1.000000 |