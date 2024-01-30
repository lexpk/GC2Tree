This repository contains the code for my Machine Learning Algorithms and Applications Project.

## Installation

The simplest way to install the required packages is to use the provided `environment.yml` file and conda:

```bash
conda env create -f environment.yml
conda activate wlclf
```

Once installed you can experiment with the code in the notebooks.

## Functionality

Currently, two main functionalities are implemented, the representation and evaluation of C2 formulas and a simple node classifier inspired by the color refinement algorithm.

### C2 Formulas

In c2.py the class Formula is used to represent and evaluate formulas in the logic C2, two-variable first order logic with counting quantifiers. The formulas can be evaluated on graphs:

```python
from c2 import *
import networkx as nx

formula = Exists(Var.x, (Exists(Var.y, E(Var.x, Var.y), 7)))
formulas.evaluate(nx.fast_gnp_random_graph(10, 0.5))
```
### Graph Classification

In GC2Tree.py a simple graph classifier is implemented:
```python
import numpy as np
import networkx as nx

graph = nx.fast_gnp_random_graph(10000, 0.001)
adj = nx.adjacency_matrix(graph)
X = np.zeros((graph.number_of_nodes(), 0))
y = np.array([1 if formula.evaluate(graph, i) else 0 for i in range(graph.number_of_nodes())])

test_samples = np.random.choice(len(y), size=int(0.2*len(y)), replace=False)
test_mask = np.zeros(len(y), dtype=bool)
test_mask[test_samples] = True

clf = GC2Tree()
clf.fit(adj, X, y, test_mask=test_mask)
clf.score(y, test_mask)
```

