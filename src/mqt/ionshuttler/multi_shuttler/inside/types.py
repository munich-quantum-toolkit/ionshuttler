import networkx as nx

Node = tuple[int, int]
Edge = tuple[Node, Node]
Graph = nx.Graph[Node]
