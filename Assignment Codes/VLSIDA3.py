class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print("%d - %d: %d" % (u, v, weight))


## STEP 1 - Plotting a graph from the Netlist

def read_gates(l,index):
    t = ''
    t = t + l[index]
    j = index + 1
    while ((l[j] != ' ') and (l[j] != ',') and (l[j] != ')')):
        t = t + l[j]
        j = j +1
    return(t)

lhs = []
rhs = []
exray = []
vega = []

abc = 0

dict = {}

file1 = open('s27.bench', 'r')
Lines = file1.readlines()

for line in Lines:
    if(len(line)>0):
        if(line[0] == 'G'):
            gate_lhs = read_gates(line, 0)
            if(dict.get(gate_lhs)):
                gate_lhs_index = dict[gate_lhs]

            else:
                gate_lhs_index = abc
                dict[gate_lhs] = abc
                abc = abc+1


            for i in range(1, len(line)):
                if (line[i] == "G"):
                    gate_rhs = read_gates(line,i)
                    if(dict.get(gate_rhs)):
                        gate_rhs_index = dict[gate_rhs]

                    else:
                        gate_rhs_index = abc
                        dict[gate_rhs] = abc
                        abc = abc+1

                    lhs.append((gate_rhs_index,gate_lhs_index))

#To get a list of all Vertices:
                    exray.append(gate_rhs)
                    exray.append(gate_lhs)

lhs.sort()
#print(lhs)

import networkx as nx
import matplotlib.pyplot as plt
from math import *

G = nx.DiGraph()

G.add_edges_from( lhs )

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size = 500)
nx.draw_networkx_edges(G, pos, edgelist = G.edges(), edge_color = 'black')
nx.draw_networkx_labels(G, pos)
plt.show()

## STEP 2 - Creating an Adjacency Matrix

import numpy as np
A = nx.to_numpy_matrix(G)

shruti = np.shape(A[0])[1]
#print(shruti + 1)

#STEP 3 : Use of Kruskal Algorithm

#list with all the vertices
exray = list(set(exray))
#print(exray)

#number of edges = length of LHS
edges = len (lhs)

#number of vertices = the variable shruti + 1
vert = shruti + 1

#since no weights are given in these graphs,
#we assume that all edges have the weight 1
weight = np.ones([vert])
print(len(exray))
for j in range(len(exray)):
    #vega.append((exray[j], weight[j]))
    print(exray[j], '\t' ,weight[j] )


g = Graph(abc)
for elem in lhs:
    g.add_edge(elem[0],elem[1],1)

g.kruskal_algo()

