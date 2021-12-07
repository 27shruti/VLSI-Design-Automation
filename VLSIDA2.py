## STEP 1 - Plotting a graph from the Netlist

def read_gates(l,index):
    t = ""
    t = t + l[index]
    for j in range(1,3):
        if(l[index+j] >= '0' and l[index+j] <= '9'):
            y = l[index + j]
            t = t + y
    return(t)

lhs = []
rhs = []

file1 = open('s27.bench', 'r')
Lines = file1.readlines()

for line in Lines:
    if(len(line)>0):
        if(line[0] == 'G'):
            gate_lhs = read_gates(line, 0)

            for i in range(1, len(line)):
                if (line[i] == "G"):
                    gate_rhs = read_gates(line,i)
                    lhs.append((gate_rhs,gate_lhs))

import networkx as nx
import matplotlib.pyplot as plt
from math import *

lhs.sort()
print(lhs)

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
print(shruti + 1)

arr = np.zeros([shruti+1,shruti+1])

for elem in lhs:
    first = int(elem[0][1:])
    second = int(elem[1][1:])
    arr[first][second] = 1

print(arr) 


#STEP 3 : Use of BellMan - Ford Algorithm
#Let src be the node from where we need to calculate the shortest distance to all other points.

#number of edges = length of LHS
edges = len (lhs)

#number of vertices = the variable shruti + 1
vert = shruti + 1

#since no weights are given in these graphs, 
#we assume that all edges have the weight 1
weight = np.ones([vert])
print(weight)

def BellmanFord(src):
	# Initializing distance from source vertex to all vertices as INFINITE and distance of source vertex as 0
	dist = [inf]*vert
	dist[src] = 0

	for t in lhs:
		u = int(t[0][1:])
		v = int(t[1][1:])
		for i in range(vert):
			if dist[u] != float("inf") and dist[u] + 1 < dist[v]:
				dist[v] = dist[u] + 1
       	 

	print('Distance from source vertex',src)
	print('Vertex \t Distance from source')
	for i in range(len(dist)):
		print(i,'\t',dist[i])



BellmanFord(0)