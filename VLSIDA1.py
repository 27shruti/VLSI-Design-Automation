def DFS(arr,shruti,visited,start): 
          
        # Print current node 
        print(start, end = ' ') 
  
        # Set current node as visited 
        visited[start] = True
  
        # For every node of the graph 
        for i in range(shruti): 
              
            # If some node is adjacent to the  
            # current node and it has not  
            # already been visited 
            if (arr[start][i] == 1 and
                    (not visited[i])): 
                DFS(arr,shruti, visited,i) 


def BFS(arr,shruti):
         
        # Visited vector to so that a
        # vertex is not visited more than 
        # once Initializing the vector to 
        # false as no vertex is visited at
        # the beginning 
        visited = [False] * shruti
        q = [0]
 
        # Set source as visited
        visited[0] = True
 
        while q:
            vis = q[0]
 
            # Print current node
            print(vis, end = ' ')
            q.pop(0)
             
            # For every adjacent vertex to 
            # the current vertex
            for i in range(shruti):
                if (arr[vis][i] == 1 and
                      (not visited[i])):
                           
                    # Push the adjacent node 
                    # in the queue
                    q.append(i)
                     
                    # set
                    visited[i] = True

            if(len(q)==0):
                d = 0
                for x in range(shruti):
                    if(visited[x] == False):
                        visited[x] = True
                        q.append(x)
                        d=1
                        x= shruti+1




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

file1 = open('c499.bench', 'r')
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
print("Abhi")
print(A)

shruti = np.shape(A[0])[1]
print(shruti)

arr = np.zeros([shruti+1,shruti+1])
#print(arr)

# map[string] = {false}
# count_var=0
# for elem in lhs:
#     if(dict[elem[0]] == false):
#         dict[elem[0]]=true;
#         count_var+=1
#     if(dict[elem[1]] == false):
#         dict[elem[1]]=true;
#         count_var+=1

# Arr[count_var][count_var] = {0}

for elem in lhs:
    first = int(elem[0][1:])
    second = int(elem[1][1:])
    arr[first][second] = 1

print(arr) 



print("BFS:")

BFS(arr,shruti+1)

print("DFS:")
visited = [False] * (shruti+1)
for i in range(shruti+1):
    if(visited[i]==False):
        visited[i]=True
        DFS(arr,shruti+1,visited,i)
