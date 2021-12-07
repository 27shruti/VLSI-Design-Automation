import networkx as nx
import matplotlib.pyplot as plt
from math import *
import numpy as np
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

def read_gates(l,index):
	t = ''
	t = t + l[index]
	j = index + 1
	while ((l[j] != ' ') and (l[j] != ',') and (l[j] != ')')):
        t = t +l[j]
    	j = j +1
	return(t)

# objective function
def objective(x):
	return x[0]**2.0

# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):

	best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

	best_eval = objective(best)
	# current working solution
	curr, curr_eval = best, best_eval
	# run the algorithm
	for i in range(n_iterations):
    	# take a step
    	candidate = curr + randn(len(bounds)) * step_size
    	# evaluate candidate point
    	candidate_eval = objective(candidate)
    	# check for new best solution
    	if candidate_eval < best_eval:
        	# store new best point
        	best, best_eval = candidate, candidate_eval
        	# report progress
        	print('>%d f(%s) = %.5f' % (i, best, best_eval))
    	# difference between candidate and current point evaluation
    	diff = candidate_eval - curr_eval
    	# calculate temperature for current epoch
    	t = temp / float(i + 1)
    	# calculate metropolis acceptance criterion
    	metropolis = exp(-diff / t)
    	# check if we should keep the new point
    	if diff < 0 or rand() < metropolis:
        	# store the new current point
        	curr, curr_eval = candidate, candidate_eval
	return [best, best_eval]

 

def load_data(filename):

	lhs = []
	rhs = []
	exray = []
	vega = []

	abc = 0

	dict = {}

	file1 = open(filename, 'r')
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
    
	G = nx.DiGraph()

	G.add_edges_from( lhs )

	pos = nx.spring_layout(G)

	nx.draw_networkx_nodes(G, pos, node_size = 500)
	nx.draw_networkx_edges(G, pos, edgelist = G.edges(), edge_color = 'black')
	nx.draw_networkx_labels(G, pos)
	plt.show()

	return lhs


def main():
    
	seed(1)
	our_data = asarray(load_data('s27bench'))

	n_iterations = 1000

	step_size = 0.1

	temp = 10

	best, score = simulated_annealing(objective, our_data, n_iterations, step_size, temp)

	print('Simulated Annealing Completed')
	print('f(%s) = %f score' % (best, score))

    
if __name__ == "__main__":
	main()
