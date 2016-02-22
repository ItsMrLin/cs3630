import os
import csv
import argparse
import numpy as np
from collections import deque
# from myro import *

class PPGraph:
	def __init__(self, filename):
		self.node = {}
		self.adjMatrix = None
		self.edges = set()
		self.nodeToInd = {}
		self.indToNode = {}
		if not os.path.exists(filename):
			raise "filepath not exists"
		with open(filename, 'r') as ppMap:
			mapReader = csv.reader(ppMap)
			mode = 0 # mode 0 for reading coordinate; mode 1 for reading edges of graph
			count = 0
			for row in mapReader:
				if len(row) == 0:
					mode = 1
					self.adjMatrix = np.zeros((count, count))
					continue
				if mode == 0:
					self.node[count] = (int(row[1]), int(row[2]))
					self.nodeToInd[row[0]] = count
					self.indToNode[count] = row[0]
					count += 1
				if mode == 1:
					first = self.nodeToInd[row[0]]
					sec = self.nodeToInd[row[1]]
					self.adjMatrix[first][sec] = 1
					self.adjMatrix[sec][first] = 1
					if first < sec:
						self.edges.add((first, sec))
					else:
						self.edges.add((sec,first))

	def plan(self):
		'''
			To find a Eulerian Path

			Assume only these two cases happen:
				1. there are only 2 nodes of odd degrees
				2. all nodes are of even degree
		'''

		node_degree = np.sum(self.adjMatrix, axis=0) % 2
		odd_node_list = np.nonzero(node_degree)[0]
		if odd_node_list.shape[0] == 0:
			return map(lambda x: self.indToNode[x], self.bfs(0,1))
		else:
			return map(lambda x: self.indToNode[x], self.bfs(odd_node_list[0], odd_node_list[1]))


	def bfs(self, start_ind, end_ind):
	    """
	    	Search the shallowest nodes in the search tree first.

	    	format of "node" in fringe: (path, node index)


	    """
	    path = [-100, start_ind]
	    fringe = deque()
	    closed = set()
	    fringe.appendleft([path, start_ind, closed])
	    total_edges = set(self.edges)
	    total_edges.add((-100, start_ind))
	    while True:
			if len(fringe) == 0:
				return None
			elem = fringe.pop()
			if elem[0][-2] < elem[0][-1]:
				current_edge = (elem[0][-2], elem[0][-1])
			else:
				current_edge = (elem[0][-1], elem[0][-2])
			closed = set(elem[2])
			if total_edges == closed:
				return elem[0][1:-1]
			elif ((elem[0][-2], elem[0][-1]) not in closed):
				if elem[0][-2] < elem[0][-1]:
					closed.add((elem[0][-2], elem[0][-1]))
				else:
					closed.add((elem[0][-1], elem[0][-2]))
				for next in np.nonzero(self.adjMatrix[elem[1]])[0]:
					temp_path = list(elem[0])
					temp_path.append(next)
					temp = [temp_path, next, closed]
					fringe.appendleft(temp)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--map', help="Specify the path to the map file")
	args = parser.parse_args()

	a = PPGraph(args.map)
	a.plan()




