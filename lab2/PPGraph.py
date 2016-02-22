import os
import csv
import numpy as np
# from myro import *

class PPGraph:
	def __init__(self, filename):
		self.node = {}
		self.adjMatrix = None
		self.nodeToInd = {}
		if not os.path.exists(filename):
			raise "path not exists"
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
					self.node[row[0]] = (int(row[1]), int(row[2]))
					self.nodeToInd[row[0]] = count
					count += 1
				if mode == 1:
					first = self.nodeToInd[row[0]]
					sec = self.nodeToInd[row[1]]
					self.adjMatrix[first][sec] = 1
					self.adjMatrix[sec][first] = 1

	def plan(self):
		'''
			To find a Eulerian Path

			Assume only these two cases happen:
				1. there are only 2 nodes of odd degrees
				2. all nodes are of even degree
		'''

		node_degree = np.sum(self.adjMatrix, axis=0) % 2
		odd_node_list = np.nonzero(node_degree)
		if odd_node_list.shape[0] == 0:
			return self.planning(0,1)
		else:
			return self.planning(odd_node_list[0], odd_node_list[1])


	def BFS(self, start_ind, end_ind):
