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
