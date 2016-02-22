from myro import *
import math
import PPGraph

def getAngleAndDistance(robotPos, robotAngle, targetPos):
	targetAngle = math.atan2(targetPos[1] - robotPos[1], targetPos[0] - robotPos[0])
	turnAngle = (math.degrees(targetAngle) - robotAngle + 360) % 360
	moveDist = math.hypot(targetPos[0] - robotPos[0], targetPos[1] - robotPos[1])
	print moveDist, int(round(turnAngle))
	return moveDist, int(round(turnAngle))


def mainLogic():
	scalingFactor = 0.6
	mapName = 'CS3630_Lab2_Map1.csv'

	graph = PPGraph.PPGraph(mapName)
	path, nodesToCoordinates = graph.plan()

	robotAngle = 0
	robotX, robotY = nodesToCoordinates[path[0]]
	for i in range(1, len(path)):
		moveDist, turnAngle = getAngleAndDistance((robotX, robotY), robotAngle, nodesToCoordinates[path[i]])
		turnBy(turnAngle, 'deg')
		wait(1)
		forward(moveDist * scalingFactor, moveDist * scalingFactor)
		wait(1)

		robotAngle = (robotAngle + turnAngle) % 360
		robotX = robotX + moveDist * math.cos(math.radians(robotAngle))
		robotY = robotY + moveDist * math.sin(math.radians(robotAngle))


def main():
	f = open('../robot_name.conf', 'r')
	init(f.readline())
	f.close()
	try:
		mainLogic()
		# turnBy(90, "deg")
	finally:
		stop()

if __name__ == "__main__":
	main()