from myro import *
init("/dev/tty.Fluke2-0B62-Fluke2")

maxTurnCount = 5

def isFollowingWall():
	checkStage = 3.0
	for i in range(int(checkStage)):
		turnLeft(1/checkStage, 0.625)
		if isWallInTheFront():
			turnRight(1/checkStage * (i+1) + 0.2, 0.625)
			return True

	turnRight(1, 0.625)

	for i in range(int(checkStage)):
		turnRight(1/checkStage, 0.625)
		if isWallInTheFront():
			turnLeft(1/checkStage * (i+1) + 0.2, 0.625)
			return True
	
	turnLeft(1, 0.625)
	return False
			


def isWallInTheFront():
	wallThreshold = 1000
	return (getObstacle("center") > wallThreshold or 
		getObstacle("middle") > wallThreshold or 
		getObstacle("left") > wallThreshold or 
		getObstacle("right") > wallThreshold or
		getObstacle(0) > wallThreshold or 
		getObstacle(1) > wallThreshold or
		getObstacle(2) > wallThreshold)

try:
	turnCount = 0
	wallCheckCount = 0
	while True:
		if isWallInTheFront():
			turnRight(1, 0.5)
			turnCount += 1
			if turnCount > maxTurnCount:
				break
		else:
			forward(1, 0.5)
			turnCount = 0
			wallCheckCount += 1
			if wallCheckCount % 3 == 0:
				wallCheckCount = 0
				if not isFollowingWall():
					break

finally:
	stop()
