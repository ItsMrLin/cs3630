from myro import *
import random
import time

def mainLogic():
	time.sleep(120)
	translate(0.5)
	while True:
		# p = 0.3
		# with p probility making a random turn
		if random.random() < 0.5:
			if random.random() < 0.5:
				rotate(0.5)
			else:
				rotate(-0.5)
			# turnBy(int(random.random() * 360), 'deg')
		else:
			# with 1-p probability moving forward 1 - 3 seconds at half speed
			rotate(0)
		time.sleep(1)

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