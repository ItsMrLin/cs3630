from myro import *
import random

def mainLogic():
	while True:
        # p = 0.3
        # with p probility making a random turn
        if random.random() < 0.3:
            turnBy(int(random.random() * 360), 'deg')
        else:
            # with 1-p probability moving forward 1 - 3 seconds at half speed
            forward(0.5, 1 + random.random() * 2)

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