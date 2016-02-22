from myro import *

def main():
	f = open('../robot_name.conf', 'r')
	init(f.readline())
	f.close()
	try:
		# do things here
		turnBy(90, 'deg')
		pass
	finally:
		stop()

if __name__ == "__main__":
	main()