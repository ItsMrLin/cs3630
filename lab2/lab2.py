from myro import *
import os


def main():
	f = open(os.path.dirname(__file__) + '/robot_name.conf', 'r')
	init(f.readline())
	try:
		# do things here
		pass
	finally:
		stop()

if __name__ == "__main__":
	main()