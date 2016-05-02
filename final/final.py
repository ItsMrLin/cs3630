from myro import *
import numpy as np
import math
import cv2
import time


def detectBlobs(im):
	""" Takes and image and locates the potential location(s) of the red marker
		on top of the robot

	Hint: bgr is the standard color space for images in OpenCV, but other color
		  spaces may yield better results

	Note: you are allowed to use OpenCV function calls here

	Returns:
	  keypoints: the keypoints returned by the SimpleBlobDetector, each of these
				 keypoints has pt and size values that should be used as
				 measurements for the particle filter
	"""

	#YOUR CODE HERE
	params = cv2.SimpleBlobDetector_Params()
 
	params.filterByColor = True
	params.blobColor = 255

	# # Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 260;
	 
	# Filter by Area.
	params.filterByArea = 1
	params.minArea = 30;
	params.maxArea = float("inf");

	# minDistBetweenBlobs = 0;

	# filterByColor = 0;
	# blobColor = 0;

	 
	# Filter by Circularity
	params.filterByCircularity = False
	 
	# Filter by Convexity
	params.filterByConvexity = False

	# Filter by Inertia
	params.filterByInertia = 0

	# print im[:,:,0]

	# in HSV
	filter_h1 = im[:,:,0] >= (179 - 13)
	filter_h2 = im[:,:,0] <= (13)
	filter_h = np.logical_or(filter_h1, filter_h2)

	filter_s1 = im[:,:,1] >= (191 - 40)
	filter_s2 = im[:,:,1] <= (191 + 40)
	filter_s = np.logical_and(filter_s1, filter_s2)

	filter_v1 = im[:,:,2] >= (128 - 40)
	filter_v2 = im[:,:,2] <= (128 - 40)
	filter_v = np.logical_and(filter_v1, filter_v2)

	filter_all = np.logical_and(filter_h, filter_s, filter_v)

	grey_img = np.zeros(im.shape[:2], dtype='uint8')
	grey_img[filter_all] = 255

	cv2.imshow("theshold", grey_img)

	detector = cv2.SimpleBlobDetector(params)
	keypoints = detector.detect(grey_img)
	keypoints = [x for x in keypoints if math.isnan(x.pt[0]) == False]
	return keypoints

def visualizeKeypoints(im, keypoints, color=(0,255,0)):
	""" Draw keypoints generated by blob detector on image in color specified
		(default is green)

	Returns:
	  im_with_keypoints: the image with keypoints overlaid
	"""	
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return im_with_keypoints

def fixKeypointsOrientation(keypoints, img_x, img_y):
	offPortion = 0
	keypoints.sort(key=lambda keypoint: keypoint.response, reverse=True)
	ptx, pty = keypoints[0].pt
	if pty < img_y/4:
		turnBy(10, 'deg')
	elif pty > img_y*3/4:
		turnBy(-10, 'deg')


def naiveLogic():
	while True:
		myro_im = takePicture()
		savePicture(myro_im, "image.png")
		opencv_im = cv2.imread("image.png")
		cv2.imshow("original", opencv_im)

		opencv_im = cv2.cvtColor(opencv_im, cv2.COLOR_BGR2HSV)
		keypoints = detectBlobs(opencv_im)
		opencv_im = visualizeKeypoints(opencv_im, keypoints)
		# cv2.imshow("seen", opencv_im)
		print keypoints
		# cv2.waitKey(0)
		if len(keypoints) > 0:
			fixKeypointsOrientation(keypoints, opencv_im.shape[0], opencv_im.shape[1])
			forward(1, 2)
			# wait(1)
		else:
			turnBy(30, 'deg')


def mainLogic():
	pass

def main():
	f = open('../robot_name.conf', 'r')
	init(f.readline())
	f.close()
	try:
		naiveLogic()
		# turnBy(90, "deg")
	finally:
		stop()

if __name__ == "__main__":
	main()