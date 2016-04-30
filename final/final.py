from myro import *

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
    params.minArea = 100;
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

 
    # U in YUV
    filter_11 = im[:,:,1] >= (158 - 17)
    filter_12 = im[:,:,1] <= (158 + 20)
    filter_1 = np.logical_and(filter_11, filter_12)
    filter_21 = im[:,:,2] >= (113 - 30)
    filter_22 = im[:,:,2] <= (113 + 30)
    filter_2 = np.logical_and(filter_21, filter_22)
    filter_all = np.logical_and(filter_1, filter_2)
    print filter_all

    grey_img = np.zeros(im.shape[:2], dtype='uint8')
    grey_img[filter_all] = 255

    detector = cv2.SimpleBlobDetector(params)
    keypoints = detector.detect(grey_img)
    keypoints = [x for x in keypoints if math.isnan(x.pt[0]) == False]
    return keypoints

def naiveLogic():
	while True:
		myro_im = takePicture()
		savePicture(picture, "image.png")
		opencv_im = cv2.imread("image.png")
		keypoints = detectBlobs(myro_im)
		if len(keypoints) >= 0:
			forward(1, 2)
			wait(1)
		else:
			turnBy(30, 'deg')


def mainLogic():
	pass

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