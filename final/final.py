from myro import *
import numpy as np
import math
import cv2

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

def predict(particles, predictSigma):
    """ Predict particles one step forward. The motion model should be additive
        Gaussian noise with sigma predictSigma
      
    Returns:
      particles: list of predicted particles (same size as input particles)
    """
    
    #YOUR CODE HERE
    particles = particles + predictSigma * np.random.randn(particles.shape[0], 2)

    return particles

def update(particles, weights, keypoints):
    """ Resample particles and update weights accordingly after particle filter
        update
      
    Returns:
      newParticles: list of resampled partcles of type np.array
      weights: weights updated after sampling
    """

    #YOUR CODE HERE
    if len(keypoints) != 0:
      for i in xrange(particles.shape[0]):
        distances = np.apply_along_axis(np.linalg.norm, 1, 
          np.array(map(lambda x: x.pt, keypoints), dtype='float') - particles[i:i+1,:]
          )
        if np.min(distances) > 100:
          weights[i] *= 0
        else:
          for dist in distances:
            weights[i] *= 1 - 100 / float( 1 + np.exp(-dist)) + 1e-6
      weights /= np.sum(weights) + 1e-9
    if np.sum(weights) == 0:
      weights = np.ones(weights.shape)/weights.shape[0]

    return particles, weights

def resample(particles, weights):
    """ Resample particles and update weights accordingly after particle filter
        update
      
    Returns:
      newParticles: list of resampled partcles of type np.array
      wegiths: weights updated after sampling
    """
    
    #YOUR CODE HERE
    newParticles = particles[np.random.choice(range(particles.shape[0]),particles.shape[0], p=weights), :]
    weights = np.ones((particles.shape[0], )) / particles.shape[0]
    return newParticles, weights

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