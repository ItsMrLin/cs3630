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
    params.filterByArea = False
    params.minArea = 0;
    params.maxArea = float("inf");
     
    # Filter by Circularity
    params.filterByCircularity = False
     
    # Filter by Convexity
    params.filterByConvexity = False

    # Filter by Inertia
    params.filterByInertia = False
    params.maxInertiaRatio = 0.5

 
    # U in YUV
    filter_11 = im[:,:,1] >= (158 - 10)
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
          weights[i] = 0
        else:
          weights[i] *= 1 - 1 / float( 1 + np.exp(-np.min(distances)) + 1e-9)
      weights /= np.sum(weights)

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

def visualizeParticles(im, particles, weights, color=(0,0,255)):
    """ Plot particles as circles with radius proportional to weight, which
        should be [0-1], (default color is red). Also plots weighted average
        of particles as blue circle. Particles should be a numpy.ndarray of
        [x, y] particle locations.

    Returns:
      im: image with particles overlaid as red circles
    """
    im_with_particles = im.copy()    
    s = (0, 0)
    for i in range(0, len(particles)):
      s += particles[i]*weights[i]
      cv2.circle(im_with_particles, tuple(particles[i].astype(int)), radius=int(weights[i]*250), color=(0,0,255), thickness=3)
    cv2.circle(im_with_particles, tuple(s.astype(int)), radius=3, color=(255,0,0), thickness=6)
    return im_with_particles

def visualizeKeypoints(im, keypoints, color=(0,255,0)):
    """ Draw keypoints generated by blob detector on image in color specified
        (default is green)

    Returns:
      im_with_keypoints: the image with keypoints overlaid
    """    
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints

if __name__ == "__main__":
  """ Iterate through a dataset of sequential images and use a blob detector and
      particle filter to track the robot(s) visible in the images. A couple
      helper functions were included to visualize blob keypoints and particles.

  """

  #some initial variables you can use
  imageSet='ImageSet1'
  imageWidth = 1280
  imageHeight = 800
  numParticles = 1000
  initialScale = 150
  predictionSigma = 150
  x0 = np.array([600, 300])  #seed location for particles
  particles = initialScale * np.random.randn(numParticles,2) + x0 #YOUR CODE HERE: make some normally distributed particles
  particles[:,0] = particles[:,0].clip(0, imageWidth)
  particles[:,1] = particles[:,1].clip(0, imageHeight)
  weights = np.ones((numParticles,)) / float(numParticles) #YOUR CODE HERE: make some weights to go along with the particles

  for i in range(0, 43):
    #read in next image
    im = cv2.imread(imageSet+'/'+imageSet+'_' + '%02d.jpg'%i)
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
 
    #visualize particles
    im_to_show = visualizeParticles(im, particles, weights)
    cv2.imshow("Current Particles", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'1_Current.jpg', im_to_show)

    #predict forward
    particles = predict(particles, predictionSigma)
    im_to_show = visualizeParticles(im, particles, weights)
    cv2.imshow("Prediction", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'2_Predicted.jpg', im_to_show)
    
    #detected keypoint in measurement
    keypoints = detectBlobs(yuv)

    #update paticleFilter using measurement if there was one
    if keypoints:
      particles, weights = update(particles, weights, keypoints)

    im_to_show = visualizeKeypoints(im, keypoints)
    im_to_show = visualizeParticles(im_to_show, particles, weights)
    cv2.imshow("Reweighted", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'3_Reweighted.jpg', im_to_show)

    #resample particles
    particles, weights = resample(particles, weights)
    im_to_show = visualizeKeypoints(im, keypoints)
    im_to_show = visualizeParticles(im_to_show, particles, weights)
    cv2.imshow("Resampled", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'4_Resampled.jpg', im_to_show)
    cv2.waitKey(0)
    
