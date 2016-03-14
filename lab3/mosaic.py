import numpy as np
import cv2
import random
import scipy.spatial.distance as ssd

CAM_KX = 720.
CAM_KY = CAM_KX
CAM_CX = 320.
CAM_CY = 200.

# DO NOT MODIFY cam_params_to_mat
def cam_params_to_mat(cam_kx, cam_ky, cam_cx, cam_cy):
    """Returns camera matrix K (3x3 numpy.matrix) from the focus and camera center parameters.
    """
    K = np.reshape(np.mat([cam_kx, 0, cam_cx, 0, cam_ky, cam_cy, 0, 0, 1]), (3,3))
    return K

# DO NOT MODIFY descript_keypt_extract
def descript_keypt_extract(img):
    """Takes an image and converts it to a list of descriptors and corresponding keypoints.

    From http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb

    Returns:
        des (np.ndarray): Nx32 array of N feature descriptors of length 32.
        kpts (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        img2 (np.array): Image which draws the locations of found keypoints.
    """

    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kpts = orb.detect(img,None)

    # compute the descriptors with ORB
    kpts, des = orb.compute(img, kpts)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kpts, color=(255,0,0), flags=0)

    kpts = [np.mat([kpt.pt[0], kpt.pt[1], 1.0]).T for kpt in kpts]

    return des, kpts, img2

# fill your in code here
def propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b):
    """Given a set of descriptors and keypoints from two images, propose good keypoint pairs.

    Feature descriptors should encode local image geometry in a way that is invariant to
    small changes in perspective. They should be comparible using an L2 metric.

    For the top N matching descrpitors, select and return the top corresponding keypoint pairs.

    Returns:
        pair_pts_a (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        pair_pts_b (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
    """
    # code here
    descripts_a = np.array(descripts_a, dtype='float')
    descripts_b = np.array(descripts_b, dtype='float')
    pair_pts_a = []
    pair_pts_b = []
    confidence = []

    proposed_pair = []
    if len(keypts_b) < len(keypts_a):
        keypts_a = keypts_a[:len(keypts_b)]
    for i, pta in enumerate(keypts_a):
        firstDist = float('inf')
        secDist = float('inf')
        firstInd = 0
        secInd = 0
        for j, ptb in enumerate(keypts_b):
            # a = np.array(descripts_a[i], dtype='int')
            # b = np.array(descripts_b[i], dtype='int')
            # dist = ssd.hamming(a,b)
            dist = np.linalg.norm(descripts_a[i, :] - descripts_b[j, :])
            if dist < firstDist and dist < secDist:
                secInd = firstInd
                secDist = firstDist
                firstInd = j
                firstDist = dist
            elif dist >= firstDist and dist < secDist:
                secInd = j
                secDist = dist
        confidence.append(firstDist/float(secDist + 1e-9))
        proposed_pair.append((i, firstInd))
    top_pairs = sorted( zip(proposed_pair, range(len(confidence))), key=lambda k: confidence[k[1]])
    # N = np.sum(np.array(confidence) <= 0.9)
    N = int(len(confidence) * 0.4)
    print N
    top_pt_inds_a = map(lambda x: x[0][0], top_pairs)
    top_pt_inds_b = map(lambda x: x[0][1], top_pairs)
    # print top_pt_inds

    pair_pts_a = [keypts_a[top_pt_inds_a[k]] for k in xrange(len(top_pt_inds_a))]
    pair_pts_b = [keypts_b[top_pt_inds_b[k]] for k in xrange(len(top_pt_inds_b))]
    return pair_pts_a, pair_pts_b

# fill your in code here
def homog_dlt(ptsa, ptsb):
    """From a list of point pairs, find the 3x3 homography between them.

    Find the homography H using the direct linear transform method. For correct
    correspondences, the points should all satisfy the equality
    w*ptb = H pta, where w > 0 is some multiplier. 

    Arguments:
        ptsa (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        ptsb (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 

    Returns:
        H (np.matrix of shape 3x3): Homography found using DLT.
    """
    # code here
    numPts = len(ptsa)
    A = np.zeros((numPts*2, 9))
    # ptsa = [np.array([223.94883728, 101.52348328, 1.]), np.array([174., 180., 1.]), np.array([446., 179., 1.]), np.array([347.04000854, 149.76000977, 1.])]
    # ptsb = [np.array([349.36022949, 104.50946808, 1.]), np.array([303.84002686, 182.88000488, 1.]), np.array([580.80004883, 177.6000061, 1.]), np.array([475.20001221, 149.76000977, 1.])]
    for i in xrange(numPts):
        A[2*i,:] = np.array([ptsa[i][0], ptsa[i][1], 1, 0, 0, 0, -ptsa[i][0] * ptsb[i][0], -ptsa[i][1] * ptsb[i][0], -ptsb[i][0]])
        A[2*i+1,:] = np.array([0, 0, 0, ptsa[i][0], ptsa[i][1], 1, -ptsa[i][0] * ptsb[i][1], -ptsa[i][1] * ptsb[i][1], -ptsb[i][1]])
    [U,D,V] = np.linalg.svd(A)
    V = V.transpose()
    H = np.matrix(np.reshape(V[:,-1],(3,3)))
    return H

# fill your in code here
def homog_ransac(pair_pts_a, pair_pts_b):
    """From possibly good keypoint pairs, determine the best homography using RANSAC.

    For a set of possible pairs, many of which are incorrect, determine the homography
    which best represents the image transformation. For the best found homography H,
    determine which points are close enough to be considered inliers for this model
    and return those.

    Returns:
        H (np.matrix of shape 3x3): Homography found using DLT and RANSAC.
        best_inliers_a (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
        best_inliers_b (list of N np.matrix of shape 3x1): List of N corresponding keypoints in 
            homogeneous form. 
    """
    # code here

    s = 4
    N = 500
    epsilon = 10
    numPts = len(pair_pts_b)
    best_H = None
    count_inliers = 0
    temp_count_inliears = 0
    best_inliers_a = []
    best_inliers_b = []
    for k in xrange(N):
        random_inds = random.sample(range(numPts), s)
        sample_a = [pair_pts_a[ind] for ind in random_inds]
        sample_b = [pair_pts_b[ind] for ind in random_inds]
        H = homog_dlt(sample_a,sample_b)
        temp_count_inliears = 0
        for i in xrange(numPts):
            target = H*pair_pts_a[i]
            target /= target[2] + 1e-9
            dist = np.linalg.norm(target - pair_pts_b[i])
            # print dist
            if dist < epsilon:
                temp_count_inliears += 1
        if temp_count_inliears >= count_inliers:
            best_H = H
            count_inliers = temp_count_inliears

    for i in xrange(numPts):
        target = best_H*pair_pts_a[i]
        target /= target[2]
        dist = np.linalg.norm(target - pair_pts_b[i])
        if dist < epsilon:
            best_inliers_a.append(pair_pts_a[i])
            best_inliers_b.append(pair_pts_b[i])
    return best_H, best_inliers_a, best_inliers_b

# DO NOT MODIFY perspect_combine
def perspect_combine(img_a, img_b, H, length, width):
    """Perspective warp and blend two images based on given homography H.

    Create img_ab by first warping all the pixels in img_a according to the relation
    img_b = H img_a.  For pixels in both img_a and img_b, average the intensity.
    Otherwise, pick the image value present, and black otherwise.
    """
    bw = img_b.shape[0]
    bl = img_b.shape[1]

    warp_a = cv2.warpPerspective(img_a, H, (length, width))
    mask_a = np.zeros((bw, bl, 3))
    for i in range(bw):
        for j in range(bl):
            if np.sum(warp_a[i][j]) > 0.:
                mask_a[i,j,:] = 1.
    img_ab = warp_a
    # img_ab[:bw,:bl,:] -= warp_a[:bw,:bl,:]/2.*mask_a[:bw,:bl,:]
    # img_ab[:bw,:bl,:] += img_b*(1-mask_a[:bw,:bl,:])
    # img_ab[:bw,:bl,:] += img_b/2.*mask_a[:bw,:bl,:]
    img_ab[:bw,:bl,:] -= np.uint8(warp_a[:bw,:bl,:]/2.*mask_a[:bw,:bl,:])
    img_ab[:bw,:bl,:] += np.uint8(img_b*(1-mask_a[:bw,:bl,:]))
    img_ab[:bw,:bl,:] += np.uint8(img_b/2.*mask_a[:bw,:bl,:])

    return img_ab

# DO NOT MODIFY img_combine_homog
def img_combine_homog(img_a, img_b, length_ab, width_ab):
    """Perspective warp and blend two images of nearby perspectives.
    """
    descripts_a, keypts_a, img_keypts_a = descript_keypt_extract(img_a)
    descripts_b, keypts_b, img_keypts_b = descript_keypt_extract(img_b)

    if True:
        cv2.imshow('Keypts A', img_keypts_a)
        cv2.imshow('Keypts B', img_keypts_b)

    pair_pts_a, pair_pts_b = propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b)
    assert(len(pair_pts_a) == len(pair_pts_b))
    assert(np.shape(pair_pts_a[0]) == (3,1) and np.shape(pair_pts_b[0]) == (3,1))
    assert(isinstance(pair_pts_a[0], np.matrix) and isinstance(pair_pts_b[0], np.matrix))

    best_H, best_inliers_a, best_inliers_b = homog_ransac(pair_pts_a, pair_pts_b)
    assert(np.shape(best_H) == (3,3) and isinstance(best_H, np.matrix))
    print len(best_inliers_a)

    fixed_H = homog_dlt(best_inliers_a, best_inliers_b)
    assert(np.shape(fixed_H) == (3,3) and isinstance(fixed_H, np.matrix))

    img_ab = perspect_combine(img_a, img_b, fixed_H, length_ab, width_ab)
    return img_ab, fixed_H

# fill your in code here
def rot_from_homog(H, K):
    """Find the rotation matrix from a homography from perspectives with identical camera centers.
    
    The rotation found should be bRa or Ra^b.

    Arguments:
        H (np.matrix of shape 3x3): Homography
        K (np.matrix of shape 3x3): Camera matrix
    Returns:
        R (np.matrix of shape 3x3): Rotation matrix from frame a to frame b
    """
    # code here
    R = np.linalg.inv(K)*H*K
    return R


# fill your in code here
def extract_y_angle(R):
    """Given a rotation matrix around the y-axis, find the angle of rotation.

    The matrix need not be perfectly in SO(3), but provides an estimate nonetheless.

    Arguments:
        R (np.matrix of shape 3x3): Rotation matrix from frame a to frame b
    Returns:
        y_ang (float): angle in radians
    """
    # code here
    y_ang = np.arccos(R[0,0])
    return y_ang



# DO NOT MODIFY single_pair_combine
def single_pair_combine(img_ai, img_bi):
    img_a = cv2.imread('image_%02d.png'%img_ai)
    img_b = cv2.imread('image_%02d.png'%img_bi)

    # decimate by 2
    img_a = img_a[::2,::2,:]
    img_b = img_b[::2,::2,:]

    length_ab, width_ab = 1000, 600
    img_ab, best_H = img_combine_homog(img_a, img_b, length_ab, width_ab)

    K = cam_params_to_mat(CAM_KX, CAM_KY, CAM_CX, CAM_CY)
    R = rot_from_homog(best_H, K)
    assert(np.shape(R) == (3,3) and isinstance(R, np.matrix))
    y_ang = extract_y_angle(R)

    print 'H'
    print best_H
    print 'K'
    print K
    print 'R'
    print R
    print 'Y angle in Radians/Degrees'
    print y_ang, np.rad2deg(y_ang)

    cv2.imshow('Image A', img_a)
    cv2.imshow('Image B', img_b)
    cv2.imshow('Combined Image', img_ab)
    cv2.waitKey(0)

# # DO NOT MODIFY multi_pair_combine
# def multi_pair_combine(beg_i, n_imgs):
#     img_ab = cv2.imread('image_%02d.png'%beg_i)
#     img_ab = img_ab[::2,::2,:]
#     for i in range(beg_i+1, beg_i+n_imgs):
#         img_b = cv2.imread('image_%02d.png'%i)

#         # decimate by 2
#         img_b = img_b[::2,::2,:]

#         length_ab, width_ab = 800+400*(i-1), 600
#         img_ab, best_H = img_combine_homog(img_ab, img_b, length_ab, width_ab)

#         cv2.imshow('Combined Image', img_ab)
#         cv2.waitKey(100)
#     cv2.imwrite('combo_%02d_%02d.png'%(beg_i, beg_i+n_imgs-1), img_ab)

def multi_pair_combine(beg_i, n_imgs):
    img_ab = cv2.imread('image_%02d.png'%beg_i)
    img_ab = img_ab[::2,::2,:]
    img_last = img_ab # NEW
    for i in range(beg_i+1, beg_i+n_imgs):
        img_b = cv2.imread('image_%02d.png'%i)

        # decimate by 2
        img_b = img_b[::2,::2,:]

        length_ab, width_ab = 800+400*(i-1), 600
        # img_ab, best_H = img_combine_homog(img_ab, img_b, length_ab, width_ab) # OLD
        img_ab_old, best_H = img_combine_homog(img_last, img_b, 1000, 600) # NEW
        img_ab = perspect_combine(img_ab, img_b, best_H, length_ab, width_ab) # NEW
        img_last = img_b # NEW

        cv2.imshow('Combined Image', img_ab)
        cv2.waitKey(100)
    cv2.imwrite('combo_%02d_%02d.png'%(beg_i, beg_i+n_imgs-1), img_ab)

# DO NOT MODIFY main
def main():
    if False:
        single_pair_combine(0, 1)

    if True:
        multi_pair_combine(1, 5)

if __name__ == "__main__":
    main()
