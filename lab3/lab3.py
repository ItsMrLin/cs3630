from myro import *
from mosaic import *
# import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')

def mainLogic():
	K = cam_params_to_mat(CAM_KX, CAM_KY, CAM_CX, CAM_CY)

	picture = takePicture("gray")
	savePicture(picture, "base_image.png")
	img_a = cv2.imread("base_image.png")
	descripts_a, keypts_a, img_keypts_a = descript_keypt_extract(img_a)

	for i in range(9):
		turnBy(5, 'deg')
		picture = takePicture("gray")
		savePicture(picture, "image.png")
		img_b = cv2.imread("image.png")
		descripts_b, keypts_b, img_keypts_b = descript_keypt_extract(img_b)
		pair_pts_a, pair_pts_b = propose_pairs(descripts_a, keypts_a, descripts_b, keypts_b)
		best_H, best_inliers_a, best_inliers_b = homog_ransac(pair_pts_a, pair_pts_b)
		R = rot_from_homog(best_H, K)
		angle = extract_y_angle(R)
		print "angle:", angle

def main():
	f = open('../robot_name.conf', 'r')
	init(f.readline())
	f.close()
	try:
		mainLogic()
	finally:
		stop()

if __name__ == "__main__":
	main()