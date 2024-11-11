import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load calibration parameters (camera matrix and distortion coefficients)
cameraMatrix = np.load('ugreen_parameters/mtx.npy')
distCoeffs = np.load('ugreen_parameters/dist.npy')

print(cameraMatrix)
print(distCoeffs)


# Read the image to rectify
img_left = cv2.imread('left_original.jpg')
img_right = cv2.imread('right_original.jpg')

img_left = cv2.resize(img_left, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
img_right = cv2.resize(img_right, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Option 1: Simple undistortion using cv2.undistort
undistorted_img_left = cv2.undistort(img_left, cameraMatrix, distCoeffs)
undistorted_img_right = cv2.undistort(img_right, cameraMatrix, distCoeffs)



# Option 2: Advanced undistortion with remapping (more control)
h, w = img_left.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), 5)
rectified_img_left = cv2.remap(img_left, mapx, mapy, cv2.INTER_LINEAR)
rectified_img_right = cv2.remap(img_right, mapx, mapy, cv2.INTER_LINEAR)


# cv2.imshow('Original Left Image', img_left)
# cv2.imshow('Original Right Image', img_right)


# Option 1: Display simple undistorted image
# cv2.imshow('Undistorted Left Image', undistorted_img_left)
# cv2.imshow('Undistorted Right Image', undistorted_img_right)

# Option 2: Display rectified image
# cv2.imshow('Rectified Left Image', rectified_img_left)
# cv2.imshow('Rectified Right Image', rectified_img_right)


stereo = cv2.StereoSGBM.create(numDisparities=96, blockSize=11)

left = cv2.cvtColor(rectified_img_left, cv2.COLOR_BGR2GRAY)
right = cv2.cvtColor(rectified_img_right, cv2.COLOR_BGR2GRAY)

disparity = stereo.compute(right, left)

cv2.imshow('disparity', disparity)



disparity[disparity == 0] = 1

# Compute the depth map
depth_map = (533.98 * 0.8) / disparity.astype(np.float32)

# Normalize depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

cv2.imshow('depth_map', depth_map)
# cv2.imshow('depth_map', depth_map_normalized)


cv2.waitKey(0)
cv2.destroyAllWindows()