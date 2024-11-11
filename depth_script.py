import cv2
import numpy as np

# Load left and right images as 8-bit unsigned integers
left_image = cv2.imread("left_original.png")
right_image = cv2.imread("right_original.png")

# Define rotation and translation vectors
rvecs = np.array([0, 0, 0])  # Keep as float32
tvecs = np.array([8.2, 0, 0])  # Keep as float32

# Define image size as a tuple of integers
imageSize = (640, 480)  # Image width and height as a tuple of integers

# Load camera parameters and ensure they're in the correct type
mtx1 = np.load('ugreen_left_parameters/mtx.npy')
dist1 = np.load('ugreen_left_parameters/dist.npy')
mtx2 = np.load('ugreen_left_parameters/mtx.npy')
dist2 = np.load('ugreen_left_parameters/dist.npy')

# Stereo rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    mtx1, dist1, mtx2, dist2, imageSize, rvecs, tvecs
)

# Initialize undistort rectify maps
map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, imageSize, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, imageSize, cv2.CV_32FC1)

# Remap the images to rectify them
rectified_left = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

# Set window size for disparity calculation
window_size = 5

# Disparity calculation with StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Should be a multiple of 16
    # blockSize=11,
    # P1=8 * 3 * window_size ** 2,  # Regularization term (adjust as needed)
    # P2=32 * 3 * window_size ** 2,  # Smoothness term (adjust as needed)
    # disp12MaxDiff=1,
    # uniquenessRatio=10,
    # speckleWindowSize=100,
    # speckleRange=32
)

# Compute the disparity map
disparity_map = stereo.compute(rectified_left, rectified_right).astype(np.float32)

# Reproject to 3D
depth_map = cv2.reprojectImageTo3D(disparity_map, Q)

# Display the disparity map
cv2.imshow("rect_left", rectified_left)
cv2.imshow("rect_right", rectified_right)
# cv2.imshow("Disparity Map", disparity_map / disparity_map.max())  # Normalize for better visualization
# cv2.imshow('depth_map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
