import cv2
import numpy as np
import glob

def objp_for_size(dimensions, reverse=False):

    x, y = dimensions

    if reverse:
        x_coords, y_coords = np.meshgrid(np.arange(x)[::-1], np.arange(y)[::-1])
    else:
        x_coords, y_coords = np.meshgrid(np.arange(x), np.arange(y))    
    
    object_points = np.stack((y_coords.flatten(), x_coords.flatten(), np.zeros_like(x_coords.flatten())), axis=-1)
    
    return object_points.astype(np.float32)

camera_name = "ugreen_right"
grid_dimensions = (10, 7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = [] 
imgpoints = [] 

imgs = glob.glob(f"{camera_name}_calibration/*.png")

for fname in imgs:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, grid_dimensions, None)

    if ret == True:
        objpoints.append(objp_for_size(grid_dimensions, reverse=True))
        corners = np.squeeze(corners)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret)
print(mtx)
print(dist)
# print(rvecs)
# print(tvecs)

np.save(f'{camera_name}_parameters/ret.npy', ret)
np.save(f'{camera_name}_parameters/mtx.npy', mtx)
np.save(f'{camera_name}_parameters/dist.npy', dist)
np.save(f'{camera_name}_parameters/rvecs.npy', rvecs)
np.save(f'{camera_name}_parameters/tvecs.npy', tvecs)