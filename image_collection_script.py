import cv2
import time
ugreen = cv2.VideoCapture(1)
# logitech = cv2.VideoCapture(cv2.CAP_DSHOW)

ugreen_left_dir = "ugreen_left_calibration/"
ugreen_right_dir = "ugreen_right_calibration/"

img_counter = 0

num_images = 30

capture_delay = 2

start_time = time.time()

while img_counter < num_images:

    ret1, ugreen_frame = ugreen.read()
    # ret2, logitech_frame = logitech.read()

    if time.time() - start_time > capture_delay:
        cv2.imwrite(ugreen_right_dir + f"img_{img_counter}.png", ugreen_frame)
        # cv2.imwrite(logitech_dir + f"img_{img_counter}.png", logitech_frame)
        img_counter += 1
        start_time = time.time()
    
    
    cv2.imshow("ugreen", ugreen_frame)
    # cv2.imshow("logitech", logitech_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

ugreen.release()
# logitech.release()
cv2.destroyAllWindows()
