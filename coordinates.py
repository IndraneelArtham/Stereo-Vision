import cv2


coordinates = []

cam = cv2.VideoCapture(cv2.CAP_DSHOW)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        coordinates.append((x, y))
        print(f"Mouse position: ({x}, {y})")


# image = cv2.imread('Images/img_stereo_straight2.png')

cv2.namedWindow('Mouse Position')
cv2.setMouseCallback('Mouse Position', mouse_callback)

while True:
    ret, frame = cam.read()
    cv2.imshow('Mouse Position', frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cv2.destroyAllWindows()