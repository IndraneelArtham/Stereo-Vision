import cv2

# Initialize the list to store the coordinates
coordinates = []

# Mouse callback function to capture coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        coordinates.append((x, y))
        print(f"Mouse position: ({x}, {y})")

# Load an image or create a blank one
image = cv2.imread('Images/img_test.png')  # Replace with your image path
# if image is None:
#     # Create a blank image if no image is loaded
#     image = cv2.imread('blank_image.png')

# Create a window and set the mouse callback function
cv2.namedWindow('Mouse Position')
cv2.setMouseCallback('Mouse Position', mouse_callback)

while True:
    cv2.imshow('Mouse Position', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()

# Optionally, save the coordinates to a file
# with open('mouse_coordinates.txt', 'w') as file:
#     for coord in coordinates:
#         file.write(f"{coord}\n")
