# Python code to read image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils.object_detection import non_max_suppression


img = cv2.imread(os.getcwd()+'/'+
    "balls.jpeg")

img = cv2.resize(img,(480,480))

img2 = img

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img, (5, 5), 0)
laplacian = cv2.Laplacian(blur, cv2.CV_64F,(5,5))

template = cv2.imread(os.getcwd()+'/'+
    'template.jpeg')

template = cv2.resize(template,(210,210))

template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


blur1 = cv2.GaussianBlur(template, (5, 5), 0)
laplacian_temp = cv2.Laplacian(blur1, cv2.CV_64F,(5,5))

laplacian = cv2.convertScaleAbs(laplacian)
laplacian_temp = cv2.convertScaleAbs(laplacian_temp)

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
laplacian = cv2.filter2D(src=laplacian, ddepth=-1, kernel=kernel)
laplacian_temp = cv2.filter2D(src=laplacian_temp, ddepth=-1, kernel=kernel)


w, h = laplacian_temp.shape[:2]


# Apply template Matching
res = cv2.matchTemplate(laplacian, laplacian_temp, cv2.TM_CCOEFF_NORMED)


threshold = 0.08


(y_points, x_points)  = np.where(res >= threshold)

boxes = list()

for (x,y) in zip(x_points, y_points):
    boxes.append((x, y, x + w, y + h))

box=non_max_suppression(np.array(boxes))

for (x1, y1, x2, y2) in box:
    print(x1)
    # draw the bounding box on the image
    cv2.rectangle(img2, (x1, y1), (x2, y2),(255, 0, 0),3)

# Second Parameter is image array
cv2.imshow("Template" ,img2)
cv2.imshow("After NMS", laplacian)

cv2.waitKey(0)


cv2.destroyAllWindows()
