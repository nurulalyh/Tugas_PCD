#Nurul Aliyah Dyah Sakhinah_F55121069

import cv2

img = cv2.imread("mammogram.tif", 0)

img_1 = 255 - img

cv2.imshow("Original Image", img)
cv2.imshow("Image Negative", img_1)

cv2.waitKey(0)
cv.destroyAllWindows()
