#Nurul ALiyah Dyah Sakhinah_F55121069

# Import Library yang dibutuhkan
import cv2

# Masukkan 2 gambar
img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

# Ubah gambar ke format grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Hitung difference image pada gambar grayscale
diff_img = cv2.absdiff(img1, img2)
diff_gray = cv2.absdiff(gray1, gray2)

# Metode histogram equalization pada difference image
eq_diff_img_gray = cv2.equalizeHist(diff_gray)

# Tampilkan hasil
cv2.imshow('Original Image 1', img1)
cv2.imshow('Original Image 2', img2)
cv2.imshow('Difference Image', diff_img)
cv2.imshow('Difference Image Gray', diff_gray)
cv2.imshow('Histogram Equalization Difference Image Gray', eq_diff_img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
