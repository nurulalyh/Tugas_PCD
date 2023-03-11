# Nama  : Nurul Aliyah Dyah Sakhinah
# NIM   : F55121069
# Tugas : Program menerapkan median filter, min filter, dan max filter

#Import semua library yang dibutuhkan
import cv2
import numpy as np

# Membaca citra dan mengubahnya ke format grayscale
img = cv2.imread('gambar.jpeg', 0)

# Menggunakan filter median
median = cv2.medianBlur(img, 5)

# Menggunakan filter min
min_filter = cv2.erode(img, np.ones((5,5),np.uint8))

# Menggunakan filter max
max_filter = cv2.dilate(img, np.ones((5,5),np.uint8))

# Mengubah ukuran citra hasil filter
resized_median = cv2.resize(median, (200, 200))
resized_min = cv2.resize(min_filter, (200, 200))
resized_max = cv2.resize(max_filter, (200, 200))

# Menyimpan citra hasil filter yang telah diubah ukurannya ke dalam file
cv2.imwrite('gambar_grayscale.jpg', img)
cv2.imwrite('hasil_median-filter.jpg', resized_median)
cv2.imwrite('hasil_min-filter.jpg', resized_min)
cv2.imwrite('hasil_max-filter.jpg', resized_max)

# Menampilkan citra asli dan hasil filter
cv2.imshow('Citra Asli', img)
cv2.imshow('Median Filter', median)
cv2.imshow('Min Filter', min_filter)
cv2.imshow('Max Filter', max_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()