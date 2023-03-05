#Nurul ALiyah Dyah Sakhinah_F55121069

# Import Library yang dibutuhkan
import cv2
import numpy as np

# Masukkan gambar
img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

# Tambahkan noise di kedua gambar
mean = 0
var = 100
sigma = var ** 0.5
noise = np.random.normal(mean, sigma, img1.shape)
img1_noisy = cv2.add(img1, noise.astype(np.uint8))

noise = np.random.normal(mean, sigma, img2.shape)
img2_noisy = cv2.add(img2, noise.astype(np.uint8))

# Gunakan filter di gambar noise
kernel = np.ones((3,3),np.float32)/9
img1_filtered = cv2.filter2D(img1_noisy,-1,kernel)
img2_filtered = cv2.filter2D(img2_noisy,-1,kernel)

# Lakukan image averaging
avg = cv2.addWeighted(img1_filtered, 0.5, img2_filtered, 0.5, 0)

# Tampilkan Hasil
cv2.imshow('Original Image 1', img1)
cv2.imshow('Original Image 2', img2)
cv2.imshow('Averaged Image', avg)
cv2.waitKey(0)
cv2.destroyAllWindows()
