import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra
img = cv2.imread('gambar.jpeg', 0)

# Menerapkan transformasi Fourier pada citra
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Menentukan ukuran filter
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

# Membuat filter Butterworth highpass
mask_butterworth = np.zeros((rows, cols), np.uint8)
r = 50
n = 2
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
        mask_butterworth[i, j] = 1 / (1 + (distance/r)**(2*n))

# Menerapkan filter Butterworth highpass pada spektrum frekuensi
fshift_filtered_butterworth = fshift * mask_butterworth

# Menerapkan inverse transformasi Fourier pada spektrum frekuensi yang difilter
img_filtered_butterworth = np.fft.ifft2(np.fft.ifftshift(fshift_filtered_butterworth))

# Membuat filter Gaussian highpass
mask_gaussian = np.zeros((rows, cols), np.uint8)
r = 50
d = 2*r
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
        mask_gaussian[i, j] = 1 - np.exp(-distance**2 / (2*d**2))

# Menerapkan filter Gaussian highpass pada spektrum frekuensi
fshift_filtered_gaussian = fshift * mask_gaussian

# Menerapkan inverse transformasi Fourier pada spektrum frekuensi yang difilter
img_filtered_gaussian = np.fft.ifft2(np.fft.ifftshift(fshift_filtered_gaussian))

# Menampilkan citra asli dan citra yang telah difilter
plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Citra Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(np.abs(img_filtered_butterworth), cmap = 'gray')
plt.title('Butterworth Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(np.abs(img_filtered_gaussian), cmap = 'gray')
plt.title('Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(mask_butterworth, cmap = 'gray')
plt.title('Filter Butterworth Highpass'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(mask_gaussian, cmap = 'gray')
plt.title('Filter Gaussian Highpass'), plt.xticks([]), plt.yticks([])
plt.show()
