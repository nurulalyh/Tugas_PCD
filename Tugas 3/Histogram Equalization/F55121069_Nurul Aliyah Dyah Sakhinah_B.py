#Nurul ALiyah Dyah Sakhinah_F55121069

#Import Library yang dibutuhkan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Masukkan Gambar
img = cv2.imread('image.jpg')

# Hitung histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Hitung fungsi distribusi kumulatif
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Terapkan equalization
img_equalized = cv2.equalizeHist(img)

# Hitung equalized histogram
hist_equalized, bins = np.histogram(img_equalized.flatten(), 256, [0, 256])

# Hitung fungsi distribusi kumulatif yang diequalized
cdf_equalized = hist_equalized.cumsum()
cdf_normalized_equalized = cdf_equalized * hist_equalized.max() / cdf_equalized.max()

# Plot histogram dan fungsi distribusi kumulatif
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.title('Histogram'), plt.xlim([0, 256]), plt.xticks(range(0, 257, 64))
plt.subplot(223), plt.imshow(img_equalized, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(img_equalized.flatten(), 256, [0, 256], color='r')
plt.title('Equalized Histogram'), plt.xlim([0, 256]), plt.xticks(range(0, 257, 64))
plt.tight_layout()
plt.show()
