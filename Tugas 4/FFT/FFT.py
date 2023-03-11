# Nama  : Nurul Aliyah Dyah Sakhinah
# NIM   : F55121069
# Tugas : Program menerapkan Fast Fourier Transform (FFT)

#Import semua library yang dibutuhkan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra dan mengubahnya ke dalam format grayscale
img = cv2.imread('gambar.jpeg', 0)

# Melakukan transformasi Fourier
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Menampilkan citra dan spektrum frekuensi
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Citra Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Spektrum Frekuensi'), plt.xticks([]), plt.yticks([])
plt.show()