# Nama  : Nurul Aliyah Dyah Sakhinah
# NIM   : F55121069
# Tugas : Program menerapkan Discrete Fourier Transform (DFT)

#Import semua library yang dibutuhkan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra dan mengubahnya ke dalam format grayscale
img = cv2.imread('gambar.jpeg', 0)

# Melakukan transformasi DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Menampilkan citra dan spektrum frekuensi
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Citra Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Spektrum Frekuensi'), plt.xticks([]), plt.yticks([])
plt.show()
