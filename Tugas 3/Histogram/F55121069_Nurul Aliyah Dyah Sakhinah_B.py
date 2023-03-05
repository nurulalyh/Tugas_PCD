#Nurul ALiyah Dyah Sakhinah_F55121069

# Import Library yang dibutuhkan
import cv2
from matplotlib import pyplot as plt

# Masukkan Gambar
img = cv2.imread('gambar.jpg')

# Menghitung histogram gambar
hist = cv2.calcHist([img],[0],None,[256],[0,256])

# Plot gambar and histogram
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Image')
axs[1].plot(hist, color='black')
axs[1].set_xlim([0, 256])
axs[1].set_title('Histogram')
axs[1].set_xlabel('Pixel Intensity')
axs[1].set_ylabel('Pixel Count')
plt.show()
