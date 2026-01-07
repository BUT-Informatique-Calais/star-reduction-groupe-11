from astropy.io import fits
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

fits_file = "./examples/HorseHead.fits"
os.makedirs("./results", exist_ok=True)

hdul = fits.open(fits_file)
data = hdul[0].data
hdul.close()

if data.ndim != 2:
    raise ValueError("Cette version Phase 2 est pour une image mono (HorseHead).")

img = data.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

plt.imsave("./results/p2_original.png", img, cmap="gray")

img_u8 = (img * 255).astype(np.uint8)

blur_bg = cv.GaussianBlur(img_u8, (0, 0), 1.5)

mask = cv.adaptiveThreshold(
    blur_bg,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    51,
    -2 
)

kernel = np.ones((3, 3), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

cv.imwrite("./results/p2_mask_binary.png", mask)

mask_soft = cv.GaussianBlur(mask, (0, 0), 2.0)
cv.imwrite("./results/p2_mask_soft.png", mask_soft)

k = 3
it = 1
ker = np.ones((k, k), np.uint8)
eroded_u8 = cv.erode(img_u8, ker, iterations=it)
cv.imwrite("./results/p2_eroded.png", eroded_u8)
 ---
M = (mask_soft.astype(np.float32) / 255.0)  # [0,1]
I0 = img.astype(np.float32)                 # [0,1]
Ie = (eroded_u8.astype(np.float32) / 255.0) # [0,1]

final = (M * Ie) + ((1.0 - M) * I0)
final = np.clip(final, 0, 1)

plt.imsave("./results/p2_final.png", final, cmap="gray")

print("OK Phase 2 saved in ./results/")
print("Try tuning: blockSize(51), C(-2), blur sigma(2.0), kernel size k(3), iterations(it=1)")
