import os
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

<<<<<<< HEAD
# Paths
fits_file = "./examples/HorseHead.fits"
os.makedirs("./results", exist_ok=True)

# --- Load FITS ---
hdul = fits.open(fits_file)
=======
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# https://docs.astropy.org/en/stable/io/fits/index.html
# Open and read the FITS file
fits_file = os.path.join(script_dir, 'examples', 'test_M31_raw.fits')
hdul = fits.open(fits_file)  # Returns an HDUList object.

# Displays information about the file
hdul.info()

# Access the data from the primary HDU
>>>>>>> 09c6666e62ba2c578ebb4b382f1494ba49b95715
data = hdul[0].data
hdul.close()

# --- Convert FITS data to image for processing ---
if data.ndim == 3:
    # Color: transpose if needed (3, H, W) -> (H, W, 3)
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
<<<<<<< HEAD

    # Save original (normalized for display)
    data_norm = (data - data.min()) / (data.max() - data.min())
    plt.imsave("./results/original.png", data_norm)

    # Convert to uint8 per channel for OpenCV
    image = np.zeros_like(data, dtype="uint8")
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype("uint8")
else:
    # Monochrome
    plt.imsave("./results/original.png", data, cmap="gray")
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

# --- Test multiple parameters ---
kernel_sizes = [3, 5, 7]
iterations_list = [1, 2, 3]

for k in kernel_sizes:
    kernel = np.ones((k, k), np.uint8)
    for it in iterations_list:
        eroded = cv.erode(image, kernel, iterations=it)
        out_path = f"./results/eroded_k{k}_it{it}.png"
        cv.imwrite(out_path, eroded)
        print("Saved:", out_path)
=======
    # If already (height, width, 3), no change needed
    
    # Normalizes the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    original_path = os.path.join(script_dir, 'results', 'original.png')
    plt.imsave(original_path, data_normalized)
    
    # Normalizes each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image 
    original_path = os.path.join(script_dir, 'results', 'original.png')
    plt.imsave(original_path, data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

# Define a kernel for erosion
kernel = np.ones((3,3), np.uint8)
# Perform erosion
eroded_image = cv.erode(image, kernel, iterations=1)

# Save the eroded image 
eroded_path = os.path.join(script_dir, 'results', 'eroded_testest.png')
cv.imwrite(eroded_path, eroded_image)

# Close the file
hdul.close()
>>>>>>> 09c6666e62ba2c578ebb4b382f1494ba49b95715
