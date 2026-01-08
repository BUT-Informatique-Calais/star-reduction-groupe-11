import os
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python erosion.py <input_fits> <output_dir>")
    sys.exit(1)

fits_file = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

# https://docs.astropy.org/en/stable/io/fits/index.html
# Open and read the FITS file
hdul = fits.open(fits_file)  # Returns an HDUList object.

# Displays information about the file
hdul.info()

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed
    
    # Normalizes the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    original_path = os.path.join(output_dir, 'original.png')
    plt.imsave(original_path, data_normalized)
    
    # Normalizes each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image 
    original_path = os.path.join(output_dir, 'original.png')
    plt.imsave(original_path, data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

# Close the file
hdul.close()

# Phase 1: Test different erosion configurations
# Define parameters to test
kernel_sizes = [3, 5, 7]  # Different kernel sizes
iterations_list = [1, 2, 3]  # Different iteration counts

# Test all parameter combinations
for ksize in kernel_sizes:
    for it_count in iterations_list:
        # Define a kernel for erosion
        kernel = np.ones((ksize, ksize), np.uint8)
        # Perform erosion with current parameters
        eroded_image = cv.erode(image, kernel, iterations=it_count)
        
        # Save the eroded image 
        eroded_path = os.path.join(output_dir, f'eroded_k{ksize}_it{it_count}.png')
        cv.imwrite(eroded_path, eroded_image)

# Create a comparison image to show erosion effect

# Use specific configuration for comparison
kernel_comparison = np.ones((5, 5), np.uint8)
eroded_comparison = cv.erode(image, kernel_comparison, iterations=2)

# Create side-by-side image
if image.ndim == 2:  # Monochrome image
    # Convert to 3 channels for text display
    original_display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    eroded_display = cv.cvtColor(eroded_comparison, cv.COLOR_GRAY2BGR)
else:  # Color image
    original_display = image.copy()
    eroded_display = eroded_comparison.copy()

# Create larger image to display both versions
height = max(original_display.shape[0], eroded_display.shape[0])
width_total = original_display.shape[1] + eroded_display.shape[1] + 20  # +20 for space

# Background image
comparison_image = np.ones((height, width_total, 3), dtype=np.uint8) * 50  # Dark gray

# Place original image on left
comparison_image[:original_display.shape[0], :original_display.shape[1]] = original_display

# Place eroded image on right
x_offset = original_display.shape[1] + 20
comparison_image[:eroded_display.shape[0], x_offset:x_offset+eroded_display.shape[1]] = eroded_display

# Add descriptive text
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(comparison_image, 'ORIGINAL', (10, 30), font, 1, (255, 255, 255), 2)
cv.putText(comparison_image, 'EROSION (k=5, it=2)', (x_offset + 10, 30), font, 1, (255, 255, 255), 2)
cv.putText(comparison_image, 'Phase 1 - Simple erosion test', (10, height - 20), font, 0.7, (200, 200, 200), 1)

# Save comparison image
comparison_path = os.path.join(output_dir, 'comparaison_phase1.png')
cv.imwrite(comparison_path, comparison_image)
"test complete"