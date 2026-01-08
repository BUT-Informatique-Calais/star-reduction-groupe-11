import os
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Open and read the FITS file
fits_file = os.path.join(script_dir, 'examples', 'test_M31_raw.fits')
hdul = fits.open(fits_file)  # Returns an HDUList object

# Access the data from the primary HDU
data = hdul[0].data

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    
    # Normalizes the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    original_path = os.path.join(script_dir, 'results', 'original_phase2.png')
    plt.imsave(original_path, data_normalized)
    
    # Normalizes each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image 
    original_path = os.path.join(script_dir, 'results', 'original_phase2.png')
    plt.imsave(original_path, data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

# Close the file
hdul.close()

# Phase 3 : Feature 2 multi size reduction
# STAR DETECTION FOR ADAPTIVE REDUCTION

# If color image, convert to grayscale for detection
if image.ndim == 3:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
else:
    gray_image = image.copy()

# Calculate statistics for thresholding
mean, median, std = sigma_clipped_stats(gray_image, sigma=3.0)

# Star detector
daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)  
sources = daofind(gray_image - median)

# Create binary masks for each star category
mask_small = np.zeros_like(gray_image, dtype=np.float32)
mask_medium = np.zeros_like(gray_image, dtype=np.float32)
mask_large = np.zeros_like(gray_image, dtype=np.float32)

if sources:
    # For each star, determine its category based on brightness
    for star in sources:
        x = int(star['xcentroid'])
        y = int(star['ycentroid'])
        
        # Measure star brightness in a small region around the center
        y_min = max(0, y-3)
        y_max = min(gray_image.shape[0], y+4)
        x_min = max(0, x-3)
        x_max = min(gray_image.shape[1], x+4)
        
        brightness = np.max(gray_image[y_min:y_max, x_min:x_max])
        
        # Determine disk size to draw
        fwhm_value = star['fwhm'] if 'fwhm' in star.colnames else 3.0
        radius = int(fwhm_value * 1.5)
        
        # Classify star based on brightness
        if brightness < 120:  # Small star
            cv.circle(mask_small, (x, y), radius, 1.0, -1)
        elif brightness < 180:  # Medium star
            cv.circle(mask_medium, (x, y), radius, 1.0, -1)
        else:  # Large star
            cv.circle(mask_large, (x, y), radius, 1.0, -1)
    
    # Apply Gaussian blur for smooth transitions
    mask_small = cv.GaussianBlur(mask_small, (5, 5), sigmaX=2)
    mask_medium = cv.GaussianBlur(mask_medium, (5, 5), sigmaX=2)
    mask_large = cv.GaussianBlur(mask_large, (5, 5), sigmaX=2)
    
    # Ensure values stay between 0 and 1
    mask_small = np.clip(mask_small, 0, 1)
    mask_medium = np.clip(mask_medium, 0, 1)
    mask_large = np.clip(mask_large, 0, 1)

# APPLY ADAPTIVE REDUCTION

# Create 3 eroded versions with different parameters
kernel_small = np.ones((3, 3), np.uint8)
image_eroded_small = cv.erode(image, kernel_small, iterations=1)

kernel_medium = np.ones((5, 5), np.uint8)
image_eroded_medium = cv.erode(image, kernel_medium, iterations=1)

kernel_large = np.ones((7, 7), np.uint8)
image_eroded_large = cv.erode(image, kernel_large, iterations=2)

# Adapt masks for color images
if image.ndim == 3:
    mask_small_3d = np.stack([mask_small] * 3, axis=2)
    mask_medium_3d = np.stack([mask_medium] * 3, axis=2)
    mask_large_3d = np.stack([mask_large] * 3, axis=2)
else:
    mask_small_3d = mask_small
    mask_medium_3d = mask_medium
    mask_large_3d = mask_large

# Combine masks and ensure total doesn't exceed 1
total_mask = mask_small_3d + mask_medium_3d + mask_large_3d
total_mask = np.clip(total_mask, 0, 1)
background_mask = 1 - total_mask

# Apply the adaptive reduction formula
final_image = (
    mask_small_3d * image_eroded_small +
    mask_medium_3d * image_eroded_medium +
    mask_large_3d * image_eroded_large +
    background_mask * image
).astype(np.uint8)

# Save the result
final_path = os.path.join(script_dir, 'results', 'result_phase2.png')
cv.imwrite(final_path, final_image)

# Feature 5 : before and after comparator

# CREATE COMPARISON TOOLS
results_dir = os.path.join(script_dir, 'results')

# Prepare images for display
if image.ndim == 2:  # Monochrome
    original_display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    final_display = cv.cvtColor(final_image, cv.COLOR_GRAY2BGR)
else:  # Color
    original_display = image.copy()
    final_display = final_image.copy()

# Before and after comparison
height = max(original_display.shape[0], final_display.shape[0])
total_width = original_display.shape[1] + final_display.shape[1] + 30

# Create comparison canvas
comparison = np.ones((height, total_width, 3), dtype=np.uint8) * 50

# Place original on left
comparison[:original_display.shape[0], :original_display.shape[1]] = original_display

# Place final on right
x_offset = original_display.shape[1] + 30
comparison[:final_display.shape[0], x_offset:x_offset+final_display.shape[1]] = final_display

# Add text
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(comparison, 'BEFORE', (20, 40), font, 1.2, (255, 255, 255), 2)
cv.putText(comparison, 'AFTER', (x_offset + 20, 40), font, 1.2, (255, 255, 255), 2)

# Save before and after comparison
comparison_path = os.path.join(results_dir, 'comparaison_phase2.png')
cv.imwrite(comparison_path, comparison)

# 4.2 Difference map (pour montrer ce qui a été enlevé)
difference = cv.absdiff(image.astype(np.float32), final_image.astype(np.float32))

# For color images, take average of channels
if difference.ndim == 3:
    difference_gray = np.mean(difference, axis=2)
else:
    difference_gray = difference

# Normalize for better visibility
diff_normalized = cv.normalize(difference_gray, None, 0, 255, cv.NORM_MINMAX)
diff_normalized = diff_normalized.astype(np.uint8)

# Apply colormap
diff_colored = cv.applyColorMap(diff_normalized, cv.COLORMAP_JET)

# Add text
cv.putText(diff_colored, "DIFFERENCE MAP", (50, 50), font, 1, (255, 255, 255), 2)

# Save difference map
diff_path = os.path.join(results_dir, 'difference.png')
cv.imwrite(diff_path, diff_colored)
