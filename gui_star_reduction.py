import tkinter as tk
from tkinter import filedialog
from astropy.io import fits
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

fits_path = None

def normalize01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def load_fits():
    global fits_path
    fits_path = filedialog.askopenfilename(
        filetypes=[("FITS files", "*.fits")]
    )
    if fits_path:
        label_file.config(text=os.path.basename(fits_path))

def apply_reduction():
    if fits_path is None:
        return
    print("Saved in:", os.getcwd())

    kernel_size = kernel_slider.get()
    iterations = iter_slider.get()

    hdul = fits.open(fits_path)
    data = hdul[0].data
    hdul.close()

    img = normalize01(data)
    img_u8 = (img * 255).astype(np.uint8)

    # mask (simple threshold)
    blur = cv.GaussianBlur(img_u8, (0, 0), 1.5)
    mask = cv.adaptiveThreshold(
        blur, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        51, -2
    )

    mask_soft = cv.GaussianBlur(mask, (0, 0), 2.0)
    M = mask_soft.astype(np.float32) / 255.0

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv.erode(img_u8, kernel, iterations=iterations)

    Ie = eroded.astype(np.float32) / 255.0
    final = (M * Ie) + ((1 - M) * img)
    final = np.clip(final, 0, 1)

    os.makedirs("results", exist_ok=True)
    plt.imsave("results/gui_result.png", final, cmap="gray")

    label_status.config(text="Image sauvegardée : gui_result.png")

# --- GUI ---
root = tk.Tk()
root.title("Star Reduction - Phase 3")

btn_load = tk.Button(root, text="Charger un fichier FITS", command=load_fits)
btn_load.pack(pady=5)

label_file = tk.Label(root, text="Aucun fichier chargé")
label_file.pack()

kernel_slider = tk.Scale(root, from_=3, to=9, orient=tk.HORIZONTAL, label="Taille du noyau")
kernel_slider.set(3)
kernel_slider.pack()

iter_slider = tk.Scale(root, from_=1, to=3, orient=tk.HORIZONTAL, label="Itérations")
iter_slider.set(1)
iter_slider.pack()

btn_apply = tk.Button(root, text="Appliquer la réduction", command=apply_reduction)
btn_apply.pack(pady=10)

label_status = tk.Label(root, text="")
label_status.pack()

btn_quit = tk.Button(root, text="Quitter", command=root.destroy)
btn_quit.pack(pady=5)
root.mainloop()
