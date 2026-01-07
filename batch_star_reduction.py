import os
import argparse
from astropy.io import fits
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def load_fits(path):
    hdul = fits.open(path)
    data = hdul[0].data
    hdul.close()

    if data is None:
        raise ValueError("FITS vide")

    data = np.array(data)

    # Color case: (3, H, W) -> (H, W, 3)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))

    return data


def normalize01(x):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + 1e-8)


def build_star_mask(gray_u8, block_size, C, sigma_bg, sigma_mask):
    # blur to help threshold
    bg = cv.GaussianBlur(gray_u8, (0, 0), sigma_bg)

    # adaptive threshold (stars -> white)
    mask = cv.adaptiveThreshold(
        bg, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        block_size,
        C
    )

    # small clean
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    # soften edges
    mask_soft = cv.GaussianBlur(mask, (0, 0), sigma_mask)

    return mask, mask_soft


def erode_and_blend(img01, mask_soft_u8, ksize, iterations):
    # img01: float [0,1]
    # mask_soft_u8: uint8 [0..255]
    M = mask_soft_u8.astype(np.float32) / 255.0

    if img01.ndim == 2:
        img_u8 = (img01 * 255).astype(np.uint8)
        ker = np.ones((ksize, ksize), np.uint8)
        eroded_u8 = cv.erode(img_u8, ker, iterations=iterations)

        I0 = img01
        Ie = eroded_u8.astype(np.float32) / 255.0

        final = (M * Ie) + ((1.0 - M) * I0)
        return np.clip(final, 0, 1)

    # color: apply erosion per channel and blend with same mask
    img_u8 = (img01 * 255).astype(np.uint8)
    ker = np.ones((ksize, ksize), np.uint8)

    eroded = np.zeros_like(img_u8)
    for c in range(img_u8.shape[2]):
        eroded[:, :, c] = cv.erode(img_u8[:, :, c], ker, iterations=iterations)

    I0 = img01
    Ie = eroded.astype(np.float32) / 255.0

    # expand mask to 3 channels
    M3 = np.repeat(M[:, :, None], 3, axis=2)
    final = (M3 * Ie) + ((1.0 - M3) * I0)
    return np.clip(final, 0, 1)


def process_one(fits_path, out_dir, args):
    base = os.path.splitext(os.path.basename(fits_path))[0]

    data = load_fits(fits_path)
    img01 = normalize01(data)

    # build grayscale for mask
    if img01.ndim == 2:
        gray01 = img01
    else:
        gray01 = np.mean(img01, axis=2)

    gray_u8 = (gray01 * 255).astype(np.uint8)

    mask_bin, mask_soft = build_star_mask(
        gray_u8,
        block_size=args.block,
        C=args.C,
        sigma_bg=args.sigma_bg,
        sigma_mask=args.sigma_mask
    )

    final = erode_and_blend(img01, mask_soft, args.kernel, args.iter)

    # save outputs
    out_final = os.path.join(out_dir, f"{base}_final.png")
    out_mask = os.path.join(out_dir, f"{base}_mask.png")
    out_mask_soft = os.path.join(out_dir, f"{base}_mask_soft.png")

    if final.ndim == 2:
        plt.imsave(out_final, final, cmap="gray")
    else:
        plt.imsave(out_final, final)

    cv.imwrite(out_mask, mask_bin)
    cv.imwrite(out_mask_soft, mask_soft)

    print("OK:", fits_path, "->", out_final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--block", type=int, default=51)     # must be odd
    parser.add_argument("--C", type=int, default=-2)
    parser.add_argument("--sigma-bg", type=float, default=1.5)
    parser.add_argument("--sigma-mask", type=float, default=2.0)
    args = parser.parse_args()

    if args.block % 2 == 0:
        raise ValueError("--block doit être impair (ex: 31, 51, 71)")

    os.makedirs(args.output_dir, exist_ok=True)

    fits_files = []
    for name in os.listdir(args.input_dir):
        if name.lower().endswith(".fits"):
            fits_files.append(os.path.join(args.input_dir, name))

    if not fits_files:
        print("Aucun .fits trouvé dans", args.input_dir)
        return

    for f in sorted(fits_files):
        process_one(f, args.output_dir, args)


if __name__ == "__main__":
    main()
