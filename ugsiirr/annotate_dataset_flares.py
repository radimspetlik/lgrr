import cv2
import numpy as np


def get_flare_mask(reflection_img: np.array, **kwargs):
    if kwargs is None:
        kwargs = dict(blur_iterations=20, ksize=5, crop_size=64)

    mask = np.zeros(reflection_img.shape[:2], bool)

    if mask_is_zero(reflection_img):
        return mask, None, None

    x, y, l, t, r, b = _get_centroid(reflection_img, **kwargs)
    paint_mask = _get_mask_from_cropped(reflection_img[b:t, l:r])

    mask[b:t, l:r][paint_mask] = 1
    return mask, y, x


def mask_is_zero(mask):
    return np.max(np.sum(mask, axis=2)) < 10 * np.mean(np.sum(mask, axis=2)) or np.max(mask) < 1


def _get_mask_from_cropped(crop):
    EPS = np.finfo(float).eps
    paint = np.sum(crop, axis=2).copy()
    mini, maxi = float(paint.min()), float(paint.max())
    thr = (maxi ** 2 - mini ** 2) / (2 * (maxi - mini) + EPS)
    paint[np.sum(crop, axis=2) > thr] = 1
    paint[np.sum(crop, axis=2) < thr] = 0
    return paint.astype(bool)


def _get_centroid(img, blur_iterations, ksize, crop_size):
    kernel = np.ones((ksize, ksize), np.float32) / 25
    dst = img.copy()
    for _ in range(blur_iterations):
        dst = cv2.filter2D(dst, -1, kernel)
    y, x = np.unravel_index(np.argmax(np.sum(dst, axis=2), axis=None), dst.shape[:2])

    b = max(0, y - crop_size // 2)
    t = min(y + crop_size // 2, img.shape[0] - 1)
    l = max(0, x - crop_size // 2)
    r = min(x + crop_size // 2, img.shape[1] - 1)

    return x, y, l, t, r, b
