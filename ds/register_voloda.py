import numpy as np
import cv2


def recover(img_src, img_ref, img_mask):
    # search for keypoints
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img_ref, None)
    kp2, des2 = brisk.detectAndCompute(img_src, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1.astype("float32"), des2.astype("float32"), k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance:
            good_matches.append(m)
    if len(good_matches) < 4:
        print("Not enough matches were found for images {} and {}: {}".format(img_src, img_reg, len(good_matches)/4))
        return
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # match homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    img_registered = cv2.warpPerspective(img_ref, M, img_src.shape[:2][::-1])  # transformed reference image

    # replace pixels in source image
    # mask = np.logical_and(img_registered< img_src, img_registered > 0)
    # mask = mask.max(axis=-1)
    kernel = np.ones((5, 5), np.uint8)
    # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1).astype(np.bool)
    mask = img_mask.astype(np.float32)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    mask = mask / mask.max()

    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask.astype(np.float32), sigma=5)
    mask = mask / mask.max()

    img_src = img_src.astype(np.float32)
    img_registered = img_registered.astype(np.float32)
    img_reconstructed = img_src.copy()
    # img_reconstructed[mask] = img_registered[mask]
    img_reconstructed = img_reconstructed * (1 - mask) + mask * (img_registered - 0)
    img_reconstructed = np.clip(img_reconstructed, 0, 255)
    return img_reconstructed.astype(np.uint8)


if __name__ == "__main__":
    """
    >>> python recover_img_with_homog.py ./data/001_src.jpg ./data/001_ref.jpg ./data/001_rec.jpg

    ./data/001_src.jpg: path to the source image to be recovered
    ./data/001_ref.jpg: path to the reference image
    ./data/001_rec.jpg: save result to
    """
    import sys

    # img_src_path, img_ref_path, img_save_to = sys.argv[1:4]
    import os
    basepath = os.path.join('c:', os.sep, 'Users', 'jarmi', 'data', 'reflections', 'db', 'SIRR_dataset_homogr', 'voloda')
    img_id = '011'
    img_src_path = os.path.join(basepath, img_id + '_src.jpg')
    img_mask_path = os.path.join(basepath, img_id + '_src_mask.jpg')
    img_rec_path = os.path.join(basepath, img_id + '_rec.jpg')
    img_reg_path = os.path.join(basepath, img_id + '_ref.jpg')

    img_src = cv2.imread(img_src_path)
    img_reg = cv2.imread(img_reg_path)
    img_mask = cv2.imread(img_mask_path)
    img_rec = recover(img_src, img_reg, img_mask)

    from matplotlib import pyplot as plt

    my_dpi = 600
    fig = plt.figure(figsize=(img_rec.shape[1] / my_dpi, img_rec.shape[0] / my_dpi), dpi=my_dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(img_rec[:, :, ::-1])
    plt.show()

    if img_rec is not None:
        cv2.imwrite(img_rec_path, img_rec)
