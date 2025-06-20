import numpy as np
import cv2


def recover(img_path, img_s_path):
    """
    estimate homography using BRISK and remove reflection from image
    :param img_path: image to be recovered, np.array
    :param img_s_path: support image for homography estimation, np.array
    :return: img_rec: recovered np.array image if at least 4 homography points were found, None otherwise
    """

    img_l = cv2.imread(img_path)
    img_r = cv2.imread(img_s_path)

    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img_l, None)
    kp2, des2 = brisk.detectAndCompute(img_r, None)
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
        print(f"Not enough matches were found for images {img_path} and {img_s_path}: {len(good_matches)}/{4}")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    img_dst = cv2.warpPerspective(img_l, M, img_r.shape[:2][::-1])
    mask = np.logical_and(img_r > img_dst, img_dst > 0)
    img_rec = img_r.copy()
    img_rec[mask] = img_dst[mask]
    return img_rec
