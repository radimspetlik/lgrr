import numpy as np
import cv2
import os

btn_down = False


def draw_mask(img, radius):
    """
    draw user-guided mask over the image
    :param return_kpoints: if True, return keypoints instead of mask
    :param img: np.array HxWxC: uint (range 0:255)
    :param radius: number, circle radius
    :return: np.array HxW: bool
    """

    def mouse_callback(event, x, y, flags, data):
        global btn_down  # TODO: this is harsh
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE) and btn_down:
            if event == cv2.EVENT_LBUTTONUP:
                btn_down = False
            dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            data['mask'][dist_from_center <= radius] = 1
            data['img'][data['mask']] = 255
            data['kpoints'].append((y, x))
            cv2.imshow("Mask", data['img'])

        elif event == cv2.EVENT_LBUTTONDOWN:
            btn_down = True

    h, w = img.shape[:2]
    mask = np.zeros((h, w), bool)
    Y, X = np.ogrid[:h, :w]
    data = dict(mask=mask, img=img.copy(), kpoints=list())
    cv2.namedWindow("Mask")
    cv2.imshow("Mask", data["img"])
    cv2.setMouseCallback("Mask", mouse_callback, data)
    cv2.waitKey(0)

    return mask, data["kpoints"]


if __name__ == '__main__':
    use_fire = False
    if use_fire:
        import Fire
        def main(img_path, radius=40):
            img = cv2.imread(img_path)
            mask = draw_mask(img, radius)
        Fire(main)
    else:
        radius = 40
        img_path = os.path.join('y:', os.sep, 'data', 'reflections', 'db', 'blender', 'COCO', 'train2017',
                                '000000000605_obj=000_camera=00000.jpg')
        img = cv2.imread(img_path)
        mask, keypoints = draw_mask(img, radius)
        mask = mask.astype(np.uint8) * 255
        rgb_mask = np.stack((mask, mask, mask), axis=-1)
        rgb_mask[:, :, 1:] = 0
        for keypoint in keypoints:
            print(keypoint)
            rgb_mask[keypoint[0], keypoint[1], 1:] = 255
        cv2.imshow("Result", rgb_mask)
        cv2.waitKey(0)
