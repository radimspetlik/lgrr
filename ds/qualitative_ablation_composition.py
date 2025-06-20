import os
from glob import glob
import cv2
import json
import numpy as np


if __name__ == '__main__':
    basedir = os.path.join('c:', os.sep, 'Users', 'jarmi', 'data', 'reflections')

    fullpath = os.path.join(basedir, 'db', 'qualitative_ablation', 'radim')

    filepaths = glob(os.path.join(fullpath, '*.png'))

    overview_image = np.empty((0, 8 * 256, 3), dtype=np.uint8)
    row_stacked = 0
    image_listed = 0
    for image_filepath in filepaths:
        image_basename = os.path.basename(image_filepath)
        # inspection
        basic_value = 256

        img = cv2.imread(image_filepath).astype(np.float32)

        x, y, w, h = 95, 50, 65, 60
        if '039.png' == image_basename:
            x, y, w, h = 155, 95, 85, 65
        if '056.png' == image_basename:
            x, y, w, h = 285-256, 90, 85, 65
        if '080.png' == image_basename:
            x, y, w, h = 270-256, 86, 75, 75

        position_to_amplify_x = slice(x, x+w)
        position_to_amplify_y = slice(y, y+h)
        position_to_amplify_x_w = slice(x-2, x + w + 2)
        position_to_amplify_y_w = slice(y-2, y + h + 2)

        for i in range(1, 8):
            current_img = img[:, i*basic_value:(i+1)*basic_value]
            to_magnify = current_img[position_to_amplify_y, position_to_amplify_x].copy()
            magnified = cv2.resize(to_magnify, (to_magnify.shape[1]*2, to_magnify.shape[0]*2), interpolation=cv2.INTER_NEAREST)
            # make white bg
            current_img[position_to_amplify_y_w, position_to_amplify_x_w] = 255
            current_img[position_to_amplify_y, position_to_amplify_x] = to_magnify
            # store magnification
            if '039.png' == image_basename:
                current_img[-magnified.shape[0] - 3:, :magnified.shape[1] + 3] = 255
                current_img[-magnified.shape[0]:, :magnified.shape[1]] = magnified
            else:
                current_img[-magnified.shape[0]-3:, -magnified.shape[1]-3:] = 255
                current_img[-magnified.shape[0]:, -magnified.shape[1]:] = magnified
            img[:, i * basic_value:(i + 1) * basic_value] = current_img

        overview_image = np.concatenate((overview_image, img), axis=0)

    cv2.imwrite(os.path.join(basedir, 'results', 'qualitative_ablation.png'), overview_image)
