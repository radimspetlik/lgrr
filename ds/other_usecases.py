import os
from glob import glob
import cv2
import json
import numpy as np


def pixelate(face):
    temp = cv2.resize(face, (4, 3), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (38, 26), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    basedir = os.path.join('c:', os.sep, 'Users', 'jarmi', 'data', 'reflections')

    datasets = {}
    datasets['glasses'] = ['192px_00_174e016b0fa0297656db58844b9e9017.jpg.png',
                           '192px_00_Celebrities-Actress-wearing-Glasses-Global-Eyeglasses.jpg.png',
                           '192px_00_Lena+Waithe+Premiere+HBO+Westworld+Season+oq71GDk2X3ml.jpg.png',
                           '192px_00_The-Top-10-Superheroes-Who-Wear-Glasses-678x381.jpg.png']

    datasets['boxes'] = ['128px_00_2337254827_9ce869d430_h.jpg.png',
                         '128px_00_DSC_0012.jpg.png',
                        # '128px_00_31251326847_35a474d8fe_b.jpg.png',
                         '128px_00_Aussie-Snow-2-e1575278468991.jpg.png']

    datasets['dust_removal'] = ['192px_03_old-photo-restoration-before.jpg.png']

    overview_image = np.empty((0, 4 * 128, 3), dtype=np.uint8)
    image_row = []
    row_stacked = 0
    image_listed = 0
    for dataset_name, image_names in datasets.items():
        for image_name in image_names:
            image_path = os.path.join(basedir, 'visual_inspection', dataset_name, image_name)

            # inspection
            basic_value = 128
            if dataset_name == 'glasses' or dataset_name == 'dust_removal':
                basic_value = 192
            img = cv2.imread(image_path).astype(np.float32)
            img_in = img[:basic_value, :basic_value]
            img_out = img[:basic_value, 3 * basic_value:4 * basic_value]

            if image_name == '192px_03_old-photo-restoration-before.jpg.png':
                y_range = slice(0, 180)
                x_range = slice(0, 180)
                img_in = img_in[y_range, x_range]
                img_out = img_out[y_range, x_range]

            img_in = cv2.resize(img_in, (128, 128)).astype(np.uint8)
            img_out = cv2.resize(img_out, (128, 128)).astype(np.uint8)

            if image_name == '128px_00_31251326847_35a474d8fe_b.jpg.png':
                y_range = slice(63, 63 + 26)
                x_range = slice(78, 78 + 38)
                img_in[y_range, x_range] = pixelate(img_in[y_range, x_range])
                img_out[y_range, x_range] = pixelate(img_out[y_range, x_range])

            image_row.append(np.concatenate((img_in, img_out), axis=0))

            if len(image_row) == 4:
                stacked = np.concatenate(image_row, axis=1)
                overview_image = np.concatenate((overview_image, stacked))

                image_row = []
                row_stacked += 1

            if row_stacked == 3:
                break
        if row_stacked == 3:
            break

    cv2.imwrite(os.path.join(basedir, 'results', 'other_usecases.png'), overview_image)
