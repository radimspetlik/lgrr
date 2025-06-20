import os
from glob import glob
import cv2
import json
import numpy as np

if __name__ == '__main__':
    basedir = os.path.join('/', os.sep, 'home.dokt', 'spetlrad', 'datagrid', 'reflections')

    datasetdir = os.path.join(basedir, 'db', 'lamps', '0000002')
    globs = glob(os.path.join(datasetdir, '*.jpg'))
    globs.extend(glob(os.path.join(datasetdir, '*.jpeg')))
    globs.extend(glob(os.path.join(datasetdir, '*.png')))
    globs.extend(glob(os.path.join(os.path.join(basedir, 'db', 'lamps', '0000005'), '*.png')))

    overview_image = np.empty((0, 6 * 128, 3), dtype=np.uint8)
    image_row = []
    row_stacked = 0
    image_listed = 0
    for image_path in sorted(globs):
        basename = os.path.basename(image_path)

        json_path = image_path + '.json'
        with open(json_path, 'r') as f:
            bboxes = json.load(f)['bboxes_x_xx_y_yy']

        for bbox in bboxes:
            image_listed += 1
            if image_listed > 9 and image_listed % 4 != 0:
                continue

            x, xx, y, yy = bbox

            # inspection
            img = cv2.imread(image_path)[y:yy, x:xx].astype(np.float32)
            img = cv2.resize(img, (128, 128))
            img = img - img.min()
            img = img / img.max()
            img = img * 255
            img = img.astype(np.uint8)

            image_row.append(img)

            if len(image_row) == 6:
                stacked = np.concatenate(image_row, axis=1)
                overview_image = np.concatenate((overview_image, stacked))

                image_row = []
                row_stacked += 1

            if row_stacked == 3:
                break
        if row_stacked == 3:
            break

    cv2.imwrite(os.path.join(basedir, 'results', 'lamps_overview.png'), overview_image)
