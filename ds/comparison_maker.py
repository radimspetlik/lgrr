import os
from glob import glob
import cv2
import json
import numpy as np

if __name__ == '__main__':
    basedir = os.path.join('/', os.sep, 'home.dokt', 'spetlrad', 'datagrid', 'reflections')

    paths = {}
    paths['voloda'] = os.path.join('visual_inspection', 'SIRR_dataset_homogr')
    paths['hifill'] = os.path.join('related', 'sample-imageinpainting-HiFill', 'GPU_CPU', 'results')
    paths['ibcln'] = os.path.join('related', 'IBCLN', 'results', 'IBCLN', 'test_final', 'images')

    available = ['003_00', '005_00', '007_00', '009_01', '011_02', '014_00', '016_01', '019_00', '020_00', '021_00', '021_07',
                 '022_00', '022_01', '025_00', '026_00', '029_00', '030_01', '031_00', '033_03', '055_00', '057_01', '058_00',
                 '061_00', '063_00']
    failure_cases = ['006_02', '028_01', #'033_01',
                     '037_02', '049_00', '053_00']

    qualitative_comparison = ['003_00', '007_00', '009_01', '011_02', '014_00', '016_01', '019_00', '021_07', '025_00', '029_06', '030_01', '063_00']

    images = {}
    for dataset_name, dataset_path in paths.items():
        datasetdir = os.path.join(basedir, dataset_path)
        globs = glob(os.path.join(datasetdir, '*.jpg'))
        globs.extend(glob(os.path.join(datasetdir, '*.jpeg')))
        globs.extend(glob(os.path.join(datasetdir, '*.png')))

        for image_path in sorted(globs):
            basename = os.path.basename(image_path).split('_')[0]
            if dataset_name == 'voloda':
                basename = os.path.basename(image_path).split('_')[2]
            if basename not in images:
                images[basename] = {}
            if dataset_name not in images[basename]:
                images[basename][dataset_name] = []

            if dataset_name == 'ibcln':
                if 'Ts_03' in image_path:
                    images[basename][dataset_name].append(image_path)
            else:
                images[basename][dataset_name].append(image_path)

    qualitative_image = np.empty((128 * 4, 0, 3), dtype=np.uint8)
    failure_image = np.empty((128 * 3, 0, 3), dtype=np.uint8)
    for image_basename, image_paths in images.items():
        if 'voloda' not in image_paths:
            print('missing {} for voloda'.format(image_basename))
            continue

        my_paths = image_paths['voloda']
        hifill_path = image_paths['hifill'][0]
        ibcln_path = image_paths['ibcln'][0]

        json_path = os.path.join(basedir, 'db', 'SIRR_dataset_homogr', 'voloda', '{}_src.jpg.json'.format(image_basename))
        with open(json_path, 'r') as f:
            bboxes = json.load(f)['bboxes_x_xx_y_yy']

        for my_path_id, my_path in enumerate(my_paths):
            x, xx, y, yy = bboxes[my_path_id][:4]

            # inspection
            img = cv2.imread(my_path)
            img_in = img[:128, :128]
            img_t = img[:128, 2 * 128:3 * 128]
            img_out = img[:128, 3 * 128:4 * 128]
            img_pred_mask = img[:128, -128:]

            # hifill
            img_hifill_o = cv2.imread(hifill_path)
            img_hifill = img_hifill_o[y:yy, x:xx]
            if img_hifill.size == 0:
                print('hifill not enough data {} {} {} {}, {}'.format(x, xx, y, yy, hifill_path))
                continue
            img_hifill = cv2.resize(img_hifill, (128, 128))

            h, w, c = img_hifill_o.shape
            if w > 2500:
                x, xx, y, yy = x // 2, xx // 2, y // 2, yy // 2

            img_ibcln = cv2.imread(ibcln_path)[y:yy, x:xx]
            if img_ibcln.size == 0:
                print('ibcln not enough data {} {} {} {}, {} {}'.format(x, xx, y, yy, img_ibcln, os.path.basename(hifill_path)))
                continue
            img_ibcln = cv2.resize(img_ibcln, (128, 128))

            stacked = np.concatenate((img_in, img_out, img_hifill, img_ibcln), axis=0)
            cv2.imwrite(os.path.join(basedir, 'results', '{}_{:02d}.png'.format(image_basename, my_path_id)), stacked)

            if '{}_{:02d}'.format(image_basename, my_path_id) in qualitative_comparison:
                qualitative_image = np.concatenate((qualitative_image, stacked), axis=1)

            if '{}_{:02d}'.format(image_basename, my_path_id) in failure_cases:
                stacked = np.concatenate((img_in, img_t, img_out), axis=0)
                failure_image = np.concatenate((failure_image, stacked), axis=1)

    cv2.imwrite(os.path.join(basedir, 'results', 'qualitative.png'), qualitative_image)
    cv2.imwrite(os.path.join(basedir, 'results', 'failure.png'), failure_image)

