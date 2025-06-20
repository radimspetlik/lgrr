import os
import cv2
import numpy as np
import json


def solve_x_gt_xx_or_y_gt_yy():
    global data
    modify = False
    with open(json_path, 'r') as f:
        data = json.load(f)
        for bbox_x_xx_y_yy in data['bboxes_x_xx_y_yy']:
            x, xx, y, yy, modify = format_bbox_x_xx_y_yy(bbox_x_xx_y_yy, modify)
            new_bboxes.append([x, xx, y, yy])
    if modify:
        with open(json_path, 'w') as f:
            json.dump({'bboxes_x_xx_y_yy': new_bboxes}, f)


def format_bbox_x_xx_y_yy(bbox_x_xx_y_yy, modify=None):
    x, xx, y, yy = bbox_x_xx_y_yy
    if x > xx:
        xt = x
        x = xx
        xx = xt
        modify = True
        # print(json_path)
    if y > yy:
        yt = y
        y = yy
        yy = yt
        modify = True
        # print(json_path)
    if modify is None:
        return x, xx, y, yy

    return x, xx, y, yy, modify


if __name__ == '__main__':
    dataset = 'boxes'
    dataset = 'lamps'
    dataset = 'SIRR_dataset_homogr'
    dataset = 'dust_removal'

    print('Do this on local, first:')
    print("rsync -avz --progress rci:'~/data/reflections/db/{}/' /mnt/c/Users/jarmi/data/reflections/db/{}/".format(dataset, dataset))
    print('Do this on local, after finished:')
    print("rsync -avz --progress /mnt/c/Users/jarmi/data/reflections/db/{}/ rci:'~/data/reflections/db/{}/'".format(dataset, dataset))
    print('and this on rci:')
    print("rsync -acv --progress /home/spetlrad/data/reflections/db/{}/ spetlrad@ritz.felk.cvut.cz:'~/datagrid/reflections/db/{}/'".format(dataset, dataset))

    drawing = False  # true if mouse is pressed
    ix, iy = -1, -1
    bboxes_x_xx_y_yy = []

    image_scaler = 0.5
    # exit()
    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode, imr, image_scaler

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            bboxes_x_xx_y_yy.append([int(x / image_scaler), x, int(y / image_scaler), y])

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                imr = resized.copy()
                cv2.rectangle(imr, (ix, iy), (x, y), (0, 255, 0), 3)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            imr = resized.copy()
            cv2.rectangle(imr, (ix, iy), (x, y), (0, 255, 0), 2)
            bboxes_x_xx_y_yy[-1][1] = int(x / image_scaler)
            bboxes_x_xx_y_yy[-1][3] = int(y / image_scaler)

            bboxes_x_xx_y_yy[-1] = format_bbox_x_xx_y_yy(bboxes_x_xx_y_yy[-1])

            print(bboxes_x_xx_y_yy)

    img = np.zeros((512, 512, 3), np.uint8)
    imr = np.zeros((512, 512, 3), np.uint8)
    resized = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    cv2.moveWindow("image", 20, 20)

    skip_existing = False

    dataset_path = os.path.join('c:', os.sep, 'Users', 'jarmi', 'data', 'reflections', 'db', '{}'.format(dataset))
    for userdir in os.listdir(dataset_path):
        userdir_path = os.path.join(dataset_path, userdir)
        if not os.path.isdir(userdir_path):
            continue
        for lamp_image_basename in os.listdir(userdir_path):
            if 'json' in lamp_image_basename:
                continue
            # if 'src' not in lamp_image_basename:
            #     continue
            json_path = os.path.join(userdir_path, lamp_image_basename + '.json')
            if os.path.exists(json_path):
                if skip_existing:
                    print('Skipping {}'.format(json_path))
                    continue
                new_bboxes = []
                # solve_x_gt_xx_or_y_gt_yy()

            bboxes_x_xx_y_yy = []
            if ('png' not in lamp_image_basename and 'jpg' not in lamp_image_basename) or 'json' in lamp_image_basename:
                continue

            print(lamp_image_basename)
            img = cv2.imread(os.path.join(userdir_path, lamp_image_basename))

            scale_percent = image_scaler * 100  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            imr = resized.copy()

            json_path = os.path.join(userdir_path, lamp_image_basename + '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for bbox_x_xx_y_yy_l in data['bboxes_x_xx_y_yy']:
                        ix, x, iy, y = bbox_x_xx_y_yy_l[:4]
                        bboxes_x_xx_y_yy.append(bbox_x_xx_y_yy_l)
                        cv2.rectangle(imr, (int(ix * image_scaler), int(iy * image_scaler)), (int(x * image_scaler), int(y * image_scaler)), (0, 255, 0), 2)

            while 1:
                imrv = imr.copy().astype(np.float) #+ 100
                imrv[imrv > 255] = 255
                cv2.imshow('image', imrv.astype(np.uint8))
                k = cv2.waitKey(1) & 0xFF
                if k == 115: #s
                    with open(json_path, 'w') as f:
                        json.dump({'bboxes_x_xx_y_yy': bboxes_x_xx_y_yy}, f)
                elif k == 32: # space
                    with open(json_path, 'w') as f:
                        json.dump({'bboxes_x_xx_y_yy': bboxes_x_xx_y_yy}, f)
                    break
                elif k == 100: #d
                    bboxes_x_xx_y_yy = []
                elif k != 255:
                    print(k)