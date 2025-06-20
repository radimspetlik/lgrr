if __name__ == '__main__':
    import cv2
    import os
    from PIL import Image
    import numpy as np

    impath = os.path.join('/', os.sep, 'home.dokt', 'spetlrad', 'datagrid', 'reflections', 'experiments', 'images', '0_val_0194_002.png')
    img = cv2.imread(impath)

    mask = img[:, 128:256]
    output = img[:, 384:512]

    mask_out = mask.astype(np.uint32) + output.astype(np.uint32)
    mask_out[mask_out > 255] = 255
    mask_out = mask_out.astype(np.uint8)

    cv2.imwrite(impath + '_e.png', np.concatenate((mask_out, output), axis=1))