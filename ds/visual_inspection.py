'''Pytorch inspection procedure of the lgrr

Usage:
  %(prog)s  [--no-cuda]
            [--json-path=<path>]

  %(prog)s (--help | -h)

Options:
    --json-path=<path>                      path to the json config file
    --no-cuda                               disables CUDA
    -h, --help                              should be help but none is given

See '%(prog)s --help' for more information.
'''


def inspect(model_to_inspect, loader, mask_to_check, visual_inspection_dir, filename):
    model_to_inspect.eval()
    with torch.no_grad():
        for batch_idx, (data, usr_mask, target, loss_mask) in enumerate(loader):
            batch_output = None
            if cuda:
                data, usr_mask, target = data.cuda(), usr_mask.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)

            for iteration in range(1):
                batch_output = model_to_inspect(
                    torch.cat((data, usr_mask), dim=1) if batch_output is None else torch.cat((batch_output, usr_mask), dim=1))

                mask_pred = batch_output[:, 3:]
                batch_output = batch_output[:, :3]

                plot_with_pyplot(0, data, usr_mask, target, batch_output, loss_mask, mask_pred, visual_inspection_dir,
                                 filename)

            del data, batch_output


from nn.trn.utils import plot_with_pyplot


def empty_mask_0_255_HWC(config):
    return np.zeros((config['model']['input_width_height'], config['model']['input_width_height'], 1), dtype=np.uint8)


def circle_mask_0_255_HWC(config):
    import cv2
    img = cv2.imread(os.path.join('/', os.sep, 'home.dokt', 'spetlrad', 'datagrid', 'reflections', '2_mask.png'))
    img[img < 255] = 0
    return img


def unknown_mask_0_255_HWC(config):
    import cv2
    img = cv2.imread(os.path.join('/', os.sep, 'home.dokt', 'spetlrad', 'datagrid', 'reflections', 'u_mask.png'))
    return img


if __name__ == '__main__':
    import sys
    from docopt import docopt
    import random
    import time
    import json
    import torch
    import os
    import logging
    import numpy as np

    prog = os.path.basename(sys.argv[0])
    completions = dict(
        prog=prog,
    )
    args = docopt(
        __doc__ % completions,
        argv=sys.argv[1:],
        version='RADIM',
    )
    args['--skip-tst-dataset'] = True
    args['--scratch-dir'] = None

    __logging_format__ = '[%(levelname)s]%(message)s'
    logFormatter = logging.Formatter(__logging_format__)

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    cuda = not bool(args['--no-cuda']) and torch.cuda.is_available()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    with open(args['--json-path']) as config_buffer:
        config = json.loads(config_buffer.read())

    if not os.path.isfile(config['trn']['continue_model']):
        dirname = os.path.dirname(config['trn']['continue_model'])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        print('Missing model file!')
        exit(0)

    from nn.trn.trn_pytorch import load_model
    model = load_model(cuda, config, logger)

    logger.info(config)
    from nn.trn.dataset import SingleImageDataset

    masks_to_check = [empty_mask_0_255_HWC]

    datasetdir = config['trn']['dataset_directory']
    visual_inspection_dir = config['trn']['visual_inspection_directory']

    if not os.path.isdir(visual_inspection_dir):
        os.makedirs(visual_inspection_dir)
    from glob import glob

    real_world_image_paths = []
    real_world_image_boxes = []
    real_world_image_counters = []
    globs = glob(os.path.join(datasetdir, '*', '*.jpg'))
    globs.extend(glob(os.path.join(datasetdir, '*', '*.jpeg')))
    globs.extend(glob(os.path.join(datasetdir, '*', '*.png')))
    for filepath in globs:
        if not os.path.isfile(filepath + '.json'):
            continue
        with open(filepath + '.json', 'r') as f:
            bboxes = json.load(f)['bboxes_x_xx_y_yy']
        if len(bboxes) == 0:
            continue
        for bbox_id, bbox in enumerate(bboxes):
            real_world_image_paths.append(filepath)
            real_world_image_boxes.append([bbox[:4]])
            real_world_image_counters.append('{:02d}'.format(bbox_id))

    for filepath, filepath_postfix, bboxes in zip(real_world_image_paths, real_world_image_counters, real_world_image_boxes):
        for mask_to_check in masks_to_check:
            logger.info(' Working on {}.'.format(filepath))
            sid_ds = SingleImageDataset(config, filepath,
                                        bboxes,
                                        mask_function_0_255_HWC=mask_to_check)
            visual_loader = torch.utils.data.DataLoader(sid_ds, batch_size=config['trn']['batch_size'], shuffle=False, num_workers=1)

            start_time = time.time()
            inspect(model, visual_loader, mask_to_check, visual_inspection_dir,
                    str(config['model']['input_width_height']) + 'px_' + filepath_postfix + '_' + os.path.basename(filepath))
            logger.info(' Finished in %.0f s.' % (time.time() - start_time))

    logger.info(' Succesfully finished...')
