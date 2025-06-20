'''Pytorch inspection procedure of the BLABLA

Usage:
  %(prog)s  [--no-cuda]
            [--json-path=<path>]

  %(prog)s (--help | -h)

Options:
    --json-path=<path>                      path to the json config file
    --no-cuda                               disables CUDA training
    -h, --help                              should be help but none is given

See '%(prog)s --help' for more information.
'''
import datetime


def test(model_to_inspect, loader):
    model_to_inspect.eval()
    errs = []
    with torch.no_grad():
        for batch_idx, (data, usr_mask, target, loss_mask, loss_mask_u) in enumerate(loader):
            if cuda:
                data, usr_mask, target = data.cuda(), usr_mask.cuda(), target.cuda()

            loss_mask_u = torch.cat((loss_mask_u,loss_mask_u,loss_mask_u), dim=1)
            mask_bool = loss_mask_u > 0.5

            batch_output = model_to_inspect(torch.cat((data, usr_mask), dim=1))

            mask_pred = batch_output[:, 3:]
            batch_output = batch_output[:, :3]

            errs.append(float(torch.abs(UGIRRDataset.unnormalize_torch(batch_output)[mask_bool] - UGIRRDataset.unnormalize_torch(target)[mask_bool]).mean()))

            # data_img = UGIRRDataset.unnormalize_torch(data).cpu().numpy()
            # batch_output_img = UGIRRDataset.unnormalize_torch(batch_output).cpu().numpy()
            # target_img = UGIRRDataset.unnormalize_torch(target).cpu().numpy()
            # joined_imgs = np.concatenate((data_img, batch_output_img, target_img), axis=-1)[:, ::-1]
            # plot_show_maybe_store(joined_imgs[0],
            #                       filename='{}'.format(batch_idx),
            #                       dir=os.path.join(experiments_directory, 'images'))

            del data, mask_pred, batch_output

    mae = float(np.array(errs).mean())
    logger.info('MAE is: {}'.format(mae))

    return mae


if __name__ == '__main__':
    import sys
    from docopt import docopt
    from nn.trn.utils import plot_with_pyplot, plot_show_maybe_store
    from torch.autograd import Variable
    import matplotlib
    import random
    import time
    import json
    import torch
    import os
    import logging
    import numpy as np
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as UGIRRDataset
    import re
    # matplotlib.use('agg')
    from matplotlib import pyplot as plt

    prog = os.path.basename(sys.argv[0])
    completions = dict(
        prog=prog,
    )
    args = docopt(
        __doc__ % completions,
        argv=sys.argv[1:],
        version='RADIM',
    )

    models_to_test = [
        '/home/spetlrad/data/reflections/experiments/2020-10-23_14-21-25-536690_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001810_val_avg-loss-best',
        '/home/spetlrad/data/reflections/experiments/2020-10-23_14-25-24-092717_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_l2=3E-01_dxy=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001728_val_avg-loss-best',
        '/home/spetlrad/data/reflections/experiments/2020-10-23_14-30-24-062617_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_l2=3E-01_dxy=1E+00_dx2=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001867_val_avg-loss-best',
        '/home/spetlrad/data/reflections/experiments/2020-10-23_15-50-16-176055_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_l2=3E-01_dxy=1E+00_dx2=1E+00_dx3=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001810_val_avg-loss-best',
        '/home/spetlrad/data/reflections/experiments/2020-10-23_15-58-25-177987_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_l2=3E-01_dxy=1E+00_dx2=1E+00_dx3=1E+00_dx4=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001959_val_avg-loss-best',
        '/home/spetlrad/data/reflections/experiments/2020-10-23_16-10-01-425404_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_l2=3E-01_dxy=1E+00_dx2=1E+00_dx3=1E+00_dx4=1E+00_dx5=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001694_val_avg-loss-best',
        '/home/spetlrad/data/reflections/experiments/2020-10-23_20-18-59-261980_corr_a=UpscaleUGSIIRR221_ic=4_o=adamw_wd=1E-03_lr=1E-03_bs=50_c=l1=1E+00_l2=3E-01_dxy=1E+00_dx2=1E+00_dx3=1E+00_dx4=1E+00_dx5=1E+00_dx6=1E+00_fgw=1E+00_bgw=1E+00_lc=100_fm=0_bof-satone_0.0-1.0_lmt=prcs_grt=rea_aug-ref_w_aug_4/epoch=001811_val_avg-loss-best'
    ]

    missing_model = False
    for model_to_test in models_to_test:
        model_to_test = model_to_test.replace('/home/spetlrad/data', '/mnt/datagrid/personal/spetlrad')
        if not os.path.isfile(model_to_test):
            missing_model = True
            dirname = os.path.dirname(model_to_test)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            rci_path = model_to_test.replace('/mnt/datagrid/personal/spetlrad', '/home/spetlrad/data')
            print('{}'.format(' '.join(['rsync', '-azv', '--progress',
                                        "spetlrad@login.rci.cvut.cz:'" + rci_path + "'",
                                        "'" + dirname + "'"])))
    if missing_model:
        exit()

    with open(args['--json-path']) as config_buffer:
        config = json.loads(config_buffer.read())

    experiments_directory = config['trn']['experiments_directory']
    args['--skip-tst-dataset'] = True
    args['--scratch-dir'] = None
    cuda = not bool(args['--no-cuda']) and torch.cuda.is_available()
    config['model']['input_width_height'] = 128
    batch_size = int(config['trn']['batch_size'])

    __logging_format__ = '[%(levelname)s]%(message)s'
    logFormatter = logging.Formatter(__logging_format__)

    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as dataset

    dataset_filenames = sorted(os.listdir(config['trn']['dataset_directory']))
    dataset_filenames = [os.path.splitext(dataset_filename)[0] for dataset_filename in dataset_filenames]
    idxs = np.random.permutation(len(dataset_filenames))[:1000]
    dataset_filenames = [dataset_filenames[i] for i in idxs]
    tst_ds = dataset(config, dataset_filenames, is_trn=False, cache_dir=None, ds_size=1000)
    tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=10)

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    models = {}
    for continue_model in models_to_test:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        config['trn']['continue_model_partly'] = ''
        config['trn']['continue_model'] = continue_model
        config['trn']['continue_model'] = config['trn']['continue_model'].replace('/home/spetlrad/data',
                                                                                  '/mnt/datagrid/personal/spetlrad')

        config['model']['architecture'] = re.search('_a=([a-zA-Z]+[0-9]+)_', config['trn']['continue_model']).groups(0)[0]

        from nn.trn.trn_pytorch import load_model, prepare_summary

        model = load_model(cuda, config, logger)

        logger.info(config)

        basename = '{}'.format('_'.join(os.path.basename(os.path.split(config['trn']['continue_model'])[0]).split('_')[:2]))
        summary, summary_filepath = prepare_summary(experiments_directory, basename)

        fh = logging.FileHandler(os.path.join(summary_filepath, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '.log'))
        logger.addHandler(fh)

        logger.info(" sys path: {}".format(sys.path))
        logger.info(' %s' % basename)

        logger.info(json.dumps(config))

        start_time = time.time()
        mae = test(model, tst_loader)
        models[basename] = mae
        logger.info(' Finished in %.0f s.' % (time.time() - start_time))

        logger.info(' Succesfully finished...')

        print('Do this on local:')
        print("rsync -avz --progress ritz:'~/datagrid/reflections/visual_inspection' /mnt/c/Users/jarmi/data/reflections/")

    print(models)
    with open(os.path.join(experiments_directory, 'maes_corr.json'), 'w') as f:
        json.dump(models, f)