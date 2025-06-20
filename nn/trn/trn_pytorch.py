'''Pytorch training procedure of the BLABLA

Usage:
  %(prog)s  [--no-cuda]
            [--json-path=<path>]
            [--skip-tst-dataset]
            [--scratch-dir=<path>]
            [--seed=<int>]
            [--augment]
            [--clear-caches]

  %(prog)s (--help | -h)

Options:
    --seed=<int>                            random seed [default: 1]
    --json-path=<path>                      path to the json config file
    --scratch-dir=<path>
    --skip-tst-dataset                      skips testing on the testing dataset
    --no-cuda                               disables CUDA training
    --augment                               turns on augmentation
    --clear-caches                          clears rci local caches
    -h, --help                              should be help but none is given

See '%(prog)s --help' for more information.
'''

import torch
import os
import sys
from nn.trn.losses import selective_loss, LossName
from nn.trn.utils import store_image_to_tensorboard, BestModel, HistogramsRecorder
from torch.utils.tensorboard import SummaryWriter


# def magic_loss(batch_target_img, batch_mask_img, batch_output, config):
#     loss = get_loss(config['trn']['criterions'])
#     return loss(batch_target_img, batch_mask_img, batch_output, config)


class OptimizerStub:
    param_groups = [{"params": []}]

    def zero_grad(self):
        pass

    def step(self):
        pass


def compute_and_store_correlations(model, model_optimizer, trn_functions, summary, training_epoch):
    keys = list(filter(lambda k: 'fg' in k and 'first_order' not in k, trn_functions.keys()))
    function_idxs = np.arange(len(keys))

    model_parameters = []
    model_parameter_names = []
    for name, param in model.named_parameters():
        model_parameters.append(param)
        model_parameter_names.append(name)

    from itertools import combinations
    comb = combinations(function_idxs, 2)
    grads = {}
    correlations = {}
    for combination in comb:
        key_0, key_1 = keys[combination[0]], keys[combination[1]]
        if key_0 not in grads:
            model.zero_grad()
            model_optimizer.zero_grad()
            map(lambda p: p.retain_grad(), model_parameters)
            trn_functions[key_0].backward(retain_graph=True)
            grads[key_0] = list(map(lambda p: p.grad.clone(), filter(lambda p: p.requires_grad, model_parameters)))

        if key_1 not in grads:
            model.zero_grad()
            model_optimizer.zero_grad()
            map(lambda p: p.retain_grad(), model_parameters)
            trn_functions[key_1].backward(retain_graph=True)
            grads[key_1] = list(map(lambda p: p.grad.clone(), filter(lambda p: p.requires_grad, model_parameters)))

        means_key_0 = list(map(torch.mean, grads[key_0]))
        means_key_1 = list(map(torch.mean, grads[key_1]))

        grad_norm_key_0 = [grads[key_0][t_id] - means_key_0[t_id] for t_id in range(len(grads[key_0]))]
        stds_key_0 = list(map(lambda t: torch.sqrt((t ** 2).mean()), grad_norm_key_0))
        grad_norm_key_1 = [grads[key_1][t_id] - means_key_1[t_id] for t_id in range(len(grads[key_1]))]
        stds_key_1 = list(map(lambda t: torch.sqrt((t ** 2).mean()), grad_norm_key_1))

        stds_mults = [stds_key_0[s_id] * stds_key_1[s_id] for s_id in range(len(stds_key_0))]

        corrs = [(grad_norm_key_0[g_id] * grad_norm_key_1[g_id]).mean() / stds_mults[g_id] for g_id in range(len(grad_norm_key_0))]

        correlations[key_0 + '-' + key_1] = corrs

    del grads

    model.zero_grad()
    model_optimizer.zero_grad()

    for correlations_name, correlations_list in correlations.items():
        summary.add_scalar('trn_abs_pcorr/{}_median'.format(correlations_name),
                           torch.median(torch.abs(torch.tensor(correlations_list))), training_epoch)
        summary.add_scalar('trn_abs_pcorr/{}_mean'.format(correlations_name), torch.mean(torch.abs(torch.tensor(correlations_list))),
                           training_epoch)
        # for parameter_idx in range(len(correlations_list)):
        #     summary.add_scalar('trn_corr/{}_{}'.format(correlations_name, model_parameter_names[parameter_idx]), correlations_list[parameter_idx])

    return correlations


def print_int(model):
    for n, p in model.named_parameters():
        if 'gamma' in n:
            print(p.grad)


def trn(model_to_train, model_optimizer, trn_epoch, trn_ds, config):
    start = time.time()
    ocn = config['model']['output_channel_num']

    model_to_train.train()
    model_to_train.zero_grad()

    trn_ds.shuffle()

    trn_losses = torch.empty((0, 4))
    if cuda: trn_losses = trn_losses.cuda()
    for batch_idx, (img, usr_mask, target, loss_mask, loss_mask_u) in enumerate(trn_loader):
        if trn_ds.augmentation is not None:
            trn_ds.augmentation.to_deterministic()
        if hasattr(trn_ds, 'reflection_augmentation') and trn_ds.reflection_augmentation is not None:
            trn_ds.reflection_augmentation.to_deterministic()
        if cuda:
            img, usr_mask, target, loss_mask, loss_mask_u = \
                img.cuda(), usr_mask.cuda(), target.cuda(), loss_mask.cuda(), loss_mask_u.cuda()

        batch_output = model_to_train(torch.cat((img, usr_mask), dim=1))

        mask_pred = batch_output[:, ocn:]
        batch_output = batch_output[:, :ocn]

        model_optimizer.zero_grad()
        model_to_train.zero_grad()

        trn_loss, trn_functions = selective_loss(target, loss_mask, batch_output, config)

        mask_pred_loss = ((mask_pred - loss_mask_u) ** 2).mean(dim=(1, 2, 3)).mean()

        trn_loss = trn_loss + mask_pred_loss

        trn_loss.backward()
        model_optimizer.step()

        trn_losses = torch.cat((trn_losses, torch.cat((trn_loss.clone().view(-1, 1),
                                                       trn_functions['loss_l1_bg'].view(-1, 1),
                                                       trn_functions['loss_l1_fg'].view(-1, 1),
                                                       torch.sqrt(mask_pred_loss).clone().view(-1, 1)
                                                       ), dim=1)), dim=0)

        if batch_idx % 100 == 0 and trn_epoch % config['trn']['store_images_every_Xth_epoch'] == 0:
            store_image_to_tensorboard('trn', trn_epoch, batch_idx, batch_output, img, usr_mask, target, loss_mask, loss_mask_u,
                                       mask_pred, summary, config, datetime_id)

        del img, usr_mask, batch_output, trn_loss, trn_functions
    end = time.time()

    return trn_losses.data.cpu().numpy(), end - start


def create_lr_loss(config, diffs_ratio, trn_loss):
    lr_loss = torch.tensor([0]).float().cuda()
    if 'l2sq' in config['trn']['lr_criterions']:
        lr_loss += config['trn']['lr_criterions']['l2sq'] * (diffs_ratio - config['trn']['target_l1_ratio']) ** 2
    if 'trn' in config['trn']['lr_criterions']:
        lr_loss += config['trn']['lr_criterions']['trn'] * trn_loss

    return lr_loss


def validate(model_to_validate, loader, training_epoch, config, log_prefix='val', store_first_N_images=5):
    ocn = config['model']['output_channel_num']
    stored_images = 0

    ratios = torch.empty((0, 1))

    start = time.time()
    model_to_validate.zero_grad()
    model_to_validate.eval()
    with torch.no_grad():
        val_losses = torch.empty((0, 4))
        if cuda: val_losses = val_losses.cuda()
        for batch_idx, (data, usr_mask, target, loss_mask, loss_mask_u) in enumerate(loader):
            if cuda:
                data, usr_mask, target, loss_mask, ratios, loss_mask_u = \
                    data.cuda(), usr_mask.cuda(), target.cuda(), loss_mask.cuda(), ratios.cuda(), loss_mask_u.cuda()

            batch_output = model_to_validate(torch.cat((data, usr_mask), dim=1))
            mask_pred = batch_output[:, ocn:]
            batch_output = batch_output[:, :ocn]
            mask_pred_loss = ((mask_pred - loss_mask_u) ** 2).mean(dim=(1, 2, 3)).mean()

            val_loss, val_functions = selective_loss(target, loss_mask, batch_output, config)
            val_losses = torch.cat((val_losses, torch.cat((val_loss.view(-1, 1),
                                                           val_functions['loss_l1_bg'].view(-1, 1),
                                                           val_functions['loss_l1_fg'].view(-1, 1),
                                                           torch.sqrt(mask_pred_loss).view(-1, 1)), dim=1)), dim=0)

            if stored_images < store_first_N_images and training_epoch % config['trn']['store_images_every_Xth_epoch'] == 0:
                for image_idx in range(config['trn']['batch_size']):
                    if stored_images == store_first_N_images:
                        break
                    store_image_to_tensorboard(log_prefix, training_epoch, batch_idx, batch_output, data, usr_mask, target, loss_mask,
                                               loss_mask_u, mask_pred, summary, config, datetime_id, image_idx_to_store=image_idx,
                                               image_idx_to_log=stored_images)
                    stored_images += 1

            del data, usr_mask, val_loss, batch_output, val_functions

    end = time.time()

    if ratios.size()[0] > 0:
        summary.add_histogram('val/histogram', ratios, training_epoch)

    return val_losses.data.cpu().numpy(), end - start


def evaluate_model(model, model_name, losses, epoch_shift, experiments_directory, trn_ds, config):
    lr = float(config['trn']['learning_rate'])

    if config['trn']['optimizer'] == "adam":
        model_optimizer = optim.Adam(  # model_param_groups,
            map(lambda t: t[1],
                filter(lambda t: t[1].requires_grad and 'gamma' not in t[0] and 'kappa' not in t[0], model.named_parameters())),
            lr=lr, weight_decay=config['trn']['weight_decay'])
    elif config['trn']['optimizer'] == "adamw":
        model_optimizer = optim.AdamW(  # model_param_groups,
            map(lambda t: t[1],
                filter(lambda t: t[1].requires_grad and 'gamma' not in t[0] and 'kappa' not in t[0], model.named_parameters())),
            lr=lr, weight_decay=config['trn']['weight_decay'])
    elif config['trn']['optimizer'] == "sgd":
        model_optimizer = optim.SGD(
            map(lambda t: t[1], filter(lambda t: t[1].requires_grad and 'gamma' not in t[0], model.named_parameters())),
            lr=lr, momentum=float(config['trn']['momentum']), weight_decay=config['trn']['weight_decay'])
    else:
        raise NotImplementedError('I don\' t know this optimizer!')

    best_models = BestModel(['loss'])

    logger.info(' Learning with the model %s...' % model_name)

    val_losses, validation_time = validate(model, val_loader, epoch_shift, config)
    val_losses, val_losses_bg, val_losses_fg, val_mask_pred_l2 = \
        val_losses[:, 0], val_losses[:, 1], val_losses[:, 2], val_losses[:, 3]
    logger.info('[{:04d}][VAL] {:.6f}, {:.1f} ({:.0f}s) mask: {:.2f}'.format(0, val_losses.mean(),
                                                                             trn_ds.unnormalize_loss(val_losses_fg.mean()),
                                                                             validation_time, val_mask_pred_l2.mean()))
    for epoch in range(epoch_shift, int(config['trn']['epochs']) + 1):
        trn_losses, training_time = trn(model, model_optimizer, epoch, trn_ds, config)
        val_losses, validation_time = validate(model, val_loader, epoch, config)
        torch.cuda.empty_cache()

        trn_losses, trn_losses_bg, trn_losses_fg, trn_mask_pred_l2 = \
            trn_losses[:, 0], trn_losses[:, 1], trn_losses[:, 2], trn_losses[:, 3]
        val_losses, val_losses_bg, val_losses_fg, val_mask_pred_l2 = \
            val_losses[:, 0], val_losses[:, 1], val_losses[:, 2], val_losses[:, 3]

        logger.info('[{:04d}][TRN] {:.6f}, {:.1f} ({:.0f}s) mask: {:.2f}'.format(epoch, trn_losses.mean(),
                                                                                 trn_ds.unnormalize_loss(trn_losses_fg.mean()),
                                                                                 training_time, trn_mask_pred_l2.mean()))
        logger.info('[{:04d}][VAL] {:.6f}, {:.1f} ({:.0f}s) mask: {:.2f}'.format(epoch, val_losses.mean(),
                                                                                 trn_ds.unnormalize_loss(val_losses_fg.mean()),
                                                                                 validation_time, val_mask_pred_l2.mean()))

        losses[epoch, :] = [trn_losses.mean(), np.median(trn_losses), val_losses.mean(), np.median(val_losses)]

        summary.add_scalar('trn/loss_avg', losses[epoch, 0], epoch)
        if 'l1' in config['trn']['criterions'] and len(config['trn']['criterions']):
            summary.add_scalar('trn/loss_unnorm_avg', trn_ds.unnormalize_loss(losses[epoch, 0]), epoch)
        summary.add_scalar('trn/loss_bg_unnorm_avg', trn_ds.unnormalize_loss(trn_losses_bg.mean()), epoch)
        summary.add_scalar('trn/loss_under_mask_unnorm_avg', trn_ds.unnormalize_loss(trn_losses_fg.mean()), epoch)
        summary.add_scalar('trn/loss_med', losses[epoch, 1], epoch)
        summary.add_scalar('trn/loss_l2_mask_pred', trn_mask_pred_l2.mean(), epoch)

        summary.add_scalar('val/loss_avg', losses[epoch, 2], epoch)
        if 'l1' in config['trn']['criterions'] and len(config['trn']['criterions']):
            summary.add_scalar('val/loss_unnorm_avg', trn_ds.unnormalize_loss(losses[epoch, 2]), epoch)
        summary.add_scalar('val/loss_bg_unnorm_avg', trn_ds.unnormalize_loss(val_losses_bg.mean()), epoch)
        summary.add_scalar('val/loss_under_mask_unnorm_avg', trn_ds.unnormalize_loss(val_losses_fg.mean()), epoch)
        summary.add_scalar('val/loss_med', losses[epoch, 3], epoch)
        summary.add_scalar('val/loss_l2_mask_pred', val_mask_pred_l2.mean(), epoch)

        if epoch > 0:
            if losses[epoch, 2] < losses[:epoch, 2].min():
                if not os.path.isdir(os.path.join(experiments_directory, model_name)):
                    os.mkdir(os.path.join(experiments_directory, model_name))
                model_filepath = os.path.join(experiments_directory, model_name, 'epoch=%06d_val_avg-loss-best' % epoch)
                torch.save(model.state_dict(), model_filepath)
                best_models.store_path('loss', model_filepath, epoch, losses[epoch, 2])

    if best_models.get_best_model_epoch('loss') is not None:
        logger.info(' Restoring model. Epoch: {} loss: {:.5f} filepath: {}'.format(best_models.get_best_model_epoch('loss'),
                                                                                   best_models.get_best_model_value('loss'),
                                                                                   best_models.get_best_model_path('loss')))
        model.load_state_dict(torch.load(best_models.get_best_model_path('loss')))

    tst_losses, tst_time = validate(model, tst_loader, epoch_shift, config, log_prefix='best_val', store_first_N_images=25)
    tst_losses, tst_losses_bg, tst_losses_under_mask = tst_losses[:, 0], tst_losses[:, 1], tst_losses[:, 2]
    logger.info('[TST] {:.6f}, {:.2f} ({:.1f} s)'.format(tst_losses.mean(), tst_losses.var(), tst_time))

    return losses


def prepare_summary(experiments_directory, basename):
    summary_filepath = os.path.join(experiments_directory, 'tensorboard', basename)
    # if os.path.exists(summary_filepath):
    #     raise FileExistsError('Are you sure you want to overwrite %s ?' % summary_filepath)
    summary = SummaryWriter(summary_filepath)

    return summary, summary_filepath


def prepare_loaders(config, logger, args):
    dataset_filenames = sorted(os.listdir(config['trn']['coco_train2017_directory']))
    dataset_filenames = [dataset_filename for dataset_filename in dataset_filenames if '_sep' not in dataset_filename]

    trn_sample_size = int(len(dataset_filenames) * 0.9)

    trn_image_filenames = dataset_filenames[:trn_sample_size]
    val_image_filenames = dataset_filenames[trn_sample_size:]

    logger.info(' There is {:d} trn, {:d} val, and {:d} tst instances.'.format(len(trn_image_filenames),
                                                                               len(val_image_filenames),
                                                                               0))
    cache_dir = args['--scratch-dir']
    if cache_dir is not None and not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            logger.warning("Can't make the cache dir, no space left on device!")
            cache_dir = None

    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as dataset
    trn_ds = dataset(config, trn_image_filenames, is_trn=True, cache_dir=cache_dir, ds_size=1000)
    val_ds = dataset(config, val_image_filenames, is_trn=False, cache_dir=cache_dir, ds_size=200)

    if not args['--skip-tst-dataset']:
        tst_ds = dataset(config, None, is_trn=False)
    else:
        logger.warning(' Using val instead of tst!')
        tst_ds = val_ds

    epochs = int(config['trn']['epochs'])
    batch_size = int(config['trn']['batch_size'])
    debug = True
    if sys.gettrace() is None:
        debug = False
    trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=12 if not debug else 0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=6 if not debug else 0)
    tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return trn_loader, val_loader, tst_loader, trn_ds, val_ds, tst_ds


def load_models_prepare_losses_file(experiments_directory, config, basename, logger, logFormatter, cuda, trn_loader, summary):
    epoch_shift = 0
    datetime_id = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))
    model_name = datetime_id + '_' + basename.replace('_', '')

    model_path = os.path.join(experiments_directory, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    fileHandler = logging.FileHandler(os.path.join(model_path, 'log'))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    model = load_model(cuda, config, logger, trn_loader, summary)

    losses_filepath = os.path.join(model_path, model_name + '_losses.npy')
    losses = np.zeros((int(config['trn']['epochs']) + 1, 4))
    if os.path.isfile(losses_filepath):
        losses = np.load(losses_filepath)
        if losses.shape[0] != int(config['trn']['epochs']) + 1:
            oldl = losses
            losses = np.zeros((int(config['trn']['epochs']) + 1, 4))
            losses[:oldl.shape[0], :] = oldl

    return model_name, model, losses, epoch_shift, datetime_id


def load_discriminator_model(cuda, config, logger, summary=None):
    model = initialize_model(config['discriminator_model']['architecture'])
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    if cuda and torch.cuda.device_count() > 1:
        logger.info(" Let's use %d GPUs!" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    logger.info(" Discrimminator model loaded.")
    return model


def load_model(cuda, config, logger, trn_loader=None, summary=None):
    model = initialize_model(config['model']['architecture'], (cuda, config))

    continue_model_path = config['trn']['continue_model']
    continue_model_path_partly = config['trn']['continue_model_partly']
    if len(continue_model_path) > 0:
        logger.info(' Loading model %s' % continue_model_path)
        model.load_state_dict(torch.load(continue_model_path, map_location='cpu'))
    elif len(continue_model_path_partly) > 0:
        logger.info(' Partly loading model %s' % continue_model_path_partly)
        model.load_state_dict_partly(torch.load(continue_model_path_partly, map_location='cpu'))

    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if summary is not None:
        images, usr_mask, labels, loss_mask, loss_mask_u = next(iter(trn_loader))
        model.eval()
        summary.add_graph(model, torch.cat((images, usr_mask), dim=1).cuda())
        summary.flush()

    # Let's use more GPU!
    if cuda and torch.cuda.device_count() > 1:
        logger.info(" Let's use %d GPUs!" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    logger.info(" Models loaded.")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(' The model has %d parameters.' % pytorch_total_params)

    return model


def initialize_scheduler(scheduler_name, init_params=()):
    module = __import__('nn.trn.optim',
                        fromlist=[scheduler_name])
    class_ = getattr(module, scheduler_name)
    scheduler = class_(*init_params)
    return scheduler


def initialize_model(architecture_name, model_init_params=()):
    import re
    base_model_classname = re.sub(r'[0-9]+', '', architecture_name)
    module = __import__('nn.trn.models.%s' % (architecture_name),
                        fromlist=[base_model_classname])
    class_ = getattr(module, base_model_classname)
    model = class_(*model_init_params)
    return model


def clear_old_cache_dirs(logger):
    import glob, os
    scratch_dirs = glob.glob(os.path.join("/", 'data', 'temporary', 'reflections_*'))
    import shutil
    for scratch_dir in scratch_dirs:
        logger.warning('Removing old cache {}.'.format(scratch_dir))
        shutil.rmtree(scratch_dir, ignore_errors=True, onerror=None)


if __name__ == '__main__':
    import datetime
    import logging
    import numpy as np
    import sys
    from docopt import docopt
    import torch.optim as optim
    import matplotlib
    import random
    import time
    import json

    matplotlib.use('agg')
    prog = os.path.basename(sys.argv[0])
    completions = dict(
        prog=prog,
    )
    args = docopt(
        __doc__ % completions,
        argv=sys.argv[1:],
        version='RADIM',
    )

    __logging_format__ = '[%(levelname)s]%(message)s'
    logFormatter = logging.Formatter(__logging_format__)

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    if bool(args['--clear-caches']):
        clear_old_cache_dirs(logger)

    cuda = not bool(args['--no-cuda']) and torch.cuda.is_available()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    with open(args['--json-path']) as config_buffer:
        config = json.loads(config_buffer.read())

    additional_info = ''
    additional_info += '_gfs={}'.format(config['model']['gabor_filter_size']) if 'Gabor' in config['model']['architecture'] else ''
    additional_info += '_lln={}'.format(config['model']['lateral_layer_num']) if 'Gabor' in config['model']['architecture'] else ''
    additional_info += '_pln={}'.format(config['model']['parallel_layer_num']) if 'Gabor' in config['model']['architecture'] else ''
    additional_info += '_bof-{}'.format(config['trn']['blend_on_fly_mode']) if config['trn']['blend_on_fly'] else ''
    additional_info += '={}-{}-{}'.format(*config['trn']['blend_on_fly_reflection_color']) if config['trn']['blend_on_fly'] == 'set' or \
                                                                                              config['trn'][
                                                                                                  'blend_on_fly'] == 'add' else ''
    additional_info += '_{:1.1f}-{:1.1f}'.format(config['trn']['blend_alpha_min'], config['trn']['blend_alpha_max']) if config['trn'][
        'blend_on_fly'] else ''
    additional_info += '_lmt={}'.format(config['trn']['loss_mask_type'])
    additional_info += '_lmks={}'.format(config['trn']['loss_mask_kernel_size']) if config['trn'][
                                                                                        'loss_mask_type'] == 'gaussian' else ''
    additional_info += '_hl={}'.format(config['model']['head_length']) if config['model'].get('head_length', None) is not None else ''
    from nn.trn.dataset import GeneratedReflectionType

    additional_info += '_grt={}'.format(config['trn']['reflection'])
    additional_info += '_grs={}'.format(config['trn']['generate_refl_size']) if config['trn'][
                                                                                    'generate_refl_type'] == GeneratedReflectionType.haar and \
                                                                                config['trn']['reflection'] == 'gen' else ''
    additional_info += '_aug-ref' if config['trn']['augment_reflection'] else ''
    additional_info += config['trn']['additional_info']
    basename = '{}_a={}_ic={:d}_o={}{}_lr={:.0E}_bs={:d}_c={}_fgw={:.0E}_bgw={:.0E}_lc={:d}_fm={:d}{}' \
        .format(
        config['trn']['task'],
        config['model']['architecture'],
        config['model']['input_channel_num'],
        config['trn']['optimizer'] if config['trn']['optimizer'] != 'sgd' else '{}-{:.2f}'.format(config['trn']['optimizer'],
                                                                                                  config['trn']['momentum']),
        '_wd={:.0E}'.format(config['trn']['weight_decay']) if config['trn']['weight_decay'] > 0 else '',
        float(config['trn']['learning_rate']),
        int(config['trn']['batch_size']),
        '_'.join(['{}={:.0E}'.format(key if 'dx' not in key else 'dx' + key[-1], val) for key, val in
                  config['trn']['criterions'].items()]).replace('+E00', ''),
        config['trn']['loss_fg_weight'],
        config['trn']['loss_bg_weight'],
        config['model']['lateral_channel_num'],
        int(config['trn']['feed_mask']),
        additional_info)

    experiments_directory = config['trn']['experiments_directory']

    summary, summary_filepath = prepare_summary(experiments_directory, basename)

    fh = logging.FileHandler(os.path.join(summary_filepath, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '.log'))
    logger.addHandler(fh)

    logger.info(" sys path: {}".format(sys.path))
    logger.info(' %s' % basename)

    logger.info(json.dumps(config))
    trn_loader, val_loader, tst_loader, trn_ds, val_ds, tst_ds = prepare_loaders(config, logger, args)

    model_name, model, losses, epoch_shift, datetime_id = \
        load_models_prepare_losses_file(experiments_directory, config, basename, logger, logFormatter, cuda, trn_loader, summary)

    start_time = time.time()
    losses = evaluate_model(model, model_name, losses, epoch_shift, experiments_directory, trn_ds, config)
    logger.info(' The whole learning finished in %.0f s.' % (time.time() - start_time))

    logger.info(' Closing summary...')
    summary.close()

    # caches
    logger.info(' Clearing caches...')
    cache_dir = args['--scratch-dir']
    if cache_dir is not None and os.path.isdir(cache_dir):
        import shutil

        shutil.rmtree(cache_dir)
    logger.info(' Cleared caches.')

    logger.info(' Succesfully finished.')
