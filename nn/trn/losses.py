from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as ds
from nn.trn.morphological import Erosion2d, Dilation2d


class ContentLoss:
    def __init__(self, criterion: Callable):
        self.criterion = criterion

    def __call__(self, inp, tar):
        return self.criterion(inp, tar)


class MaskContentLoss:
    def __init__(self, criterion: Callable):
        self.criterion = criterion
        self.normalize = ds.normalize  # TODO: this is weird :/
        self.unnormalize = ds.unnormalize  # TODO: move normalization from dataset to model

    def __call__(self, inp, tar, mask=None):
        mask = torch.cat((mask, mask, mask), dim=1).cpu().numpy()
        mask = self.unnormalize(mask)
        mask = torch.from_numpy(mask).cuda()
        mask = torch.clamp(mask, 0., 1.)
        return self.criterion(inp[mask > 0], tar[mask > 0])


def zero_one_unnormalize_with_ds_values(whaat):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as ds
    mean = torch.from_numpy(ds.mean[np.newaxis]).cuda()
    std = torch.from_numpy(ds.std[np.newaxis]).cuda()
    whaat = whaat * std
    whaat += mean
    return whaat


def l1(batch_target_img, batch_mask_img, batch_output, config):
    loss = torch.abs(batch_target_img - batch_output)

    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    return loss.mean(), loss[:, :3][batch_mask_img < 0.5].mean(), loss[:, :3][batch_mask_img > 0.5].mean()


def l1_finetune(batch_target_img, batch_mask_img, batch_output, config):
    loss = torch.abs(batch_target_img - batch_output)

    crossval_loss_mask_trn, crossval_loss_mask_val = batch_mask_img[:, :3], batch_mask_img[:, 3:]

    crossval_loss_mask_trn = zero_one_unnormalize_with_ds_values(crossval_loss_mask_trn)
    crossval_loss_mask_trn = torch.clamp(crossval_loss_mask_trn, 0, 1.0)

    crossval_loss_mask_val = zero_one_unnormalize_with_ds_values(crossval_loss_mask_val)
    crossval_loss_mask_val = torch.clamp(crossval_loss_mask_val, 0, 1.0)

    return loss[crossval_loss_mask_trn > 0.5].mean(), loss[crossval_loss_mask_val > 0.5].mean(), loss[
        crossval_loss_mask_val > 0.5].mean()


def l1l(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    loss = torch.abs(batch_target_img - batch_output)
    loss_under_mask = config['trn']['loss_mask_term'] * loss[:, :3][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
    loss_whole_image = config['trn']['loss_whole-image_term'] * loss[:, :3].mean()
    for i in range(1, batch_target_img.size()[1] // 3):
        loss_under_mask += config['trn']['loss_mask_term'] * loss[:, 3 * i:3 * (i + 1)][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
        loss_whole_image += config['trn']['loss_whole-image_term'] * loss[:, 3 * i:3 * (i + 1)].mean()
    return loss_whole_image + loss_under_mask, \
           loss[:, :3][batch_mask_img < 0.5].mean() if (batch_mask_img < 0.5).sum() > 0 else torch.tensor([0]).float().cuda(), \
           loss[:, :3][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()


def l1ld(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    dx_t = batch_target_img[:, :, :, 1:] - batch_target_img[:, :, :, :-1]
    dy_t = batch_target_img[:, :, 1:] - batch_target_img[:, :, :-1]

    dx_o = batch_output[:, :, :, 1:] - batch_output[:, :, :, :-1]
    dy_o = batch_output[:, :, 1:] - batch_output[:, :, :-1]

    dx_loss = torch.abs(dx_t - dx_o)
    dy_loss = torch.abs(dy_t - dy_o)

    loss = torch.abs(batch_target_img - batch_output)
    loss_under_mask = config['trn']['loss_mask_term'] * loss[:, :3][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
    loss_whole_image = config['trn']['loss_whole-image_term'] * loss[:, :3].mean()
    for i in range(1, batch_target_img.size()[1] // 3):
        loss_under_mask += config['trn']['loss_mask_term'] * loss[:, 3 * i:3 * (i + 1)][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
        loss_whole_image += config['trn']['loss_whole-image_term'] * loss[:, 3 * i:3 * (i + 1)].mean()
    loss_under_mask += config['trn']['loss_dxdy_term'] * dx_loss[batch_mask_img[:, :, :, 1:] > 0.5].mean()
    loss_under_mask += config['trn']['loss_dxdy_term'] * dy_loss[batch_mask_img[:, :, 1:, :] > 0.5].mean()
    return loss_whole_image + loss_under_mask, \
           loss[:, :3][batch_mask_img < 0.5].mean() if (batch_mask_img < 0.5).sum() > 0 else torch.tensor([0]).float().cuda(), \
           loss[:, :3][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()


def l1lds(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    dx_t = batch_target_img[:, :, :, 1:] - batch_target_img[:, :, :, :-1]
    dy_t = batch_target_img[:, :, 1:] - batch_target_img[:, :, :-1]

    dx_o = batch_output[:, :, :, 1:] - batch_output[:, :, :, :-1]
    dy_o = batch_output[:, :, 1:] - batch_output[:, :, :-1]

    dx_loss = torch.abs(dx_t - dx_o)
    dy_loss = torch.abs(dy_t - dy_o)

    loss = torch.abs(batch_target_img - batch_output)
    loss_l2 = torch.sqrt(torch.sum((batch_target_img - batch_output) ** 2, dim=1))
    loss_under_mask = config['trn']['loss_mask_term'] * loss[:, :3][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
    loss_under_mask = loss_under_mask + config['trn']['loss_l2_term'] * loss_l2[batch_mask_img[:, 0] > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
    loss_whole_image = config['trn']['loss_whole-image_term'] * loss[:, :3].mean()
    for i in range(1, batch_target_img.size()[1] // 3):
        loss_under_mask += config['trn']['loss_mask_term'] * loss[:, 3 * i:3 * (i + 1)][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()
        loss_whole_image += config['trn']['loss_whole-image_term'] * loss[:, 3 * i:3 * (i + 1)].mean()
    loss_under_mask += config['trn']['loss_dxdy_term'] * dx_loss[batch_mask_img[:, :, :, 1:] > 0.5].mean()
    loss_under_mask += config['trn']['loss_dxdy_term'] * dy_loss[batch_mask_img[:, :, 1:, :] > 0.5].mean()
    return loss_whole_image + loss_under_mask, \
           loss[:, :3][batch_mask_img < 0.5].mean() if (batch_mask_img < 0.5).sum() > 0 else torch.tensor([0]).float().cuda(), \
           loss[:, :3][batch_mask_img > 0.5].mean() if (batch_mask_img > 0.5).sum() > 0 else torch.tensor([0]).float().cuda()


def zero_t():
    return torch.tensor([0]).float().cuda()


def selective_loss(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    dilation = Dilation2d(3, 3, 6, False).to(batch_mask_img.device)
    batch_mask_img_fg = dilation.forward(batch_mask_img)

    functions = {}

    bg_num = (batch_mask_img_fg < 0.5).sum()
    fg_num = (batch_mask_img > 0.5).sum()

    # from matplotlib import pyplot as plt
    # m = batch_mask_img_fg.detach().cpu().numpy()[0]
    # my_dpi = 600
    # fig = plt.figure(figsize=(m.shape[2] / my_dpi, m.shape[1] / my_dpi), dpi=my_dpi, frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(m.transpose(1, 2, 0))
    # plt.savefig('/mnt/datagrid/personal/spetlrad/reflections/tst.png')

    bg_bool_mask = batch_mask_img < 0.5
    fg_bool_mask = batch_mask_img > 0.5

    x_y_diff = batch_target_img - batch_output
    l1_loss = torch.abs(x_y_diff)
    l1_loss_first_order = x_y_diff.clone()
    l1_loss_first_order = torch.tanh(10000.0 * l1_loss_first_order)
    functions['loss_l1_fg_first_order'] = l1_loss_first_order * fg_bool_mask
    l1_loss_first_order_exact = x_y_diff.clone()
    l1_loss_first_order_exact[l1_loss_first_order_exact <= 0] = -1
    l1_loss_first_order_exact[l1_loss_first_order_exact > 0] = 1
    functions['loss_l1_fg_first_order_exact'] = l1_loss_first_order_exact * fg_bool_mask
    functions['loss_l1_fg'] = l1_loss[fg_bool_mask].mean() if fg_num > 0 else zero_t()
    functions['loss_l1_bg'] = l1_loss[bg_bool_mask].mean() if bg_num > 0 else zero_t()

    loss_fg = torch.tensor([0]).float().cuda()
    loss_bg = torch.tensor([0]).float().cuda()
    if 'l1' in config['trn']['criterions']:
        loss_fg = loss_fg + config['trn']['criterions']['l1'] * functions['loss_l1_fg']
        loss_bg = loss_bg + config['trn']['criterions']['l1'] * functions['loss_l1_bg']

    if 'l2' in config['trn']['criterions']:
        loss_l2 = torch.sqrt(torch.sum((batch_target_img - batch_output) ** 2, dim=1).abs() + 0.000000001)
        functions['loss_l2_fg'] = loss_l2[fg_bool_mask[:, 0]].mean() if fg_num > 0 else zero_t()
        functions['loss_l2_bg'] = loss_l2[bg_bool_mask[:, 0]].mean() if bg_num > 0 else zero_t()
        loss_fg = loss_fg + config['trn']['criterions']['l2'] * functions['loss_l2_fg']
        loss_bg = loss_bg + config['trn']['criterions']['l2'] * functions['loss_l2_bg']

    if 'l2sq' in config['trn']['criterions']:
        loss_l2sq = torch.sum((batch_target_img - batch_output) ** 2, dim=1)
        functions['loss_l2sq_fg'] = loss_l2sq[fg_bool_mask[:, 0]].mean() if fg_num > 0 else zero_t()
        functions['loss_l2sq_bg'] = loss_l2sq[bg_bool_mask[:, 0]].mean() if bg_num > 0 else zero_t()
        loss_fg = loss_fg + config['trn']['criterions']['l2sq'] * functions['loss_l2sq_fg']
        loss_bg = loss_bg + config['trn']['criterions']['l2sq'] * functions['loss_l2sq_bg']

    dxdys = {'': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
    for dxdy_key, dxdy_val in dxdys.items():
        if 'dx{}dy{}'.format(dxdy_key, dxdy_key) in config['trn']['criterions']:
            loss_fg, loss_bg = compute_dxdy(config, dxdy_val, batch_output, batch_target_img,
                                            bg_bool_mask, bg_num,
                                            fg_bool_mask, fg_num,
                                            functions,
                                            loss_fg, loss_bg)

    return config['trn']['loss_fg_weight'] * loss_fg + config['trn']['loss_bg_weight'] * loss_bg, functions


def compute_dxdy(config, order, batch_output, batch_target_img, bg_bool_mask, bg_num, fg_bool_mask, fg_num, functions, loss_fg,
                 loss_bg):
    dx_t = batch_target_img[:, :, :, order:] - batch_target_img[:, :, :, :-order]
    dy_t = batch_target_img[:, :, order:] - batch_target_img[:, :, :-order]
    dx_o = batch_output[:, :, :, order:] - batch_output[:, :, :, :-order]
    dy_o = batch_output[:, :, order:] - batch_output[:, :, :-order]

    dx_loss = torch.abs(dx_t - dx_o)
    dy_loss = torch.abs(dy_t - dy_o)

    x_order = 'x' if order == 1 else 'x{}'.format(order)
    y_order = 'y' if order == 1 else 'y{}'.format(order)

    functions['loss_d{}_fg'.format(x_order)] = dx_loss[fg_bool_mask[:, :, :, order:]].mean() if fg_num > 0 else zero_t()
    functions['loss_d{}_fg'.format(y_order)] = dy_loss[fg_bool_mask[:, :, order:, :]].mean() if fg_num > 0 else zero_t()

    loss_fg = loss_fg + config['trn']['criterions']['d{}d{}'.format(x_order, y_order)] * functions['loss_d{}_fg'.format(x_order)]
    loss_fg = loss_fg + config['trn']['criterions']['d{}d{}'.format(x_order, y_order)] * functions['loss_d{}_fg'.format(y_order)]

    functions['loss_d{}_bg'.format(x_order)] = dx_loss[bg_bool_mask[:, :, :, order:]].mean() if bg_num > 0 else zero_t()
    functions['loss_d{}_bg'.format(y_order)] = dy_loss[bg_bool_mask[:, :, order:, :]].mean() if bg_num > 0 else zero_t()

    loss_bg = loss_bg + config['trn']['criterions']['d{}d{}'.format(x_order, y_order)] * functions['loss_d{}_bg'.format(x_order)]
    loss_bg = loss_bg + config['trn']['criterions']['d{}d{}'.format(x_order, y_order)] * functions['loss_d{}_bg'.format(y_order)]

    return loss_fg, loss_bg


def l1l_g_mask(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    loss = torch.abs(batch_target_img - batch_output)
    loss = loss[batch_mask_img < 0.5]
    return loss.mean()


def l1l_freq(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    batch_output_freq = torch.fft(torch.stack([batch_output.unsqueeze(-1), batch_output.unsqueeze(-1)], dim=-1),
                                  signal_ndim=2, normalized=True)
    batch_target_img_freq = torch.fft(torch.stack([batch_target_img.unsqueeze(-1), batch_target_img.unsqueeze(-1)], dim=-1),
                                      signal_ndim=2, normalized=True)
    loss_freq = torch.abs(batch_output_freq - batch_target_img_freq)

    loss = torch.abs(batch_target_img - batch_output)
    loss = loss.mean() + config['trn']['loss_mask_term'] * loss[batch_mask_img > 0].mean()
    return loss.mean() + config['trn']['loss_freq_term'] * loss_freq.mean()


def l1l_freq_max(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    batch_output_freq = torch.fft(torch.stack([batch_output.unsqueeze(-1), batch_output.unsqueeze(-1)], dim=-1),
                                  signal_ndim=2, normalized=True)
    batch_target_img_freq = torch.fft(torch.stack([batch_target_img.unsqueeze(-1), batch_target_img.unsqueeze(-1)], dim=-1),
                                      signal_ndim=2, normalized=True)
    loss_freq = torch.abs(batch_output_freq - batch_target_img_freq)

    loss_max = torch.max(torch.max(torch.abs(batch_target_img - batch_output), dim=-1)[0], dim=-1)[0]
    loss = loss_max.mean() + config['trn']['loss_mask_term'] * torch.max(
        torch.abs(batch_target_img - batch_output)[batch_mask_img > 0]).mean()
    return loss.mean() + config['trn']['loss_freq_term'] * torch.max(torch.max(torch.max(loss_freq, dim=-1)[0], dim=-1)[0], dim=-1)[
        0].mean()


def find_mask_indices_single_dim(mask):
    mask[mask > 0.5] = 1.0
    mask[mask < 0.5] = 0.0

    mask = torch.cumsum(mask, -1)
    out, inv_indices = torch.unique(mask, sorted=True, return_inverse=True)
    starts = inv_indices[:, :, 1]
    ends = inv_indices[:, :, -1]

    return starts, ends


def l1l_freq_limited(batch_target_img, batch_mask_img, batch_output, config):
    batch_mask_img = zero_one_unnormalize_with_ds_values(batch_mask_img)
    batch_mask_img = torch.clamp(batch_mask_img, 0, 1.0)

    max_vals_x, max_inds_x = torch.max(batch_mask_img, dim=-1)
    s, e = find_mask_indices_single_dim(max_vals_x)

    batch_output_freq = torch.fft(torch.stack([batch_output.unsqueeze(-1), batch_output.unsqueeze(-1)], dim=-1),
                                  signal_ndim=2, normalized=True)
    batch_target_img_freq = torch.fft(torch.stack([batch_target_img.unsqueeze(-1), batch_target_img.unsqueeze(-1)], dim=-1),
                                      signal_ndim=2, normalized=True)
    loss_freq = torch.abs(batch_output_freq - batch_target_img_freq)

    loss = torch.abs(batch_target_img - batch_output)
    loss = loss.mean() + config['trn']['loss_mask_term'] * loss[batch_mask_img > 0].mean()
    return loss.mean() + config['trn']['loss_freq_term'] * loss_freq.mean()


class LossName:
    l1 = 'l1'
    mse = 'mse'
    l1l = 'l1l'
    l1ld = 'l1ld'
    l1lds = 'l1lds'
    l1l_freq = 'l1l-freq'
    l1l_freq_limited = 'l1l-freq-l'
    l1_g_mask = 'l1-g-mask'
    l1l_freq_max = 'l1l-freq-max'
    l1_finetune = 'l1-fine'


def get_loss(loss_name, mask_loss=False):
    if loss_name == LossName.l1:
        return l1
    elif loss_name == LossName.l1_finetune:
        return l1_finetune
    elif loss_name == LossName.mse:
        criterion = nn.MSELoss()
    elif loss_name == LossName.l1l:
        return l1l
    elif loss_name == LossName.l1ld:
        return l1ld
    elif loss_name == LossName.l1lds:
        return l1lds
    elif loss_name == LossName.l1l_freq:
        return l1l_freq
    elif loss_name == LossName.l1l_freq_limited:
        return l1l_freq_limited
    elif loss_name == LossName.l1_g_mask:
        return l1l_g_mask
    elif loss_name == LossName.l1l_freq_max:
        return l1l_freq_max
    else:
        raise NotImplementedError(f"No such loss: {loss_name}")
    if not mask_loss:
        return ContentLoss(criterion)
    else:
        return MaskContentLoss(criterion)


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """

    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        # return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1. - pos)) + torch.mean(F.relu(1. + neg)))