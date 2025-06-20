import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class PixGen:
    def __init__(self, window_width=3, window_height=3, type='type-4'):
        from skimage.feature import haar_like_feature_coord
        self.ww, self.wh = window_width, window_height
        self.coords, self.types = haar_like_feature_coord(self.wh, self.ww)

    def generate_images(self, white_pixels_count=None):
        images = np.empty((0, self.wh, self.ww), dtype=np.uint8)
        for idx in range(len(self.coords)):
            img = np.zeros((1, self.wh, self.ww), dtype=np.uint8)
            for q in range(len(self.coords[idx])):
                a_from_x, a_from_y, a_to_x, a_to_y = self.collect_indexes(self.coords[idx][q])
                img[0, a_from_y:a_to_y, a_from_x:a_to_x] = 255
            if white_pixels_count is None or (img[0, :, :] > 0).sum() == white_pixels_count:
                images = np.concatenate((images, img), axis=0)

        return images

    def collect_indexes(self, coords):
        a_from, a_to = coords
        a_from_y, a_from_x = a_from
        a_to_y, a_to_x = a_to
        a_to_y, a_to_x = a_to_y + 1, a_to_x + 1
        return a_from_x, a_from_y, a_to_x, a_to_y


if __name__ == '__main__':
    a = PixGen(3, 3)
    from matplotlib import pyplot as plt

    images = a.generate_images()
    for i in range(images.shape[0]):
        plt.imshow(images[i])
        plt.show()


def store_image_to_tensorboard(prefix, epoch, batch_idx, batch_output, data, usr_mask, target, loss_mask, loss_mask_u, mask_pred,
                               summary, config, datetime_id, image_idx_to_store=0, image_idx_to_log=0):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as ds

    mask = torch.zeros(data[:, :3].shape)
    for i in range(3):
        mask[:, i:i + 1] = usr_mask

    mask_pred = mask_pred[image_idx_to_store].clone().detach().float().cpu().numpy() * 255.
    mask_pred_3c = np.zeros(data[0, :3].shape)
    for i in range(3):
        mask_pred_3c[i:i + 1] = mask_pred

    loss_mask_u = loss_mask_u[image_idx_to_store].clone().detach().float().cpu().numpy() * 255.
    loss_mask_u3c = np.zeros(data[0, :3].shape)
    for i in range(3):
        loss_mask_u3c[i:i + 1] = loss_mask_u

    loss_mask_vis = loss_mask[image_idx_to_store].clone().detach().float().cpu().numpy()
    loss_mask_vis = ds.unnormalize(loss_mask_vis)

    if epoch == 1 and batch_idx == 0:
        store_sanity_image(batch_idx, data, epoch, mask, prefix, summary)

    viz = []
    for output_ch_idx in range(batch_output.size()[1] // 3):
        viz_3channel = []
        for item in (data[:, output_ch_idx * 3:(output_ch_idx + 1) * 3],
                     mask,
                     target[:, output_ch_idx * 3:(output_ch_idx + 1) * 3],
                     batch_output[:, output_ch_idx * 3:(output_ch_idx + 1) * 3]):
            item = item[image_idx_to_store].clone().detach().float().cpu().numpy()
            item = ds.unnormalize(item)[::-1]
            viz_3channel.append(item)

        viz_3channel.append(loss_mask_vis)
        # iffed_output = viz_3channel[0].copy()
        # iffed_output[loss_mask_vis > 128] = viz_3channel[3][loss_mask_vis > 128]
        # viz_3channel.append(iffed_output)
        viz_3channel.append(mask_pred_3c)
        viz_3channel.append(loss_mask_u3c)

        diff_image, colorbar = create_difference_image_and_colorbar(viz_3channel[3], viz_3channel[2])
        viz_3channel.append(diff_image)
        viz_3channel.append(colorbar)

        viz.append(np.concatenate(viz_3channel, axis=2))

    viz = np.concatenate(viz, axis=1)

    sp = ' ' * 5
    subtitle_field_text = '{}'.format(sp + 'input'
                                      + sp + 'input_mask'
                                      + sp * 2 + 'target'
                                      + sp * 2 + 'output'
                                      + sp + 'loss_mask'
                                      + sp + 'usr_mask_pred'
                                      + sp + 'target-output')
    subtitle_field = np.ones((20, viz.shape[2], 3), dtype=np.uint8) * 50
    cv2.putText(subtitle_field, subtitle_field_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    viz = np.concatenate((viz, subtitle_field.transpose((2, 0, 1))), axis=1)

    id_field = create_identification_image(image_idx_to_store, epoch, prefix, viz, datetime_id)
    viz = np.concatenate((viz, id_field.transpose((2, 0, 1))), axis=1)

    summary.add_image("{}/{}".format(prefix, image_idx_to_log), viz.astype('float32') / 255., dataformats='CHW', global_step=epoch)
    from socket import gethostname
    img_path = os.path.join(config['trn']['experiments_directory'], 'images',
                            '{}_{}_{}_{:04d}_{:03d}.png'.format(gethostname(), os.environ.get('CUDA_VISIBLE_DEVICES'), prefix, epoch,
                                                                image_idx_to_store))
    if not os.path.isdir(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))
    cv2.imwrite(img_path, viz.transpose((1, 2, 0))[:, :, ::-1])


def store_sanity_image(batch_idx, data, epoch, mask, prefix, summary):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as ds
    idxs = data.size()[1] // 3
    sanity_viz = []
    for item_idx in range(idxs):
        item = data[0, item_idx * 3:(item_idx + 1) * 3].clone().detach().float().cpu().numpy()
        item = ds.unnormalize(item)[::-1]
        sanity_viz.append(item)
    sanity_viz.append(ds.unnormalize(mask[0].clone().detach().float().cpu().numpy())[::-1])
    sanity_viz = np.concatenate(sanity_viz, axis=2)
    summary.add_images("sanity_{}/{}".format(prefix, batch_idx), sanity_viz.astype('float32') / 255., dataformats='CHW',
                       global_step=epoch)


def create_identification_image(batch_idx, epoch, prefix, viz, datetime_id):
    id_field_text = '{}_{}_{}:{}'.format(prefix, datetime_id, epoch, batch_idx)
    id_field = np.ones((20, viz.shape[2], 3), dtype=np.uint8) * 20
    cv2.putText(id_field, id_field_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return id_field


def create_difference_image_and_colorbar(source, target, max_val=30):
    diff = np.abs(source - target)
    diff = np.mean(diff, axis=0, keepdims=True)
    diff = np.concatenate([diff, diff, diff], axis=0)
    max_val_constant = (255. / float(max_val))
    diff *= max_val_constant  # max je 30, cokoliv pod je relativne
    diff[diff > 255] = 255
    diff_max = diff.max() + 10 ** -7
    diff /= diff_max

    colorbar = apply_color_map(np.repeat(np.linspace(255, 0, diff.shape[2], dtype=np.uint8).reshape(-1, 1), 20, axis=1)) \
        .transpose((1, 2, 0))
    colorbar = np.rot90(colorbar).copy()
    cv2.putText(colorbar, '{:.1f}'.format(diff_max / max_val_constant), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return apply_color_map(diff[0]), np.fliplr(np.flipud(np.rot90(colorbar))).transpose((2, 0, 1))


def apply_color_map(grayscale_image):
    from matplotlib import cm
    cm = cm.get_cmap('viridis')
    return cm(grayscale_image).transpose((2, 0, 1))[:3] * 255


class BestModel(object):
    def __init__(self, measure_names):
        self.measures = {}
        for measure_name in measure_names:
            self.measures[measure_name] = {'path_to_model': '', 'value': None, 'epoch': None}

    def store_path(self, measure_name, path_to_model, epoch=None, value=None):
        self.measures[measure_name]['path_to_model'] = path_to_model
        self.measures[measure_name]['value'] = value
        self.measures[measure_name]['epoch'] = epoch

    def get_best_model_path(self, measure_name):
        return self.measures[measure_name]['path_to_model']

    def get_best_model_value(self, measure_name):
        return self.measures[measure_name]['value']

    def get_best_model_epoch(self, measure_name):
        return self.measures[measure_name]['epoch']


class HistogramsRecorder:
    def __init__(self, begin=0.0, end=1.0, step=0.01, number_of_bins_for_value_recording=None):
        self.begin = begin
        self.end = end
        self.step = step
        self.bins = np.arange(begin, end, step)
        self.number_of_bins_for_value_recording = len(self.bins)
        if number_of_bins_for_value_recording is not None:
            self.number_of_bins_for_value_recording = number_of_bins_for_value_recording
        self.datasets = ['all']
        self.histograms = {}
        for dataset_name in self.datasets:
            self.histograms[dataset_name] = {}
            for class_name in ['positive', 'negative']:
                self.histograms[dataset_name][class_name] = {'bins': self.bins, 'heights': np.zeros(
                    (self.number_of_bins_for_value_recording,))}

        self.computed = False

    def positive_bins(self, dataset='all'):
        return self.histograms[dataset]['positive']['bins']

    def negative_bins(self, dataset='all'):
        return self.histograms[dataset]['negative']['bins']

    def positive_heights(self, dataset='all'):
        return self.histograms[dataset]['positive']['heights']

    def negative_heights(self, dataset='all'):
        return self.histograms[dataset]['negative']['heights']

    def record_positive(self, values, dataset_names_for_values=()):
        self.__record_values(self.histograms['all']['positive']['heights'], values)
        for dataset_name in self.datasets:
            if 'all' in dataset_name or len(dataset_names_for_values) == 0: continue
            self.__record_values(self.histograms[dataset_name]['positive']['heights'],
                                 values[dataset_name == dataset_names_for_values])

    def record_negative(self, values, dataset_names_for_values=()):
        self.__record_values(self.histograms['all']['negative']['heights'], values)
        for dataset_name in self.datasets:
            if 'all' in dataset_name or len(dataset_names_for_values) == 0: continue
            self.__record_values(self.histograms[dataset_name]['negative']['heights'],
                                 values[dataset_name == dataset_names_for_values])

    def mark_as_computed(self):
        self.computed = True

    def number_of_bins(self):
        return len(self.bins)

    def __record_values(self, where, values):
        if self.computed or np.sum(~np.isnan(values)) == 0:
            return

        histogram = \
            np.histogram(values[~np.isnan(values)], bins=self.number_of_bins_for_value_recording, range=(self.begin, self.end))[0]
        where += histogram

    def normalize(self):
        if self.computed:
            return

        for class_name in ['positive', 'negative']:
            for dataset_name in self.datasets:
                self.histograms[dataset_name][class_name]['heights'] /= np.sum(
                    self.histograms[dataset_name][class_name]['heights']) + 0.00000001


def plot_with_pyplot(figure_id, data, usr_mask, target, batch_output, loss_mask, mask_pred, visual_inspection_dir, filename,
                     image_idx_to_store=0, finetuning_output=None):
    import cv2

    mask = torch.zeros(data[:, :3].shape)
    for i in range(3):
        mask[:, i:i + 1] = usr_mask

    mask_pred = mask_pred[image_idx_to_store].clone().detach().float().cpu().numpy() * 255.
    mask_pred_3c = np.zeros(data[0, :3].shape)
    for i in range(3):
        mask_pred_3c[i:i + 1] = mask_pred

    viz = []
    for item in (data[:, :3], mask, target, batch_output):
        item = numpy_and_unnormalize(item[image_idx_to_store])
        viz.append(item)

    from nn.trn.utils import create_difference_image_and_colorbar
    # diff_image, colorbar = create_difference_image_and_colorbar(viz[0], viz[2])
    # viz.insert(3, diff_image)
    # viz.insert(4, colorbar)

    if finetuning_output is not None:
        viz.append(finetuning_output)

    diff_image, colorbar = create_difference_image_and_colorbar(viz[3], viz[2], max_val=255)
    viz.append(diff_image)
    viz.append(colorbar)

    # iffed output + diff
    # loss_mask = mask[0].clone().detach().float().cpu().numpy()
    # iffed_output = viz[0].copy()
    # iffed_output[loss_mask > 0] = viz[3][loss_mask > 0]
    viz.append(mask_pred_3c)

    # diff_image, colorbar = create_difference_image_and_colorbar(viz[-1], viz[2])
    # viz.append(diff_image)
    # viz.append(colorbar)

    viz = np.concatenate(viz, axis=2)

    sp = ' ' * 5
    id_field_text = '{}'.format(sp + 'input'
                                + sp * 2 + 'mask  '
                                + sp * 2 + 'target'
                                # + sp + '  ' + 'input-target'
                                + sp * 2 + 'output'
                                # + sp + 'target-output'
                                # + sp + 'iffed output'  # +
                                + (sp + 'Sfinetuned' if finetuning_output is not None else '')
                                + sp + 'target-output'
                                + sp + 'predicted msk'
                                )
    id_field = np.ones((20, viz.shape[2], 3), dtype=np.uint8) * 20
    cv2.putText(id_field, id_field_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    viz = np.concatenate((viz, id_field.transpose((2, 0, 1))), axis=1)

    plot_show_maybe_store(viz, filename=filename, dir=visual_inspection_dir)


def plot_show_maybe_store(viz, filename=None, dir=None):
    my_dpi = 600
    fig = plt.figure(figsize=(viz.shape[2] / my_dpi, viz.shape[1] / my_dpi), dpi=my_dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(viz.transpose((1, 2, 0)) / 255.)
    if filename is not None and dir is not None:
        plt.savefig(os.path.join(dir, filename + '.png'))
    plt.show()


def normalize_torch(item):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as UGIRRDataset
    item = UGIRRDataset.unnormalize_torch(item)
    return item


def unnormalize_torch(item):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as UGIRRDataset
    item = UGIRRDataset.unnormalize_torch(item)
    return item


def numpy_and_unnormalize(item):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as UGIRRDataset
    item = item.clone().detach().float().cpu().numpy()
    item = UGIRRDataset.unnormalize(item)[::-1]
    return item


def numpy_and_min_max_unnormalize(item):
    from nn.trn.dataset import UserGuidedIsolatedReflectionRemovalDatasetFromCOCO as UGIRRDataset
    item = item.clone().detach().float().cpu().numpy()
    item = item - item.min()
    item = item / item.max()
    item = item * 255.0
    return item
