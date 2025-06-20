import logging
import time
from glob import glob

import cv2
import json
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from ugsiirr.annotate_dataset_flares import get_flare_mask
from nn.trn.utils import PixGen
from scipy.stats import beta

logger = logging.getLogger("training")


def convert_reflection_to_user_interaction(img, kernel_wh=55, threshold=0.2, plot=False, gaussian_blur_sigma=3.0):
    if img.max() < 5:
        return np.zeros(img.shape[:2])

    mask_image = img.astype(np.float32) @ np.array([[0.3], [0.3], [0.3]])
    mask_image = mask_image[:, :, 0]

    kernel = np.zeros((kernel_wh, kernel_wh))
    cv2.circle(kernel, (kernel.shape[0] // 2, kernel.shape[1] // 2), kernel.shape[0] // 2, (255), -1)

    mask_image = cv2.filter2D(mask_image, -1, kernel)
    mask_image /= mask_image.max()
    mask_image[mask_image <= threshold] = 0
    mask_image[mask_image > threshold] = 255
    if gaussian_blur_sigma > 0:
        mask_image = cv2.GaussianBlur(mask_image, (15, 15), gaussian_blur_sigma)

    if plot:
        from matplotlib import pyplot as plt
        plt.figure(1)
        plt.imshow(img)
        plt.show()

        plt.figure(2)
        plt.imshow(mask_image, cmap='Greys')
        plt.show()

    return mask_image


class GeneratedReflectionType:
    many1px_rndcol = 'many1px_rnd-col'
    onepx_rndcol = '1px_rnd-col'
    haar = 'haar'


class UserGuidedIsolatedReflectionRemovalDatasetFromCOCO(Dataset):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    mean_torch = torch.tensor(mean).view((1, 3, 1, 1))
    std_torch = torch.tensor(std).view((1, 3, 1, 1))

    def __len__(self):
        return len(self.idx_map)

    def __init__(self, config, image_filenames, is_trn=True, cache_dir=None, ds_size=None):
        self.config = config
        self.image_c = config['model']['input_channel_num']
        self.coco_train2017_dir = config['trn']['coco_train2017_directory']
        self.lamp_directory = config['trn']['lamp_dataset_directory']
        self.lamp_paths = []
        self.lamp_bbox_paths = glob(os.path.join(self.lamp_directory, '*', '*.json'))
        self.lamp_bboxes = {'path_id': [], 'bbox': []}
        for lamp_bbox_path_id, lamp_bbox_path in enumerate(self.lamp_bbox_paths):
            self.lamp_paths.append(os.path.splitext(lamp_bbox_path)[0])
            with open(lamp_bbox_path, 'r') as f:
                data = json.load(f)
                for bbox_x_xx_y_yy in data['bboxes_x_xx_y_yy']:
                    self.lamp_bboxes['path_id'].append(lamp_bbox_path_id)
                    self.lamp_bboxes['bbox'].append(bbox_x_xx_y_yy)

        self.blend_on_fly = config["trn"].get("blend_on_fly", False)
        self.blend_on_fly_mode = config["trn"].get("blend_on_fly_mode", "add")
        self.blend_on_fly_alpha_min = config["trn"].get("blend_alpha_min", 0.0)
        self.blend_on_fly_alpha_max = config["trn"].get("blend_alpha_max", 1.0)
        self.is_trn = is_trn
        self.cache_dir = cache_dir
        self.image_filenames = image_filenames
        self.image_crop_positions = [[]] * len(image_filenames)
        self.augmentation = None
        self.ds_size = ds_size
        self.haar_perturbations = PixGen(config['trn']['generate_refl_size'], config['trn']['generate_refl_size']).generate_images()
        if self.ds_size is None:
            self.idx_map = list(range(len(self.image_filenames)))
        else:
            self.idx_map = np.random.permutation(np.arange(len(self.image_filenames), dtype=np.int32))[
                           :min(len(self.image_filenames), self.ds_size)]

        logger.info(' During an epoch, actively using {} samples.'.format(len(self.idx_map)))

        if config['trn']['augment_reflection']['active'] and is_trn:
            self.reflection_augmentation = self._prepare_reflection_augmentation()
        if config['trn']['augment'] and is_trn:
            self.augmentation = self._prepare_augmentation()

        if not is_trn:
            self.image_id_crops = []
            for src_img_filename in self.image_filenames:
                src_img_path = os.path.join(self.coco_train2017_dir, src_img_filename)
                src_img = cv2.imread(src_img_path)
                wh = self.config['model']['input_width_height']
                h, w, _ = src_img.shape
                b, l, r, t = self.get_crop_idxs(wh, h, w)
                self.image_id_crops.append([b, l, r, t])

            if self.blend_on_fly_mode == 'nosatexp':
                x = lambda sample: sample if sample <= 1.0 else 1.0
                self.blend_alphas = [x(np.random.exponential() / 5.0) for i in range(len(self.image_filenames))]
            elif self.blend_on_fly_mode == 'nosatbeta':
                self.blend_alphas = beta.ppf(np.random.rand(len(self.image_filenames)), 0.75, 0.45)

        start_time = time.time()

        logger.info(' Loading the %d files finished after %.2f s...' % (len(self.image_filenames), time.time() - start_time))

    def shuffle(self):
        if self.ds_size is not None:
            self.idx_map = np.random.permutation(np.arange(len(self.image_filenames), dtype=np.int32))[:self.ds_size]

    def normalize(self, img):
        img = img.copy()
        for cycle_id in range(img.shape[0] // 3):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img[idx_from:idx_to] /= 255.0
            img[idx_from:idx_to] -= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean
            img[idx_from:idx_to] /= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std

        return img

    def normalize_torch(self, img_batch):
        img_batch = img_batch.clone()
        for cycle_id in range(img_batch.size()[1] // 3):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img_batch[:, idx_from:idx_to] = img_batch[:, idx_from:idx_to] / 255.0
            img_batch[:, idx_from:idx_to] = img_batch[:, idx_from:idx_to] - UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean_torch.to(img_batch.device)
            img_batch[:, idx_from:idx_to] = img_batch[:, idx_from:idx_to] / UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std_torch.to(img_batch.device)

        return img_batch

    @staticmethod
    def unnormalize(img):
        img = img.copy()
        img = img * UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std
        img = img + UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean
        img = img * 255.0
        img[img < 0] = 0
        img[img > 255] = 255

        return img

    @staticmethod
    def unnormalize_torch(img_batch):
        img = img_batch.clone().detach()
        img = img * UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std_torch.to(img_batch.device)
        img = img + UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean_torch.to(img_batch.device)
        img = img * 255.0
        img[img < 0] = 0
        img[img > 255] = 255

        return img

    @staticmethod
    def unnormalize_loss(loss):
        loss = loss * UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std.mean()
        loss *= 255.0
        return loss

    def prepare_imgs(self, image_id):
        src_img, refl_img, usr_mask, separate_refl_img = self.load_img(image_id)

        if self.augmentation is not None:
            src_img = self.augmentation.augment_image(src_img)
            refl_img = self.augmentation.augment_image(refl_img)

        if self.config['trn']['reflection'] != 'gen' and self.config['trn']['reflection'] != 'rea':
            refl_img, separate_refl_img, src_img, usr_mask = self.crop_images_wrt_mask(refl_img, separate_refl_img, src_img, usr_mask)

        if src_img.shape[0] < self.config['model']['input_width_height'] \
                or src_img.shape[1] < self.config['model']['input_width_height']:
            logger.warning(' Experienced a crop with size {}.'.format(src_img.shape))
            src_img = cv2.resize(src_img, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
            refl_img = cv2.resize(refl_img, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
            usr_mask = cv2.resize(usr_mask, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
            separate_refl_img = cv2.resize(separate_refl_img,
                                           (self.config['model']['input_width_height'], self.config['model']['input_width_height']))

        src_img = src_img.transpose((2, 0, 1)).astype(np.float32)
        refl_img = refl_img.transpose((2, 0, 1)).astype(np.float32)
        usr_mask = usr_mask.transpose((2, 0, 1)).astype(np.float32)
        separate_refl_img = separate_refl_img.transpose((2, 0, 1)).astype(np.float32)

        return src_img, refl_img, usr_mask, separate_refl_img

    def crop_images_wrt_mask(self, refl_img, separate_refl_img, src_img, usr_mask):
        crop_size = self.config['model']['input_width_height']
        mask, y, x = get_flare_mask(usr_mask, blur_iterations=20, ksize=5, crop_size=crop_size)
        h, w, _ = usr_mask.shape
        b, l, r, t = self.get_crop_idxs(crop_size, h, w, x, y)
        src_img = src_img[t:b, l:r]
        refl_img = refl_img[t:b, l:r]
        usr_mask = usr_mask[t:b, l:r]
        separate_refl_img = separate_refl_img[t:b, l:r]
        return refl_img, separate_refl_img, src_img, usr_mask

    def get_crop_idxs(self, crop_size, h, w, x=None, y=None):
        if x is None:
            y, x = np.random.randint(0, h - 1), np.random.randint(0, w - 1)
        l = max(0, x - crop_size // 2)
        t = max(0, y - crop_size // 2)
        r = min(w, x + crop_size // 2)
        b = min(h, y + crop_size // 2)
        if l == 0: r = crop_size
        if t == 0: b = crop_size
        if r == w: l = w - crop_size
        if b == h: t = h - crop_size
        return b, l, r, t

    def load_img(self, image_id):
        separate_refl_img_cache_path = os.path.join(str(self.cache_dir), self.image_filenames[image_id] + '_sep_refl.png')

        cache_read_successful = False
        if os.path.exists(separate_refl_img_cache_path):
            separate_refl_img = cv2.imread(separate_refl_img_cache_path)
            cache_read_successful = not separate_refl_img is None

        src_img_path = os.path.join(self.coco_train2017_dir, self.image_filenames[image_id])
        src_img = cv2.imread(src_img_path)
        if src_img.shape[0] < self.config['model']['input_width_height'] or \
                src_img.shape[1] < self.config['model']['input_width_height']:
            logger.warning(' Experienced a crop with size {}.'.format(src_img.shape))
            src_img = cv2.resize(src_img, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))

        wh = self.config['model']['input_width_height']
        h, w, _ = src_img.shape
        b, l, r, t = self.get_crop_idxs(wh, h, w)
        if not self.is_trn:
            b, l, r, t = self.image_id_crops[image_id]
        src_img = src_img[t:b, l:r]

        # src_img = np.ones_like(src_img) * np.array([217,155,79])[np.newaxis, np.newaxis]

        if not cache_read_successful:
            if self.config['trn']['reflection'] == 'gen':
                separate_refl_img, src_img, usr_mask = self.generate_reflection_on_fly(src_img, image_id)
            elif self.config['trn']['reflection'] == 'rea':
                lamp_bboxes_id = image_id % len(self.lamp_bboxes['path_id'])
                lamp_path = self.lamp_paths[self.lamp_bboxes['path_id'][lamp_bboxes_id]]
                lamp_bbox_x_xx_y_yy = self.lamp_bboxes['bbox'][lamp_bboxes_id]

                separate_refl_img = cv2.imread(lamp_path)
                if separate_refl_img is None:
                    logger.error(' The image {} is very much bad!'.format(lamp_path))
                separate_refl_img = separate_refl_img.astype(np.float32)
                separate_refl_img /= separate_refl_img.max()
                separate_refl_img *= 255
                x, xx, y, yy = lamp_bbox_x_xx_y_yy[:4]
                separate_refl_img = separate_refl_img[y:yy, x:xx]
                separate_refl_img = cv2.resize(separate_refl_img, (wh, wh))

                usr_mask = np.zeros(separate_refl_img.shape)
                separate_refl_img_max = np.max(separate_refl_img, axis=-1)
                usr_mask[separate_refl_img_max > 25] = 255
            elif self.config['trn']['reflection'] == 'ble':
                separate_refl_img_path = os.path.join(self.dataset_dir,
                                                      self.image_filenames[image_id].replace('.png', '_sep.png').replace('.jpg',
                                                                                                                         '_sep.jpg') + '0001.jpg')
                separate_refl_img = cv2.imread(separate_refl_img_path).astype(np.float32)
                if separate_refl_img.max() < 30:
                    separate_refl_img = np.zeros_like(separate_refl_img).astype(np.float32)
                else:
                    separate_refl_img /= separate_refl_img.max()
                separate_refl_img *= 255
            else:
                raise NotImplementedError("Don't know this type of reflection generation.")

            if not self.config['trn']['exact_mask']:
                usr_mask = convert_reflection_to_user_interaction(separate_refl_img, kernel_wh=5,
                                                                  gaussian_blur_sigma=self.config['trn']['mask_gaussian_blur_sigma'])
                usr_mask = np.stack((usr_mask, usr_mask, usr_mask), axis=-1)

            if self.cache_dir is not None:
                cv2.imwrite(separate_refl_img_cache_path, separate_refl_img)

        if hasattr(self, "reflection_augmentation"):
            separate_refl_img = self.reflection_augmentation.augment(image=separate_refl_img.astype(np.uint8))

        separate_refl_img = separate_refl_img.astype(np.float32)
        separate_refl_img -= separate_refl_img.min()
        separate_refl_img = separate_refl_img / (separate_refl_img.max() + 0.000001)
        separate_refl_img *= 255

        usr_mask = np.zeros(separate_refl_img.shape)
        separate_refl_img_max = np.max(separate_refl_img, axis=-1)
        usr_mask[separate_refl_img_max > 50] = 255

        if self.blend_on_fly:
            refl_img, src_img = self.blend(image_id, src_img, usr_mask, separate_refl_img, alpha_min=self.blend_on_fly_alpha_min,
                                           alpha_max=self.blend_on_fly_alpha_max, mode=self.blend_on_fly_mode)
        else:
            refl_img_path = os.path.join(self.dataset_dir, self.image_filenames[image_id])
            refl_img = cv2.imread(refl_img_path)

        return src_img, refl_img, usr_mask, separate_refl_img

    def put_image_in_center(self, separate_refl_img, src_img):
        iwh = self.config['model']['input_width_height']
        size_scalar = min(iwh / float(separate_refl_img.shape[1]), iwh / float(separate_refl_img.shape[0]))
        separate_refl_img = cv2.resize(separate_refl_img, (int(size_scalar * separate_refl_img.shape[1]),
                                                           int(size_scalar * separate_refl_img.shape[0])))
        w = (src_img.shape[1] - separate_refl_img.shape[1]) // 2
        h = (src_img.shape[0] - separate_refl_img.shape[0]) // 2
        separate_refl_img = cv2.copyMakeBorder(separate_refl_img, h,
                                               src_img.shape[0] - separate_refl_img.shape[0] - h, w,
                                               src_img.shape[1] - separate_refl_img.shape[1] - w, cv2.BORDER_REFLECT)
        return separate_refl_img

    def generate_reflection_on_fly(self, src_img, image_id):
        separate_refl_img = np.zeros(src_img.shape)
        usr_mask = np.zeros(src_img.shape, dtype=np.uint8)

        meshgrid = np.mgrid[0:self.config['model']['input_width_height'], 0:self.config['model']['input_width_height']]
        meshgrid = meshgrid[:, 15:-15:15, 15:-15:15]
        move_around = np.random.randint(-5, 5, meshgrid.shape)
        meshgrid = meshgrid + move_around
        blend_on_fly_reflection_color = np.random.randint(0, 255, (meshgrid.shape[1], meshgrid.shape[2], 3))
        if -1 not in self.config['trn']['blend_on_fly_reflection_color']:
            blend_on_fly_reflection_color = self.config['trn']['blend_on_fly_reflection_color']
        if self.config['trn']['generate_refl_type'] == GeneratedReflectionType.haar:
            perturbation_id = image_id % self.haar_perturbations.shape[0]
            half_shift = self.config['trn']['generate_refl_size'] // 2
            for i in range(self.config['trn']['generate_refl_size']):
                for j in range(self.config['trn']['generate_refl_size']):
                    if self.haar_perturbations[perturbation_id, i, j] > 0:
                        separate_refl_img[(meshgrid[0] + i - half_shift, meshgrid[1] + j - half_shift)] = blend_on_fly_reflection_color
                        usr_mask[(meshgrid[0] + i - half_shift, meshgrid[1] + j - half_shift)] = 255
        elif self.config['trn']['generate_refl_type'] == GeneratedReflectionType.many1px_rndcol:
            separate_refl_img[(meshgrid[0], meshgrid[1])] = blend_on_fly_reflection_color  # setting the color
            usr_mask[(meshgrid[0], meshgrid[1])] = 255
        elif self.config['trn']['generate_refl_type'] == GeneratedReflectionType.onepx_rndcol:
            pos = image_id % meshgrid[0].shape[0]
            separate_refl_img[meshgrid[0][pos, pos], meshgrid[1][pos, pos]] = blend_on_fly_reflection_color[
                pos, pos]  # setting the color
            usr_mask[meshgrid[0][pos, pos], meshgrid[1][pos, pos]] = 255

        kernel_size = self.config['trn']['generate_refl_size'] + 2
        assert kernel_size % 2 == 1 and kernel_size >= 3
        if self.config['trn']['generate_refl_type'] == "line":
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, 1] = 1.0
            separate_refl_img = cv2.filter2D(separate_refl_img, -1, kernel)
        if self.config['trn']['generate_refl_type'] == "box":
            kernel = np.zeros((kernel_size + 2, kernel_size + 2))
            kernel[kernel.shape[0]:-kernel.shape[0], kernel.shape[1]:-kernel.shape[1]] = 1.0
            separate_refl_img = cv2.filter2D(separate_refl_img, -1, kernel)
        return separate_refl_img, src_img, usr_mask

    def __getitem__(self, idx):
        return self._getitem(idx)

    def _getitem(self, idx):
        idx = self.idx_map[idx]
        src_img, refl_img, usr_mask, separate_refl_img = self.prepare_imgs(idx)
        loss_mask = usr_mask.copy()
        if self.config['trn']['loss_mask_type'] == 'gaussian':
            ksize = self.config['trn']['loss_mask_kernel_size']
            sigma = self.config['trn']['loss_mask_gaussian_blur_sigma']
            kernel = cv2.getGaussianKernel(ksize, sigma) * cv2.getGaussianKernel(ksize, sigma).transpose()
            loss_mask = cv2.filter2D(loss_mask.transpose((1, 2, 0)), -1, kernel).transpose((2, 0, 1))
            loss_mask = loss_mask / loss_mask.max()
            loss_mask *= 255.
        elif self.config['trn']['loss_mask_type'] == 'ones':
            loss_mask = np.ones(loss_mask.shape, dtype=usr_mask.dtype) * 255.
        if not self.config['trn']['feed_mask']:
            usr_mask = np.zeros(usr_mask.shape, dtype=usr_mask.dtype)
        if self.config['model']['input_channel_num'] == 10:
            usr_mask_01 = usr_mask.copy()
            usr_mask_01 /= 255.0
            inv_usr_mask = usr_mask_01.copy()
            inv_usr_mask -= 1.0
            inv_usr_mask *= -1
            refl_img = np.concatenate((refl_img, refl_img * usr_mask_01, refl_img * inv_usr_mask), axis=0)

        refl_img = self.normalize(refl_img)
        src_img = self.normalize(src_img)
        usr_mask = self.normalize(usr_mask)
        loss_mask_n = self.normalize(loss_mask)
        separate_refl_img = self.normalize(separate_refl_img)

        if self.config['trn']['task'] == 'sep':
            src_img = np.concatenate((src_img, separate_refl_img), axis=0)

        return refl_img, usr_mask[:1], src_img, loss_mask_n, loss_mask[:1] / 255.0

    def _prepare_reflection_augmentation(self):
        from imgaug import augmenters as iaa
        r = self.config['trn']['augment_reflection']
        return iaa.Sequential(
            [
                iaa.Affine(rotate=r['AffineRotate'], shear=r['AffineShear'], scale=r['AffineScale'], mode='edge'),
                iaa.contrast.LinearContrast(r['LinearContrast'], per_channel=False),  # improve or worsen the contrast
                iaa.Sometimes(r['pChangeColorTemperature'],
                              iaa.ChangeColorTemperature(r['ChangeColorTemperature'], from_colorspace=iaa.CSPACE_BGR)
                              ),
                iaa.Add(r['addValue'], per_channel=False),
                iaa.Fliplr(p=r['pFliplr']),
                iaa.Flipud(p=r['pFlipud']),
                iaa.Sometimes(
                    r['pBlendAlpha'],
                    iaa.BlendAlpha(
                        factor=r['BlendAlphaFactor'],
                        foreground=iaa.Affine(translate_px={"x": r['BlendAlphaX'], "y": r['BlendAlphaY']}),
                        per_channel=False
                    )
                ),
                iaa.Sometimes(
                    r['pGaussianBlur'],
                    iaa.GaussianBlur(sigma=r['GaussianBlurSigma'])
                )
            ],
            random_order=True
        )

    def _prepare_augmentation(self):
        from imgaug import augmenters as iaa
        return iaa.Sequential(
            [
                iaa.contrast.LinearContrast((0.3, 2.0), per_channel=False),  # improve or worsen the contrast
                iaa.Add((-50, 50), per_channel=False),
                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            ],
            random_order=True
        )

    def blend(self, image_id, src_img, usr_mask, separate_refl_img, alpha_min=.8, alpha_max=1, mode='add'):
        assert mode == 'add' or mode == 'set' or mode == 'scr' or mode == 'zha' or mode == 'nosat' or mode == 'nosatone' or mode == 'satone' or mode == 'nosatexp' or mode == 'nosatbeta' or mode == 'imp' or mode == 'imp-rnd'
        assert 0 <= alpha_min <= alpha_max <= 1
        alpha = np.random.random() * (alpha_max - alpha_min) + alpha_min
        if mode == 'add':
            blend = src_img + separate_refl_img * alpha
        elif mode == 'set':
            blend = src_img.copy()
            blend[usr_mask > 0] = separate_refl_img[usr_mask > 0]
        elif mode == 'scr':
            blend = 1. - (1. - src_img / 255.) * (1. - (separate_refl_img * alpha) / 255.)
            blend *= 255
        elif mode == 'nosat':
            if not self.is_trn:
                alpha = self.blend_alphas[image_id]
            blend = src_img + separate_refl_img * alpha
            b_max = blend.max()
            if b_max > 255:
                b_max = 255. / b_max
                src_img = src_img * b_max
                blend = blend * b_max
        elif mode == 'satone':
            alpha = 1.0
            blend = self.blend_unlog(alpha, separate_refl_img, src_img)
        elif mode == 'nosatone':
            alpha = 1.0
            blend, src_img = self.blend_unlog_nosat(alpha, separate_refl_img, src_img)
        elif mode == 'nosatexp':
            if self.is_trn:
                sample = np.random.exponential() / 5.0
                rx = sample if sample <= 1.0 else 1.0
            else:
                rx = self.blend_alphas[image_id]
            alpha = (1.0 - rx) * (alpha_max - alpha_min) + alpha_min
            blend, src_img = self.blend_unlog_nosat(alpha, separate_refl_img, src_img)
        elif mode == 'nosatbeta':
            if self.is_trn:
                rx = beta.ppf(np.random.rand(), 0.75, 0.45)
            else:
                rx = self.blend_alphas[image_id]
            alpha = rx * (alpha_max - alpha_min) + alpha_min
            blend, src_img = self.blend_unlog_nosat(alpha, separate_refl_img, src_img)
        elif mode == 'imp':
            blend = src_img.copy()
            blend[usr_mask > 0] = 0
        elif mode == 'imp-rnd':
            blend = src_img.copy()
            how_many_rnd_cols = np.sum(usr_mask > 0)
            blend[usr_mask > 0] = np.random.randint(0, 255, (how_many_rnd_cols,))
        # https://arxiv.org/pdf/1806.05376.pdf based on https://arxiv.org/pdf/1708.03474.pdf
        elif mode == 'zha':
            att = 1.1  # modification - intensity decay fixed
            t = src_img / 255.
            r = alpha * separate_refl_img / 255.
            t = np.power(t, 2.2)
            r = np.power(r, 2.2)
            blend = t + r
            for i in range(3):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r[:, :, i] = r[:, :, i] - (mean_i - 1) * att
            r[r >= 1] = 1
            r[r <= 0] = 0

            # modification due to unnecessary t dimming
            blend = r + t

            blend = np.power(blend, 1 / 2.2)
            blend *= 255

        blend[blend < 0] = 0
        blend[blend > 255] = 255
        return blend, src_img

    def blend_unlog_nosat(self, alpha, separate_refl_img, src_img):
        blend = self.blend_unlog(alpha, separate_refl_img, src_img)
        b_max = blend.max()
        if b_max > 255:
            b_max = 255. / b_max
            src_img = src_img * b_max
            blend = blend * b_max
        return blend, src_img

    def blend_unlog(self, alpha, separate_refl_img, src_img):
        t = src_img / 255.
        r = alpha * separate_refl_img / 255.
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)
        blend = t + r
        blend = np.power(blend, 1 / 2.2)
        blend *= 255
        return blend


class Randomizer():
    def __init__(self, len=64):
        self.image_section_indexes_map = np.arange(len)

    def update_len(self, len):
        self.image_section_indexes_map = np.arange(len)

    def shuffle(self):
        self.image_section_indexes_map = np.random.permutation(np.arange(len(self.image_section_indexes_map)))

    def __len__(self):
        return len(self.image_section_indexes_map)

    def __getitem__(self, item):
        assert isinstance(item, int)

        return self.image_section_indexes_map[item]


class UserGuidedIsolatedReflectionRemovalDatasetFromCOCOSingleImage(UserGuidedIsolatedReflectionRemovalDatasetFromCOCO):
    def __init__(self, config, image_filenames, is_trn=True, cache_dir=None, ds_size=None, randomizer=None):
        super().__init__(config, image_filenames, is_trn, cache_dir, ds_size)

        self.stored_data = []
        self.image_section_indexes = self.__prepare_indexes_of_image_sections_for_crossval()
        self.image_section_randomizer = randomizer
        self.image_section_randomizer.update_len(len(self.image_section_indexes))
        self.crossval_split_idx = int(len(self.image_section_indexes) * 0.9)

        self.image_section_randomizer.shuffle()

        assert self.config['trn'].get('finetune_image_idx', None) is not None

    def __prepare_indexes_of_image_sections_for_crossval(self):
        indexes = []
        for y in range(0, self.config['model']['input_width_height'], 8):
            for x in range(0, self.config['model']['input_width_height'], 8):
                indexes.append((y, x))

        return indexes

    def shuffle(self):
        assert self.is_trn
        # self.image_section_randomizer.shuffle()

    def __getitem__(self, idx):
        if len(self.stored_data) == 0:
            refl_img, src_img, loss_mask = self._getitem(self.config['trn']['finetune_image_idx'])
            self.stored_data = [refl_img, src_img, loss_mask]

        loss_mask = super().unnormalize(self.stored_data[-1])
        loss_mask /= loss_mask.max()

        crossval_loss_mask = np.zeros(loss_mask.shape, dtype=loss_mask.dtype)
        isim_idx_range = range(self.crossval_split_idx)
        if not self.is_trn:
            isim_idx_range = range(self.crossval_split_idx, len(self.image_section_randomizer))

        for isim_idx in isim_idx_range:
            y, x = self.image_section_indexes[self.image_section_randomizer[isim_idx]]
            crossval_loss_mask[:, y:y + 8, x:x + 8] = 1.

        crossval_loss_mask_trn = crossval_loss_mask * (1. - loss_mask)
        crossval_loss_mask_val = crossval_loss_mask * loss_mask

        crossval_loss_mask_trn = self.normalize(crossval_loss_mask_trn * 255.)
        crossval_loss_mask_val = self.normalize(crossval_loss_mask_val * 255.)

        return self.stored_data[0], self.stored_data[1], np.concatenate((crossval_loss_mask_trn, crossval_loss_mask_val), axis=0)

    def __len__(self):
        return 1


class SingleImageDataset:
    def __init__(self, config, image_filename, bboxes_x_xx_y_yy=(), mask_function_0_255_HWC=None, loss_mask_filename=None):
        self.config = config
        self.image_filename = image_filename
        self.image = cv2.imread(self.image_filename)
        self.loss_mask = np.ones_like(self.image)
        self.loss_mask_filename = loss_mask_filename
        if loss_mask_filename is not None:
            self.loss_mask = cv2.imread(self.loss_mask_filename)

        self.target_filename = image_filename.replace('_src', '_rec')
        self.target = None
        if os.path.isfile(self.target_filename):
            self.target = cv2.imread(self.target_filename)
            if np.any(self.target.shape != self.image.shape):
                logger.warning('Target of {} does not have the same dimensions as the src. {} vs {}'.format(self.target_filename,
                                                                                                            str(self.image.shape),
                                                                                                            str(self.target.shape)))
                self.target = None

        self.bboxes_x_xx_y_yy = bboxes_x_xx_y_yy if len(bboxes_x_xx_y_yy) > 0 \
            else [[0, config['model']['input_width_height'], 0, config['model']['input_width_height']]]

        self.mask_function_0_255_HWC = mask_function_0_255_HWC
        self.augmentation = None

    def __len__(self):
        return len(self.bboxes_x_xx_y_yy)

    def normalize(self, img):
        img = img.copy()
        for cycle_id in range(img.shape[0] // 3):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img[idx_from:idx_to] /= 255.0
            img[idx_from:idx_to] -= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean
            img[idx_from:idx_to] /= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std

        return img

    def shuffle(self):
        pass

    @staticmethod
    def unnormalize_loss(img_batch):
        return UserGuidedIsolatedReflectionRemovalDatasetFromCOCOSingleImage.unnormalize_loss(img_batch)

    def __getitem__(self, item):
        x, xx, y, yy = self.bboxes_x_xx_y_yy[item]
        img = self.image[y:yy, x:xx]
        loss_mask = self.loss_mask[y:yy, x:xx]

        img = cv2.resize(img, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
        loss_mask = cv2.resize(loss_mask, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))

        mask = np.zeros_like(img)
        if self.mask_function_0_255_HWC is not None:
            mask = self.mask_function_0_255_HWC(self.config)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        img = img.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        loss_mask = loss_mask.transpose((2, 0, 1)).astype(np.float32)

        img = self.normalize(img)
        mask = self.normalize(mask)
        loss_mask = self.normalize(loss_mask)

        target = img.copy()
        if self.target is not None:
            target = self.target[y:yy, x:xx]
            target = cv2.resize(target, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
            target = target.transpose((2, 0, 1)).astype(np.float32)
            target = self.normalize(target)

        return img, mask[:1], target, loss_mask


class SingleImageDatasetCrossVal:
    def __init__(self, config, image_filename, is_trn, randomizer, bboxes_x_xx_y_yy=(), mask_function_0_255_HWC=None,
                 loss_mask_filename=None,
                 image_section_randomizer=None):
        self.config = config
        self.is_trn = is_trn
        self.image_filename = image_filename
        self.image = cv2.imread(self.image_filename)
        self.target_filename = image_filename.replace('_src', '_rec')
        self.target = None
        if os.path.isfile(self.target_filename):
            self.target = cv2.imread(self.target_filename)

        self.loss_mask = np.ones_like(self.image)
        self.loss_mask_filename = loss_mask_filename
        if loss_mask_filename is not None:
            self.loss_mask = cv2.imread(self.loss_mask_filename)
        self.bboxes_x_xx_y_yy = bboxes_x_xx_y_yy if len(bboxes_x_xx_y_yy) > 0 \
            else [[0, config['model']['input_width_height'], 0, config['model']['input_width_height']]]

        self.mask_function_0_255_HWC = mask_function_0_255_HWC
        self.augmentation = None

        self.stored_data = []
        self.image_section_indexes = self.__prepare_indexes_of_image_sections_for_crossval()

        self.image_section_randomizer = image_section_randomizer
        if image_section_randomizer is None:
            self.image_section_randomizer = randomizer
            self.image_section_randomizer.update_len(len(self.image_section_indexes))
            self.image_section_randomizer.shuffle()

        self.crossval_split_idx = int(len(self.image_section_indexes) * 0.9)

    def __len__(self):
        return len(self.bboxes_x_xx_y_yy)

    def normalize(self, img):
        img = img.copy()
        for cycle_id in range(img.shape[0] // 3):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img[idx_from:idx_to] /= 255.0
            img[idx_from:idx_to] -= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean
            img[idx_from:idx_to] /= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std

        return img

    def shuffle(self):
        assert self.is_trn
        # self.image_section_randomizer.shuffle()

    @staticmethod
    def unnormalize_loss(img_batch):
        return UserGuidedIsolatedReflectionRemovalDatasetFromCOCOSingleImage.unnormalize_loss(img_batch)

    def __prepare_indexes_of_image_sections_for_crossval(self):
        indexes = []
        for y in range(0, self.config['model']['input_width_height'], 8):
            for x in range(0, self.config['model']['input_width_height'], 8):
                indexes.append((y, x))

        return indexes

    def __getitem__(self, item):
        x, xx, y, yy = self.bboxes_x_xx_y_yy[item]
        img = self.image[y:yy, x:xx]
        loss_mask = 255 - self.loss_mask[y:yy, x:xx]
        usr_mask = self.loss_mask[y:yy, x:xx]

        img = cv2.resize(img, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
        loss_mask = cv2.resize(loss_mask, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
        usr_mask = cv2.resize(usr_mask, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))

        loss_mask = loss_mask / loss_mask.max()

        crossval_loss_mask = np.zeros_like(loss_mask)
        isim_idx_range = range(self.crossval_split_idx)
        if not self.is_trn:
            isim_idx_range = range(self.crossval_split_idx, len(self.image_section_randomizer))

        for isim_idx in isim_idx_range:
            isim_y, isim_x = self.image_section_indexes[self.image_section_randomizer[isim_idx]]
            crossval_loss_mask[isim_y:isim_y + 8, isim_x:isim_x + 8, :] = 1.

        crossval_loss_mask = crossval_loss_mask * loss_mask
        crossval_loss_mask = crossval_loss_mask * 255.

        mask = np.zeros_like(img)
        if self.mask_function_0_255_HWC is not None:
            mask = self.mask_function_0_255_HWC(self.config)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        usr_mask = usr_mask.astype(np.float32) / 255.0
        # usr_mask = 1.0 - loss_mask

        img = img.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        usr_mask = usr_mask.transpose((2, 0, 1)).astype(np.float32)
        crossval_loss_mask = crossval_loss_mask.transpose((2, 0, 1)).astype(np.float32)

        img = self.normalize(img)
        mask = self.normalize(mask)
        crossval_loss_mask = self.normalize(crossval_loss_mask)

        target = img.copy()
        if self.target is not None:
            target = self.target[y:yy, x:xx]
            target = cv2.resize(target, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
            target = target.transpose((2, 0, 1)).astype(np.float32)
            target = self.normalize(target)

        return img, mask[:1], target, crossval_loss_mask, usr_mask[:1]


class SingleImageAugmentedDataset:
    def __init__(self, config, image_filename, bboxes_x_xx_y_yy=(), mask_function_0_255_HWC=None, loss_mask_filename=None, ds_size=1,
                 is_trn=False):
        self.config = config
        self.is_trn = is_trn
        self.ds_size = ds_size
        self.image_filename = image_filename
        self.image = cv2.imread(self.image_filename)
        self.loss_mask = np.ones_like(self.image)
        self.loss_mask_filename = loss_mask_filename
        if loss_mask_filename is not None:
            self.loss_mask = cv2.imread(self.loss_mask_filename)

        self.target_filename = image_filename.replace('_src', '_rec')
        self.target = None
        if os.path.isfile(self.target_filename):
            self.target = cv2.imread(self.target_filename)
            if np.any(self.target.shape != self.image.shape):
                logger.warning('Target of {} does not have the same dimensions as the src. {} vs {}'.format(self.target_filename,
                                                                                                            str(self.image.shape),
                                                                                                            str(self.target.shape)))
                self.target = None

        self.bboxes_x_xx_y_yy = bboxes_x_xx_y_yy if len(bboxes_x_xx_y_yy) > 0 \
            else [[0, config['model']['input_width_height'], 0, config['model']['input_width_height']]]

        self.mask_function_0_255_HWC = mask_function_0_255_HWC
        self.augmentation = None
        if self.is_trn:
            self.augmentation = self.__prepare_augmentation()

    def __len__(self):
        return self.ds_size

    def normalize(self, img):
        img = img.copy()
        for cycle_id in range(img.shape[0] // 3):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img[idx_from:idx_to] /= 255.0
            img[idx_from:idx_to] -= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.mean
            img[idx_from:idx_to] /= UserGuidedIsolatedReflectionRemovalDatasetFromCOCO.std

        return img

    def shuffle(self):
        pass

    @staticmethod
    def unnormalize_loss(img_batch):
        return UserGuidedIsolatedReflectionRemovalDatasetFromCOCOSingleImage.unnormalize_loss(img_batch)

    def __getitem__(self, item):
        x, xx, y, yy = self.bboxes_x_xx_y_yy[item % len(self.bboxes_x_xx_y_yy)]
        img = self.image[y:yy, x:xx]
        loss_mask = self.loss_mask[y:yy, x:xx]
        if self.is_trn:
            loss_mask = 255 - self.loss_mask[y:yy, x:xx]

        img = cv2.resize(img, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
        loss_mask = cv2.resize(loss_mask, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))

        mask = np.zeros_like(img)
        if self.mask_function_0_255_HWC is not None:
            mask = self.mask_function_0_255_HWC(self.config)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        if self.augmentation is not None:
            img = self.augmentation.augment_image(img)

        img = img.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        loss_mask = loss_mask.transpose((2, 0, 1)).astype(np.float32)

        # plot_show_maybe_store(np.concatenate((img, mask, loss_mask), axis=-1)[::-1],
        #                       filename='myplot1.png', dir='/home.dokt/spetlrad/cvut83-reflections/')

        img = self.normalize(img)
        mask = self.normalize(mask)
        loss_mask = self.normalize(loss_mask)

        target = img.copy()
        if self.target is not None:
            target = self.target[y:yy, x:xx]
            target = cv2.resize(target, (self.config['model']['input_width_height'], self.config['model']['input_width_height']))
            target = target.transpose((2, 0, 1)).astype(np.float32)
            target = self.normalize(target)

        return img, mask[:1], target, loss_mask, mask[:1]

    @staticmethod
    def __prepare_augmentation():
        from imgaug import augmenters as iaa
        return iaa.Sequential(
            [
                iaa.contrast.LinearContrast((0.3, 2.0), per_channel=False),  # improve or worsen the contrast
                iaa.Add((-50, 50), per_channel=False),
                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            ],
            random_order=True
        )