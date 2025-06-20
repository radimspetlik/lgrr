import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UpscaleUGSIIRR(nn.Module):
    def __init__(self,
                 use_cuda,
                 config,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        super(UpscaleUGSIIRR, self).__init__()

        if block is None:
            block = InvertedResidual
        self.input_channel_num = config['model']['input_channel_num']
        self.output_channel_num = config['model']['output_channel_num']
        lateral_channel_num = config['model']['lateral_channel_num']

        if inverted_residual_setting is None:
            self.inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(self.inverted_residual_setting) == 0 or len(self.inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(self.inverted_residual_setting))

        # building first layer
        last_channel = 1280
        input_channel = 32
        self.top_lateral_layer = nn.Conv2d(_make_divisible(input_channel * width_mult, round_nearest),
                                           lateral_channel_num, kernel_size=1, stride=1, padding=0)
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(self.input_channel_num, input_channel, stride=1)]

        self.inverted_residual_blocks = []
        # building inverted residual blocks
        for t, c, n, s in self.inverted_residual_setting:
            inverted_residual_blocks = []
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                inverted_residual_block = block(input_channel, output_channel, stride, expand_ratio=t)
                features.append(inverted_residual_block)
                inverted_residual_blocks.append(inverted_residual_block)
                input_channel = output_channel
            self.inverted_residual_blocks.append(nn.Sequential(*inverted_residual_blocks))
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.last_lateral_layer = nn.Conv2d(_make_divisible(last_channel * width_mult, round_nearest),
                                            lateral_channel_num, kernel_size=1, stride=1, padding=0)
        self.inverted_residual_blocks = nn.ModuleList(self.inverted_residual_blocks)
        # make features nn.Sequential
        self.features = nn.Sequential(*features)

        # Lateral layers
        # exclude last setting as this lateral connection is the the top layer
        # build layer only if resolution has decreased (stride > 1)
        self.lateral_setting = [setting for setting in self.inverted_residual_setting[:-1]
                                if setting[3] > 1]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(_make_divisible(setting[1] * width_mult, round_nearest),
                      lateral_channel_num, kernel_size=1, stride=1, padding=0)
            for setting in self.lateral_setting])

        # Smooth layers
        # n = lateral layers + 1 for top layer
        self.smooth_layers = nn.ModuleList([nn.Sequential(nn.ReflectionPad2d(1),
                                                          nn.Conv2d(lateral_channel_num, lateral_channel_num, kernel_size=3,
                                                                    stride=1))] *
                                           (len(self.lateral_layers) + 2))

        # reduce feature maps to one pixel
        # allows to upsample semantic information of every part of the image
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # UPSCALE PART
        self.concatenate_up_to_layer = 3
        self.upscale_left_blocks = nn.ModuleList([
            ConvBNReLU(lateral_channel_num * (1 + self.concatenate_up_to_layer), lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            nn.Sequential(nn.ReflectionPad2d(1),
                          nn.Conv2d(lateral_channel_num, self.output_channel_num, kernel_size=3, stride=1))])

        self.upscale_left_mult_blocks = nn.ModuleList([
            ConvBNReLU(lateral_channel_num * (1 + self.concatenate_up_to_layer), lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            nn.Sequential(nn.ReflectionPad2d(1),
                          nn.Conv2d(lateral_channel_num, self.output_channel_num, kernel_size=3, stride=1))])

        self.upscale_right_blocks = nn.ModuleList([
            ConvBNReLU(lateral_channel_num * (1 + self.concatenate_up_to_layer), lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            nn.Sequential(nn.ReflectionPad2d(1),
                          nn.Conv2d(lateral_channel_num, self.output_channel_num, kernel_size=3, stride=1))])

        self.upscale_right_mult_blocks = nn.ModuleList([
            ConvBNReLU(lateral_channel_num * (1 + self.concatenate_up_to_layer), lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            ConvBNReLU(lateral_channel_num, lateral_channel_num, kernel_size=3, stride=1),
            nn.Sequential(nn.ReflectionPad2d(1),
                          nn.Conv2d(lateral_channel_num, self.output_channel_num, kernel_size=3, stride=1))])

        self.mask_pred_conv = nn.Conv2d(2 * self.output_channel_num, 1, kernel_size=1, stride=1)

        self._initialize_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def load_state_dict_partly(self, state_dict, fix_params=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'upscale_block_top' in name:
                name = 'upscale_left_blocks.3.' + name.split('_')[-1][4:]
            elif 'upscale_block_' in name:
                name = 'upscale_left_blocks.'+name.split('_')[-1]
            if name not in own_state:
                logging.getLogger('training').warning(' Did not find param %s' % name)
                continue
            if param.size() != own_state[name].size():
                logging.getLogger('training').warning(' Skipping incompatible sized param %s' % name)
                continue
            own_state[name].copy_(param)
            if own_state[name].dtype == torch.float32 or own_state[name].dtype == torch.float64:
                own_state[name].requires_grad = not fix_params

    def forward(self, x):
        # storing x size to use later
        bs, ch, h, w = x.size()

        # bottom up
        residual = self.features[0](x)

        # loop through inverted_residual_blocks (mobile_netV2)
        # save lateral_connections to lateral_tensors
        # track how many lateral connections have been made
        lateral_tensors = [self.top_lateral_layer(residual)]
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(residual)  # run block of mobile_net_V2
            if self.inverted_residual_setting[i][3] > 1 and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append(self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            residual = output

        # connect m_layer with previous m_layer and lateral layers recursively
        m_layers = [self.last_lateral_layer(self.features[-1](residual))]
        # reverse lateral tensor order for top down
        lateral_tensors.reverse()
        for lateral_tensor in lateral_tensors:
            m_layers.append(self._upsample_add(m_layers[-1], lateral_tensor))

        # smooth all m_layers
        assert len(self.smooth_layers) == len(m_layers)
        smoothed_lateral_layers = [smooth_layer(m_layer) for smooth_layer, m_layer in zip(self.smooth_layers, m_layers)]

        _, __, H, W = smoothed_lateral_layers[self.concatenate_up_to_layer].size()
        for p_layer_id in range(self.concatenate_up_to_layer):
            smoothed_lateral_layers[p_layer_id] = F.interpolate(smoothed_lateral_layers[p_layer_id], size=(H, W), mode='bilinear',
                                                                align_corners=False)

        residual = torch.cat(tuple(p_layer for p_layer in smoothed_lateral_layers[:self.concatenate_up_to_layer + 1]), 1)

        residual_left = self.upscale_smooth(h, residual, self.upscale_left_blocks, smoothed_lateral_layers, w)
        residual_left_mult = self.upscale_smooth(h, residual, self.upscale_left_mult_blocks, smoothed_lateral_layers, w)
        residual_right = self.upscale_smooth(h, residual, self.upscale_right_blocks, smoothed_lateral_layers, w)
        residual_right_mult = self.upscale_smooth(h, residual, self.upscale_right_mult_blocks, smoothed_lateral_layers, w)

        img =  x[:, :self.output_channel_num] \
               + torch.sigmoid(residual_left_mult) * residual_left \
               - torch.sigmoid(residual_right_mult) * residual_right

        mask_pred = self.mask_pred_conv(torch.cat((torch.sigmoid(residual_left_mult), - torch.sigmoid(residual_right_mult)), dim=1))

        return torch.cat((img, torch.sigmoid(mask_pred)), dim=1)

    def upscale_smooth(self, h, residual, upscale_left_blocks, smoothed_lateral_layers, w):
        residual = upscale_left_blocks[0](residual)
        residual = self._upsample_add(residual, smoothed_lateral_layers[self.concatenate_up_to_layer + 1])

        residual = upscale_left_blocks[1](residual)
        residual = F.interpolate(residual, size=(h, w), mode='bilinear', align_corners=False)

        residual = upscale_left_blocks[2](residual)
        residual = self._upsample_add(residual, smoothed_lateral_layers[self.concatenate_up_to_layer + 2])

        residual = upscale_left_blocks[-1](residual)
        return residual

    def _concat_with_resized_mask(self, mask, residual):
        _, __, H, W = residual.size()
        mask_to_add = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        residual = torch.cat((residual, mask_to_add), 1)
        return residual

    def zero_grad_custom(self):
        pass
