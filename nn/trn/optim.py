from torch.optim.optimizer import Optimizer
import math


class NoSchedulerV0(object):
    def step(self, epoch, data=None):
        pass


class L1SchedulerV1(object):
    def __init__(self, optimizer, logger, verbose=False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose
        self.logger = logger

        self.change_amount = 2.0
        self.last_ratio_of_sign_change = 0.0

    def step(self, epoch, ratio_of_sign_change):
        if math.isnan(ratio_of_sign_change):
            self.logger.info('Epoch {:5d}: not changing learning rate, got NaN'.format(epoch))
            return

        if self.last_ratio_of_sign_change > ratio_of_sign_change:
            if self.change_amount > 1:
                self.change_amount = 0.5
            else:
                self.change_amount = 2.0
        old_lr, new_lr = self._change_lr(epoch)

        if self.verbose:
            self.logger.info('Epoch {:5d}: changing learning rate'
                             ' from {:.4e} to {:.4e} (old ratio: {:.1f}, new ratio: {:.1f}).'.format(epoch, old_lr, new_lr,
                                                                                                   self.last_ratio_of_sign_change,
                                                                                                   ratio_of_sign_change))
        self.last_ratio_of_sign_change = ratio_of_sign_change

    def _change_lr(self, epoch):
        old_lr = -1
        new_lr = -1
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * float(self.change_amount)

            param_group['lr'] = new_lr
        return old_lr, new_lr
