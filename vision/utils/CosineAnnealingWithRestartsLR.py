import math

from torch.optim.lr_scheduler import _LRScheduler

"""
pytorch 1.0.1 version中，暂时官方未支持CosineAnnealingWithRestartsLR学习率调度策略，
1.1.0 version中，已直接支持   lijixiang
"""


class CosineAnnealingWithRestartsLR(_LRScheduler):
    def __init__(self, optimizer, t0, lr_min=0., last_epoch=-1, t_mul=1.0, t_decay=1.0):
        self.T_max = t0
        self.T_mult = t_mul
        self.next_restart = t0
        self.eta_min = lr_min
        self.last_restart = 0
        self.gamma = 0
        self.decay = t_decay
        super(CosineAnnealingWithRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        Tcur = self.last_epoch - self.last_restart
        # print("Tcur={}, last_epoch={}, last_restart={}".format(Tcur, self.last_epoch, self.last_restart))
        # if Tcur >= self.next_restart:
        #     self.next_restart *= self.T_mult
        #     self.last_restart = self.last_epoch
        result = [(self.eta_min +
                 (base_lr - self.eta_min) * (1 + math.cos(math.pi * Tcur / self.next_restart)) / 2)
                for base_lr in self.base_lrs]
        scale = pow(self.decay, self.gamma)
        if Tcur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = self.last_epoch
            self.gamma += 1

        return result*scale


if __name__ == '__main__':
    # for  test
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch.optim as optim

    # example 0
    dummy_net_1 = nn.Sequential(nn.Linear(3, 2))
    optimizer = optim.SGD(dummy_net_1.parameters(), lr=0.05, momentum=0.9)
    scheduler = CosineAnnealingWithRestartsLR(optimizer, t0=160, lr_min=1e-6, t_mul=1.0)
    lrs = []
    for i in range(160):
        scheduler.step()
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
    plt.plot(lrs)
    plt.show()
