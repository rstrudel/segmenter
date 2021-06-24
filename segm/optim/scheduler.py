from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler.scheduler import Scheduler


class PolynomialLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size,
        iter_warmup,
        iter_max,
        power,
        min_lr=0,
        last_epoch=-1,
    ):
        self.step_size = step_size
        self.iter_warmup = int(iter_warmup)
        self.iter_max = int(iter_max)
        self.power = power
        self.min_lr = min_lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        iter_cur = float(self.last_epoch)
        if iter_cur < self.iter_warmup:
            coef = iter_cur / self.iter_warmup
            coef *= (1 - self.iter_warmup / self.iter_max) ** self.power
        else:
            coef = (1 - iter_cur / self.iter_max) ** self.power
        return (lr - self.min_lr) * coef + self.min_lr

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]

    def step_update(self, num_updates):
        self.step()
