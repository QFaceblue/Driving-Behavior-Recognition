import matplotlib.pyplot as plt
import math
import torch.optim as optim
import torch


def change_lr1(epoch, T=5, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr2(epoch, T=7, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    elif epoch < T * 5:
        mul = mul * factor * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr_d1(epoch, T=6, factor=0.3, min=1e-3):
    mul = 1.
    n = epoch / T
    while n > 1:
        mul *= factor
        n -= 1
    return max((1 + math.cos(math.pi * (epoch % T) / T)) * mul / 2, min)


def change_lr3(epoch, T=5, factor=1, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr_d2(epoch, T=5, factor=0.3, min=1e-3):
    mul = 1.
    n = epoch / T
    while n > 1:
        mul *= factor
        n -= 1
    return max((1 + math.cos(math.pi * (epoch % T) / T)) * mul / 2, min)


def getXY(func, epoches, start_lr):
    x = []
    y = []
    for i in range(epoches):
        x.append(i)
        y.append(start_lr * func(i))
    return x, y


def getXY_scheduler(scheduler, epoches):
    x = []
    y = []
    for i in range(epoches):
        x.append(i)
        y.append(scheduler.get_last_lr())
        scheduler.step()
    return x, y


def change_0(epoch):
    return 1.


def change_1(epoch, T=9, factor=0.1, min=1e-3):
    mul = 1.
    n = epoch / T
    while n > 1:
        mul *= factor
        n -= 1
    return max(mul, min)


def change_2(epoch, T=9, min=1e-3):
    return max((1 + math.cos(math.pi * epoch / T)) / 2, min)


def change_3(epoch, T=9, factor=0.3, min=1e-3):
    mul = 1.
    n = (epoch - T) / (2 * T)
    while n > 0:
        mul *= factor
        n -= 1
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


optimizer = optim.SGD([torch.Tensor([1, 1])], lr=1e-3, momentum=0, dampening=0, weight_decay=0)
optimizer = optim.RMSprop([torch.Tensor([1, 1])], lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
optimizer = optim.Adam([torch.Tensor([1, 1])], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.3)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1)
scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr3)
scheduler3 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr1)

# x1, y1 = getXY(change_lr1, epoches=16, start_lr=1e-3)
# x2, y2 = getXY(change_lr2, epoches=36, start_lr=1e-3)
# x3, y3 = getXY(change_lr_d1, epoches=18, start_lr=1e-3)

# x1, y1 = getXY_scheduler(scheduler1, epoches=16)
# x2, y2 = getXY_scheduler(scheduler2, epoches=16)
# x3, y3 = getXY_scheduler(scheduler3, epoches=16)

scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_1)
scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_2)
scheduler3 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_3)
x1, y1 = getXY_scheduler(scheduler1, epoches=27)
x2, y2 = getXY_scheduler(scheduler2, epoches=27)
x3, y3 = getXY_scheduler(scheduler3, epoches=27)
# fig = plt.figure(figsize=(16, 9))
# plt.plot(x2, y2)
# plt.show()
fig = plt.figure(figsize=(16, 9))
sub1 = fig.add_subplot(1, 3, 1)
# sub1.set_title("changelr1")
sub1.set_title("(a)", y=-0.1, fontsize=18)
sub1.plot(x1, y1)
sub2 = fig.add_subplot(1, 3, 2)
# sub2.set_title("changelr2")
sub2.set_title("(b)", y=-0.1, fontsize=18)
sub2.plot(x2, y2)
sub3 = fig.add_subplot(1, 3, 3)
# sub2.set_title("changelr3")
sub3.set_title("(c)", y=-0.1, fontsize=18)
sub3.plot(x3, y3)
plt.show()
