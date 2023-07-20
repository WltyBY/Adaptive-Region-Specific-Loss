import torch
from torch import nn
import numpy as np
import time


class Adaptive_Region_Specific_TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-5, region_size=(16, 16, 16), do_bg=True, A=0.3, B=0.4):
        """
        3D region_size's axis in (x, y, z)
        2D region_size's axis in (x, y)
        """
        super(Adaptive_Region_Specific_TverskyLoss, self).__init__()
        self.smooth = smooth
        self.region_size = region_size
        self.do_bg = do_bg
        self.dim = len(region_size)
        assert self.dim in [2, 3], "The num of dim must be 2 or 3."
        if self.dim == 3:
            self.pool = nn.AvgPool3d(kernel_size=(region_size[2], region_size[0], region_size[1]),
                                     stride=(region_size[2], region_size[0], region_size[1]))
        elif self.dim == 2:
            self.pool = nn.AvgPool2d(kernel_size=region_size, stride=region_size)

        self.A = A
        self.B = B
        self.each_box_num = np.prod(self.region_size)

    def forward(self, x, y):
        # 默认x是未经过softmax的。2D/3D: [batchsize, c, (z,) x, y]
        x = torch.softmax(x, dim=1)

        shp_x, shp_y = x.shape, y.shape
        assert self.dim == (len(shp_x) - 2), "The region size must match the data's size."

        # get num of boxes in each axis
        if self.dim == 2:
            assert x.shape[-2] % self.region_size[0] == 0, "x The data/region_size must be divisible"
            assert x.shape[-1] % self.region_size[1] == 0, "y The data/region_size must be divisible"
        elif self.dim == 3:
            assert x.shape[-2] % self.region_size[0] == 0, "x The data/region_size must be divisible"
            assert x.shape[-1] % self.region_size[1] == 0, "y The data/region_size must be divisible"
            assert x.shape[-3] % self.region_size[2] == 0, "z The data/region_size must be divisible"

        if not self.do_bg:
            x = x[:, 1:]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # the three in [batchsize, class_num, (z,) x, y]
        tp = x * y_onehot
        fp = x * (1 - y_onehot)
        fn = (1 - x) * y_onehot

        # the three in [batchsize, class_num, (z/region_z,) x/region_x, y/region_y]
        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        # [batchsize, class_num, (z/region_z,) x/region_x, y/region_y]
        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        # [batchsize, class_num, (z / region_z,) x / region_x, y / region_y]
        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)
        region_tversky = 1 - region_tversky

        # [batchsize, class_num]
        region_tversky = region_tversky.sum(list(range(2, len(shp_x))))

        region_tversky = region_tversky.mean()

        return region_tversky


if __name__ == "__main__":
    # the number of regions in 3D-test and 2D-test is 1024
    loss = Adaptive_Region_Specific_TverskyLoss(region_size=(16, 14, 8))
    size = (2, 2, 64, 128, 224)
    pre = torch.softmax(torch.rand(size), dim=1)
    label = torch.randint(0, 2, size)
    start = time.time()
    print(loss(pre, label))
    time_length = time.time() - start
    print("Time cost: {}s".format(time_length))

    loss = Adaptive_Region_Specific_TverskyLoss(region_size=(16, 16))
    size = (2, 2, 512, 512)
    pre = torch.softmax(torch.rand(size), dim=1)
    label = torch.randint(0, 2, size)
    start = time.time()
    print(loss(pre, label))
    time_length = time.time() - start
    print("Time cost: {}s".format(time_length))
