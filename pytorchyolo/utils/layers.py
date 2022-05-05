import torch 
import torch.nn as nn 
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers, weight=False):
        super().__init__()
        self.layers = layers 
        self.weight = weight 
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)

        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]

            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, :nx]
        return x 


class FeatureConcat(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.layers = layers 
		self.multiple = len(layers) > 1

	def forward(self, x, outputs):
		return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Mish(nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))




class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()