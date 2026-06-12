import torch
import torch.nn as nn

from .CamoFormer import CamoFormer


class CamoFormerMSI8(nn.Module):
    def __init__(self, cfg):
        super(CamoFormerMSI8, self).__init__()
        self.adapter = nn.Conv2d(8, 3, kernel_size=1, bias=False)
        self.camoformer = CamoFormer(cfg)
        self.initialize_adapter()

    def initialize_adapter(self):
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[0, 4, 0, 0] = 1.0
            self.adapter.weight[1, 2, 0, 0] = 1.0
            self.adapter.weight[2, 1, 0, 0] = 1.0

    def forward(self, x, shape=None, name=None):
        x = self.adapter(x)
        return self.camoformer(x, shape, name)
