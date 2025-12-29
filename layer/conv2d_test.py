
import unittest
import torch
from torch import Tensor
from layer.conv2d import Conv2dLayer
from config.layer import Conv2dLayerConfig

class TestConv2dLayer(unittest.TestCase):
    def test_basic_forward(self):
        cfg = Conv2dLayerConfig(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        layer = Conv2dLayer(cfg)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)
        self.assertEqual(out.shape, (1, 16, 32, 32))

    def test_strided(self):
        cfg = Conv2dLayerConfig(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1
        )
        layer = Conv2dLayer(cfg)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)
        self.assertEqual(out.shape, (1, 16, 16, 16))

    def test_padding_modes(self):
        for mode in ["zeros", "reflect", "replicate", "circular"]:
            cfg = Conv2dLayerConfig(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding=1,
                padding_mode=mode  # type: ignore
            )
            layer = Conv2dLayer(cfg)
            x = torch.randn(1, 1, 10, 10)
            _ = layer(x)

if __name__ == '__main__':
    unittest.main()
