
import unittest
import torch
from layer.dense import DenseLayer
from config.layer import DenseLayerConfig

class TestDenseLayer(unittest.TestCase):
    def test_basic(self):
        cfg = DenseLayerConfig(d_in=10, d_out=5)
        layer = DenseLayer(cfg)
        x = torch.randn(2, 10)
        out = layer(x)
        self.assertEqual(out.shape, (2, 5))

    def test_full_features(self):
        cfg = DenseLayerConfig(
            d_in=10,
            d_out=5,
            activation="relu",
            normalization="layer_norm",
            dropout=0.1
        )
        layer = DenseLayer(cfg)
        x = torch.randn(2, 10)
        out = layer(x)
        self.assertEqual(out.shape, (2, 5))

if __name__ == '__main__':
    unittest.main()
