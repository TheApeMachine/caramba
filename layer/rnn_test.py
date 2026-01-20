
import unittest
import torch
from layer.rnn import RNNLayer
from config.layer import RNNLayerConfig

class TestRNNLayer(unittest.TestCase):
    def test_lstm_forward(self):
        cfg = RNNLayerConfig(
            input_size=10,
            hidden_size=20,
            cell_type="lstm",
            num_layers=1,
            batch_first=True
        )
        layer = RNNLayer(cfg)
        x = torch.randn(2, 5, 10)  # B, T, D
        out, (h, c) = layer(x)
        self.assertEqual(out.shape, (2, 5, 20))
        self.assertEqual(h.shape, (1, 2, 20))  # D*layers, B, H

    def test_gru_forward(self):
        cfg = RNNLayerConfig(
            input_size=10,
            hidden_size=20,
            cell_type="gru",
            num_layers=2,
            batch_first=True
        )
        layer = RNNLayer(cfg)
        x = torch.randn(2, 5, 10)
        out, h = layer(x)
        self.assertEqual(out.shape, (2, 5, 20))
        self.assertEqual(h.shape, (2, 2, 20))

if __name__ == '__main__':
    unittest.main()
