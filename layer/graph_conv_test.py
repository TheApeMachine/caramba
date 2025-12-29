
import unittest
import torch
from layer.graph_conv import GraphConvLayer
from config.layer import GraphConvLayerConfig

class TestGraphConvLayer(unittest.TestCase):
    def test_gcn_forward(self):
        cfg = GraphConvLayerConfig(
            in_features=16,
            out_features=8,
            kind="gcn"
        )
        layer = GraphConvLayer(cfg)
        
        # 5 nodes
        x = torch.randn(5, 16)
        # Random adjacency
        adj = torch.eye(5)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        
        out = layer((x, adj))
        self.assertEqual(out.shape, (5, 8))

    def test_gcn_sparse(self):
        cfg = GraphConvLayerConfig(
            in_features=16,
            out_features=8,
            kind="gcn"
        )
        layer = GraphConvLayer(cfg)
        
        x = torch.randn(5, 16)
        adj = torch.eye(5).to_sparse()
        
        out = layer((x, adj))
        self.assertEqual(out.shape, (5, 8))

    def test_gat_forward(self):
        cfg = GraphConvLayerConfig(
            in_features=16,
            out_features=8,
            kind="gat",
            heads=2,
            concat=True
        )
        layer = GraphConvLayer(cfg)
        
        # 5 nodes
        x = torch.randn(5, 16)
        adj = torch.eye(5)
        
        out = layer((x, adj))
        # With concat=True, out is heads * (out_features // heads) = out_features
        # Wait, implementation says: head_dim = out_features // heads
        # out = heads * head_dim = out_features
        self.assertEqual(out.shape, (5, 8))

if __name__ == '__main__':
    unittest.main()
