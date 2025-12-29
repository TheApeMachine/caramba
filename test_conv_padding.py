
from config.layer import Conv2dLayerConfig
from layer.conv2d import Conv2dLayer

def test_conv2d_padding_mode():
    print("Testing Conv2d padding_mode...")
    
    # Test valid modes
    for mode in ["zeros", "reflect", "replicate", "circular"]:
        cfg = Conv2dLayerConfig(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding_mode=mode  # type: ignore
        )
        layer = cfg.build()
        assert isinstance(layer, Conv2dLayer)
        print(f"✓ padding_mode='{mode}' accepted")

if __name__ == "__main__":
    test_conv2d_padding_mode()
