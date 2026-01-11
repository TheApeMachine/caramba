import pytest
from pydantic import ValidationError, TypeAdapter
from caramba.config.compute import LocalComputeConfig, VastAIComputeConfig, ComputeConfig
from caramba.config.target import ExperimentTargetConfig

def test_local_compute_config():
    cfg = LocalComputeConfig(device="cuda")
    assert cfg.type == "local"
    assert cfg.device == "cuda"

def test_vast_ai_compute_config():
    cfg = VastAIComputeConfig(gpu_name="H100", max_price_per_hr=2.5)
    assert cfg.type == "vast_ai"
    assert cfg.gpu_name == "H100"
    assert cfg.max_price_per_hr == 2.5


def test_vast_ai_zero_price() -> None:
    with pytest.raises(ValidationError):
        VastAIComputeConfig(max_price_per_hr=0.0)


def test_compute_config_union():
    # Test discriminators
    adapter = TypeAdapter(ComputeConfig)
    cfg1 = adapter.validate_python({"type": "local", "device": "mps"})
    assert isinstance(cfg1, LocalComputeConfig)
    
    cfg2 = adapter.validate_python({"type": "vast_ai", "gpu_name": "L4"})
    assert isinstance(cfg2, VastAIComputeConfig)

def test_invalid_compute_config():
    adapter = TypeAdapter(ComputeConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown"})
    
    with pytest.raises(ValidationError):
        # max_price_per_hr must be positive
        VastAIComputeConfig(max_price_per_hr=-1.0)

def test_target_with_compute():
    # Verify that ExperimentTargetConfig accepts compute
    data = {
        "name": "test-target",
        "backend": "lightning",
        "compute": {
            "type": "vast_ai",
            "gpu_name": "A100"
        },
        "task": {"ref": "task.test"},
        "data": {"ref": "dataset.test", "config": {"path": "test.npy", "block_size": 128}},
        "system": {"ref": "system.test", "config": {"model": {"type": "test"}}},
        "objective": {"ref": "objective.test"},
        "trainer": {"ref": "trainer.test"},
        "runs": []
    }
    target = ExperimentTargetConfig.model_validate(data)
    assert target.compute.type == "vast_ai"
    assert target.compute.gpu_name == "A100"
    assert target.backend == "lightning"
