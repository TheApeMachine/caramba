
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
from benchmark.multi_model_artifacts import MultiModelArtifactGenerator, MultiModelComparisonSummary
from benchmark.memory import MemoryResult

def reproduce():
    # Mock data
    summary = MultiModelComparisonSummary(
        model_names=["model1", "model2"],
        throughputs={"model1": 100, "model2": 200},
        kv_bytes_per_token={"model1": 10, "model2": 20},
    )
    model_names = ["model1", "model2"]
    colors = ["red", "blue"]

    # Mock MemoryResult
    # The error happens when accessing .peak_allocated_mb on a MemoryResult object
    # inside _plot_overall_comparison

    # Let's try to simulate what _plot_overall_comparison does
    res = MemoryResult(model_name="test")
    try:
        print(f"Accessing peak_allocated_mb: {res.peak_allocated_mb}")
    except AttributeError as e:
        print(f"Caught expected error: {e}")

    try:
        print(f"Accessing peak_memory_mb: {res.peak_memory_mb}")
        print("Successfully accessed peak_memory_mb")
    except AttributeError as e:
        print(f"Unexpected error accessing peak_memory_mb: {e}")

    # Now let's try to reproduce the warning
    # The warning happens in _generate_charts -> ... -> _plot_overall_view (not _plot_overall_comparison for the warning, but likely _plot_overall_view for the bar charts)

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = MultiModelArtifactGenerator(Path(tmpdir))
        # We can't easily call private methods, but we can check if we can trigger it via public ones
        # or just inspecting the code is enough given the clear warning message.

        # Let's just create a small plot to verify fixing the warning
        fig, ax = plt.subplots()
        try:
            # reproducible warning
            ax.set_xticklabels(["a", "b"])
        except UserWarning:
            pass # Use simplefilter to catch it if needed, but it prints to stderr

if __name__ == "__main__":
    reproduce()
