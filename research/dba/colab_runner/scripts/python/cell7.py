# Display Results Summary
from pathlib import Path

dashboard_file = output_dir / "report.html"
if dashboard_file.exists():
    print(f"Dashboard saved to: {dashboard_file}")
else:
    print("No dashboard file found.")

print("\nOutput files:")
for f in sorted(output_dir.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  - {f.name} ({size_kb:.1f} KB)")
