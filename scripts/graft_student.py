import torch
import sys
import shutil
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python graft_student.py <student_ckpt> <output_dir>")
        sys.exit(1)

    student_ckpt = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    print(f"Grafting {student_ckpt} -> {output_dir}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We need to create a checkpoint that StandardTrainer will recognize as a "resume" point.
    # StandardTrainer looks for `{run_id}_{phase}_final.pt` or similar.
    # Let's just copy it to `runs/instruct/sft_standard_final.pt` isn't quite right.
    # It looks for `latest` via Checkpointer.
    
    # Actually, we can just use the `load_state_dict` approach in a custom script if we wanted,
    # but to stay in the framework:
    
    # The framework's Checkpointer saves: {'system_state_dict': ..., 'run_id': ..., 'step': ...}
    # Your student checkpoint should already have this format if it came from UpcycleTrainer.
    
    try:
        ckpt = torch.load(student_ckpt, map_location="cpu")
        if "system_state_dict" not in ckpt:
            print("⚠️ Checkpoint missing 'system_state_dict'. Wrapping it...")
            # Assume it's a raw state dict
            ckpt = {
                "system_state_dict": ckpt,
                "run_id": "sft",
                "step": 0
            }
            torch.save(ckpt, output_dir / "sft_standard_0.pt")
        else:
            # It's a valid checkpoint. Reset step to 0 for the new run.
            ckpt["step"] = 0
            ckpt["run_id"] = "sft"
            torch.save(ckpt, output_dir / "sft_standard_0.pt")
            
        print("✅ Created graft checkpoint: sft_standard_0.pt")
        print("You can now run the training manifest.")
        
    except Exception as e:
        print(f"❌ Failed to graft: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
