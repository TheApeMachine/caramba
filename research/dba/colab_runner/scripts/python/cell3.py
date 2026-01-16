# Clone/Update Repository
import os
from pathlib import Path

REPO_DIR = Path("/content/caramba")

if REPO_DIR.exists():
    print("Updating existing repo...")
    !cd {REPO_DIR} && git fetch && git reset --hard origin/{GITHUB_BRANCH}
else:
    print("Cloning repo...")
    !git clone --depth 1 -b {GITHUB_BRANCH} https://github.com/{GITHUB_REPO}.git {REPO_DIR}

print(f"\nRepo ready at {REPO_DIR}")
