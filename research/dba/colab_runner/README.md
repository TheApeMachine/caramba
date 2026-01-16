# DBA Colab Benchmark Dispatcher

Dispatch DBA benchmark jobs from your local machine to run on Google Colab's free GPU.

## How It Works

1. **Local Machine**: Creates a configured Jupyter notebook with your settings
2. **Playwright**: Automates Chrome to upload and run the notebook on Colab
3. **Google Colab**: Executes the benchmark using free GPU (T4/P100)
4. **Google Drive**: Stores checkpoints (input) and results (output)

## Setup

### 1. Install Dependencies

```bash
pip install playwright
playwright install chromium
```

### 2. Prepare Google Drive

Upload your checkpoints to Google Drive:
```
My Drive/
└── DBA/
    ├── checkpoints/
    │   └── 100k/
    │       ├── baseline/
    │       │   └── a100_fw1b_l22_baseline_s42_100k.pt
    │       └── sem8geo32v40/
    │           └── a100_fw1b_l22_dba_s42_100k.pt
    └── results/  (created automatically)
```

## Usage

### Basic Usage

```bash
cd research/dba

# Run benchmark on 100k checkpoints
python colab_runner/dispatch.py --checkpoint-dir "/DBA/checkpoints/100k"

# Or use Makefile
make colab-dispatch CHECKPOINT_DIR="/DBA/checkpoints/100k"
```

### All Options

```bash
python colab_runner/dispatch.py \
    --checkpoint-dir "/DBA/checkpoints/100k" \
    --results-dir "/DBA/results" \
    --tests-per-category 30 \
    --seed 42 \
    --webhook "https://hooks.slack.com/services/..." \
    --timeout 7200 \
    --headless
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | (required) | Google Drive path to checkpoints |
| `--results-dir` | `/DBA/results` | Google Drive path for results |
| `--tests-per-category` | 30 | Number of tests per category |
| `--seed` | 42 | Random seed for reproducibility |
| `--webhook` | None | Slack/Discord webhook for notifications |
| `--github-repo` | `theapemachine/caramba` | GitHub repo to clone |
| `--github-branch` | `main` | Branch to use |
| `--headless` | False | Run browser without visible window |
| `--timeout` | 3600 | Max execution time (seconds) |
| `--notebook-only` | False | Only generate notebook, don't run |

## What Happens During Execution

1. **Browser Launch**: Chrome opens (visible unless `--headless`)
2. **Google Sign-In**: If not already signed in, you'll be prompted
3. **Notebook Upload**: Configured notebook is uploaded to Colab
4. **GPU Runtime**: Automatically selects GPU accelerator
5. **Drive Mount**: You may need to authorize Drive access
6. **Execution**: All cells run sequentially
7. **Monitoring**: Progress is shown in terminal
8. **Completion**: Results saved to Drive, optional webhook notification

## Example Workflow

```bash
# 1. Quick test (5 tests per category)
python colab_runner/dispatch.py \
    --checkpoint-dir "/DBA/checkpoints/100k" \
    --tests-per-category 5

# 2. Full benchmark with notification
python colab_runner/dispatch.py \
    --checkpoint-dir "/DBA/checkpoints/100k" \
    --tests-per-category 30 \
    --webhook "https://hooks.slack.com/services/XXX/YYY/ZZZ" \
    --timeout 7200

# 3. Headless overnight run
python colab_runner/dispatch.py \
    --checkpoint-dir "/DBA/checkpoints/100k" \
    --headless \
    --timeout 14400
```

## Retrieving Results

Results are saved to Google Drive at the configured `--results-dir`:

```
My Drive/DBA/results/
└── behavioral_20240115_123456/
    ├── results.json          # Full benchmark results
    ├── test_suite.json       # Test definitions
    ├── report.html           # Interactive dashboard
    └── attention_samples.json # Attention data (if enabled)
```

Download from Drive and open `report.html` locally, or sync with Google Drive desktop app.

## Troubleshooting

### "Playwright not found"
```bash
pip install playwright
playwright install chromium
```

### Sign-in issues
- Don't use `--headless` for first run
- Complete Google sign-in manually
- Browser stays open for inspection

### GPU not available
- Colab free tier has usage limits
- Try again later or use Colab Pro
- Check Runtime > Change runtime type

### Drive mount fails
- Click "Allow" when prompted in Colab
- Make sure Drive paths are correct

### Timeout
- Increase `--timeout` for large benchmarks
- Use `--tests-per-category 5` for testing
- Check Colab output for errors

## Notebook Template

The notebook (`benchmark_notebook.ipynb`) contains:

1. **Configuration**: Editable form fields
2. **GPU Check**: Verifies CUDA is available
3. **Dependencies**: Installs tiktoken, pyyaml
4. **Repository**: Clones caramba from GitHub
5. **Checkpoint Discovery**: Finds all .pt files
6. **Benchmark Run**: Executes behavioral_suite_v2
7. **Notification**: Sends webhook (optional)
8. **Results Summary**: Lists output files

You can also manually upload and run the notebook:
```bash
python colab_runner/dispatch.py --notebook-only --checkpoint-dir "/DBA/checkpoints/100k"
# Then manually upload the generated notebook to Colab
```
