
```bash
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██
  by dernet     ((BETA TESTING))
```

## What is this?

Side-Step is a CLI that runs to the side of ACE-Step 1.5 root folder (hence the name) that lets you do LoRA fine-tuning for ACE-Step models 1.5 on your own music.
You give it audio files, it learns the style, and produces a small LoRA
adapter file you can load at inference time to generate music in that style.

This was mainly built with cloud computing in mind, but optimizers with lower VRAM requirements have been added for those who want to use it locally on underpowered gear.

## For the initial commit

What works and what doesn't?

-Works: Vanilla training loop, fixed training loop.

-Doesn't work: Gradient estimations, preprocessing from the CLI.

-DO NOT USE: the TUI unless you want to test it. It is currently broken and i don't want you wasting your time trying to get it to work.

## Before you start

You need to have the original ACE-Step 1.5 folder and all requirements from that project installed. Side step **Will not work** if you do not have these installed.

## How to install?

Clone this repository with
```bash
git clone https://github.com/koda-dernet/Side-Step.git
```

## Prerequisites

- Everything that the original ACE-Step 1.5 repository requires, namely:
- Python 3.11+
- PyTorch 2.10 with CUDA support (NVIDIA GPU recommended, 16GB+ VRAM)
- The ACE-Step 1.5 model checkpoints downloaded to `./checkpoints/`
- Your audio dataset preprocessed into `.pt` tensor files (see Step 1 below)
- Install ACE-Step's core dependencies first (see ACE-Step's own README), then:

```bash
pip install -r requirements-sidestep.txt
```

or if you are using uv (which is recommended for this project)

```bash
uv pip install -r requirements-sidestep.txt
```

This installs the Side-Step extras: `rich` (CLI formatting) and `textual` (TUI).


### Optional dependencies

For low-VRAM 8-bit optimizers:
```bash
pip install bitsandbytes>=0.45.0
```

For the Prodigy adaptive optimizer (auto-tunes learning rate):
```bash
pip install prodigyopt>=1.1.2
```

You will get a folder named "training_v2" and three other files. **Do not attempt to run them on their own**. They need to be inside your ACE-Step 1.5 folder. It is a **non-destructive parallel module** (`acestep/training_v2/`)
that lives alongside the existing `acestep/training/` directory. The existing
training code is **never modified** -- Side-Step imports from it where needed.

```
train.py                          CLI entry point (repo root)
sidestep_tui.py                   TUI entry point (repo root)
requirements-sidestep.txt         Side-Step extra dependencies
    |
    v
acestep/training_v2/
    cli/
        common.py                 Shared argparse, validation, config construction
        train_vanilla.py          vanilla subcommand -> original LoRATrainer
        train_fixed.py            fixed subcommand -> FixedLoRATrainer
    tui/
        app.py                    Main Textual application
        state.py                  Centralized reactive state management
        theme.py                  CSS-like styling
        screens/
            dashboard.py          Dashboard with quick actions, runs, stats
            training_config.py    Interactive training form with VRAM profiles
            training_monitor.py   Live training progress display
            dataset_browser.py    Dataset browsing + preprocessing
            preprocess_monitor.py Preprocessing progress monitor
            estimate.py           Gradient sensitivity estimation UI
            run_history.py        Run history viewer
            settings.py           User preferences
        widgets/
            gpu_gauge.py          Live GPU utilization gauge
            log_viewer.py         Scrolling log display
            loss_sparkline.py     Inline loss graph
            file_picker.py        File/directory picker
    ui/
        banner.py                 CLI startup banner
        progress.py               Rich live training display
        wizard.py                 Interactive wizard mode
        errors.py                 Error display utilities
        gpu_monitor.py            GPU monitoring for CLI
    configs.py                    LoRAConfigV2, TrainingConfigV2 (extend originals)
    optim.py                      Optimizer & scheduler factory functions
    gpu_utils.py                  Auto GPU detection, VRAM query
    model_loader.py               Per-phase model loading
    timestep_sampling.py          Corrected sample_t_r() + CFG dropout
    trainer_fixed.py              FixedLoRAModule + FixedLoRATrainer
    trainer_vanilla.py            VanillaTrainer adapter for TUI
    estimate.py                   Gradient sensitivity estimation module
    preprocess.py                 Audio preprocessing module
    tensorboard_utils.py          TrainingLogger wrapper
    make_test_fixtures.py         Synthetic test data generator (for testing and debugging)
```


### Platform support

Side-Step works on both **Linux** and **Windows**. On Windows, `num_workers`
defaults to 0 automatically. Configuration is stored in `%APPDATA%\sidestep\`
on Windows and `~/.config/sidestep/` on Linux. 
MacOS hasn't been yet tested.



## Quick Start

### Step 1: Prepare your dataset

Before training, your audio files need to be preprocessed into tensor format.
If you've already built a dataset using the ACE-Step Dataset Builder (from the
Gradio UI or the dataset builder CLI), you'll have a directory full of `.pt`
files (optionally with a `manifest.json`). That directory is your `--dataset-dir`.
Note: `manifest.json` is optional -- if missing, the loader automatically scans
for `.pt` files in the directory.

### Step 2: Train

First, opend a command prompt or bash in your ACE-Step 1.5 installation and then you have two ways to start training:

**Option A: Interactive wizard** (just run `train.py` with no arguments)

```bash
python train.py
```
or 

```bash
uv run train.py
```

This launches a step-by-step wizard that walks you through every setting
with keyboard prompts and sane defaults. Press Enter to accept defaults,
or type a new value.

The wizard offers two configuration modes:

- **Basic mode** (default): Asks only essential questions -- paths, LoRA settings,
  basic training parameters. Uses sensible defaults for everything else.
- **Advanced mode**: Exposes ALL settings including device/precision selection,
  weight decay, gradient clipping, data loader tuning, and logging options.

First-time users should use Basic mode; power users who want full control can
choose Advanced mode.

**Option B: Command-line arguments** (one-liner)

```bash
python train.py fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --dataset-dir ./preprocessed_tensors/my_dataset \
    --output-dir ./lora_output/my_lora \
    --epochs 200
```

Both methods produce the same result. The wizard is great for first-time
users; CLI arguments are better for scripts and repeat runs.

This uses the recommended defaults (rank 64, alpha 128). When it finishes,
your LoRA adapter is in `./lora_output/my_lora/final/adapter/`.

**Option C: Interactive TUI** (full terminal interface) **(IN ALPHA TESTING, UNSTABLE AND HONESTLY BROKEN, DO NOT USE)**

```bash
python sidestep_tui.py
```

This launches a full interactive terminal app with visual forms, live GPU
monitoring, progress bars, and run history. Navigate with keyboard shortcuts:
- **[F]** Fixed training, **[V]** Vanilla training
- **[P]** Preprocess, **[E]** Estimate
- **[D]** Dashboard, **[H]** History, **[S]** Settings
- **[?]** Help, **[Q]** Quit

The TUI includes a **VRAM Profile** selector that auto-fills recommended
settings based on your GPU's VRAM, and an **Expert Mode** toggle for
advanced low-level settings.



## Training Modes

There are two available training modes. Use **fixed** unless you have a
specific reason to use vanilla.

### `fixed` (recommended)

Corrected training that matches how the model was originally trained:
- Continuous timestep sampling (not discrete 8-step)
- CFG dropout (randomly drops conditions 15% of the time during training)

```bash
python train.py fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --dataset-dir ./preprocessed_tensors/jazz_piano \
    --output-dir ./lora_output/jazz_fixed \
    --epochs 200
```

### `vanilla`

Reproduces the original training behavior for backward compatibility. This
mode has known differences from how the model was actually trained. A warning
is always printed. Use only if you need to reproduce results from older LoRAs.

```bash
python train.py vanilla \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --dataset-dir ./preprocessed_tensors/jazz_piano \
    --output-dir ./lora_output/jazz_vanilla \
    --epochs 200
```

## Complete Argument Reference

Every argument, its default, and what it does.

### Global Flags

Available in: all subcommands (placed **before** the subcommand name)

| Argument | Default | Description |
|----------|---------|-------------|
| `--plain` | `False` | Disable Rich output; use plain text. Also set automatically when stdout is piped |
| `--yes` or `-y` | `False` | Skip the confirmation prompt and start training immediately |

### Model and Paths

Available in: vanilla, fixed, estimate

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | **(required)** | Path to the root checkpoints directory (contains `acestep-v15-turbo/`, etc.) |
| `--model-variant` | `turbo` | Which model to use: `turbo`, `base`, or `sft` |
| `--dataset-dir` | **(required)** | Directory containing your preprocessed `.pt` tensor files and `manifest.json` |

### Device and Precision

Available in: all subcommands

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `auto` | Which device to train on. Options: `auto`, `cuda`, `cuda:0`, `cuda:1`, `mps`, `xpu`, `cpu`. Auto-detection priority: CUDA > MPS (Apple Silicon) > XPU (Intel) > CPU |
| `--precision` | `auto` | Floating point precision. Options: `auto`, `bf16`, `fp16`, `fp32`. Auto picks: bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU |

### LoRA Settings

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--rank` or `-r` | `64` | LoRA rank. Higher = more capacity and more VRAM. Recommended: 64 (ACE-Step dev recommendation) |
| `--alpha` | `128` | LoRA scaling factor. Controls how strongly the adapter affects the model. Usually 2x the rank. Recommended: 128 |
| `--dropout` | `0.1` | Dropout probability on LoRA layers. Helps prevent overfitting. Range: 0.0 to 0.5 |
| `--attention-type` | `both` | Which attention layers to target. Options: `both` (self + cross attention, 192 modules), `self` (self-attention only, audio patterns, 96 modules), `cross` (cross-attention only, text conditioning, 96 modules) |
| `--target-modules` | `q_proj k_proj v_proj o_proj` | Which projection layers get LoRA adapters. Space-separated list. Combined with `--attention-type` to determine final target modules |
| `--bias` | `none` | Whether to train bias parameters. Options: `none` (no bias training), `all` (train all biases), `lora_only` (only biases in LoRA layers) |

### Training Hyperparameters

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` or `--learning-rate` | `0.0001` | Initial learning rate. For Prodigy optimizer, set to `1.0` |
| `--batch-size` | `1` | Number of samples per training step. Usually 1 for music generation (audio tensors are large) |
| `--gradient-accumulation` | `4` | Number of steps to accumulate gradients before updating weights. Effective batch size = batch-size x gradient-accumulation |
| `--epochs` | `100` | Maximum number of training epochs (full passes through the dataset) |
| `--warmup-steps` | `100` | Number of optimizer steps where the learning rate ramps up from 10% to 100% |
| `--weight-decay` | `0.01` | Weight decay (L2 regularization). Helps prevent overfitting |
| `--max-grad-norm` | `1.0` | Maximum gradient norm for gradient clipping. Prevents training instability from large gradients |
| `--seed` | `42` | Random seed for reproducibility. Same seed + same data = same results |
| `--optimizer-type` | `adamw` | Optimizer: `adamw`, `adamw8bit` (saves VRAM), `adafactor` (minimal state), `prodigy` (auto-tunes LR) |
| `--scheduler-type` | `cosine` | LR schedule: `cosine`, `linear`, `constant`, `constant_with_warmup`. Prodigy auto-forces `constant` |
| `--gradient-checkpointing` | `False` | Recompute activations during backward to save VRAM (~40-60% less activation memory, ~30% slower) |
| `--offload-encoder` | `False` | Move encoder/VAE to CPU after setup. Frees ~2-4GB VRAM with minimal speed impact |

### Corrected Training (fixed mode only)

Available in: fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--cfg-ratio` | `0.15` | Classifier-free guidance dropout rate. With this probability, each sample's condition is replaced with a null embedding during training. This teaches the model to work both with and without text prompts. The model was originally trained with 0.15 |

### Data Loading

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-workers` | `4` | Number of parallel data loading worker processes. Set to 0 on machines with limited RAM |
| `--pin-memory` / `--no-pin-memory` | `True` | Pin loaded tensors in CPU memory for faster GPU transfer. Disable if you're low on RAM |
| `--prefetch-factor` | `2` | Number of batches each worker prefetches in advance |
| `--persistent-workers` / `--no-persistent-workers` | `True` | Keep data loading workers alive between epochs instead of respawning them |

### Checkpointing

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | **(required)** | Directory where LoRA weights, checkpoints, and TensorBoard logs are saved |
| `--save-every` | `10` | Save a full checkpoint (LoRA weights + optimizer + scheduler state) every N epochs |
| `--resume-from` | *(none)* | Path to a checkpoint directory to resume training from. Restores LoRA weights, optimizer state, and scheduler state |

### Logging and Monitoring

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-dir` | `{output-dir}/runs` | Directory for TensorBoard log files. View with `tensorboard --logdir <path>` |
| `--log-every` | `10` | Log loss and learning rate every N optimizer steps |
| `--log-heavy-every` | `50` | Log per-layer gradient norms every N optimizer steps. These are more expensive to compute but useful for debugging |
| `--sample-every-n-epochs` | `0` | Generate an audio sample every N epochs during training. 0 = disabled. (Not yet implemented) |

### Preprocessing (optional)

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--preprocess` | `False` (flag) | If set, run audio preprocessing before training |
| `--audio-dir` | *(none)* | Source directory containing audio files (for preprocessing) |
| `--dataset-json` | *(none)* | Path to labeled dataset JSON file (for preprocessing) |
| `--tensor-output` | *(none)* | Output directory where preprocessed .pt tensor files will be saved |
| `--max-duration` | `240` | Maximum audio duration in seconds. Longer files are truncated |
