#!/usr/bin/env python3
"""
ACE-Step Training V2 -- CLI Entry Point

Usage:
    python train.py <subcommand> [args]

Subcommands:
    vanilla          Reproduce existing (bugged) training for backward compatibility
    fixed            Corrected training: continuous timesteps + CFG dropout
    selective        Corrected training with dataset-specific module selection
    estimate         Gradient sensitivity analysis (no training)
    compare-configs  Compare module config JSON files

Examples:
    python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo \\
        --dataset-dir ./preprocessed_tensors/jazz --output-dir ./lora_output/jazz

    python train.py --help
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup (before any library imports that might configure logging)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def _has_subcommand() -> bool:
    """Check if sys.argv contains a recognized subcommand or --help."""
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        return True  # let argparse handle help
    known = {"vanilla", "fixed", "selective", "estimate", "compare-configs"}
    return bool(known & set(args))


def main() -> int:
    # -- Interactive wizard when no subcommand is given -----------------------
    if not _has_subcommand():
        from acestep.training_v2.ui.wizard import run_wizard

        args = run_wizard()
        if args is None:
            return 0
    else:
        from acestep.training_v2.cli.common import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args()

    from acestep.training_v2.cli.common import validate_paths

    # -- Preprocessing (wizard only, not yet implemented) --------------------
    if getattr(args, "preprocess", False):
        print("[INFO] Preprocessing is not yet implemented.")
        print("[INFO] Use the Gradio UI or manual scripts to preprocess audio,")
        print("[INFO] then run:  python train.py fixed --dataset-dir <tensor_dir> ...")
        return 0

    # -- Dispatch -----------------------------------------------------------
    sub = args.subcommand

    # compare-configs has its own validation
    if sub == "compare-configs":
        return _run_compare_configs(args)

    # All other subcommands need path validation
    if not validate_paths(args):
        return 1

    if sub == "vanilla":
        from acestep.training_v2.cli.train_vanilla import run_vanilla
        return run_vanilla(args)

    elif sub == "fixed":
        from acestep.training_v2.cli.train_fixed import run_fixed
        return run_fixed(args)

    elif sub == "selective":
        return _run_selective(args)

    elif sub == "estimate":
        return _run_estimate(args)

    else:
        print(f"[FAIL] Unknown subcommand: {sub}", file=sys.stderr)
        return 1


# ===========================================================================
# Placeholder subcommands (Conversation C / D)
# ===========================================================================

def _run_selective(args) -> int:
    """Run selective training (placeholder -- full implementation in Conversation C)."""
    print("[INFO] Selective training is not yet implemented.")
    print("[INFO] Use 'fixed' for corrected training, or 'estimate' for module analysis.")
    return 0


def _run_estimate(args) -> int:
    """Run gradient estimation (placeholder -- full implementation in Conversation C)."""
    print("[INFO] Estimation is not yet implemented.")
    return 0


def _run_compare_configs(args) -> int:
    """Compare module config JSON files (placeholder -- full implementation in Conversation D)."""
    from acestep.training_v2.cli.common import validate_paths
    if not validate_paths(args):
        return 1
    print("[INFO] compare-configs is not yet implemented.")
    return 0


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
