"""
This script:
- Defines canonical config entries (CONFIGS). Add the names your sweep uses.
- Accepts CLI overrides for bit-width, sparsity, exceptions, etc.
- Builds a unified run_cfg dict and calls trainer.train_and_evaluate
- Ensures resume behavior by default and does device checks (CPU/GPU).
- Does not assume a particular compression pipeline; the trainer/compression framework
"""

import argparse
import os
import sys
import torch
import wandb

from dataloader import get_cifar10_loaders

# Try to import a project-level config; otherwise we'll fall back to a small default.
try:
    from compression_config import CONFIGS as PROJECT_CONFIGS
except Exception:
    PROJECT_CONFIGS = {}

# Canonical configs (extend/modify as needed).
# Each config entry defines tasks to perform. CLI overrides will replace values when provided.
DEFAULT_CONFIGS = {
    "BASELINE": {
        "description": "Train FP32 baseline (no pruning, no quant).",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.0,
        "bit_width": 32,
        "act_bit_width": 32,
        "prune_exceptions": [],  # list of module-name substrings to exclude from pruning
    },
    "PRUNE": {
        "description": "Apply global magnitude pruning (one-shot) and fine-tune in FP32.",
        "enable_pruning": True,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.6,
        "bit_width": 32,
        "act_bit_width": 32,
        "prune_exceptions": [],
    },
    "QAT": {
        "description": "Apply QAT (fake-quant) to weights (and optionally activations) without pruning.",
        "enable_pruning": False,
        "enable_qat": True,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.0,
        "bit_width": 8,
        "act_bit_width": 8,
        "prune_exceptions": [],
    },
    "ACTIVATION_ONLY": {
        "description": "Quantize activations only (fake-quant) during training; keep weights FP32.",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.0,
        "bit_width": 32,
        "act_bit_width": 8,
        "prune_exceptions": [],
    },
    "INT8_STATIC": {
        "description": "Post-training static INT8 quantization (CPU-only evaluation or export).",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": True,
        "enable_activation_quant": True,
        "prune_sparsity": 0.0,
        "bit_width": 8,
        "act_bit_width": 8,
        "prune_exceptions": [],
    },
    "PRUNE_QAT": {
        "description": "Global pruning followed by QAT (weights+activations fake-quant).",
        "enable_pruning": True,
        "enable_qat": True,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.6,
        "bit_width": 8,
        "act_bit_width": 8,
        "prune_exceptions": [],
    },
    "HYBRID": {
        "description": "Prune + activation-only quantization during fine-tune (example hybrid).",
        "enable_pruning": True,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.6,
        "bit_width": 8,
        "act_bit_width": 8,
        "prune_exceptions": [],
    }
}


CONFIGS = {**DEFAULT_CONFIGS, **PROJECT_CONFIGS}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='Name of the config to run (must appear in CONFIGS).')
    parser.add_argument('--bit-width', type=int, default=None,
                        help='Weight bit-width override. If omitted, uses config default.')
    parser.add_argument('--act-bit-width', type=int, default=None,
                        help='Activation bit-width override.')
    parser.add_argument('--sparsity', type=float, default=None,
                        help='Pruning sparsity override (fraction pruned, e.g., 0.6 means 60%% weights pruned).')
    parser.add_argument('--width-mult', type=float, default=1.0,
                        help='Network width multiplier (for mobilenet-like nets).')
    parser.add_argument('--exceptions', type=str, default='',
                        help='Comma-separated substrings of module/parameter names to exclude from pruning.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb-project', type=str, default='Assign_3_cifar10-compression')
    parser.add_argument('--wandb-entity', type=str, default='adithya968-iitmaana')
    parser.add_argument('--disable-wandb', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Enable resume from baseline or previous ckpt (default behavior).')
    parser.add_argument('--calibration-batches', type=int, default=50,
                        help='Number of batches to use for activation calibration (if needed).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate config name
    if args.config not in CONFIGS:
        print(f"ERROR: Unknown config '{args.config}'. Known configs: {list(CONFIGS.keys())}")
        sys.exit(1)

    # Build run-level config (start from config entry, then apply CLI overrides)
    entry = CONFIGS[args.config].copy()
    run_cfg = entry.copy()
    # CLI overrides 
    if args.bit_width is not None:
        run_cfg['bit_width'] = args.bit_width
    if args.act_bit_width is not None:
        run_cfg['act_bit_width'] = args.act_bit_width
    if args.sparsity is not None:
        run_cfg['prune_sparsity'] = args.sparsity
    if args.exceptions:
        run_cfg['prune_exceptions'] = [s.strip() for s in args.exceptions.split(',') if s.strip()]

    # Add runtime parameters
    run_cfg['epochs'] = args.epochs
    run_cfg['batch_size'] = args.batch_size
    run_cfg['lr'] = args.lr
    run_cfg['width_mult'] = args.width_mult
    run_cfg['calibration_batches'] = args.calibration_batches

    # Device selection (get_device should fallback to cpu if cuda not available)
    device = get_device(args.device)  # trainer.get_device provides fallback

    # If run requests static INT8 and device is CUDA, fallback to CPU and warn.
    if run_cfg.get('enable_static_int8', False) and device.type == 'cuda':
        print("WARNING: static INT8 tasks are often CPU-only in our environment. Falling back to CPU.")
        device = torch.device('cpu')

    # Start WandB
    wandb_run = None
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   config=run_cfg)
        wandb_run = wandb

    # Load dataset
    train_loader, val_loader = get_cifar10_loaders(batch_size=run_cfg['batch_size'])

    comp = None

    # Call trainer 
    train_and_evaluate(cfg=run_cfg,
                       comp=comp,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       epochs=run_cfg['epochs'],
                       lr=run_cfg['lr'],
                       device=device,
                       wandb_run=wandb_run,
                       resume=True)

    if wandb_run:
        wandb.finish()
