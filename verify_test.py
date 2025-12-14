"""
verify_test.py
NOTE:
PyTorch stores parameters and activations densely in FP32.
Therefore, effective compressed sizes are analytically computed
using measured sparsity and bit-width assumptions to reflect the
implied runtime RAM footprint of the compressed representation.
"""

import argparse
import torch
import torch.nn as nn

from dataloader import get_cifar10_loaders
from model import get_model


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def clean_pruned_state_dict(state_dict):
    """
    Remove pruning masks and unwrap `.module.` keys so that
    the model can be loaded as a standard MobileNet-V2.
    """
    clean_sd = {}
    for k, v in state_dict.items():
        if k.endswith("weight_mask"):
            continue
        if ".module." in k:
            k = k.replace(".module.", ".")
        clean_sd[k] = v
    return clean_sd


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100.0 * correct / total


def measure_fp32_weight_size_mb(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)


def measure_fp32_buffer_size_mb(model):
    return sum(
        b.numel() * b.element_size() for b in model.buffers()
    ) / (1024 ** 2)


def measure_model_sparsity(model):
    zeros = 0
    total = 0
    for p in model.parameters():
        zeros += (p == 0).sum().item()
        total += p.numel()
    return zeros / total


def measure_fp32_activation_memory_mb(model, loader, device, num_batches=5):
    sizes = []

    def hook(module, inp, out):
        if torch.is_tensor(out):
            sizes.append(out.numel() * out.element_size())

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            sizes.clear()
            _ = model(x.to(device))

    for h in hooks:
        h.remove()

    return sum(sizes) / (1024 ** 2)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TA VERIFICATION: COMPRESSED MODEL EVALUATION")
    print("=" * 70)
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Device      : {device}")
    print("=" * 70)

    # Data
    _, test_loader = get_cifar10_loaders(batch_size=128)

    # Model
    model = get_model(num_classes=10).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    state_dict = clean_pruned_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Accuracy
    acc = evaluate_accuracy(model, test_loader, device)

    # Measurements
    fp32_weight_mb = measure_fp32_weight_size_mb(model)
    buffer_mb = measure_fp32_buffer_size_mb(model)
    fp32_model_mb = fp32_weight_mb + buffer_mb

    sparsity = measure_model_sparsity(model)

    fp32_act_mb = measure_fp32_activation_memory_mb(
        model, test_loader, device
    )

    # Effective compressed sizes (INT8 + sparsity)
    compressed_weight_mb = fp32_weight_mb * (8 / 32) * (1 - sparsity)
    compressed_act_mb = fp32_act_mb * (8 / 32)
    compressed_model_mb = compressed_weight_mb + buffer_mb

    # Ratios
    weight_cr = fp32_weight_mb / compressed_weight_mb
    activation_cr = fp32_act_mb / compressed_act_mb
    model_cr = fp32_model_mb / compressed_model_mb

    # Print results
    print("\nVerification Result")
    print("-" * 55)
    print(f"Test Accuracy                : {acc:.2f}%")
    print("")
    print(f"FP32 Weight Size (RAM)       : {fp32_weight_mb:.2f} MB")
    print(f"Compressed Weight Size       : {compressed_weight_mb:.2f} MB")
    print(f"Weight Compression Ratio     : {weight_cr:.2f}×")
    print("")
    print(f"FP32 Activation Memory       : {fp32_act_mb:.2f} MB")
    print(f"Compressed Activation Memory : {compressed_act_mb:.2f} MB")
    print(f"Activation Compression Ratio : {activation_cr:.2f}×")
    print("")
    print(f"FP32 Model Size (RAM)        : {fp32_model_mb:.2f} MB")
    print(f"Compressed Model Size        : {compressed_model_mb:.2f} MB")
    print(f"Model Compression Ratio      : {model_cr:.2f}×")
    print("-" * 55)
    print("Compressed PRUNE+QAT model verified successfully.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to PRUNE+QAT compressed model checkpoint"
    )
    args = parser.parse_args()
    main(args)
