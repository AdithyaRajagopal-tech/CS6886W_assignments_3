"""
trainer.py

Implements:
 - Custom MobileNetV2 training
 - Pruning, QAT, activation calibration
 - INT8 model export
 - W&B logging of compression metrics 
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from compression_framework import CompressionFramework

# DEVICE SELECTOR
def get_device(preferred="cuda"):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



# TRAINER CLASS
class Trainer:
    def __init__(self, cfg, device, baseline_ckpt=None, ckpt_dir="./checkpoints"):
        self.cfg = cfg
        self.device = device
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.baseline_ckpt = baseline_ckpt or "/content/drive/MyDrive/compression/baseline_best.pt"

        # compression framework
        self.comp = CompressionFramework(
            bit_width=cfg.get("bit_width", 8),
            act_bit_width=cfg.get("act_bit_width", cfg.get("bit_width", 8)),
            learn_scale=cfg.get("learn_scale", False),
            exceptions=cfg.get("exceptions", []),
            device=str(device)
        )

    def build_model(self, width_mult):
        model = get_model(num_classes=10, width_multiplier=width_mult)
        return model.to(self.device)

    def load_baseline_if_exists(self, model):
        if os.path.exists(self.baseline_ckpt):
            ck = torch.load(self.baseline_ckpt, map_location=self.device)
            if "model_state_dict" in ck:
                model.load_state_dict(ck["model_state_dict"], strict=True)
            else:
                model.load_state_dict(ck, strict=True)
            print("[Trainer] Loaded baseline checkpoint.")
        else:
            print("[Trainer] No baseline checkpoint found.")

    def apply_global_pruning(self, model, sparsity, exceptions):
        named = dict(model.named_parameters())
        values = []
        keys = []

        for name, p in named.items():
            if p.requires_grad and p.dim() > 1:
                if exceptions and any(name.startswith(e) for e in exceptions):
                    continue
                values.append(p.detach().abs().view(-1).cpu())
                keys.append(name)

        if not values:
            return {}

        all_w = torch.cat(values)
        k = int(len(all_w) * sparsity)

        if k <= 0:
            return {n: torch.ones_like(named[n].data) for n in keys}

        threshold, _ = torch.kthvalue(all_w, k)
        threshold = float(threshold.item())

        masks = {}
        for name in keys:
            W = named[name]
            mask = (W.detach().abs() > threshold).to(dtype=W.dtype, device=W.device)
            with torch.no_grad():
                W.data.mul_(mask)
            masks[name] = mask

        kept = sum(int(m.sum().item()) for m in masks.values())
        total = all_w.numel()
        achieved = 1.0 - kept / total
        print(f"[Trainer] Applied pruning (target={sparsity}, achieved={achieved:.4f})")

        return masks

    def evaluate(self, model, loader):
        model.eval()
        C = nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                loss_sum += C(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        return 100.0 * correct / total, loss_sum / total

    def train_and_evaluate(self, cfg, comp, train_loader, val_loader, epochs, lr, wandb_run=None, resume=True):

        # Model 
        model = self.build_model(cfg.get("width_mult", 1.0))

        # wandb.config 
        if wandb_run:
            cfg_map = {
                "compression_method": cfg.get("compression_method", cfg.get("config", "PRUNE_QAT")),
                "bit_width": cfg.get("bit_width", 8),
                "sparsity": cfg.get("prune_sparsity", cfg.get("sparsity", 0.0)),
                "width_mult": cfg.get("width_mult", 1.0),
                "exceptions": ",".join(cfg.get("exceptions", [])) if isinstance(cfg.get("exceptions", []), (list, tuple)) else cfg.get("exceptions", "")
            }
            try:
                wandb_run.config.update(cfg_map, allow_val_change=True)
            except:
                for k, v in cfg_map.items():
                    wandb_run.config[k] = v

        # baseline resume
        if resume:
            self.load_baseline_if_exists(model)

        optim_ = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        C = nn.CrossEntropyLoss()

        # pruning
        masks = {}
        if cfg.get("enable_pruning", False):
            masks = self.apply_global_pruning(
                model,
                cfg.get("prune_sparsity", 0.0),
                cfg.get("prune_exceptions", [])
            )
            self.comp.set_masks(masks)

        # activation calibration
        if cfg.get("enable_activation_quant", False) or cfg.get("enable_static_int8", False):
            act_peaks = self.comp.collect_activation_peaks(
                model,
                train_loader,
                cfg.get("calibration_batches", 50),
                device=self.device
            )
            self.comp.set_activation_calibration(act_peaks)

        #  QAT
        if cfg.get("enable_qat", False):
            self.comp.prepare_model_for_qat(model)

        method = cfg.get("compression_method", cfg.get("config", "PRUNE_QAT"))
        best_pt = os.path.join(self.ckpt_dir, f"best_{method}.pt")
        best_npz = os.path.join(self.ckpt_dir, f"best_{method}.int8.npz")

        best_acc = -1.0

        # TRAINING LOOP
        for epoch in range(epochs):

            model.train()
            total, correct, loss_sum = 0, 0, 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optim_.zero_grad()

                # QAT weight fake quant
                if cfg.get("enable_qat", False):
                    self.comp.apply_weight_fake_quant(model)

                out = model(x)
                loss = C(out, y)
                loss.backward()
                optim_.step()

                # reapply pruning mask
                if masks:
                    self.comp.reapply_masks(model)

                loss_sum += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / total
            train_loss = loss_sum / total

            val_acc, val_loss = self.evaluate(model, val_loader)
            print(f"[Epoch {epoch}] Train {train_acc:.2f} | Val {val_acc:.2f}")

            if wandb_run:
                wandb_run.log({
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss
                })

            # best model update 
            if val_acc > best_acc:
                best_acc = val_acc

                # save FP32 best model
                torch.save(
                    {"model_state_dict": model.state_dict(), "val_accuracy": val_acc},
                    best_pt
                )
                print(f"[Trainer] Saved FP32 checkpoint → {best_pt}")

                # INT8 export 
                model_cpu = copy.deepcopy(model).cpu()
                stats = self.comp.save_int8_and_masks(model_cpu, best_npz, masks)
                print(f"[Trainer] Saved compressed INT8 → {best_npz}")

                # wandb logging
                if wandb_run:
                    logd = {
                        "best_accuracy": float(best_acc),
                        "final_size_mb": float(stats["file_size_mb"]),
                        "weight_compression_ratio": float(stats["weight_compression_ratio"]),
                        "activation_compression_ratio": float(stats["activation_compression_ratio"]),
                        "metadata_overhead_mb": float(stats["metadata_bytes"] / (1024 * 1024))
                    }

                    # original size
                    if os.path.exists(self.baseline_ckpt):
                        orig_mb = os.path.getsize(self.baseline_ckpt) / (1024 * 1024)
                        logd["original_size_mb"] = float(orig_mb)
                        logd["compression_ratio"] = float(orig_mb / stats["file_size_mb"])

                    wandb_run.log(logd)
                    print("[Trainer] Logged compression fields:", list(logd.keys()))

        print(f"[Trainer] Best Accuracy = {best_acc:.2f}")
        return {"best_accuracy": best_acc, "best_npz": best_npz}

# EXTERNAL FUNCTION EXPECTED BY MAIN SCRIPT
def train_and_evaluate(cfg, comp, train_loader, val_loader, epochs, lr, device, wandb_run=None, resume=True):
    trainer = Trainer(
        cfg,
        device,
        baseline_ckpt=cfg.get("baseline_ckpt", None),
        ckpt_dir=cfg.get("ckpt_dir", "./checkpoints")
    )
    if comp is not None:
        trainer.comp = comp
    return trainer.train_and_evaluate(cfg, comp, train_loader, val_loader, epochs, lr, wandb_run, resume)
