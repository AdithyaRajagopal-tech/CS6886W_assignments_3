"""
compression_framework.py

"""
import os
import copy
import torch
import torch.nn as nn
import numpy as np

# --- FakeQuant STE (unchanged semantics) ---
class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        if scale == 0 or scale is None:
            return x
        x_div = x / scale
        x_q = torch.clamp(torch.round(x_div), qmin, qmax)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class FakeQuantModule(nn.Module):
    def __init__(self, bit_width=8, learn_scale=False):
        super().__init__()
        self.bit_width = int(bit_width)
        self.learn_scale = bool(learn_scale)
        init_scale = torch.tensor(1.0)
        if self.learn_scale:
            self.scale = nn.Parameter(init_scale)
        else:
            self.register_buffer('scale', init_scale)

    def forward(self, x):
        qmax = 2 ** (self.bit_width - 1) - 1
        if self.learn_scale:
            scale_val = float(self.scale.detach().cpu().item())
            return FakeQuantSTE.apply(x, scale_val, -qmax, qmax)
        else:
            max_val = x.detach().abs().max().item()
            scale_val = (max_val / qmax) if max_val > 0 else 1.0
            # update buffer
            # store as tensor buffer for device consistency
            self.scale = torch.tensor(scale_val, device=x.device)
            return FakeQuantSTE.apply(x, scale_val, -qmax, qmax)


class CompressionFramework:
    def __init__(self, bit_width=8, act_bit_width=8, learn_scale=False, exceptions=None, device='cpu'):
        self.bit_width = int(bit_width)
        self.act_bit_width = int(act_bit_width)
        self.learn_scale = bool(learn_scale)
        self.exceptions = exceptions or []
        self.masks = {}
        self.device = device
        self._act_hooks = []
        self._act_peaks = {}

    def set_masks(self, masks: dict):
        self.masks = masks or {}

    def reapply_masks(self, model: nn.Module):
        if not self.masks:
            return
        named = dict(model.named_parameters())
        for name, mask in self.masks.items():
            if name in named:
                with torch.no_grad():
                    named[name].data.mul_(mask.to(named[name].device))

    #  QAT hooks 
    def prepare_model_for_qat(self, model: nn.Module):
        self.remove_qat_hooks()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fq = FakeQuantModule(bit_width=self.act_bit_width, learn_scale=self.learn_scale)
                module._qat_fake_quant = fq
                def make_hook(fq_mod):
                    def hook(mod, inp, outp):
                        return fq_mod(outp)
                    return hook
                h = module.register_forward_hook(make_hook(fq))
                self._act_hooks.append(h)
        print(f"[CompressionFramework] Registered {len(self._act_hooks)} activation QAT hooks.")

    def remove_qat_hooks(self):
        for h in self._act_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._act_hooks = []

    def apply_weight_fake_quant(self, model: nn.Module):
        qmax = 2 ** (self.bit_width - 1) - 1
        for name, p in model.named_parameters():
            if p.dim() == 0:
                continue
            skip = False
            for exc in self.exceptions or []:
                if exc and name.startswith(exc):
                    skip = True
                    break
            if skip:
                continue
            with torch.no_grad():
                data = p.data
                max_abs = data.abs().max().item()
                scale = (max_abs / qmax) if max_abs > 0 else 1.0
                p.copy_(FakeQuantSTE.apply(data, scale, -qmax, qmax))

    #  Activation peaks collection 
    def collect_activation_peaks(self, model: nn.Module, data_loader, num_batches: int, device=None):
        device = device or self.device
        self._act_peaks = {}
        hooks = []
        def make_capture(name):
            def hook(module, inp, outp):
                if isinstance(outp, torch.Tensor):
                    max_abs = outp.detach().abs().max().item()
                    numel = outp.detach().numel()
                else:
                    max_abs = 0.0
                    numel = 0
                    for t in outp:
                        if isinstance(t, torch.Tensor):
                            max_abs = max(max_abs, t.detach().abs().max().item())
                            numel += t.detach().numel()
                prev = self._act_peaks.get(name, {'max_abs': 0.0, 'numel': 0})
                prev['max_abs'] = max(prev['max_abs'], max_abs)
                prev['numel'] = max(prev['numel'], numel)
                self._act_peaks[name] = prev
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6)):
                hooks.append(module.register_forward_hook(make_capture(name)))

        model.eval()
        batches = 0
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                model(images)
                batches += 1
                if batches >= num_batches:
                    break

        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        peaks = {}
        for k, v in self._act_peaks.items():
            peaks[k] = {'max_abs': float(v['max_abs']), 'numel': int(v['numel'])}
        return peaks

    def set_activation_calibration(self, act_peaks: dict):
        self._act_peaks = act_peaks or {}

    # EXPORT: save int8 + masks + metadata; return size stats 
    def save_int8_and_masks(self, model: nn.Module, fname: str, masks: dict = None):
        out = {}
        metadata = {}
        qmax = 2 ** (self.bit_width - 1) - 1

        total_weight_bytes = 0
        total_mask_bytes = 0
        total_metadata_bytes = 0

        for name, p in model.named_parameters():
            data = p.detach().cpu().numpy()
            max_abs = float(np.abs(data).max())
            scale = (max_abs / qmax) if max_abs > 0 else 1.0
            data_q = np.round(data / scale).astype(np.int8)
            out[f"{name}.int8"] = data_q
            out[f"{name}.scale"] = np.array(scale, dtype=np.float32)
            total_weight_bytes += data_q.nbytes

            if masks and name in masks:
                mask_arr = masks[name].detach().cpu().numpy().astype(np.uint8)
            else:
                mask_arr = np.ones_like(data_q, dtype=np.uint8)
            out[f"{name}.mask"] = mask_arr
            total_mask_bytes += mask_arr.nbytes

            metadata[name] = {"shape": data.shape, "bit_width": self.bit_width}

        # activation peaks
        act_keys = []
        act_max = []
        act_numel = []
        for k, v in (self._act_peaks or {}).items():
            act_keys.append(k)
            act_max.append(float(v.get('max_abs', 0.0)))
            act_numel.append(int(v.get('numel', 0)))
        out['act_keys'] = np.array(act_keys, dtype=object)
        out['act_max'] = np.array(act_max, dtype=np.float32)
        out['act_numel'] = np.array(act_numel, dtype=np.int32)

        # metadata arrays
        meta_keys = list(metadata.keys())
        meta_vals = [str(metadata[k]) for k in meta_keys]
        out['meta_keys'] = np.array(meta_keys, dtype=object)
        out['meta_vals'] = np.array(meta_vals, dtype=object)

        total_metadata_bytes += out['meta_keys'].nbytes
        total_metadata_bytes += out['meta_vals'].nbytes
        total_metadata_bytes += out['act_keys'].nbytes
        total_metadata_bytes += out['act_max'].nbytes
        total_metadata_bytes += out['act_numel'].nbytes

        # write compressed npz
        np.savez_compressed(fname, **out)
        # assert file finished on disk
        file_size_mb = float(os.path.getsize(fname)) / (1024 * 1024)

        # ratios (FP32 assumptions)
        weight_compression_ratio = 32 / self.bit_width if self.bit_width > 0 else 1.0
        activation_compression_ratio = 32 / self.act_bit_width if self.act_bit_width > 0 else 1.0

        print(f"[CompressionFramework] Saved int8 + masks + metadata to {fname}")

        return {
            "file_size_mb": file_size_mb,
            "weight_bytes": int(total_weight_bytes),
            "mask_bytes": int(total_mask_bytes),
            "metadata_bytes": int(total_metadata_bytes),
            "weight_compression_ratio": float(weight_compression_ratio),
            "activation_compression_ratio": float(activation_compression_ratio),
        }
