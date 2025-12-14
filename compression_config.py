"""
I Kept  project's global configs (model/dataset/training/etc.)
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'name': 'MobileNetV2',
    'num_classes': 10,
    'input_size': 32,  # CIFAR-10
    'pretrained': False,
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    'name': 'CIFAR-10',
    'data_dir': './data',
    'num_classes': 10,
    'train_split': 0.8,
    'val_split': 0.2,
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    'epochs': 100,  # Baseline: 350, Fine-tune: 50 epoch
    'batch_size': 128,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_workers': 2,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'seed': 42,
}

# ============================================================================
# REQUIRED RUN-LEVEL CONFIGURATIONS (used by main + trainer)
# ============================================================================

CONFIGS = {

    # ---------------------- BASELINES ----------------------
    "BASELINE": {
        "description": "FP32 baseline training.",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.0,
        "bit_width": 32,
        "act_bit_width": 32,
        "prune_exceptions": [],
    },

    # ------------------------ PRUNING -----------------------
    "PRUNING": {
        "description": "Magnitude pruning without quantization.",
        "enable_pruning": True,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.5,
        "bit_width": 32,
        "act_bit_width": 32,
        "prune_exceptions": [],
    },

    # ------------------- WIDTH MULTIPLIER -------------------
    "WIDTH_MULTIPLIER": {
        "description": "Change width multiplier only (no compression).",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.0,
        "bit_width": 32,
        "act_bit_width": 32,
    },

    # ------------------------ FP16 --------------------------
    "FP16": {
        "description": "Train model in FP16 precision (weights + activations).",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.0,
        "bit_width": 16,
        "act_bit_width": 16,
    },

    # ------------------- ACT FP16 ONLY ----------------------
    "ACT_FP16_ONLY": {
        "description": "Only activations are FP16; weights remain FP32.",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.0,
        "bit_width": 32,
        "act_bit_width": 16,
    },

    # ------------------------ HYBRID ------------------------
    "HYBRID": {
        "description": "Hybrid pruning + activation quantization.",
        "enable_pruning": True,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.3,
        "bit_width": 32,
        "act_bit_width": 8,
        "prune_exceptions": [],
    },

    # --------------------- PRUNE + QAT ----------------------
    "PRUNE_QAT": {
        "description": "Pruning + QAT (fake quant for weights & activations).",
        "enable_pruning": True,
        "enable_qat": True,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.6,
        "bit_width": 8,
        "act_bit_width": 8,
        "prune_exceptions": [],
    },

    # ------------------------- QAT --------------------------
    "QAT": {
        "description": "Pure QAT (no pruning).",
        "enable_pruning": False,
        "enable_qat": True,
        "enable_static_int8": False,
        "enable_activation_quant": True,
        "prune_sparsity": 0.0,
        "bit_width": 8,
        "act_bit_width": 8,
    },

    # -------------------- INT8 STATIC -----------------------
    "INT8_STATIC": {
        "description": "Post-training static INT8 quantization.",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": True,
        "enable_activation_quant": True,
        "prune_sparsity": 0.0,
        "bit_width": 8,
        "act_bit_width": 8,
    },

    # -------------------- INT8 DYNAMIC ----------------------
    "INT8_DYNAMIC": {
        "description": "Dynamic INT8 quantization (weights only).",
        "enable_pruning": False,
        "enable_qat": False,
        "enable_static_int8": False,
        "enable_activation_quant": False,
        "prune_sparsity": 0.0,
        "bit_width": 8,
        "act_bit_width": 8,
    },
}

# ============================================================================
# PROJECT-LEVEL COMPRESSION DEFAULTS (unchanged from your file)
# ============================================================================

COMPRESSION_CONFIG = {
    'methods': ['INT8_STATIC', 'INT8_DYNAMIC', 'FP16', 'INT4', 'PRUNING', 'WIDTH_MULTIPLIER', 'HYBRID'],
    'defaults': {
        'bit_width': 8,
        'sparsity_ratio': 0.3,
        'width_multiplier': 0.75,
        'static_quantization': False,
        'exception_layers': ['features.0', 'classifier'],
    }
}

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================

CHECKPOINT_CONFIG = {
    'baseline_ckpt': '/content/drive/MyDrive/compression/baseline_best.pt',
    'compression_ckpt_dir': '/content/drive/MyDrive/compression_checkpoints',
    'save_best_only': True,
    'save_frequency': 10,
}

# ============================================================================
# W&B CONFIGURATION
# ============================================================================

WB_CONFIG = {
    'project': 'cifar10-compression-assign3',
    'entity': None,
    'tags': ['compression', 'cifar10', 'mobilenetv2'],
    'notes': 'Model compression with fine-tuning on CIFAR-10',
}

# ============================================================================
# METRICS (unchanged)
# ============================================================================

METRICS_TO_LOG = [
    'test_accuracy',
    'best_accuracy',
    'train_loss',
    'test_loss',
    'compression_ratio',
    'weight_compression_ratio',
    'activation_compression_ratio',
    'original_size_mb',
    'final_size_mb',
    'metadata_overhead_mb',
    'inference_time_ms',
]

# ============================================================================
# HELPERS (unchanged)
# ============================================================================

def get_config(config_name):
    all_configs = {
        'model': MODEL_CONFIG,
        'dataset': DATASET_CONFIG,
        'training': TRAINING_CONFIG,
        'compression': COMPRESSION_CONFIG,
        'checkpoint': CHECKPOINT_CONFIG,
        'wb': WB_CONFIG,
        'metrics': METRICS_TO_LOG,
        'run_modes': CONFIGS,
    }
    return all_configs.get(config_name, None)


def get_run_name(compression_method, bit_width, sparsity_ratio, width_multiplier):
    return f"{compression_method}_bw{bit_width}_sp{sparsity_ratio:.1f}_wm{width_multiplier:.2f}"


def print_config():
    print("CONFIGURATION SUMMARY")
    print("MODEL:", MODEL_CONFIG)
    print("DATASET:", DATASET_CONFIG)
    print("TRAINING:", TRAINING_CONFIG)
    print("AVAILABLE RUN MODES:", list(CONFIGS.keys()))


if __name__ == "__main__":
    print_config()
