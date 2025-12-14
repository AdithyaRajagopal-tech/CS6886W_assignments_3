"""
Model Definition - MobileNetV2 for CIFAR-10

Custom MobileNetV2 implementation

Adapted for CIFAR-10 (32x32 images, 10 classes)
- Input: 32x32 RGB images
- Output: 10 class probabilities

Layers typically compressed:
- features.0: Initial Conv2d 3 tO32 
- features[1-7]: Inverted residual blocks (depthwise + pointwise)
- classifier: Final linear layer (can be compressed)

Exception layers (often kept uncompressed):
- features.0: First convolution (impacts all features)
"""

import torch
import torch.nn as nn


# BatchNorm configuration
BN_MOMENTUM = 0.1
BN_EPSILON = 1e-5


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU6 block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPSILON),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNet-v2 building block)"""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPSILON),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV2(nn.Module):
    """
    MobileNet-v2 architecture for CIFAR-10
    
    Configuration:
    - Width multiplier: configurable (1.0 baseline)
    - Dropout rate: 0.2
    - BatchNorm momentum: 0.1
    - Total parameters: ~2.3M (baseline)
    """
    
    def __init__(self, num_classes=10, width_multiplier=1.0, dropout_rate=0.2):
        """
        Initialize MobileNet-v2
        
        Args:
            num_classes: Number of output classes
            width_multiplier: Width multiplier for channel scaling
            dropout_rate: Dropout rate before classifier
        """
        super().__init__()
        
        # Inverted residual settings: (expand_ratio, channels, blocks, stride)
        inverted_residual_setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        
        input_channel = 32
        last_channel = 1280
        
        input_channel = int(input_channel * width_multiplier)
        last_channel = int(last_channel * max(1.0, width_multiplier))
        
        # First conv layer
        features = [ConvBNReLU(3, input_channel, kernel_size=3, stride=1)]
        
        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        # Final conv layer
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes=10, width_multiplier=1.0, dropout_rate=0.2):
    """
    Get MobileNetV2 model for CIFAR-10
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        width_multiplier: Width multiplier for channel reduction
        dropout_rate: Dropout rate in classifier
    
    Returns:
        MobileNetV2 model instance
    """
    return MobileNetV2(
        num_classes=num_classes,
        width_multiplier=width_multiplier,
        dropout_rate=dropout_rate
    )


def count_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info():
    """
    Print comprehensive model information for documentation
    """
    model = get_model(num_classes=10)
    
    print("="*80)
    print("MODEL ARCHITECTURE: MobileNetV2 for CIFAR-10")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print("\nLAYER STRUCTURE:")
    print("-" * 80)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:50s} | Params: {params:,}")
    
    print("\n" + "="*80)
    
    print("\nMODEL COMPRESSION ANNOTATIONS:")
    print("-" * 80)
    print("exception_layers = ['features.0', 'classifier']")
    print("\nBY COMPRESSION METHOD:")
    print("  - INT8/INT4: Quantize all except features.0 and classifier")
    print("  - PRUNING: Apply to features[1-7], be careful with classifier")
    print("  - WIDTH_MULTIPLIER: Reduce channels in depthwise convolutions")
    print("  - HYBRID: INT8 + 30% pruning on most layers")


if __name__ == "__main__":
    model = get_model(num_classes=10)
    print(model)
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    print("\n")
    get_model_info()
