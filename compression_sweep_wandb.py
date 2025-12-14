"""
W&B Sweep Integration and Analysis
Handles W&B integration, experiment tracking, and result analysis
"""

import wandb
import yaml
import sys
import os

def sweep_run_wrapper():
    """Wrapper function for W&B sweep runs """
    
    # Initialize W&B run FIRST
    wandb.init()
    config = wandb.config
    
    # Extract parameters from W&B config
    compression_config = {
        'config': config.get('config', 'INT8_DYNAMIC'),
        'bit_width': int(config.get('bit-width', 8)),
        'static': bool(config.get('static', False)),
        'sparsity': float(config.get('sparsity', 0.0)),
        'width_multiplier': float(config.get('width-mult', 1.0)),
        'exceptions': config.get('exceptions', 'features.0'),
    }
    
    cmd = [
        sys.executable,
        'run_compression_experiments.py',
        '--config', compression_config['config'],
        '--bit-width', str(compression_config['bit_width']),
        '--sparsity', str(compression_config['sparsity']),
        '--epochs', '50',
        '--exceptions', compression_config['exceptions'],
        '--disable-wandb',
    ]
    
    # Run compression pipeline
    import subprocess
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode == 0


def run_sweep(sweep_config_path):
    """
    Initialize and run W&B sweep
    
    Args:
        sweep_config_path: Path to sweep config YAML file
    """
    
    if not os.path.exists(sweep_config_path):
        print(f" Sweep config not found: {sweep_config_path}")
        return False
    
    print(f"\nInitializing W&B Sweep from {sweep_config_path}...")
    
    # Load sweep config
    with open(sweep_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f" Sweep Config Loaded:")
    print(f"  - Project: {config.get('project')}")
    print(f"  - Name: {config.get('name')}")
    print(f"  - Method: {config.get('method')}")
    
    # Create sweep
    sweep_id = wandb.sweep(config, project=config['project'])
    
    print(f"\n{'='*80}")
    print(f" Sweep created: {config['project']}/sweeps/{sweep_id}")
    print(f" Sweep ID: {sweep_id}")
    print(f" Sweep URL: https://wandb.ai/adithya968-iitmaana/{config['project']}/sweeps/{sweep_id}")
    print(f"{'='*80}\n")
    
    # Run sweep agent
    print("Starting sweep agent... (Press Ctrl+C to stop)\n")
    wandb.agent(sweep_id, function=sweep_run_wrapper, count=10)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run W&B sweep for model compression")
    parser.add_argument('--sweep-config', type=str, required=True, help='Path to sweep config YAML')
    args = parser.parse_args()
    
    success = run_sweep(args.sweep_config)
    sys.exit(0 if success else 1)