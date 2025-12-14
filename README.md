# CS6886W_assignments_3
Assignment3_submissions
# ENV:
Python ≥ 3.9
PyTorch ≥ 1.13
CUDA (if available)
# Best configuration:
Method: PRUNE + QAT
Bit-width: INT8
Sparsity: 0.6
Width multiplier: 1.0
# Execution:
python verify_test.py \
  --ckpt PRUNE_QAT_bw8_sp0.6_wm1.00_best_.pt
