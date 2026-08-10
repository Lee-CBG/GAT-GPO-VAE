import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', nargs='+', required=True)
parser.add_argument('--labels', nargs='+', required=True)
parser.add_argument('--output', default='histogram.png')
parser.add_argument('--n_g0', type=int, default=383, help='Number of G0 genes (first n rows/cols)')
args = parser.parse_args()

n = len(args.checkpoints)
fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
if n == 1:
    axes = axes.reshape(2, 1)

for col, (ckpt_path, label) in enumerate(zip(args.checkpoints, args.labels)):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    key = [k for k in state_dict.keys() if 'q_mask_logits' in k][0]
    logits = state_dict[key].float().numpy()
    print(f"{label}: shape {logits.shape}")

    # Full matrix
    probs_full = 1 / (1 + np.exp(-logits.flatten()))
    ax = axes[0, col]
    ax.hist(probs_full, bins=100, range=(0,1), color='steelblue', edgecolor='none')
    ax.axvline(0.5, color='red', linestyle='--')
    ax.set_title(f"{label}\nFull matrix")
    ax.set_xlabel('sigmoid(W_hat)')
    ax.set_ylabel('count')
    ax.legend(title=f'edges>0.5: {(probs_full>0.5).sum()}')

    # G0 submatrix
    g0 = args.n_g0
    logits_g0 = logits[:g0, :g0]
    # Remove diagonal (no self-loops)
    mask = ~np.eye(g0, dtype=bool)
    probs_g0 = 1 / (1 + np.exp(-logits_g0[mask].flatten()))
    ax = axes[1, col]
    ax.hist(probs_g0, bins=100, range=(0,1), color='darkorange', edgecolor='none')
    ax.axvline(0.5, color='red', linestyle='--')
    ax.set_title(f"{label}\nG° submatrix ({g0}×{g0}, no self-loops)")
    ax.set_xlabel('sigmoid(W_hat)')
    ax.set_ylabel('count')
    ax.legend(title=f'edges>0.5: {(probs_g0>0.5).sum()}')

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
