"""
Peak GPU memory for a full evaluation pass (2500 particles).
Usage: python profile_eval.py <run_dir> [devices]
Wraps eval.evaluate_local_experiment so the measured path is the real one.
"""
import sys, time, torch, cupy as cp
from eval import evaluate_local_experiment

RUN = sys.argv[1]
DEV = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cp.cuda.Device(DEV).use()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()

evaluate_local_experiment(
    RUN, average_treatment_effect_method="perturbseq",
    batch_size=128, ate_n_particles=2500, qc_pass=False, thr=3, devices=DEV,
)

print("\n" + "=" * 60)
print("run              %s" % RUN)
print("peak allocated   %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("peak reserved    %.2f GB" % (torch.cuda.max_memory_reserved() / 1024**3))
print("wall time        %.1f s" % (time.time() - t0))
print("=" * 60)
