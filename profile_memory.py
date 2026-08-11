"""
Profile parameter count, peak GPU memory and per-step time: GAT vs MLP encoder.
Usage: python profile_memory.py <config.yaml> [n_steps]
Runs real training steps via the standard path, capped by max_steps.
"""
import sys, time, yaml, torch, pytorch_lightning as pl
from train_rpe1 import preprocess_config, get_data_module, add_data_info_to_config
from gpo_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)

CFG = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

config = preprocess_config(yaml.safe_load(open(CFG)))
dm = get_data_module(config)
config = add_data_info_to_config(config, dm)
pl.seed_everything(config["seed"])

lm = TrainConfigPerturbationLightningModule(
    config=config,
    D_obs_counts_train=dm.get_train_perturbation_obs_counts(),
    D_obs_counts_val=dm.get_val_perturbation_obs_counts(),
    D_obs_counts_test=dm.get_test_perturbation_obs_counts(),
    qc_obs_counts_train=dm.get_train_qc_obs_counts(),
    qc_obs_counts_val=dm.get_val_qc_obs_counts(),
    qc_obs_counts_test=dm.get_test_qc_obs_counts(),
    x_var_info=dm.get_x_var_info().index,
    use_scheduler=False,
)

n_tot = sum(p.numel() for p in lm.parameters())
n_tr = sum(p.numel() for p in lm.parameters() if p.requires_grad)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()

trainer = pl.Trainer(
    accelerator="gpu", devices=[0], logger=False, enable_checkpointing=False,
    max_steps=STEPS, max_epochs=-1, num_sanity_val_steps=0, val_check_interval=1.0,
    enable_progress_bar=False,
    gradient_clip_val=config.get("gradient_clip_norm"),
)
trainer.fit(lm, train_dataloaders=dm.train_dataloader())

elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
resv = torch.cuda.max_memory_reserved() / 1024**3

print("\n" + "=" * 60)
print("config              %s" % CFG)
print("params (total)      %,d".replace(",", "") % n_tot if False else "params (total)      {:,}".format(n_tot))
print("params (trainable)  {:,}".format(n_tr))
print("peak allocated      %.2f GB" % peak)
print("peak reserved       %.2f GB" % resv)
print("steps               %d" % STEPS)
print("time/step           %.3f s" % (elapsed / max(STEPS, 1)))
print("=" * 60)
