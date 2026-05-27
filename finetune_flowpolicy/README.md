# Fine-tuning FlowPolicy (Franka Kitchen) via ReinFlow PPO

Jalankan dari **akar repositori** (`kripsy12/`), bukan dari folder `finetune_flowpolicy/`.

## Setup (sekali)

```bash
cd /path/ke/kripsy12
conda activate flowpolicy-kitchen
pip install psutil pretty_errors
```

## Fine-tuning

**Smoke test** (cepat, verifikasi pipeline):

```bash
conda activate flowpolicy-kitchen
cd /path/ke/kripsy12

python finetune_flowpolicy/scripts/run_ft_ppo.py \
    env.n_envs=2 \
    train.n_train_itr=2 \
    train.n_steps=5 \
    train.batch_size=10 \
    train.update_epochs=1 \
    train.val_freq=10 \
    train.save_model_freq=2 \
    train.n_critic_warmup_itr=0 \
    train.use_bc_loss=false \
    train.target_kl=null \
    env.max_episode_steps=40 \
    device=cuda:0
```

**Training penuh** (default config, checkpoint pretrained baseline):

```bash
conda activate flowpolicy-kitchen
cd /path/ke/kripsy12

python finetune_flowpolicy/scripts/run_ft_ppo.py \
    seed=101 \
    device=cuda:0 \
    base_policy_path=outputs/baseline_seed101_standard/checkpoints/latest-001.ckpt
```

Override umum:

```bash
python finetune_flowpolicy/scripts/run_ft_ppo.py \
    env.n_envs=8 \
    train.n_train_itr=200 \
    device=cuda:0 \
    base_policy_path=outputs/baseline_seed101_standard/checkpoints/latest-001.ckpt
```

Output: `outputs/ft_ppo_flowpolicy/<nama_run>/<timestamp>/checkpoint/`

- `best_flowpolicy.ckpt` — untuk evaluasi dengan `infer_kitchen.py`
- `last.pt` / `best.pt` — untuk resume training ReinFlow

## Evaluasi

```bash
conda activate flowpolicy-kitchen
cd /path/ke/kripsy12

python FlowPolicy/infer_kitchen.py \
    --checkpoint outputs/ft_ppo_flowpolicy/<nama_run>/<timestamp>/checkpoint/best_flowpolicy.ckpt \
    --metrics-json outputs/ft_ppo_flowpolicy/<nama_run>/<timestamp>/eval_metrics.json \
    --n-infer-episodes 50 \
    --seed 42
```

Ganti `<nama_run>/<timestamp>` sesuai folder di `outputs/ft_ppo_flowpolicy/`.

## Konversi `.pt` → `.ckpt` (opsional)

Jika hanya punya checkpoint ReinFlow `.pt`:

```bash
python finetune_flowpolicy/scripts/export_to_flowpolicy_ckpt.py \
    --pt-ckpt outputs/ft_ppo_flowpolicy/<nama_run>/<timestamp>/checkpoint/best.pt \
    --src-ckpt outputs/baseline_seed101_standard/checkpoints/latest-001.ckpt \
    --out outputs/ft_ppo_flowpolicy/<nama_run>/<timestamp>/checkpoint/best_flowpolicy.ckpt
```
