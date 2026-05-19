# `finetune_flowpolicy/` — RL Fine-Tuning FlowPolicy via ReinFlow PPO (Franka Kitchen)

Paket berdiri sendiri yang membungkus checkpoint **FlowPolicy** (`ConditionalUnet1D` +
`FlowPolicyEncoder`, pretrained pada Franka Kitchen point-cloud) sehingga bisa di-fine-tune
dengan **PPO ReinFlow** tanpa memodifikasi `FlowPolicy/` maupun `ReinFlow/`.

Strategi inti:

- **Velocity network adapter** (`adapters/flowpolicy_velocity_adapter.py`): bungkus
  `ConditionalUnet1D` dengan interface `FlowMLP` (ReinFlow) — `forward`, `sample_action`,
  atribut `cond_enc_dim`, `time_dim`, `act_dim_total`, `horizon_steps`, `action_dim`.
- **Frozen encoder + normalizer** (`envs/encoder_wrapper.py`): pre-encode dict obs
  `{point_cloud, agent_pos}` ke flat feature `{"state": (128,)}` di env wrapper.
  Buffer PPO ReinFlow yang stateful low-dim langsung kompatibel (`obs_dim=128`).
- **PPOFlowAdapter** (`model/ppo_flowpolicy.py`): subclass `PPOFlow` ReinFlow yang
  override `load_policy` agar membaca format checkpoint FlowPolicy (`state_dicts.ema_model`).
- **TrainPPOFlowPolicyAgent** (`train/train_ppo_flowpolicy_agent.py`): subclass
  `TrainPPOFlowAgent` yang monkey-patch lokal `env.gym_utils.make_async` untuk
  membangun vec env Franka Kitchen, serta menambah hook penulisan checkpoint dalam
  format `*.ckpt` FlowPolicy (sehingga `FlowPolicy/infer_kitchen.py` bisa langsung dipakai).

Inference steps fine-tuning = **K=1** (sesuai pretraining ConsistencyFM). Eksplorasi
disuplai oleh:

- initial noise `x0 ~ N(0, I)` (`account_for_initial_stochasticity=true`),
- noise injection state-conditioned `σ(s)` via `NoisyFlowMLP` + `ExploreNoiseNet`
  (`noise_scheduler_type=learn_decay`, range `[min_std, max_std]`).

## Prasyarat

Pakai environment proyek yang sama (`conda activate flowpolicy-kitchen`). Dependensi inti
sudah dipasang oleh `FlowPolicy/requirements-franka-kitchen.txt`. Tambahkan dua paket runtime
yang dibutuhkan ReinFlow tapi tidak dipasang FlowPolicy:

```bash
conda activate flowpolicy-kitchen
pip install psutil pretty_errors
```

Verifikasi import end-to-end:

```bash
cd /path/ke/kripsy12
python -c "import finetune_flowpolicy.paths; \
  from finetune_flowpolicy.train.train_ppo_flowpolicy_agent import TrainPPOFlowPolicyAgent; \
  print('OK')"
```

## Quick smoke test (laptop / debugging)

Dari akar repositori (`/path/ke/kripsy12`):

```bash
conda activate flowpolicy-kitchen
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

Smoke test ini membuka 2 env Franka Kitchen secara sinkron dan melakukan 2 iterasi PPO
(evaluasi di itr 0, update di itr 1). Verifikasi end-to-end:

- Loader checkpoint FlowPolicy (`utils/ckpt_io.py`).
- FlowPolicyEncoder + LinearNormalizer di env wrapper (`envs/encoder_wrapper.py`).
- ConditionalUnet1D adapter ke NoisyFlowMLP (`adapters/flowpolicy_velocity_adapter.py`).
- PPO loop & loss (ReinFlow `train_ppo_flow_agent.py`, reused).
- Hook checkpoint output: `last.pt`, `state_<itr>.pt`, `best.pt` (ReinFlow) **dan**
  `last_flowpolicy.ckpt`, `best_flowpolicy.ckpt` (FlowPolicy-compatible, untuk
  `infer_kitchen.py`).

Catatan: pada n_train_itr < 3 scheduler cosine ReinFlow akan menghasilkan
`anneal_steps=0` → ZeroDivisionError. Gunakan `n_train_itr>=2` agar
`anneal_steps=int(n_train_itr * 0.65) >= 1`.

## Training full

```bash
conda activate flowpolicy-kitchen
python finetune_flowpolicy/scripts/run_ft_ppo.py \
    seed=101 \
    device=cuda:0 \
    base_policy_path=outputs/baseline_seed101_standard/checkpoints/latest-001.ckpt
```

Output ada di `outputs/ft_ppo_flowpolicy/<name>/<timestamp>/`:

- `checkpoint/last.pt` — checkpoint penuh ReinFlow (untuk resume).
- `checkpoint/last_flowpolicy.ckpt` — format FlowPolicy (untuk evaluasi).
- `checkpoint/best.pt` & `checkpoint/best_flowpolicy.ckpt` — evaluasi terbaik.

## Evaluasi pakai `infer_kitchen.py` (zero-modification)

```bash
python FlowPolicy/infer_kitchen.py \
    --checkpoint outputs/ft_ppo_flowpolicy/.../checkpoint/best_flowpolicy.ckpt \
    --metrics-json outputs/ft_ppo_flowpolicy/.../eval_metrics.json \
    --n-infer-episodes 50 \
    --seed 42
```

Karena `best_flowpolicy.ckpt` mempertahankan struktur `state_dicts.ema_model` (UNet fine-tuned
+ encoder + normalizer asli + cfg pretrained), `infer_kitchen.py` & `KitchenRunner` bekerja
tanpa adaptasi apapun.

## Konversi manual checkpoint `.pt` → `.ckpt`

```bash
python finetune_flowpolicy/scripts/export_to_flowpolicy_ckpt.py \
    --pt-ckpt outputs/ft_ppo_flowpolicy/.../checkpoint/best.pt \
    --src-ckpt outputs/baseline_seed101_standard/checkpoints/latest-001.ckpt \
    --out outputs/ft_ppo_flowpolicy/.../checkpoint/best_flowpolicy.ckpt
```

## Struktur

```
finetune_flowpolicy/
├── README.md                 # dokumen ini
├── paths.py                  # setup sys.path untuk FlowPolicy/ & ReinFlow/
├── adapters/
│   └── flowpolicy_velocity_adapter.py
├── envs/
│   ├── encoder_wrapper.py
│   └── franka_kitchen_vec.py
├── model/
│   └── ppo_flowpolicy.py
├── train/
│   └── train_ppo_flowpolicy_agent.py
├── utils/
│   └── ckpt_io.py
├── cfg/
│   └── ft_ppo_flowpolicy_kitchen.yaml
└── scripts/
    ├── run_ft_ppo.py
    └── export_to_flowpolicy_ckpt.py
```

## Risiko & mitigasi singkat

- **K=1 = sinyal grad tipis**: BC loss W2 koefisien kecil, `noise_scheduler_type=learn_decay`,
  `account_for_initial_stochasticity=true`. Naikkan `min_std`/`max_std` bila stagnan.
- **Reward sparse Franka Kitchen**: `reward_scale_running=true`, GAE `lambda=0.95`,
  critic warmup ≥5 iter.
- **Action overflow**: `clip_intermediate_actions=true`, `denoised_clip_value=1.0` di ruang
  ternormalisasi `[-1, 1]`; unnormalize di env wrapper.
- **Memori GPU**: `n_envs=8` default. ConditionalUnet1D ~80 M params (down_dims [512,1024,2048])
  membutuhkan ≥16 GB. Untuk GPU lebih kecil, set `down_dims=[256,512,1024]` di config _hanya
  jika checkpoint pretrained juga pakai ukuran itu_ (mis. baseline_seed101_standard pakai
  [512,1024,2048] — jangan diubah, mismatch akan gagal `load_unet_state_dict(strict=True)`).
- **Sync vs Async env**: default `asynchronous: false` (lebih stabil; MuJoCo + encoder
  dijalankan di main process). Aktifkan `env.asynchronous=true` untuk speed-up bila
  multiprocess di Linux stabil.
