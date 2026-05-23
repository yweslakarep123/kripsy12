# FlowPolicy

Implementasi **Flow Policy** untuk kontrol robotik dengan observasi **point cloud** (antara lain Franka Kitchen via Gymnasium-Robotics). Proses training memakai [Hydra](https://hydra.cc/) dan logging [Weights & Biases](https://wandb.ai/).

Struktur repositori:

```text
<akar-repo>/                # root Git (folder berisi scripts + FlowPolicy)
├── scripts/                # orkestrator eksperimen (baseline + random search)
│   ├── run_experiment.py
│   ├── run_experiment.sh         # pintasan CLI: baseline lalu random search
│   ├── run_baseline_only.sh      # hanya baseline (6 run default)
│   ├── run_hyperband_only.sh     # hanya random search + rerun pemenang (nama legacy)
│   ├── run_hyperband_laptop_smoke.sh   # smoke test random search (laptop)
│   ├── verify_hyperband_no_gpu.sh      # cek logika Hyperband lama (tanpa GPU)
│   ├── random_search.py          # random search berpusat di baseline
│   ├── hyperband_search.py       # Hyperband legacy (tidak dipakai orkestrator utama)
│   ├── cv_splits.py
│   ├── summarize.py
│   ├── plot_results.py
│   └── experiment_constants.py
└── FlowPolicy/             # train.py, infer_kitchen.py, paket flow_policy_3d
    ├── train.py
    ├── infer_kitchen.py
    ├── setup.py
    ├── requirements-franka-kitchen.txt
    └── flow_policy_3d/
```

- Perintah **training tunggal** (`train.py`): dari **`FlowPolicy/`** (folder yang berisi `train.py`).
- **Pipeline eksperimen** (`scripts/run_experiment.py`): dijalankan dari **akar repositori** (folder induk `scripts/` dan `FlowPolicy/`).
- **Cheat sheet perintah** (conda, random search minimal, smoke test laptop, Vast.ai): lihat [Referensi perintah penting](#referensi-perintah-penting).

## Prasyarat

- **Linux** (disarankan Ubuntu 22.04+); training headless dengan MuJoCo / EGL umum dipakai di cloud GPU.
- **NVIDIA GPU** dengan driver CUDA yang kompatibel dengan PyTorch yang Anda pasang.
- **Python 3.10** (sesuai lingkungan yang dipakai proyek ini).
- Akun **Weights & Biases** (opsional: set `WANDB_API_KEY` atau `wandb offline`).

## Instalasi (lokal atau VM / Vast.ai)

### 1. Buat environment

Disarankan **Miniconda/Mambaforge**:

```bash
conda create -n flowpolicy-kitchen python=3.10 -y
conda activate flowpolicy-kitchen
```

### 2. Pasang PyTorch (sesuaikan versi CUDA host Anda)

Contoh untuk CUDA 12.4 (sesuaikan dengan [PyTorch Get Started](https://pytorch.org/get-started/locally/)):

```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### 3. Dependensi proyek + editable install

```bash
cd FlowPolicy
pip install -U pip
pip install -r requirements-franka-kitchen.txt
pip install -e .
```

**PyTorch3D:** jika `pip install pytorch3d` gagal, coba:

```bash
conda install pytorch3d -c pytorch3d
```

## Dataset (zarr)

Task Franka Kitchen membutuhkan dataset **zarr** (lihat `flow_policy_3d/config/task/franka_kitchen_complete4.yaml`, field `task.dataset.zarr_path`).

- Default config mengarah ke `FlowPolicy/data/kitchen_complete_from_minari.zarr` (relatif dari folder berisi `train.py`; pada layout repo ini setara dengan `FlowPolicy/FlowPolicy/data/kitchen_complete_from_minari.zarr`).
- Anda bisa mengganti path lewat override Hydra, misalnya data hasil konversi Minari:

```bash
task.dataset.zarr_path=FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Pastikan file zarr ada di path tersebut (atau gunakan path absolut di instance Vast.ai).

## Referensi perintah penting

Semua perintah di bawah ini dijalankan dari **akar repositori** (folder yang berisi `scripts/` dan `FlowPolicy/`), kecuali `train.py` / `infer_kitchen.py` yang dijalankan dari **`FlowPolicy/`**.

Path dataset default (relatif terhadap folder berisi `train.py`):

```text
FlowPolicy/data/kitchen_complete_from_minari.zarr
```

### Persiapan (setiap sesi terminal baru)

```bash
cd /path/ke/kripsy12          # ganti dengan path clone Anda
conda activate flowpolicy-kitchen
```

Cek GPU dan PyTorch (wajib sebelum training nyata):

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Bantuan CLI orkestrator:

```bash
python scripts/run_experiment.py --help
```

Jadikan skrip shell dapat dieksekusi (sekali saja):

```bash
chmod +x scripts/*.sh
```

### Random search saja (profil **minimal**, disarankan)

Fase pencarian hiperparameter memakai **satu seed × profil `minimal`** (tanpa augmentasi observasi), dengan **early stopping** aktif (eval simulasi Kitchen setiap 200 epoch). Pemenang (`val_loss` terkecil) di-rerun penuh (train + inferensi) pada `--seeds × --profiles`.

**Skrip pintasan** (default: N=16 trial, search di `seed=0` × `minimal`, rerun di 3 seed × 2 profil):

```bash
conda activate flowpolicy-kitchen
chmod +x scripts/run_hyperband_only.sh

./scripts/run_hyperband_only.sh \
  --output-dir outputs/search_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Python langsung** (semua flag eksplisit):

```bash
python scripts/run_experiment.py \
  --search-only \
  --output-dir outputs/search_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --random-search-n 16 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-train-seed 0 \
  --search-profile minimal \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --early-stop-rollout-every 200 \
  --max-batch-size 128 \
  --dataloader-num-workers 4 \
  --cv-seed 12345
```

**Hanya profil `minimal` di seluruh pipeline** (search + baseline/rerun), tambahkan:

```bash
  --profiles minimal
```

(pada skrip pintasan, argumen setelah `"$@"` bisa meng-override: `./scripts/run_hyperband_only.sh --profiles minimal`)

| Aspek | Nilai default | Keterangan |
|-------|---------------|------------|
| Profil search | `minimal` | `--search-profile minimal` |
| Seed search | `0` | `--search-train-seed 0` (satu run training per trial) |
| Trial | `16` | `--random-search-n 16` |
| Early stop | **on** | `--disable-early-stop` untuk mematikan |
| Rollout eval | setiap **200** epoch | `--early-stop-rollout-every 200` |
| Sinyal pemenang | `val_loss_final` | dari `training_final.json` per trial |
| Video MP4 | hanya fase **inferensi** rerun | `runs/.../inference_videos/infer_ep_*.mp4`; eval training tidak menyimpan MP4 |
| Resume search | `random_search_state.json` | jangan ganti `N`, `--random-search-seed`, `--search-train-seed`, atau `--search-profile` di tengah jalan |

Folder trial search: `runs/hb_cfg<idx>_seed0_minimal/` (cfg_idx ≥ 1000). Rerun pemenang: `runs/hb_best_seed<seed>_<profile>/` (`cfg_idx=-3`).

### Verifikasi logika Hyperband legacy (tanpa GPU, cepat)

Memastikan rumus bracket, Successive Halving, dan `hyperband_state.json` benar — **tidak** melatih model:

```bash
conda activate flowpolicy-kitchen
./scripts/verify_hyperband_no_gpu.sh
```

### Smoke test di laptop (8 GB VRAM, end-to-end ringan)

Setelah `torch.cuda.is_available()` bernilai `True`. N=2 trial, epoch sedikit via early stop / config — hanya memastikan pipeline jalan:

```bash
conda activate flowpolicy-kitchen
./scripts/run_hyperband_laptop_smoke.sh
```

Keluaran: `outputs/laptop_hyperband_smoke/` (`random_search_state.json`, `runs/hb_cfg*_seed0_minimal/`).

Override path keluaran / zarr:

```bash
OUTPUT_DIR=outputs/smoke1 ZARR_PATH=FlowPolicy/data/kitchen_complete_from_minari.zarr \
  ./scripts/run_hyperband_laptop_smoke.sh
```

### Pipeline eksperimen — tiga mode utama

| Mode | Skrip pintasan | Isi |
|------|----------------|-----|
| Baseline + random search + rerun pemenang | `./scripts/run_experiment.sh` | Fase 1→2→3 (default produksi) |
| **Random search + rerun** (profil search `minimal`) | `./scripts/run_hyperband_only.sh` | Lewati baseline; N=16 trial default |
| Hanya baseline (6 run) | `./scripts/run_baseline_only.sh` | Lewati random search |
| Smoke test laptop | `./scripts/run_hyperband_laptop_smoke.sh` | N kecil, VRAM 8 GB |

**Produksi (default, dari laptop kuat atau Vast.ai):**

```bash
conda activate flowpolicy-kitchen
./scripts/run_experiment.sh \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Hanya baseline:**

```bash
./scripts/run_baseline_only.sh \
  --output-dir outputs/baseline_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Hanya random search (+ rerun top-1 di 3 seed × 2 profil):**

```bash
./scripts/run_hyperband_only.sh \
  --output-dir outputs/search_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Contoh di Vast.ai / cloud (sesuaikan path):

```bash
cd /workspace/kripsy12
conda activate flowpolicy-kitchen
./scripts/run_hyperband_only.sh \
  --output-dir outputs/search_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --random-search-n 16
```

### Anggaran waktu (random search)

Asumsi kalibrasi (satu run penuh = **3000 epoch**, **1 seed × 1 profil**, `batch_size` ≤ 128):

| GPU efektif | Waktu per run penuh | Catatan |
|-------------|---------------------|---------|
| **~30 TFLOPS** | ~2,7 jam | Early stop sering menghentikan lebih awal |
| **~100 TFLOPS** | ~0,8 jam | |

**Fase random search (default):** `N=16` trial × **1 seed × profil `minimal`**. Setiap trial dilatih sampai `training.num_epochs` (default 3000) atau berhenti lebih awal via early stop. Perkiraan kasar: `16 × waktu_per_run` (bisa jauh lebih sedikit jika early stop aktif).

**Fase rerun pemenang:** **6 run** (3 seed × 2 profil) train + inferensi penuh.

**Kalibrasi wajib di mesin Anda:** jalankan satu trial search atau satu baseline, hitung jam untuk satu run, lalu skala.

#### Resume

Jalankan ulang **perintah yang sama** dengan `--output-dir` yang sama; random search melanjutkan dari `random_search_state.json`. Jangan ganti `--random-search-n`, `--random-search-seed`, `--search-train-seed`, atau `--search-profile` di tengah jalan.

#### Laptop 8 GB — knob VRAM (tambahkan ke perintah di atas)

```bash
  --max-batch-size 16 \
  --dataloader-num-workers 0 \
  --skip-inference-videos
```

Contoh random search di laptop 8 GB (lambat; disarankan smoke test dulu):

```bash
./scripts/run_hyperband_only.sh \
  --output-dir outputs/laptop_search \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --random-search-n 4 \
  --max-batch-size 16 \
  --dataloader-num-workers 0 \
  --skip-inference-videos
```

### GPU 16 GB (Vast.ai / desktop)

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 64 \
  --dataloader-num-workers 2
```

### Melanjutkan eksperimen / folder baru

Jalankan ulang perintah yang sama di `--output-dir` yang sama — job selesai (`metrics.json` atau `status=ok` di CSV) dilewati; random search melanjutkan dari `random_search_state.json`.

Mulai dari nol (folder baru):

```bash
mkdir -p outputs/experiment_fresh
./scripts/run_experiment.sh --output-dir outputs/experiment_fresh \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Hapus manual isi folder lama jika ingin train ulang semua di path yang sama: `runs/`, `results.csv`, `configs.json`, `random_search_state.json`, `cv_splits.json`.

### Agregasi dan plot (tanpa training ulang)

```bash
python scripts/summarize.py --output-dir outputs/experiment
python scripts/plot_results.py --output-dir outputs/experiment
```

Dengan `results.csv` kustom:

```bash
python scripts/summarize.py --output-dir outputs/experiment --results-csv outputs/experiment/results.csv
python scripts/plot_results.py --output-dir outputs/experiment --results-csv outputs/experiment/results.csv
```

### Training / inferensi tunggal (di luar orkestrator)

Dari folder **`FlowPolicy/`**:

```bash
cd FlowPolicy
conda activate flowpolicy-kitchen

# Satu run training
python train.py task=franka_kitchen_complete4 \
  task.dataset.zarr_path=FlowPolicy/data/kitchen_complete_from_minari.zarr \
  logging.mode=offline

# Inferensi dari checkpoint
python infer_kitchen.py \
  --checkpoint path/ke/checkpoints/latest.ckpt \
  --metrics-json path/ke/metrics.json \
  --n-infer-episodes 50 \
  --seed 42 \
  --warmup-steps 20
```

### Vast.ai (ringkas)

```bash
conda activate flowpolicy-kitchen
cd /workspace/kripsy12
./scripts/run_hyperband_only.sh \
  --output-dir outputs/search_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --random-search-n 16 \
  --max-batch-size 128 \
  --dataloader-num-workers 4
```

Variabel lingkungan opsional: `WANDB_API_KEY`, `WANDB_MODE=offline`.

Detail instalasi cloud: [Menjalankan di Vast.ai](#menjalankan-di-vastai).

---

## Menjalankan training

Dari **`FlowPolicy/`**:

```bash
python train.py task=franka_kitchen_complete4 \
  task.dataset.zarr_path=FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Override umum lain:

| Override | Keterangan |
|----------|------------|
| `training.device=cuda:0` | Device PyTorch (sesuaikan jika multi-GPU). |
| `training.debug=true` | Mode debug Hydra (epoch/step dibatasi di kode). |
| `logging.mode=offline` | W&B tanpa upload (berguna di mesin tanpa kredensial). |

Checkpoint dan log Hydra biasanya di bawah `FlowPolicy/data/outputs/` (atau sesuai `hydra.run.dir`).

## Pipeline eksperimen (baseline + random search, tanpa k-fold)

Pelatihan **tidak** memakai validasi silang berlipat (k-fold). Episode dibagi **sekali** menjadi train / validation / test (`scripts/cv_splits.py`): satu partisi tetap, dapat direproduksi dengan `--cv-seed`.

Skrip **`scripts/run_experiment.py`** menjalankan tiga fase **berurutan**:

| Fase | Isi | Jumlah run (default) |
|------|-----|------------------------|
| **1. Baseline** | Hyperparameter default FlowPolicy (`experiment_constants.DEFAULT_BASELINE_HPARAMS`) × **3 seed** × **2 profil preprocessing** | **6** |
| **2. Random search** | `N` trial (default **16**) disampling di sekitar baseline; setiap trial dilatih **1 seed × 1 profil** (`--search-train-seed 0`, `--search-profile minimal`). Sinyal pemenang = **`val_loss_final` terkecil**. Early stop aktif (eval simulasi tiap 200 epoch). State: `random_search_state.json`. | **16** trial (tidak ditulis ke `results.csv`) |
| **3. Rerun pemenang** | Konfigurasi pemenang di-rerun **penuh** (train + infer + `results.csv` `status=ok`) pada **3 seed × 2 profil**. Baris CSV: `cfg_idx = -3`. | **6** |

**Total default:** 6 baseline + 16 trial search + 6 rerun = **12 run tercatat di `results.csv`** + evaluasi intermediate search di `random_search_state.json`.

Profil preprocessing: **`standard`** (augmentasi/noise observasi) dan **`minimal`** (tanpa augmentasi). **Fase search default memakai `minimal` saja**; baseline dan rerun tetap bisa keduanya lewat `--profiles`.

### Random search singkat

- **`--random-search-n`**: jumlah trial (default 16).
- **`--random-search-seed`**: seed RNG sampling konfigurasi (default 99).
- **`--random-search-sigma`**: lebar sampling Gaussian di sekitar baseline (default 1.0).
- **`--search-train-seed`** / **`--search-profile`**: seed dan profil **satu-satunya** dipakai saat melatih setiap trial (default `0` × **`minimal`**).
- Setiap trial dilatih penuh hingga `training.num_epochs` (dari config, default 3000) atau berhenti lebih awal via **early stopping** (monitor `success_rate_k1…k4` di simulasi).
- Sinyal `val_loss` dari `training_final.json.val_loss_final` (`training.compute_val_loss=true`).
- **Inferensi rollout + MP4** hanya pada fase **baseline / rerun pemenang**, bukan selama trial search.

Flag **`--search-only`** / **`--hyperband-only`** (alias legacy): lewati baseline, jalankan random search + rerun saja.

### Menjalankan dari akar repositori

Folder akar adalah yang berisi **`scripts/`** dan **`FlowPolicy/`** (kode training ada di `FlowPolicy/train.py`).

**Opsi A — skrip pintasan (disarankan):**

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Opsi B — memanggil Python langsung:**

```bash
python scripts/run_experiment.py \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Argumen `--zarr-path` **relatif terhadap folder berisi `train.py`** (lihat `KitchenDataset._resolve_zarr_path`). Untuk dataset di `FlowPolicy/FlowPolicy/data/`, gunakan nilai `FlowPolicy/data/kitchen_complete_from_minari.zarr` atau path absolut.

### Hanya random search (tanpa baseline)

Gunakan **`--search-only`** di `run_experiment.py`, atau skrip pintasan **`scripts/run_hyperband_only.sh`** (nama file legacy; isinya random search).

```bash
./scripts/run_hyperband_only.sh \
  --output-dir outputs/search_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Flag **`--baseline-only`** dan **`--search-only`** / **`--hyperband-only`** saling eksklusif.

**Catatan:** opsi **`--results-csv`** mengubah lokasi **`results.csv`** untuk baseline dan rerun pemenang. Trial random search **tidak** ditulis ke `results.csv` — state-nya ada di **`random_search_state.json`**.

Penjelasan **`--results-csv`**: jika **tidak** diberikan, file CSV default adalah `<output-dir>/results.csv`. Kombinasi `(cfg_idx, seed, profile, fold)` yang sudah **`status=ok`** dilewati saat rerun/baseline.

### Training ulang: hanya baseline, folder baru (laptop, tanpa melanjutkan run lama)

Orchestrator **melewati** job yang sudah selesai jika di `--output-dir` yang sama sudah ada **`metrics.json`**. Untuk **baseline** dan **rerun pemenang**, lewati juga jika **`results.csv`** memuat baris `status=ok`. Untuk **fase random search**, resume dari **`random_search_state.json`**. Folder baru = mulai dari nol:

```bash
mkdir -p outputs/baseline_laptop_fresh
./scripts/run_baseline_only.sh \
  --output-dir outputs/baseline_laptop_fresh \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 16 \
  --dataloader-num-workers 0
```

Setara tanpa skrip shell:

```bash
mkdir -p outputs/baseline_laptop_fresh
python scripts/run_experiment.py \
  --baseline-only \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --output-dir outputs/baseline_laptop_fresh \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 16 \
  --dataloader-num-workers 0
```

- Ganti nama **`outputs/baseline_laptop_fresh`** sesuai keinginan Anda (tanggal / mesin).
- Jika Anda **sengaja** memakai ulang folder lama tetapi ingin train ulang semua, hapus dulu isinya (**`runs/`**, **`results.csv`**, **`configs.json`**, **`random_search_state.json`**, **`cv_splits.json`**) — hati-hati: data metrik lama hilang.

### Opsi untuk GPU 16 GB

Model ini berat; pada GPU **16 GB** kurangi beban memori bertahap jika muncul OOM:

| Knob | Saran untuk 16 GB | Catatan |
|------|-------------------|---------|
| `--max-batch-size` | **`64`** (paling aman), lalu coba **`96`** | Membatasi batch train dan validation secara bersamaan |
| `--dataloader-num-workers` | **`2`** atau **`0`** | Mengurangi salinan batch di RAM CPU |
| `--checkpoint-every` | tetap default atau lebih besar | Tidak mengurangi VRAM; hanya frekuensi simpan ckpt |

Contoh **konservatif (VRAM 16 GB)**:

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 64 \
  --dataloader-num-workers 2
```

Contoh **sedikit lebih agresif** (setelah 64 berjalan stabil):

```bash
python scripts/run_experiment.py \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 96 \
  --dataloader-num-workers 4
```

Default orchestrator memakai **`--max-batch-size 128`**; itu cocok untuk VRAM **≥ ~24 GB**. Untuk **16 GB**, mulai dari **`64`** atau **`96`**.

### Opsi untuk GPU 8 GB (laptop)

Pada **VRAM 8 GB** (umum di laptop gaming ringan / mobile GPU), ruang sangat sempit untuk model besar + simulasi Kitchen. Perkirakan **OOM** lebih sering; selalu mulai dari batch **kecil** dan worker **minimal**:

| Knob | Saran untuk 8 GB | Catatan |
|------|------------------|---------|
| `--max-batch-size` | **`16`** (paling aman), lalu coba **`32`** jika stabil | Lebih kecil dari 16 GB; hindari `64` kecuali Anda sudah verifikasi tidak OOM |
| `--dataloader-num-workers` | **`0`** (disarankan) atau **`1`** | Worker lebih banyak menambah salinan batch di **RAM sistem** laptop |
| `--checkpoint-every` | bisa dinaikkan (mis. `400`) | Mengurangi frekuensi I/O disk, tidak menolong VRAM banyak |

Contoh **konservatif (VRAM 8 GB / laptop)**:

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 16 \
  --dataloader-num-workers 0
```

Jika masih OOM setelah **`16`**, turunkan **`--random-search-n`** atau **`--max-batch-size`**, atau gunakan GPU cloud (lihat [Vast.ai](#menjalankan-di-vastai)).

**Tips laptop:** tutup aplikasi berat (browser dengan banyak tab, IDE lain), hindari sleep/hibernasi saat training panjang, dan pastikan daya AC terhubung (thermal GPU turun bisa memicu error atau throttling).

### Opsi CLI yang sering dipakai

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--seeds` | `0 42 101` | Seed untuk baseline / rerun pemenang (train + inferensi). |
| `--profiles` | `standard minimal` | Profil preprocessing baseline + rerun (`standard` = augmentasi; `minimal` = tanpa). |
| `--search-only` | (off) | Hanya random search + rerun pemenang; lewati baseline. |
| `--hyperband-only` | (off) | Alias legacy untuk `--search-only`. |
| `--random-search-n` | `16` | Jumlah trial random search. |
| `--random-search-seed` | `99` | Seed RNG sampling konfigurasi. |
| `--random-search-sigma` | `1.0` | Lebar sampling di sekitar baseline. |
| `--search-train-seed` | `0` | Seed training **selama fase search** (satu seed). |
| `--search-profile` | **`minimal`** | Profil preprocessing **selama fase search** (satu profil). |
| `--disable-early-stop` | (off) | Matikan early stopping (baseline, search, rerun). |
| `--early-stop-rollout-every` | `200` | Interval epoch eval simulasi untuk early stop. |
| `--cv-seed` | `12345` | Seed pembagian episode train/val/test. |
| `--n-infer-episodes` | `50` | Episode evaluasi inferensi test (rerun/baseline). |
| `--skip-inference-videos` | (off) | Lewati MP4 `infer_ep_*.mp4`. |
| `--output-dir` | `outputs/experiment` | Relatif terhadap akar repo. |
| `--results-csv` | (off) | Lokasi `results.csv` kustom; job `status=ok` dilewati. |
| `--max-batch-size` | `128` | Plafon batch train/val. |
| `--dataloader-num-workers` | `4` | Workers DataLoader. |
| `--checkpoint-every` | `200` | Frekuensi simpan checkpoint (resume). |
| `--baseline-only` | (off) | Hanya baseline; lewati random search. |

### Keluaran

Di `--output-dir` (mis. `outputs/experiment/`):

- `configs.json` — baseline (`baseline_config_dict()`) untuk fase-1 dan rerun pemenang.
- `random_search_state.json` — state random search: trial, `val_loss` per cfg, pemenang. Resume search.
- `cv_splits.json` — satu partisi train/val (+ meta `split_mode`).
- `results.csv` — baseline (`cfg_idx=-1`) dan rerun pemenang (`cfg_idx=-3`). Trial search **tidak** ada di CSV.
- `runs/<nama_run>/` — Hydra output, `checkpoints/`, `metrics.json`, `training_final.json`, `inference_videos/` (setelah inferensi).
- `summary.csv`, `plots/*` — dari `summarize.py` / `plot_results.py`.

Nama folder run:

- Baseline: `baseline_seed<seed>_<profile>` (`cfg_idx=-1`).
- Rerun pemenang search: `hb_best_seed<seed>_<profile>` (`cfg_idx=-3`).
- Trial random search: `hb_cfg<idx>_seed<search_seed>_<search_profile>` (mis. `hb_cfg1000_seed0_minimal`).

### Resume setelah mesin mati

- **Baseline** dan **rerun pemenang**: dilewati jika ada **`metrics.json`** atau baris **`status=ok`** di `results.csv`.
- **Random search**: resume dari **`random_search_state.json`** — trial dengan `done=true` dilewati.
- Training terputus (`latest.ckpt`, belum `training_final.json`) → dilanjutkan (`training.resume=true`).
- Training selesai, inferensi belum → hanya `infer_kitchen.py`.

Konfigurasi tiap job dicetak ke **terminal** sebelum `train` / `infer`.

### Inferensi manual (checkpoint tunggal)

Dari **`FlowPolicy/`**:

```bash
python infer_kitchen.py \
  --checkpoint path/ke/checkpoints/latest.ckpt \
  --metrics-json path/ke/metrics.json \
  --n-infer-episodes 50 \
  --seed 42 \
  --warmup-steps 20
```

### Agregasi / plot saja (tanpa train ulang)

```bash
python scripts/summarize.py --output-dir outputs/experiment
python scripts/plot_results.py --output-dir outputs/experiment
```

Jika **`--results-csv`** menunjuk ke file **di luar** `<output-dir>/results.csv`, sertakan opsi yang sama:

```bash
python scripts/summarize.py --output-dir outputs/experiment --results-csv path/ke/results.csv
python scripts/plot_results.py --output-dir outputs/experiment --results-csv path/ke/results.csv
```

(`--output-dir` dan path relatif **`--results-csv`** diukur dari akar repo; path absolut juga boleh.)

## Menjalankan di [Vast.ai](https://vast.ai/)

1. **Pilih template** dengan CUDA + PyTorch yang sudah mendekati kebutuhan, atau image Ubuntu + CUDA lalu ikuti langkah instalasi di atas.
2. **Clone repo** ke disk instance (mis. `/workspace`):

   ```bash
   git clone https://github.com/<user>/<repo>.git
   cd <repo>
   ```

3. **Variabel lingkungan** (di UI Vast atau di shell):

   - `WANDB_API_KEY` — jika memakai W&B online.
   - Opsional: `CUDA_VISIBLE_DEVICES=0` jika hanya satu GPU yang ingin dipakai (catatan: `train.py` juga mengatur `CUDA_VISIBLE_DEVICES` di blok `if __name__ == "__main__"`).

4. **Data zarr:** unggah ke volume instance atau unduh dari penyimpanan Anda; gunakan path absolut di override `task.dataset.zarr_path` agar tidak membingungkan working directory Hydra.

5. **VRAM:** untuk penyetelan **16 GB** lihat [Opsi untuk GPU 16 GB](#opsi-untuk-gpu-16-gb); untuk **laptop 8 GB** lihat [Opsi untuk GPU 8 GB (laptop)](#opsi-untuk-gpu-8-gb-laptop). Utamakan menurunkan **`--max-batch-size`** pada `run_experiment.py`. Jika masih OOM, persempit batch di override Hydra atau gunakan GPU dengan memori lebih besar. Urutan inisialisasi di `train.py` sudah mengutamakan memuat bobot ke GPU sebelum membuat environment simulasi Kitchen (mengurangi bentrok VRAM dengan MuJoCo/rendering).

6. **Headless:** pastikan tidak ada ketergantungan pada display; rendering `rgb_array` via MuJoCo biasanya berjalan di server GPU.

Contoh **On-start script** ringkas:

```bash
#!/bin/bash
set -euo pipefail
cd /workspace/<repo>/FlowPolicy   # folder yang berisi train.py
pip install -r requirements-franka-kitchen.txt
pip install -e .
python train.py task=franka_kitchen_complete4 task.dataset.zarr_path=/data/kitchen.zarr
```

## Push ke GitHub

1. Buat repositori kosong di GitHub.
2. Di mesin lokal (dari **akar repositori**):

   ```bash
   git init   # jika belum
   git remote add origin https://github.com/<user>/<repo>.git
   git add README.md scripts FlowPolicy
   git commit -m "Add README and FlowPolicy training code"
   git branch -M main
   git push -u origin main
   ```

   Hindari meng-commit folder besar seperti `data/outputs/` atau file zarr raksasa; gunakan `.gitignore` bila perlu.

## Lisensi / atribusi

Sesuaikan bagian ini dengan lisensi asli proyek upstream Anda (jika ada).

## Kontak

Sesuaikan dengan informasi kontributor Anda.
