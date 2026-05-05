# FlowPolicy

Implementasi **Flow Policy** untuk kontrol robotik dengan observasi **point cloud** (antara lain Franka Kitchen via Gymnasium-Robotics). Proses training memakai [Hydra](https://hydra.cc/) dan logging [Weights & Biases](https://wandb.ai/).

Struktur repositori:

```text
FlowPolicy/                 # root Git (repo ini)
├── scripts/                # orkestrator eksperimen (baseline + random search + CV)
│   ├── run_experiment.py
│   ├── cv_splits.py
│   ├── summarize.py
│   ├── plot_results.py
│   └── experiment_constants.py
└── FlowPolicy/             # kode Python, train.py, paket flow_policy_3d
    ├── train.py
    ├── infer_kitchen.py
    ├── setup.py
    ├── requirements-franka-kitchen.txt
    └── flow_policy_3d/
```

- Perintah **training tunggal** (`train.py`): dari **`FlowPolicy/FlowPolicy`**.
- **Pipeline eksperimen** (`scripts/run_experiment.py`): dari **akar repo** (`FlowPolicy/`, induk dari folder `FlowPolicy/` yang berisi `train.py`).

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
cd FlowPolicy/FlowPolicy
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

- Default config mengarah ke `data/franka_kitchen_complete4_expert.zarr` (relatif dari `FlowPolicy/FlowPolicy`).
- Anda bisa mengganti path lewat override Hydra, misalnya data hasil konversi Minari:

```bash
task.dataset.zarr_path=FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Pastikan file zarr ada di path tersebut (atau gunakan path absolut di instance Vast.ai).

## Menjalankan training

Dari **`FlowPolicy/FlowPolicy`**:

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

Checkpoint dan log Hydra biasanya di bawah `FlowPolicy/FlowPolicy/data/outputs/`.

## Pipeline eksperimen (baseline + random search + CV)

Skrip **`scripts/run_experiment.py`** mengorkestrasi:

1. **Baseline** — hyperparameter default FlowPolicy (`experiment_constants.DEFAULT_BASELINE_HPARAMS`) untuk tiap kombinasi **seed × preprocessing × lipatan CV**.
2. **Random search** — konfigurasi yang disampling sekali (disimpan di `configs.json`), dipakai bersama untuk semua seed dan profil preprocessing.

Preprocessing: **`standard`** (noise observasi) dan **`minimal`** (tanpa augmentasi). Tiap run punya folder sendiri di bawah `runs/`.

### Menjalankan dari akar repositori

```bash
cd FlowPolicy    # folder yang berisi scripts/ dan subfolder FlowPolicy/
python scripts/run_experiment.py \
  --output-dir outputs/experiment \
  --zarr-path data/kitchen_complete_from_minari.zarr
```

Argumen `--zarr-path` bersifat **relatif terhadap `FlowPolicy/FlowPolicy`** (tempat `train.py`). Sesuaikan jika dataset Anda di lokasi lain (mis. path absolut).

### Opsi CLI yang sering dipakai

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--seeds` | `0 42 101` | Tiga seed untuk inisialisasi / shuffle / inferensi |
| `--profiles` | `standard minimal` | Profil dataset (dengan / tanpa noise observasi) |
| `--n-configs` | `10` | Jumlah kombinasi hyperparameter random search |
| `--n-folds` | `5` | Lipatan CV pada level episode |
| `--sampling-seed` | `99` | Seed untuk sampling random search (reproducible `configs.json`) |
| `--cv-seed` | `12345` | Seed pembagian episode train/val/test |
| `--n-infer-episodes` | `50` | Episode evaluasi setelah training |
| `--output-dir` | `outputs/experiment` | Relatif terhadap akar repo |
| `--max-batch-size` | `128` | **Plafon** batch train/val (turunkan jika VRAM ~16 GB kewalahan, mis. `96` atau `64`) |
| `--dataloader-num-workers` | `4` | Workers DataLoader (turunkan jika RAM host penuh) |
| `--checkpoint-every` | `200` | Simpan checkpoint berkala agar bisa dilanjut setelah mesin mati |

Contoh untuk GPU **VRAM ~16 GB**:

```bash
python scripts/run_experiment.py \
  --output-dir outputs/experiment \
  --zarr-path data/kitchen_complete_from_minari.zarr \
  --max-batch-size 96 \
  --dataloader-num-workers 4
```

### Keluaran

Di `--output-dir` (mis. `outputs/experiment/`):

- `configs.json` — baseline + daftar konfigurasi random search (`version: 2`).
- `cv_splits.json` — definisi lipatan episode.
- `results.csv` — satu baris per run (hyperparameter + metrik + `status`).
- `runs/<nama_run>/` — Hydra output, `checkpoints/`, `metrics.json`, `training_final.json`.
- `summary.csv`, `plots/*.png` dan `*.pdf` — dibuat otomatis di akhir (`summarize.py`, `plot_results.py`).

Nama folder baseline: `baseline_seed<seed>_<profile>_fold<f>`; random search: `cfg<idx>_seed<seed>_<profile>_fold<f>`.

### Resume setelah mesin mati

Run **dilewati** jika sudah selesai: ada **`metrics.json`** di folder run, atau **`results.csv`** sudah punya baris dengan kombinasi yang sama dan **`status=ok`**.

- Training terputus (ada **`latest.ckpt`**, belum ada **`training_final.json`**) → training **dilanjutkan** (`training.resume=true`).
- Training selesai (**`training_final.json`** + ckpt) tetapi inferensi belum → hanya **`infer_kitchen.py`** yang dijalankan.

Konfigurasi tiap job dicetak ke **terminal** sebelum `train` / `infer`.

### Inferensi manual (checkpoint tunggal)

Dari **`FlowPolicy/FlowPolicy`**:

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

(`--output-dir` relatif terhadap akar repo.)

## Menjalankan di [Vast.ai](https://vast.ai/)

1. **Pilih template** dengan CUDA + PyTorch yang sudah mendekati kebutuhan, atau image Ubuntu + CUDA lalu ikuti langkah instalasi di atas.
2. **Clone repo** ke disk instance (mis. `/workspace`):

   ```bash
   git clone https://github.com/<user>/FlowPolicy.git
   cd FlowPolicy/FlowPolicy
   ```

3. **Variabel lingkungan** (di UI Vast atau di shell):

   - `WANDB_API_KEY` — jika memakai W&B online.
   - Opsional: `CUDA_VISIBLE_DEVICES=0` jika hanya satu GPU yang ingin dipakai (catatan: `train.py` juga mengatur `CUDA_VISIBLE_DEVICES` di blok `if __name__ == "__main__"`).

4. **Data zarr:** unggah ke volume instance atau unduh dari penyimpanan Anda; gunakan path absolut di override `task.dataset.zarr_path` agar tidak membingungkan working directory Hydra.

5. **VRAM:** model ini besar (~255M parameter). Jika masih OOM, kurangi `dataloader.batch_size` / `val_dataloader.batch_size` di override Hydra atau gunakan GPU dengan memori lebih besar. Urutan inisialisasi di `train.py` sudah mengutamakan memuat bobot ke GPU sebelum membuat environment simulasi Kitchen (mengurangi bentrok VRAM dengan MuJoCo/rendering).

6. **Headless:** pastikan tidak ada ketergantungan pada display; rendering `rgb_array` via MuJoCo biasanya berjalan di server GPU.

Contoh **On-start script** ringkas:

```bash
#!/bin/bash
set -euo pipefail
cd /workspace/FlowPolicy/FlowPolicy   # sesuaikan path clone Anda
pip install -r requirements-franka-kitchen.txt
pip install -e .
python train.py task=franka_kitchen_complete4 task.dataset.zarr_path=/data/kitchen.zarr
```

## Push ke GitHub

1. Buat repositori kosong di GitHub.
2. Di mesin lokal (dari root repo `FlowPolicy/`):

   ```bash
   git init   # jika belum
   git remote add origin https://github.com/<user>/<repo>.git
   git add README.md FlowPolicy
   git commit -m "Add README and FlowPolicy training code"
   git branch -M main
   git push -u origin main
   ```

   Hindari meng-commit folder besar seperti `data/outputs/` atau file zarr raksasa; gunakan `.gitignore` bila perlu.

## Lisensi / atribusi

Sesuaikan bagian ini dengan lisensi asli proyek upstream Anda (jika ada).

## Kontak

Sesuaikan dengan informasi kontributor Anda.
