# FlowPolicy

Implementasi **Flow Policy** untuk kontrol robotik dengan observasi **point cloud** (antara lain Franka Kitchen via Gymnasium-Robotics). Proses training memakai [Hydra](https://hydra.cc/) dan logging [Weights & Biases](https://wandb.ai/).

Struktur repositori:

```text
<akar-repo>/                # root Git (folder berisi scripts + FlowPolicy)
├── scripts/                # orkestrator eksperimen (baseline + random search)
│   ├── run_experiment.py
│   ├── run_experiment.sh   # pintasan CLI dengan default 66 run
│   ├── run_experiment_random_search.sh   # hanya random search (tanpa baseline)
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

Skrip **`scripts/run_experiment.py`** menjalankan dua fase **berurutan**:

| Fase | Isi | Jumlah run (default) |
|------|-----|------------------------|
| **1. Baseline** | Hyperparameter default FlowPolicy (`experiment_constants.DEFAULT_BASELINE_HPARAMS`) × **3 seed** × **2 profil preprocessing** | **6** |
| **2. Random search** | Konfigurasi yang disampling sekali (`configs.json`, kolom `sampled`) × **3 seed** × **2 profil** | **60** (`--n-configs 10` → 10 × 3 × 2) |

**Total default: 66 run** (6 baseline + 60 random search).

Profil preprocessing (proxy “dengan / tanpa” augmentasi observasi): **`standard`** (noise observasi) dan **`minimal`** (tanpa noise tersebut). Tiap run punya folder sendiri di `runs/`.

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

Untuk **melewati baseline** dan hanya menjalankan fase random search, gunakan flag **`--random-search-only`** di `run_experiment.py`, atau skrip pintasan **`scripts/run_experiment_random_search.sh`**.

Skrip shell tersebut memakai urutan seed **`0` → `42` → `1010` → `0`** (empat posisi per siklus, lalu berulang untuk kombinasi berikutnya dalam grid), dua profil **`standard`** dan **`minimal`**, serta **`--n-configs 10`** seperti default pipeline penuh. Total run random search dengan default skrip: **10 × 4 × 2 = 80** (bukan 60).

**Sampling grid hiperparameter:** jika variabel lingkungan **`SAMPLING_SEED`** belum diset, skrip mengisi seed sampling **secara acak** tiap kali dijalankan (berguna untuk grid RS baru). Untuk mengulang grid yang sama, set eksplisit, misalnya `SAMPLING_SEED=99`, atau biarkan **`configs.json`** yang sudah ada di `--output-dir` (file itu dipakai ulang; tidak di-resample).

**Opsi A — skrip pintasan:**

```bash
./scripts/run_experiment_random_search.sh \
  --output-dir outputs/experiment_rs \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Argumen tambahan diteruskan ke `run_experiment.py` (misalnya `--max-batch-size 64 --dataloader-num-workers 2`).

Reproduksi grid sampling (seed sampling tetap):

```bash
SAMPLING_SEED=99 ./scripts/run_experiment_random_search.sh \
  --output-dir outputs/experiment_rs \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Opsi B — Python langsung** (parameter seed / sampling bisa Anda ubah):

```bash
python scripts/run_experiment.py \
  --random-search-only \
  --seeds 0 42 1010 0 \
  --profiles standard minimal \
  --n-configs 10 \
  --sampling-seed 99 \
  --output-dir outputs/experiment_rs \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Flag **`--baseline-only`** dan **`--random-search-only`** saling meniadakan; jangan dipakai bersamaan.

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

Jika masih OOM setelah **`16`**, tidak ada pengaturan aman lain di orchestrator selain **menurunkan hiperparameter batch di config** (mis. sampel random search yang memakai `batch_size` besar di `configs.json`) atau **menjalankan training tunggal** dengan override Hydra lebih agresif — pertimbangkan juga **instans GPU cloud** (lihat [Vast.ai](#menjalankan-di-vastai)) untuk pipeline **66 run** penuh agar waktu dan stabilitas lebih masuk akal.

**Tips laptop:** tutup aplikasi berat (browser dengan banyak tab, IDE lain), hindari sleep/hibernasi saat training panjang, dan pastikan daya AC terhubung (thermal GPU turun bisa memicu error atau throttling).

### Opsi CLI yang sering dipakai

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--seeds` | `0 42 101` | Tiga seed untuk training / dataset / inferensi |
| `--profiles` | `standard minimal` | Profil preprocessing dataset |
| `--n-configs` | `10` | Jumlah sampel random search; total run RS = **n × jumlah seed × jumlah profil** (= 60 dengan default) |
| `--sampling-seed` | `99` | Seed sampling random search (agar `configs.json` reproducible) |
| `--cv-seed` | `12345` | Seed **satu** pembagian episode train/val/test (bukan k-fold) |
| `--n-infer-episodes` | `50` | Episode evaluasi setelah training |
| `--output-dir` | `outputs/experiment` | Relatif terhadap akar repo |
| `--max-batch-size` | `128` | Plafon batch train/val; turunkan untuk **GPU 16 GB** atau **laptop 8 GB** (lihat bagian di atas) |
| `--dataloader-num-workers` | `4` | Workers DataLoader |
| `--checkpoint-every` | `200` | Simpan checkpoint berkala (resume jika mesin mati) |
| `--baseline-only` | (off) | Hanya baseline; random search tidak dijalankan |
| `--random-search-only` | (off) | Hanya random search; baseline tidak dijalankan |

### Keluaran

Di `--output-dir` (mis. `outputs/experiment/`):

- `configs.json` — baseline + daftar konfigurasi random search (`version: 2`).
- `cv_splits.json` — **satu** partisi train/val (+ meta `split_mode`, bukan daftar lipatan k-fold penuh).
- `results.csv` — satu baris per run (hyperparameter + metrik + `status`).
- `runs/<nama_run>/` — output Hydra, `checkpoints/`, `metrics.json`, `training_final.json`.
- `summary.csv`, `plots/*.png` dan `*.pdf` — dibuat otomatis di akhir (`summarize.py`, `plot_results.py`).

Nama folder run:

- Baseline: `baseline_seed<seed>_<profile>`
- Random search: `cfg<idx>_seed<seed>_<profile>`

### Resume setelah mesin mati

Run **dilewati** jika sudah selesai: ada **`metrics.json`** di folder run, atau **`results.csv`** sudah punya baris dengan kombinasi yang sama dan **`status=ok`**.

- Training terputus (ada **`latest.ckpt`**, belum ada **`training_final.json`**) → training **dilanjutkan** (`training.resume=true`).
- Training selesai (**`training_final.json`** + ckpt) tetapi inferensi belum → hanya **`infer_kitchen.py`** yang dijalankan.

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

(`--output-dir` relatif terhadap akar repo.)

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
