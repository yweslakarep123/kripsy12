# FlowPolicy

Implementasi **Flow Policy** untuk kontrol robotik dengan observasi **point cloud** (antara lain Franka Kitchen via Gymnasium-Robotics). Proses training memakai [Hydra](https://hydra.cc/) dan logging [Weights & Biases](https://wandb.ai/).

Struktur repositori:

```text
<akar-repo>/                # root Git (folder berisi scripts + FlowPolicy)
├── scripts/                # orkestrator eksperimen (baseline + Hyperband)
│   ├── run_experiment.py
│   ├── run_experiment.sh         # pintasan CLI: baseline lalu Hyperband
│   ├── run_baseline_only.sh      # hanya baseline (6 run default)
│   ├── run_hyperband_only.sh     # hanya Hyperband + rerun pemenang top-1
│   ├── run_hyperband_laptop_smoke.sh   # smoke test Hyperband (laptop, R kecil)
│   ├── verify_hyperband_no_gpu.sh      # cek logika Hyperband tanpa GPU
│   ├── hyperband_search.py       # implementasi Hyperband (Li et al., 2018)
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
- **Cheat sheet perintah** (conda, smoke test laptop, Hyperband, Vast.ai): lihat [Referensi perintah penting](#referensi-perintah-penting).

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

### Verifikasi logika Hyperband (tanpa GPU, cepat)

Memastikan rumus bracket, Successive Halving, dan `hyperband_state.json` benar — **tidak** melatih model:

```bash
conda activate flowpolicy-kitchen
./scripts/verify_hyperband_no_gpu.sh
```

### Smoke test di laptop (8 GB VRAM, end-to-end ringan)

Setelah `torch.cuda.is_available()` bernilai `True`. Epoch sedikit (`R=4`), satu bracket, satu seed — hanya untuk memastikan pipeline jalan:

```bash
conda activate flowpolicy-kitchen
./scripts/run_hyperband_laptop_smoke.sh
```

Keluaran: `outputs/laptop_hyperband_smoke/` (`hyperband_state.json`, `runs/hb_cfg*/`, `runs/hb_best_seed0_standard/`).

Override path keluaran / zarr:

```bash
OUTPUT_DIR=outputs/smoke1 ZARR_PATH=FlowPolicy/data/kitchen_complete_from_minari.zarr \
  ./scripts/run_hyperband_laptop_smoke.sh
```

### Pipeline eksperimen — tiga mode utama

| Mode | Skrip pintasan | Isi |
|------|----------------|-----|
| Baseline + Hyperband + rerun pemenang | `./scripts/run_experiment.sh` | Fase 1→2→3 (default produksi) |
| Hanya baseline (6 run) | `./scripts/run_baseline_only.sh` | Lewati Hyperband |
| Hanya Hyperband + rerun pemenang | `./scripts/run_hyperband_only.sh` | Lewati baseline |

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

**Hanya Hyperband (+ rerun top-1 di 3 seed × 2 profil):**

```bash
./scripts/run_hyperband_only.sh \
  --output-dir outputs/hyperband_only \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Hyperband hemat waktu (≤ ~2 hari, single-bracket SHA):**

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/exp_fast \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --hyperband-s-max 2 \
  --hyperband-s-min 2
```

Setara memanggil Python langsung (semua flag tersedia):

```bash
python scripts/run_experiment.py \
  --output-dir outputs/experiment \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-min 0 \
  --hyperband-seed 99 \
  --hyperband-search-train-seed 0 \
  --hyperband-search-profile standard
```

### Laptop 8 GB — knob VRAM (tambahkan ke perintah di atas)

```bash
  --max-batch-size 16 \
  --dataloader-num-workers 0 \
  --skip-inference-videos
```

Contoh pipeline penuh di laptop 8 GB (lambat; disarankan smoke test dulu):

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/laptop_full \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 16 \
  --dataloader-num-workers 0 \
  --hyperband-s-max 2 \
  --hyperband-s-min 2 \
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

Jalankan ulang perintah yang sama di `--output-dir` yang sama — job selesai (`metrics.json` atau `status=ok` di CSV) dilewati; Hyperband melanjutkan dari `hyperband_state.json`.

Mulai dari nol (folder baru):

```bash
mkdir -p outputs/experiment_fresh
./scripts/run_experiment.sh --output-dir outputs/experiment_fresh \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Hapus manual isi folder lama jika ingin train ulang semua di path yang sama: `runs/`, `results.csv`, `configs.json`, `hyperband_state.json`, `cv_splits.json`.

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
cd /workspace/<repo>
./scripts/run_experiment.sh \
  --output-dir outputs/vast_exp \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr \
  --max-batch-size 64 \
  --dataloader-num-workers 2 \
  --hyperband-s-max 2 \
  --hyperband-s-min 2
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

## Pipeline eksperimen (baseline + Hyperband, tanpa k-fold)

Pelatihan **tidak** memakai validasi silang berlipat (k-fold). Episode dibagi **sekali** menjadi train / validation / test (`scripts/cv_splits.py`): satu partisi tetap, dapat direproduksi dengan `--cv-seed`.

Skrip **`scripts/run_experiment.py`** menjalankan tiga fase **berurutan**:

| Fase | Isi | Jumlah run (default) |
|------|-----|------------------------|
| **1. Baseline** | Hyperparameter default FlowPolicy (`experiment_constants.DEFAULT_BASELINE_HPARAMS`) × **3 seed** × **2 profil preprocessing** | **6** |
| **2. Hyperband** | Sampling random konfigurasi dari `SEARCH_SPACE` (tanpa `training.num_epochs` — itu resource), evaluasi dengan **`val_loss`** sebagai sinyal early-stopping antar-rung, di **1 seed × 1 profile** (default seed=0, profile=`standard`). State: `hyperband_state.json`. | tergantung `R`, `eta`, `s_min`, `s_max` — lihat tabel di bawah |
| **3. Rerun pemenang** | Konfigurasi pemenang Hyperband (val_loss terkecil di antara semua evaluasi) di-rerun **penuh** (train + infer + simpan ke `results.csv` `status=ok`) pada **3 seed × 2 profil** dengan `training.num_epochs = R`. Baris CSV: `cfg_idx = -3`. | **6** |

**Total default:** 6 baseline + (Hyperband, ~14–24 baseline-equivalent tergantung `s_min`) + 6 rerun = **12 run tercatat di `results.csv` + sekitar 50–130 evaluasi Hyperband intermediate** (tidak ditulis ke `results.csv`; ada di `hyperband_state.json`).

Profil preprocessing (proxy "dengan / tanpa" augmentasi observasi): **`standard`** (noise observasi) dan **`minimal`** (tanpa noise tersebut). Tiap run punya folder sendiri di `runs/`.

### Hyperband (Li et al., 2018) singkat

Hyperband (https://arxiv.org/pdf/1603.06560) menggabungkan **random search + Successive Halving** dengan alokasi resource adaptif. Untuk Kitchen, **resource = jumlah epoch training**. Kunci:

- `R` (`--hyperband-max-epochs`): jumlah epoch maksimum per konfigurasi (default **3000** = baseline default).
- `eta` (`--hyperband-eta`): rasio downsampling antar-rung (default **3** sesuai paper).
- `s_max_native = floor(log_eta(R))` → untuk `R=3000, eta=3`: `s_max_native = 7`.
- `B = (s_max_native + 1) * R` → anggaran teoritis penuh Hyperband.
- Untuk tiap bracket `s in {s_max, ..., s_min}`:
  - `n_s = ceil( int(B/R/(s+1)) * eta^s )` konfigurasi disampling acak.
  - `r_s = R * eta^(-s)` epoch awal per konfigurasi.
  - Successive Halving rung `i in {0..s}`: latih `n_s * eta^(-i)` konfigurasi ke `r_s * eta^i` epoch, lalu pertahankan **top floor(n_i / eta)** by val_loss.
- Pemenang: konfigurasi dengan **val_loss terkecil di antara semua evaluasi** (sesuai paper Algorithm 1: "smallest intermediate loss seen so far").

Sinyal `val_loss` diambil dari `training_final.json.val_loss_final` (sudah ditulis FlowPolicy `train.py` pada `training.compute_val_loss=true`). Inference rollout **tidak** dijalankan selama Hyperband — hanya pada fase **Rerun pemenang** (top-1) di 3 seeds × 2 profiles, agar `results.csv`/`summary.csv`/`plots/` memiliki metrik test lengkap yang dapat dibandingkan dengan baseline.

**Deviasi praktis dari paper (didokumentasikan paper Section 5):** training **inkremental** antar-rung — konfigurasi yang lolos cull melanjutkan training dari `latest.ckpt` (`training.resume=true`) dengan `training.num_epochs = r_{i+1} - r_i`, BUKAN melatih ulang dari nol. Konfigurasi yang ter-cull, folder `runs/hb_cfg{idx}/` dihapus untuk hemat disk.

### Anggaran waktu Hyperband (R=3000, eta=3)

Cost ≈ jumlah epoch yang dilatih total. Asumsikan 1 "baseline-equivalent" = 3000 epoch training pada 1 seed × 1 profile.

| `s_max` | `s_min` | Bracket dijalankan | Epoch-equivalent | Baseline-equivalent |
|---|---|---|---|---|
| 7 (native) | 0 | s = 7, 6, 5, 4, 3, 2, 1, 0 | ~131000 | ~44 |
| 4 | 0 | s = 4, 3, 2, 1, 0 | ~87000 | ~29 |
| 2 (default `s_max`) | 0 (default `s_min`) | s = 2, 1, 0 | ~58000 | ~19 |
| 2 | 2 | **s = 2 saja** (single-bracket SHA) | ~14000 | **~4.7** |
| 1 | 1 | s = 1 saja | ~20000 | ~6.7 |
| 0 | 0 | s = 0 saja (random search 8 config @ R) | ~24000 | ~8 |

**Penting (anggaran 2 hari):** jika 1 baseline run ≈ 8 jam, maka 2 hari = 6 baseline-equivalent. Konfigurasi paling aman untuk fit **≤ 2 hari**: **`--hyperband-s-max 2 --hyperband-s-min 2`** (single-bracket SHA s=2, ~4.7 baseline-equivalent ≈ **38 jam**). Default `s_min=0` di shell pintasan menjalankan multi-bracket Hyperband penuh sesuai paper, tetapi membutuhkan compute lebih besar.

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

### Hanya Hyperband (tanpa baseline)

Untuk **melewati baseline** dan hanya menjalankan Hyperband + rerun pemenang, gunakan flag **`--hyperband-only`** di `run_experiment.py`, atau skrip pintasan **`scripts/run_hyperband_only.sh`**.

**Opsi A — skrip pintasan:**

```bash
./scripts/run_hyperband_only.sh \
  --output-dir outputs/hb \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Opsi B — Python langsung (single-bracket SHA, fit ≤ 2 hari):**

```bash
python scripts/run_experiment.py \
  --hyperband-only \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-max 2 \
  --hyperband-s-min 2 \
  --hyperband-seed 99 \
  --hyperband-search-train-seed 0 \
  --hyperband-search-profile standard \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --output-dir outputs/hb_fast \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

**Opsi C — multi-bracket Hyperband penuh (sesuai paper, lebih mahal):**

```bash
python scripts/run_experiment.py \
  --hyperband-only \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-max 2 \
  --hyperband-s-min 0 \
  --output-dir outputs/hb_full \
  --zarr-path FlowPolicy/data/kitchen_complete_from_minari.zarr
```

Flag **`--baseline-only`** dan **`--hyperband-only`** saling eksklusif (maksimal satu aktif).

**Catatan:** opsi **`--results-csv`** juga mengubah lokasi **`results.csv`** untuk baseline dan rerun pemenang Hyperband. `summarize.py` / `plot_results.py` mendukung **`--results-csv`** yang sama jika Anda menjalankan agregasi manual. Hyperband fase intermediate **tidak** ditulis ke `results.csv` — state-nya ada di `hyperband_state.json`.

Penjelasan **`--results-csv`**: jika **tidak** diberikan, file CSV default adalah `<output-dir>/results.csv`. Jika **diberikan** (relatif ke akar repo atau path absolut), semua baris metrik baseline + rerun pemenang Hyperband ditulis ke file itu, dan kombinasi `(cfg_idx, seed, profile, fold)` yang sudah **`status=ok`** di file tersebut **tidak** di-train / di-infer ulang — cocok untuk melanjutkan tanpa menduplikasi run yang sama.

### Training ulang: hanya baseline, folder baru (laptop, tanpa melanjutkan run lama)

Orchestrator **melewati** job yang sudah selesai jika di `--output-dir` yang sama, per kombinasi run, sudah ada **`metrics.json`** di folder run tersebut. Untuk **baseline** dan **rerun pemenang Hyperband**, lewati juga berlaku jika **`results.csv`** (lihat **`--results-csv`** di bawah) memuat baris dengan kombinasi yang sama dan **`status=ok`**. Untuk **fase Hyperband intermediate**, resume mengandalkan **`hyperband_state.json`** (lihat `<output-dir>/hyperband_state.json`). Agar dianggap **mulai dari nol**, pakai **folder keluaran yang baru** (path yang belum dipakai), misalnya:

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
- Jika Anda **sengaja** memakai ulang folder lama tetapi ingin train ulang semua, hapus dulu isinya (**`runs/`**, **`results.csv`** atau file yang Anda set di **`--results-csv`**, **`configs.json`**, **`hyperband_state.json`**, **`cv_splits.json`**) — hati-hati: data metrik lama hilang. Menghapus **`results.csv`** saja **tidak** memaksa ulang semua job: selama **`metrics.json`** masih ada di suatu folder run, job itu tetap dilewati; hapus juga folder run yang bersangkutan di **`runs/`** jika ingin train ulang dari awal.

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

Jika masih OOM setelah **`16`**, tidak ada pengaturan aman lain di orchestrator selain **menurunkan hiperparameter batch di config** (mis. konfigurasi Hyperband yang memakai `batch_size` besar di `hyperband_state.json`) atau **menjalankan training tunggal** dengan override Hydra lebih agresif — pertimbangkan juga **instans GPU cloud** (lihat [Vast.ai](#menjalankan-di-vastai)) untuk pipeline Hyperband penuh agar waktu dan stabilitas lebih masuk akal.

**Tips laptop:** tutup aplikasi berat (browser dengan banyak tab, IDE lain), hindari sleep/hibernasi saat training panjang, dan pastikan daya AC terhubung (thermal GPU turun bisa memicu error atau throttling).

### Opsi CLI yang sering dipakai

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--seeds` | `0 42 101` | Tiga seed untuk training / dataset / inferensi (baseline + rerun pemenang Hyperband). |
| `--profiles` | `standard minimal` | Profil preprocessing dataset (baseline + rerun pemenang Hyperband). |
| `--cv-seed` | `12345` | Seed **satu** pembagian episode train/val/test (bukan k-fold). |
| `--n-infer-episodes` | `50` | Episode evaluasi setelah training. |
| `--output-dir` | `outputs/experiment` | Relatif terhadap akar repo. |
| `--results-csv` | (off) | Jalur `results.csv` (relatif repo atau absolut). Default bila tidak diisi: `<output-dir>/results.csv`. Baseline & rerun pemenang Hyperband: job dengan `status=ok` di file ini dilewati. |
| `--max-batch-size` | `128` | Plafon batch train/val; turunkan untuk **GPU 16 GB** atau **laptop 8 GB** (lihat bagian di atas). |
| `--dataloader-num-workers` | `4` | Workers DataLoader. |
| `--checkpoint-every` | `200` | Simpan checkpoint berkala (resume jika mesin mati). |
| `--baseline-only` | (off) | Hanya baseline; fase Hyperband tidak dijalankan. |
| `--hyperband-only` | (off) | Hanya Hyperband + rerun pemenang top-1; baseline dilewati. |
| `--hyperband-max-epochs` | `3000` | Resource maksimum Hyperband per konfigurasi (`R`). |
| `--hyperband-eta` | `3` | Rasio downsampling antar-rung Hyperband (`eta`, sesuai paper). |
| `--hyperband-s-min` | `0` | Bracket terkecil yang dijalankan; naikkan ke `2` (single-bracket SHA) untuk fit ≤ 2 hari. |
| `--hyperband-s-max` | `None` | Bracket terbesar; default = `floor(log_eta(R))`. Bisa di-cap (mis. `2`) untuk batasi compute. |
| `--hyperband-seed` | `99` | Seed RNG sampling konfigurasi Hyperband. |
| `--hyperband-search-train-seed` | `0` | Seed training selama fase Hyperband (1 seed saja). |
| `--hyperband-search-profile` | `standard` | Profil preprocessing selama fase Hyperband (1 profil saja). |

### Keluaran

Di `--output-dir` (mis. `outputs/experiment/`):

- `configs.json` — **`version: 5`**: berisi baseline (`baseline_config_dict()`) yang dipakai fase-1 dan rerun pemenang Hyperband.
- `hyperband_state.json` — state lengkap Hyperband: parameter (`R`, `eta`, `s_min`, `s_max`), daftar bracket, konfigurasi per bracket, evaluasi val_loss per rung, dan pemenang final. Dipakai untuk resume mesin-mati Hyperband.
- `cv_splits.json` — **satu** partisi train/val (+ meta `split_mode`, bukan daftar lipatan k-fold penuh).
- `results.csv` — default di folder `--output-dir`, kecuali Anda set **`--results-csv`**. Satu baris per run baseline (`cfg_idx=-1`) atau rerun pemenang Hyperband (`cfg_idx=-3`). Hyperband intermediate (cfg_idx `>= 1000`) **tidak** ditulis ke CSV.
- `runs/<nama_run>/` — output Hydra, `checkpoints/`, `metrics.json`, `training_final.json`.
- `summary.csv`, `plots/*.png` dan `*.pdf` — dibuat otomatis di akhir (`summarize.py`, `plot_results.py`).

Nama folder run:

- Baseline: `baseline_seed<seed>_<profile>` (`cfg_idx=-1`).
- Rerun pemenang Hyperband: `hb_best_seed<seed>_<profile>` (`cfg_idx=-3`).
- Hyperband intermediate: `hb_cfg<idx>/` (cfg_idx `>= 1000`, hanya bertahan untuk konfigurasi yang lolos cull; folder yang ter-cull dihapus otomatis untuk hemat disk).

### Resume setelah mesin mati

- **Baseline** dan **rerun pemenang Hyperband**: run dilewati jika ada **`metrics.json`** di folder run, atau jika **`results.csv`** sudah memuat baris dengan kombinasi `(cfg_idx, seed, profile, fold)` yang sama dan **`status=ok`**.
- **Hyperband fase intermediate**: state ada di **`hyperband_state.json`** — rung yang sudah punya `val_loss` (sukses) dilewati. Folder `runs/hb_cfg<idx>/` yang bertahan dipakai ulang untuk training inkremental rung berikutnya.
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
