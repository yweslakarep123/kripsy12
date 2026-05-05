# FlowPolicy

Implementasi **Flow Policy** untuk kontrol robotik dengan observasi **point cloud** (antara lain Franka Kitchen via Gymnasium-Robotics). Proses training memakai [Hydra](https://hydra.cc/) dan logging [Weights & Biases](https://wandb.ai/).

Struktur repositori:

```text
FlowPolicy/                 # root Git (repo ini)
└── FlowPolicy/             # kode Python, train.py, paket flow_policy_3d
    ├── train.py
    ├── setup.py
    ├── requirements-franka-kitchen.txt
    └── flow_policy_3d/
```

Semua perintah di bawah diasumsikan dijalankan dari direktori **`FlowPolicy/FlowPolicy`** (folder yang berisi `train.py`).

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
