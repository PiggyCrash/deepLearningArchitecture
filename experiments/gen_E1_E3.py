"""Generate notebook E1 (baseline), E2 (data fix), E3 (augmentasi + split)."""
from make_nb import make_notebook

# ----------------------------------------------------------------- E1
make_notebook(
    "E1_baseline_original.ipynb",
    "# E1 — Baseline: replika notebook asli, jalan di GPU\n"
    "Semua config **persis** notebook `img_clf.ipynb` asli, termasuk bug-nya. "
    "Bedanya hanya: device CPU → **CUDA**, epoch 1 → 3 (supaya sinyal training terlihat), "
    "dan test dibuat batched (256) supaya tidak 400 detik.",
    [
        "## Apa yang TIDAK diubah (sengaja)\n"
        "- Model: `ConvNet` asli (conv 16/32/64, fc 128/64/10, sigmoid)\n"
        "- Loss: `binary_cross_entropy` pada output sigmoid\n"
        "- Optimizer: SGD lr=0.0025, momentum=0.9, batch_size=10\n"
        "- Data: train batch 1,3,5 / val batch 2,4, tanpa augmentasi\n"
        "- **Bug slicing channel dipertahankan** (`fix_channels=False`): di notebook asli, "
        "`img_green` dan `img_blue` sama-sama mengambil `data[0:1024]` (channel merah), "
        "jadi ketiga channel identik — model efektif cuma melihat 1 channel.",
        "## Run",
    ],
    [
        "",
        "from exp_common import *\n"
        "model = ConvNet(out_act='sigmoid')\n"
        "res = run_experiment(\n"
        "    'E1_baseline', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_5'],\n"
        "    val_files=['data_batch_2', 'data_batch_4'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='bce', augment=False, fix_channels=False)\n",
    ],
)

# ----------------------------------------------------------------- E2
make_notebook(
    "E2_data_fix.ipynb",
    "# E2 — Perubahan #1 (data processing): perbaiki slicing channel RGB\n"
    "Satu-satunya perubahan vs E1: `fix_channels=True`, yaitu green=`data[1024:2048]` "
    "dan blue=`data[2048:3072]` seperti seharusnya. Model, loss, optimizer, batch, "
    "epoch, split — semua sama dengan E1.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNet(out_act='sigmoid')\n"
        "res = run_experiment(\n"
        "    'E2_data_fix', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_5'],\n"
        "    val_files=['data_batch_2', 'data_batch_4'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='bce', augment=False, fix_channels=True)\n",
    ],
)

# ----------------------------------------------------------------- E3
make_notebook(
    "E3_augmentasi_split.ipynb",
    "# E3 — Perubahan #2 (data processing + config split): augmentasi + train lebih banyak\n"
    "Perubahan vs E2:\n"
        "- `augment=True`: random horizontal flip + random crop (pad 4) pada data train\n"
        "- Train memakai 4 batch (1,3,4,5 = 40.000 sample), val = batch 2 (10.000). "
        "Di E1/E2 train cuma 30.000 dan val 20.000.\n"
        "- Sisanya (model, loss, optimizer, batch, epoch) sama.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNet(out_act='sigmoid')\n"
        "res = run_experiment(\n"
        "    'E3_augmentasi', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='bce', augment=True, fix_channels=True)\n",
    ],
)
