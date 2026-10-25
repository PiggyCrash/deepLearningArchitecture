"""Generate notebook E8 (scratch long-run) dan E9 (transfer learning proper).
Target: train/val/test acc semuanya > 85%."""
from make_nb import make_notebook

# ----------------------------------------------------------------- E8
make_notebook(
    "E8_final_scratch.ipynb",
    "# E8 — FINAL scratch: ResNet-18 (CIFAR) + resep training penuh\n"
    "**Target: train/val/test acc > 85%, train from scratch.**\n\n"
    "Kenapa eksperimen sebelumnya mentok di ~80%:\n"
    "- E5-C (ConvNetD, 8 epoch): kurva masih naik di epoch terakhir → kehabisan epoch\n"
    "- ConvNetD cuma 297rb param → kapasitas terbatas\n"
    "- lr tetap 0.001 → akhir training masih 'loncat-loncat', tidak halus konvergen\n\n"
    "**Resep E8 (gabungan semua pelajaran E1–E7):**\n"
    "- **ResNet-18 versi CIFAR** (conv 3x3 stem tanpa maxpool, ~11 jt param) — "
    "residual connection (pelajaran E4-D) + kapasitas besar (pelajaran E4-C, tapi "
    "kali ini dioptimasi dengan benar)\n"
    "- SGD momentum=0.9, lr=0.05 + **cosine annealing** → lr besar di awal "
    "(belajar cepat), mengecil halus ke 0 (konvergen rapi)\n"
    "- weight_decay=5e-4 (regularisasi, antisipasi overfit pada model besar)\n"
    "- batch 128, **40 epoch** (~15-20 menit di RTX 4050)\n"
    "- data: fix channel + augmentasi flip/crop, 40k train / 10k val (setup E3)",
    ["## Run (±15-20 menit)"],
    [
        "from exp_common import *\n"
        "model = ResNet18CIFAR()\n"
        "res = run_experiment(\n"
        "    'E8_final_scratch', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=40, batch_size=128, lr=0.05, optimizer_type='sgd',\n"
        "    momentum=0.9, weight_decay=5e-4, scheduler='cosine',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)

# ----------------------------------------------------------------- E9
make_notebook(
    "E9_final_transfer.ipynb",
    "# E9 — FINAL transfer learning: EfficientNet-B0 full fine-tune, config benar\n"
    "**Target: train/val/test acc > 85%.**\n\n"
    "Kenapa E6 (transfer learning gaya notebook asli) cuma 54%:\n"
    "- sigmoid + BCE → gradient lemah (pelajaran E4-A)\n"
    "- SGD lr=0.0025 → adaptasi lambat; 3 epoch tidak cukup untuk 4 jt parameter\n\n"
    "**Resep E9:**\n"
    "- EfficientNet-B0 pretrained ImageNet, **full fine-tune** (semua bobot dilatih)\n"
    "- output **logits + CrossEntropy** (bukan sigmoid+BCE)\n"
    "- **AdamW lr=3e-4** (lr lebih kecil dari scratch — bobot pretrained sudah bagus, "
    "tinggal diadaptasi pelan-pelan) + cosine annealing\n"
    "- batch 64, **12 epoch** (~12 menit)\n"
    "- data: fix channel + augmentasi, 40k train / 10k val",
    ["## Run (±10-15 menit)"],
    [
        "from exp_common import *\n"
        "model = EffNetFT()\n"
        "res = run_experiment(\n"
        "    'E9_final_transfer', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=12, batch_size=64, lr=3e-4, optimizer_type='adam',\n"
        "    weight_decay=1e-4, scheduler='cosine',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
