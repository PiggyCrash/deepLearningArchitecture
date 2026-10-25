"""Generate notebook E4-B: arch chain step B (padding + BatchNorm)."""
from make_nb import make_notebook

make_notebook(
    "E4_arch_B_padding_bn.ipynb",
    "# E4-B — Arsitektur, modif B: padding=1 + BatchNorm\n"
    "**Interpretasi dari E4-A:** mengganti sigmoid+BCE → logits+CE langsung menaikkan "
    "test acc dari 10% → **43.7%**. Tapi kurva masih menanjak lambat (train acc baru 37.8% "
    "di epoch 3, loss masih turun) → model *underfitting*, training belum konvergen. "
    "Dugaan: tanpa normalisasi antar-layer, distribusi aktivasi bergeser terus "
    "(internal covariate shift) sehingga lr kecil 0.0025 lambat merambat.\n\n"
    "**Modif B:** pada arsitektur yang sama, tambahkan:\n"
    "- `padding=1` di semua conv → spatial size terjaga (32→16→8, bukan menyusut 32→30→28→…), "
    "tepi gambar ikut terbaca\n"
    "- `BatchNorm2d` setelah tiap conv → aktivasi distabilkan, gradient mengalir lebih baik, "
    "training lebih cepat konvergen\n\n"
    "Hyperparameter tetap SGD lr=0.0025, batch 10 — isolasi murni efek arsitektur.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetB()\n"
        "res = run_experiment(\n"
        "    'E4_arch_B', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
