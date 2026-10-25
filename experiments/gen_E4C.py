"""Generate notebook E4-C: arch chain step C (deeper/wider + dropout)."""
from make_nb import make_notebook

make_notebook(
    "E4_arch_C_deeper_dropout.ipynb",
    "# E4-C — Arsitektur, modif C: lebih dalam & lebih lebar + Dropout\n"
    "**Interpretasi dari E4-B:** padding+BatchNorm menaikkan test acc 43.7% → **56.7%**, "
    "dan kurva train/val masih turun-turunnya di epoch 3 (train acc 52.1% vs val 57.0% — "
    "val lebih tinggi karena train diaugmentasi). Artinya: masih *underfitting*, "
    "bottleneck sekarang di **kapasitas** — hanya 3 conv sempit (16/32/64 channel).\n\n"
    "**Modif C:**\n"
    "- 4 conv block (32/64/128/128 channel) — representasi lebih kaya\n"
    "- Head fc 128→256→10 (lebih besar dari 64→128→64→10)\n"
    "- `Dropout(0.3)` sebelum fc — antisipasi overfitting karena kapasitas naik ~8x\n\n"
    "Hyperparameter tetap SGD lr=0.0025, batch 10.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetC(dropout=0.3)\n"
        "res = run_experiment(\n"
        "    'E4_arch_C', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
