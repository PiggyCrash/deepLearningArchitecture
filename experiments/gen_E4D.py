"""Generate notebook E4-D: arch chain step D (residual connections)."""
from make_nb import make_notebook

make_notebook(
    "E4_arch_D_residual.ipynb",
    "# E4-D — Arsitektur, modif D: residual connection (mini-ResNet)\n"
    "**Interpretasi dari E4-C:** menambah kapasitas malah menurunkan test acc "
    "56.7% → **54.6%** dan waktu training naik 2.3x (67s → 152s). Masalahnya: plain "
    "deep network dengan lr kecil sulit dioptimasi — gradient harus melewati 8 layer "
    "berurutan; kapasitas ekstra tidak terpakai dalam 3 epoch.\n\n"
    "**Modif D:** ganti blok conv biasa dengan **residual block** (shortcut `x + F(x)` "
    "ala ResNet), downsample pakai stride-2 conv:\n"
    "- 3 stage (32/64/128 channel), tiap stage = conv+BN+ReLU + 1 ResBlock\n"
    "- Dropout 0.3 + fc tunggal 128→10 sebagai head\n\n"
    "Shortcut membuat gradient 'jalan tol' ke layer awal, jadi depth benar-benar terpakai. "
    "Hyperparameter tetap SGD lr=0.0025, batch 10.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetD(dropout=0.3)\n"
        "res = run_experiment(\n"
        "    'E4_arch_D', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
