"""Generate notebook E5-A: hyperparameter chain step A (lr + batch size)."""
from make_nb import make_notebook

make_notebook(
    "E5_hyperA_lr_batch.ipynb",
    "# E5-A — Hyperparameter, modif A: lr 0.0025→0.01, batch 10→64\n"
    "**Interpretasi dari E4-D:** residual net mencapai test acc **66.6%**, tapi di akhir "
    "epoch 3 train acc (63.3%) masih < val acc (67.1%) dan loss masih turun tajam — "
    "model belum konvergen. Penyebab efisiensi buruk: lr 0.0025 sangat kecil + batch 10 "
    "membuat gradient noisy dan 4000 step/epoch lambat (70s/epoch).\n\n"
    "**Modif A (hyperparameter saja, arsitektur tetap ConvNetD):**\n"
    "- learning rate 0.0025 → **0.01** (4x, langkah lebih berarti per update)\n"
    "- batch size 10 → **64** (gradient lebih stabil, GPU utilization naik, "
    "step/epoch turun 4000 → 625)\n",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetD(dropout=0.3)\n"
        "res = run_experiment(\n"
        "    'E5_hyperA_lr_batch', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=64, lr=0.01, optimizer_type='sgd',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
