"""Generate notebook E5-B: hyperparameter chain step B (optimizer SGD -> Adam)."""
from make_nb import make_notebook

make_notebook(
    "E5_hyperB_adam.ipynb",
    "# E5-B — Hyperparameter, modif B: SGD → Adam\n"
    "**Interpretasi dari E5-A:** lr 0.01 + batch 64 memang 4x lebih cepat (52s vs 211s), "
    "tapi test acc turun 66.6% → **61.8%**. Penyebab: total update turun dari 12.000 step "
    "(4000×3) jadi 1.875 step (625×3) — model undertrained; dan train acc (65.5%) kini "
    "> val acc (62.4%) → mulai sedikit overfit. SGD+momentum butuh banyak step untuk "
    "konvergen; dengan step sedikit, optimizer adaptif seharusnya lebih efisien.\n\n"
    "**Modif B:** ganti optimizer **SGD → Adam** (lr=0.001, default β). Adam menyesuaikan "
    "learning rate per-parameter dari momentum orde-1 & 2, jadi tiap step lebih efektif. "
    "Batch tetap 64.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetD(dropout=0.3)\n"
        "res = run_experiment(\n"
        "    'E5_hyperB_adam', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=64, lr=0.001, optimizer_type='adam',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
