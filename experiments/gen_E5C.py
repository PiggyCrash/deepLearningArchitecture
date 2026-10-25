"""Generate notebook E5-C: hyperparameter chain step C (epochs 3 -> 8)."""
from make_nb import make_notebook

make_notebook(
    "E5_hyperC_epochs.ipynb",
    "# E5-C — Hyperparameter, modif C: epoch 3 → 8\n"
    "**Interpretasi dari E5-B:** Adam menyamai akurasi E4-D (**66.5%** vs 66.6%) dengan "
    "waktu 3.6x lebih cepat (59s vs 211s). Train acc (67.9%) ≈ val acc (67.0%) — model "
    "seimbang dan loss masih turun → masih ada ruang; bottleneck berikutnya bukan "
    "arsitektur/optimizer, melainkan **budget training** (baru 3 epoch).\n\n"
    "**Modif C:** konfigurasi identik dengan E5-B (ConvNetD, Adam lr=0.001, batch 64), "
    "hanya epoch 3 → **8**. Yang diamati: apakah akurasi masih naik, dan di epoch ke "
    "berapa mulai overfit (train acc ≫ val acc).",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetD(dropout=0.3)\n"
        "res = run_experiment(\n"
        "    'E5_hyperC_epochs', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=8, batch_size=64, lr=0.001, optimizer_type='adam',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
