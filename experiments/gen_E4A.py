"""Generate notebook E4-A: arch chain step A (activation/loss change)."""
from make_nb import make_notebook

make_notebook(
    "E4_arch_A_logits_ce.ipynb",
    "# E4-A — Arsitektur, modif A: sigmoid+BCE → logits+CrossEntropy\n"
    "**Interpretasi dari E1–E3:** baseline selalu kolaps ke ~10% (memprediksi 1 kelas saja), "
    "di semua varian data. Loss turun sedikit lalu macet di ~0.325 (≈ -ln(0.9)·... pola khas "
    "output konstan). Penyebab: output `sigmoid` + `binary_cross_entropy` + lr kecil 0.0025 "
    "membuat gradient sangat lemah — model nyaman di minimum dangkal 'jawab rata-rata'.\n\n"
    "**Modif A:** arsitektur SAMA PERSIS dengan ConvNet asli, hanya hilangkan sigmoid "
    "(output = logits mentah) dan ganti loss ke `CrossEntropyLoss` (softmax internal). "
    "Data tetap setup E3 (fix channel + augmentasi, 40k train). Hyperparameter tetap "
    "SGD lr=0.0025, batch 10 — supaya murni mengisolasi efek ganti loss/aktivasi.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = ConvNetA()\n"
        "res = run_experiment(\n"
        "    'E4_arch_A', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=10, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
