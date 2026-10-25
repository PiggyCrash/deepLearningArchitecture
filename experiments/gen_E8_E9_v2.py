"""Generate notebook E8 v2 (ConvNet ASLI, resep penuh) dan E9 v2 (EffNet-B0 asli, 20 epoch).
Sesuai instruksi user: TIDAK mengganti model — pakai model asli dari notebook baseline.
Target: train/val/test acc > 85%."""
from make_nb import make_notebook

# ----------------------------------------------------------------- E8 v2
make_notebook(
    "E8_final_scratch.ipynb",
    "# E8 — FINAL scratch: ConvNet ASLI (tanpa ubah arsitektur), resep training penuh\n"
    "**Target: train/val/test acc > 85%, train from scratch, model = ConvNet asli.**\n\n"
    "Model **persis** notebook baseline: conv 3→16→32→64 (kernel 3), GAP, fc 64→128→64→10. "
    "Tidak ada layer yang ditambah/diubah. Satu-satunya perubahan 'model': output dibuat "
    "logits (tanpa sigmoid) supaya bisa pakai CrossEntropy — pelajaran E4-A.\n\n"
    "Yang diubah murni **resep training** (pelajaran dari E1–E7):\n"
    "- SGD momentum=0.9, lr=0.05 + **cosine annealing** ke 0\n"
    "- weight_decay=5e-4 (regularisasi)\n"
    "- batch 128, **100 epoch** (model kecil, cepat: ±5 detik/epoch)\n"
    "- data: fix channel + augmentasi flip/crop, 40k train / 10k val\n\n"
    "**Catatan jujur:** model sekecil ini (41rb param, tanpa BatchNorm/residual) punya "
    "batas kapasitas — target 85% sangat ambisius; kita lihat seberapa dekat resep "
    "training yang benar bisa membawanya.",
    ["## Run (±10-15 menit)"],
    [
        "from exp_common import *\n"
        "model = ConvNetA()  # layer identik dengan ConvNet asli, minus sigmoid\n"
        "print('params:', count_params(model))\n"
        "res = run_experiment(\n"
        "    'E8_final_scratch', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=100, batch_size=128, lr=0.05, optimizer_type='sgd',\n"
        "    momentum=0.9, weight_decay=5e-4, scheduler='cosine',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)

# ----------------------------------------------------------------- E9 v2
make_notebook(
    "E9_final_transfer.ipynb",
    "# E9 — FINAL transfer learning: EfficientNet-B0 ASLI (model baseline), config benar\n"
    "**Target: train/val/test acc > 85%.**\n\n"
    "Model = `EffNet_clf` asli dari notebook baseline: **EfficientNet-B0 pretrained "
    "ImageNet**, classifier diganti 10 kelas, full fine-tune. Tidak ada model lain.\n\n"
    "E9 percobaan pertama (12 epoch, AdamW lr=3e-4) dapat train 87.1% / val 84.6% / "
    "test 84.3% — **nyaris 85%**, dan kurva masih naik. Maka v2: resep sama persis, "
    "hanya **epoch 12 → 20** (lr 3e-4 + cosine tetap).",
    ["## Run (±8-10 menit)"],
    [
        "from exp_common import *\n"
        "model = EffNetFT()  # EfficientNet-B0 asli, output logits untuk CE\n"
        "print('params:', count_params(model))\n"
        "res = run_experiment(\n"
        "    'E9_final_transfer', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=20, batch_size=64, lr=3e-4, optimizer_type='adam',\n"
        "    weight_decay=1e-4, scheduler='cosine',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
