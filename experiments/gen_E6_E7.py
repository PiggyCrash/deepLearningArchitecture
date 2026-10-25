"""Generate notebook E6 (transfer learning asli) dan E7 (frozen backbone)."""
from make_nb import make_notebook

make_notebook(
    "E6_transfer_effnet_asli.ipynb",
    "# E6 — Transfer learning: replika config EffNet notebook asli (di GPU)\n"
    "Persis desain `EffNet_clf` asli: EfficientNet-B0 pretrained ImageNet, classifier "
    "diganti 10 kelas, output **sigmoid + BCE**, full fine-tune (semua bobot dilatih), "
    "SGD lr=0.0025 momentum=0.9. Data: setup E3 (fix channel + augmentasi, 40k train).\n\n"
    "Dua penyesuaian yang terpaksa diubah (dan alasannya):\n"
    "- batch 10 → 64: EffNet-B0 ~100x lebih berat dari ConvNet; batch 10 di 40k sample "
    "= 4000 step/epoch akan sangat lama\n"
    "- input di-scale /255 sebelum ImageNet-normalizer (di notebook asli nilai 0–255 "
    "langsung dinormalisasi dengan mean/std ImageNet — secara matematis keliru)",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = EffNet_clf()\n"
        "res = run_experiment(\n"
        "    'E6_transfer_effnet', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=64, lr=0.0025, optimizer_type='sgd',\n"
        "    loss_mode='bce', augment=True, fix_channels=True)\n",
    ],
)

make_notebook(
    "E7_transfer_frozen.ipynb",
    "# E7 — Transfer learning modif: backbone DIBEKUKAN + head baru + Adam + CE\n"
    "Hipotesis: fitur ImageNet sudah sangat umum; untuk CIFAR-10 32x32 mungkin cukup "
    "dipakai apa adanya. Modifikasi vs E6:\n"
    "- Semua `features` EfficientNet-B0 di-`requires_grad=False` (feature extractor beku)\n"
    "- Head baru: Dropout(0.3) → Linear 1280→128 → ReLU → Linear 128→10\n"
    "- Output logits + CrossEntropy (bukan sigmoid+BCE)\n"
    "- Adam lr=0.001 (bukan SGD lr=0.0025)\n\n"
    "Konsekuensi: parameter trainable turun drastis (dari ~4 juta jadi ~165 ribu), "
    "backprop hanya lewat head → jauh lebih cepat per epoch. Pertanyaannya: berapa "
    "akurasi yang hilang (atau justru naik?) dibanding full fine-tune E6.",
    ["## Run"],
    [
        "from exp_common import *\n"
        "model = EffNetFrozen()\n"
        "res = run_experiment(\n"
        "    'E7_transfer_frozen', model,\n"
        "    train_files=['data_batch_1', 'data_batch_3', 'data_batch_4', 'data_batch_5'],\n"
        "    val_files=['data_batch_2'],\n"
        "    epochs=3, batch_size=64, lr=0.001, optimizer_type='adam',\n"
        "    loss_mode='ce', augment=True, fix_channels=True)\n",
    ],
)
