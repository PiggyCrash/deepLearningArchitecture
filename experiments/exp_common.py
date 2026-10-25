"""
exp_common.py — shared code untuk semua eksperimen img_clf.

Berisi: dataset, model (scratch + transfer learning), fungsi train/val/test,
dan runner eksperimen. Setiap notebook eksperimen hanya mengatur config lalu
memanggil run_experiment().
"""
import os
import time
import json
import pickle as pk
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# PATH & KONSTANTA
# ------------------------------------------------------------------
ROOT_DATA = "C:/Users/dgash/Downloads/ml_task_cnn/cifar-10-batches-py/"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Agar semua eksperimen comparable: seed sama => init & shuffle sama."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        d = pk.load(fo, encoding='bytes')
    return d


def one_hot(lbl, n=10):
    arr = np.zeros(n, dtype=np.float32)
    arr[lbl] = 1
    return arr


class CIFARData(Dataset):
    """
    Dataset CIFAR-10 dari file batch pickle.

    fix_channels=True  -> slicing channel yang BENAR (R:0-1023, G:1024-2047, B:2048-3071)
    fix_channels=False -> reproduksi bug notebook asli (G dan B salah ambil slice 0-1023,
                          alias ketiga channel = channel merah semua)

    augment=True -> random horizontal flip + random crop (pad 4) pada data training.
    """
    def __init__(self, batch_files, root_data=ROOT_DATA, fix_channels=True, augment=False):
        labels, data = [], []
        for bf in batch_files:
            d = unpickle(root_data + bf)
            k = list(d.keys())
            labels.append(d[k[1]])
            data.append(d[k[2]])
        self.labels = np.concatenate(labels, axis=0)
        self.data = np.concatenate(data, axis=0)
        self.fix_channels = fix_channels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _to_rgb(self, flat):
        if self.fix_channels:
            r = np.array(flat[0:1024]).reshape((32, 32, 1))
            g = np.array(flat[1024:2048]).reshape((32, 32, 1))
            b = np.array(flat[2048:3072]).reshape((32, 32, 1))
        else:  # bug asli: semua channel = channel merah
            r = np.array(flat[0:1024]).reshape((32, 32, 1))
            g = np.array(flat[0:1024]).reshape((32, 32, 1))
            b = np.array(flat[0:1024]).reshape((32, 32, 1))
        return np.concatenate((r, g, b), axis=2)  # HWC uint8

    def _augment(self, img):
        # random horizontal flip
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]
        # random crop dengan padding 4
        padded = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        i, j = np.random.randint(0, 9, size=2)
        return padded[i:i + 32, j:j + 32, :]

    def __getitem__(self, index):
        img = self._to_rgb(self.data[index])
        if self.augment:
            img = self._augment(img)
        return {
            'image': img.transpose(2, 0, 1),                 # CHW
            'label': one_hot(self.labels[index]),            # one-hot (untuk BCE)
            'cls': int(self.labels[index]),                  # index (untuk CrossEntropy)
        }


# ------------------------------------------------------------------
# MODEL — TRAIN FROM SCRATCH
# ------------------------------------------------------------------
class ConvNet(nn.Module):
    """Arsitektur ASLI notebook: 3x(conv+relu+maxpool) -> GAP -> 3xfc.
    out_act='sigmoid' mereproduksi notebook asli; 'none' mengeluarkan logits (untuk CE)."""
    def __init__(self, out_act='sigmoid'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.out_act = out_act
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = x / 255
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(self.globalpool(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.out_act == 'sigmoid':
            x = self.sigm(x)
        return x


class ConvNetA(ConvNet):
    """ARCH-A: ConvNet asli, tapi output logits + dipakai dengan CrossEntropy."""
    def __init__(self):
        super().__init__(out_act='none')


class ConvNetB(nn.Module):
    """ARCH-B: ARCH-A + padding=1 (spatial size terjaga) + BatchNorm setelah tiap conv."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x / 255
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # 32x32 -> 16x16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # 16x16 -> 8x8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # 8x8   -> 4x4
        x = self.flatten(self.globalpool(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ConvNetC(nn.Module):
    """ARCH-C: ARCH-B + lebih dalam & lebih lebar (4 conv block, 32/64/128/128 ch)
    + Dropout untuk regularisasi."""
    def __init__(self, dropout=0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
            )
        self.b1 = block(3, 32)
        self.b2 = block(32, 64)
        self.b3 = block(64, 128)
        self.b4 = block(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x / 255
        x = self.pool(self.b2(self.b1(x)))   # 32 -> 16
        x = self.pool(self.b4(self.b3(x)))   # 16 -> 8
        x = self.flatten(self.globalpool(x))
        x = self.relu(self.fc1(self.drop(x)))
        return self.fc2(x)


class ResBlock(nn.Module):
    """Residual block sederhana: x + conv-bn-relu-conv-bn(x)."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class ConvNetD(nn.Module):
    """ARCH-D: ARCH-C + residual connection (mini-ResNet).
    Downsample via stride-2 conv, residual block per stage."""
    def __init__(self, dropout=0.3):
        super().__init__()
        def stage(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                ResBlock(cout),
            )
        self.s1 = stage(3, 32, 1)      # 32x32
        self.s2 = stage(32, 64, 2)     # 16x16
        self.s3 = stage(64, 128, 2)    # 8x8
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x / 255
        x = self.s3(self.s2(self.s1(x)))
        x = self.flatten(self.globalpool(x))
        return self.fc(self.drop(x))


class ResNet18CIFAR(nn.Module):
    """ResNet-18 versi CIFAR (conv 3x3 tanpa maxpool di awal).
    ~11 jt parameter. Dipakai untuk E8: scratch dengan target >85%."""
    def __init__(self, num_classes=10):
        super().__init__()

        def basic_block(cin, cout, stride):
            downsample = None
            if stride != 1 or cin != cout:
                downsample = nn.Sequential(
                    nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(cout),
                )
            return nn.ModuleDict({
                'conv1': nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                'bn1': nn.BatchNorm2d(cout),
                'conv2': nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                'bn2': nn.BatchNorm2d(cout),
                'down': downsample if downsample is not None else nn.Identity(),
            })

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 2 block per stage, sesuai ResNet-18 [2,2,2,2]
        self.stages = nn.ModuleList()
        channels = [(64, 64, 1), (64, 64, 1),      # stage1: 64
                    (64, 128, 2), (128, 128, 1),   # stage2: 128
                    (128, 256, 2), (256, 256, 1),  # stage3: 256
                    (256, 512, 2), (512, 512, 1)]  # stage4: 512
        for cin, cout, stride in channels:
            self.stages.append(basic_block(cin, cout, stride))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x / 255)
        for b in self.stages:
            identity = b['down'](x)
            out = self.relu(b['bn1'](b['conv1'](x)))
            out = b['bn2'](b['conv2'](out))
            x = self.relu(out + identity)
        return self.fc(self.flatten(self.pool(x)))


# ------------------------------------------------------------------
# MODEL — TRANSFER LEARNING
# ------------------------------------------------------------------
class EffNet_clf(nn.Module):
    """Transfer learning ASLI notebook: EfficientNet-B0 pretrained ImageNet,
    classifier diganti 10 kelas, output sigmoid (dipakai dengan BCE)."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        import torchvision.transforms as transforms
        self.rgb_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.RGB_encoder.classifier[1] = nn.Linear(1280, 10)
        self.flatten = nn.Flatten()
        self.sigm = nn.Sigmoid()

    def forward(self, rgb):
        rgb = self.rgb_normalizer(rgb / 255)
        f = self.RGB_encoder.features(rgb)
        return self.sigm(self.RGB_encoder.classifier(
            self.flatten(self.RGB_encoder.avgpool(f))))


class EffNetFrozen(nn.Module):
    """EfficientNet-B0 dengan backbone DIBEKUKAN (feature extractor),
    hanya head baru yang dilatih. Output logits + CrossEntropy."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        import torchvision.transforms as transforms
        self.rgb_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b0(weights='IMAGENET1K_V1')
        for p in self.RGB_encoder.features.parameters():
            p.requires_grad = False
        self.RGB_encoder.classifier = nn.Identity()
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, rgb):
        rgb = self.rgb_normalizer(rgb / 255)
        f = self.flatten(self.RGB_encoder.avgpool(self.RGB_encoder.features(rgb)))
        return self.head(f)


class EffNetFT(nn.Module):
    """EfficientNet-B0 pretrained, full fine-tune, output LOGITS + CrossEntropy.
    Config yang benar untuk transfer learning (dipakai di E9)."""
    def __init__(self, dropout=0.3):
        super().__init__()
        import torchvision.models as models
        import torchvision.transforms as transforms
        self.rgb_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.RGB_encoder.classifier[1] = nn.Linear(1280, 10)
        self.RGB_encoder.classifier[0] = nn.Dropout(dropout, inplace=True)
        self.flatten = nn.Flatten()

    def forward(self, rgb):
        rgb = self.rgb_normalizer(rgb / 255)
        f = self.RGB_encoder.features(rgb)
        return self.RGB_encoder.classifier(self.flatten(self.RGB_encoder.avgpool(f)))


# ------------------------------------------------------------------
# TRAIN / EVAL / TEST
# ------------------------------------------------------------------
def compute_loss(pred, data, loss_mode):
    if loss_mode == 'bce':
        return F.binary_cross_entropy(pred, data['label'].to(pred.device, torch.float))
    return F.cross_entropy(pred, data['cls'].to(pred.device))


def accuracy(pred, data):
    return (pred.argmax(1) == data['cls'].to(pred.device)).float().mean().item()


def run_epoch(data_loader, model, device, loss_mode, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for data in data_loader:
            img = data['image'].to(device, dtype=torch.float)
            data_dev = {k: v.to(device) for k, v in data.items()}
            if train_mode:
                optimizer.zero_grad()
            pred = model(img)
            loss = compute_loss(pred, data_dev, loss_mode)
            if train_mode:
                loss.backward()
                optimizer.step()
            bs = img.size(0)
            tot_loss += loss.item() * bs
            tot_acc += accuracy(pred, data_dev) * bs
            n += bs
    return tot_loss / n, tot_acc / n


def test_model(model, device, batch_size=256, fix_channels=True):
    """Test ke seluruh 10000 sample test_batch, batched (jauh lebih cepat dari
    notebook asli yang batch_size=1). Return acc + per-class acc."""
    test_set = CIFARData(['test_batch'], fix_channels=fix_channels)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct = np.zeros(10)
    total = np.zeros(10)
    with torch.no_grad():
        for data in loader:
            img = data['image'].to(device, dtype=torch.float)
            pred = model(img).argmax(1).cpu().numpy()
            gt = data['cls'].numpy()
            for c in range(10):
                m = gt == c
                total[c] += m.sum()
                correct[c] += (pred[m] == c).sum()
    per_class = correct / total
    return float(correct.sum() / total.sum()), per_class


def count_params(model, trainable_only=False):
    ps = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


# ------------------------------------------------------------------
# RUNNER
# ------------------------------------------------------------------
def run_experiment(name, model, train_files, val_files, epochs=3, batch_size=64,
                   lr=0.0025, optimizer_type='sgd', momentum=0.9, loss_mode='ce',
                   augment=False, fix_channels=True, device=DEVICE, seed=42,
                   weight_decay=0.0, scheduler='none'):
    """
    Jalankan 1 eksperimen penuh: train -> val per epoch -> test -> simpan hasil.
    Semua hasil disimpan ke experiments/results/<name>.json
    """
    set_seed(seed)
    print(f"===== {name} =====")
    print(f"device={device}, epochs={epochs}, batch={batch_size}, lr={lr}, "
          f"opt={optimizer_type}, loss={loss_mode}, augment={augment}, "
          f"fix_channels={fix_channels}")
    print(f"params total={count_params(model):,} trainable={count_params(model, True):,}")

    train_set = CIFARData(train_files, fix_channels=fix_channels, augment=augment)
    val_set = CIFARData(val_files, fix_channels=fix_channels, augment=False)
    print(f"train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(trainable, lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
             if scheduler == 'cosine' else None)

    model = model.to(device, dtype=torch.float)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        te = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, model, device, loss_mode, optimizer)
        vl_loss, vl_acc = run_epoch(val_loader, model, device, loss_mode)
        if sched is not None:
            sched.step()
        dt = time.time() - t0
        epoch_time = time.time() - te
        lr_now = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=vl_loss, val_acc=vl_acc, cum_time=dt,
                            epoch_time_s=epoch_time, lr=lr_now,
                            train_samples_per_s=len(train_set) / epoch_time))
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} "
              f"({epoch_time:.1f}s, {len(train_set)/epoch_time:.0f} samples/s)")

    train_time = time.time() - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if device.type == 'cuda' else 0.0)
    print(f"peak GPU mem: {peak_mem_mb:.0f} MB")

    t0 = time.time()
    test_acc, per_class = test_model(model, device, fix_channels=fix_channels)
    test_time = time.time() - t0
    print(f"TEST ACC = {test_acc:.4f}  (eval {test_time:.1f}s, train {train_time:.1f}s)")

    result = dict(
        name=name,
        config=dict(epochs=epochs, batch_size=batch_size, lr=lr,
                    optimizer=optimizer_type, momentum=momentum, loss=loss_mode,
                    weight_decay=weight_decay, scheduler=scheduler,
                    augment=augment, fix_channels=fix_channels,
                    train_files=train_files, val_files=val_files,
                    params=count_params(model), params_trainable=count_params(model, True)),
        history=history,
        train_time_s=train_time,
        test_time_s=test_time,
        peak_gpu_mem_mb=peak_mem_mb,
        test_acc=test_acc,
        per_class_acc={CLASSES[i]: float(per_class[i]) for i in range(10)},
    )
    out = os.path.join(RESULTS_DIR, name + ".json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved:", out)
    torch.cuda.empty_cache()
    return result
