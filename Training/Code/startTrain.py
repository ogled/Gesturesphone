import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


# -----------------------------
# Dataset с .npy кэшем + дополнительные аугментации
# -----------------------------
class GestureDataset(Dataset):
    def __init__(
        self,
        annotations_path,
        data_dir,
        train=True,
        augment=False,
        cache_dir=None,
        max_len=None,
        frame_dropout=0.1,
        noise_std=0.05,
        scale_jitter=0.05,
        time_shift=5,
    ):
        self.annotations = pd.read_csv(
            annotations_path, sep=None, engine="python", on_bad_lines="skip"
        )
        self.annotations = self.annotations[
            self.annotations["train"] == train
        ].reset_index(drop=True)
        self.data_dir = data_dir
        self.labels = sorted(self.annotations["text"].unique())
        self.label2idx = {c: i for i, c in enumerate(self.labels)}
        self.augment = augment
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.frame_dropout = frame_dropout
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter
        self.time_shift = time_shift  # shift frames left/right
        if cache_dir:
            ensure_dir(cache_dir)

    def __len__(self):
        return len(self.annotations)

    def _cache_path(self, attachment_id):
        return (
            os.path.join(self.cache_dir, f"{attachment_id}.npy")
            if self.cache_dir
            else None
        )

    def _load_tensor(self, attachment_id, csv_path):
        cache_path = self._cache_path(attachment_id)
        if cache_path and os.path.exists(cache_path):
            try:
                arr = np.load(cache_path)
                return torch.from_numpy(arr)
            except Exception:
                pass  # если npy битый — перезагрузим из csv
        df = pd.read_csv(csv_path)
        X = torch.tensor(df.drop(columns=["frame"]).values.astype("float32"))
        if cache_path:
            np.save(cache_path, X.numpy())
        return X

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        subdir = "train" if row["train"] else "test"
        csv_path = os.path.join(self.data_dir, subdir, f"{row['attachment_id']}.csv")
        attachment_id = row["attachment_id"]
        X = self._load_tensor(attachment_id, csv_path)
        # нормализация по каждому сэмплу
        X = (X - X.mean()) / (X.std() + 1e-6)
        seq_len = X.shape[0]

        # Обрезка или паддинг
        if self.max_len is not None:
            if seq_len > self.max_len:
                start = (
                    np.random.randint(0, seq_len - self.max_len + 1)
                    if self.augment
                    else 0
                )
                X = X[start : start + self.max_len]
            elif seq_len < self.max_len:
                pad_len = self.max_len - seq_len
                X = torch.cat([X, torch.zeros(pad_len, X.size(1))], dim=0)

        # Аугментации (временные и пространственные)
        if self.augment:
            # случайный тайм-шифт
            if self.time_shift and self.max_len:
                shift = np.random.randint(-self.time_shift, self.time_shift + 1)
                if shift > 0:
                    X = torch.cat([X[shift:], torch.zeros(shift, X.size(1))], dim=0)
                elif shift < 0:
                    X = torch.cat([torch.zeros(-shift, X.size(1)), X[:shift]], dim=0)

            # frame dropout (случайно удаляем кадры)
            if self.frame_dropout > 0.0 and X.shape[0] > 1:
                keep_mask = torch.rand(X.shape[0]) > self.frame_dropout
                if keep_mask.sum() == 0:
                    keep_mask[0] = True
                X = X[keep_mask]

            # масштабирование признаков
            if self.scale_jitter > 0.0:
                s = 1.0 + (torch.randn(1).item() * self.scale_jitter)
                X = X * s

            # шум
            if self.noise_std > 0.0:
                X = X + torch.randn_like(X) * self.noise_std

            # возможно: горизонтальный/вертикальный jitter (если признаки координаты)
            # можно добавить здесь, если признаки имеют смысл пространственных координат

        y = self.label2idx[row["text"]]
        return X, torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, labels


# -----------------------------
# Temporal + Transformer hybrid
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.3
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.resample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        res = x if self.resample is None else self.resample(x)
        return torch.relu(out + res)


class TransformerAttention(nn.Module):
    """Лёгкий Transformer attention блок для усиления TCN"""

    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm(x)))
        return x


class TCNModel(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        num_channels=[128, 256, 512],
        kernel_size=5,
        dropout=0.3,
        num_heads=4,
    ):
        super().__init__()

        # --- TCN backbone ---
        tcn_layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2**i
            tcn_layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*tcn_layers)

        # --- Transformer head ---
        self.attn = TransformerAttention(
            num_channels[-1], num_heads=num_heads, dropout=dropout
        )

        # --- Classification head ---
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_channels[-1]),
            nn.Dropout(0.4),
            nn.Linear(num_channels[-1], num_classes),
        )

    def forward(self, x, lengths=None):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)  # -> [B, F, T]
        x = self.tcn(x)  # -> [B, C, T]
        x = x.permute(0, 2, 1)  # -> [B, T, C]
        x = self.attn(x)  # -> [B, T, C]
        x = x.permute(0, 2, 1)  # -> [B, C, T]
        x = self.pool(x)
        out = self.fc(x)
        return out


# -----------------------------
# Utility: EMA (экспоненциальное скользящее среднее) для стабилизации валид.
# -----------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = self._clone_model(model)
        self.decay = decay
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _clone_model(model):
        # глубоко копируем структуру и веса
        ema = (
            type(model)(
                **{k: v for k, v in model.__dict__.get("_init_args", {}).items()}
            )
            if hasattr(model, "_init_args")
            else None
        )
        # Если модель не предоставляет _init_args, просто clone using state_dict
        import copy

        ema = copy.deepcopy(model)
        return ema

    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * decay + msd[k].detach() * (1.0 - decay))

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# -----------------------------
# Loss: Focal (опционально)
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# Mixup helper
# -----------------------------
def mixup_data(x, y, alpha=0.2, device="cpu"):
    if alpha <= 0:
        return x, y, 1.0, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# -----------------------------
# Вспомогательные функции
# -----------------------------
def make_sampler(dataset):
    counts = {}
    for lbl in dataset.annotations["text"]:
        counts[lbl] = counts.get(lbl, 0) + 1
    weights = [
        1.0 / counts[dataset.annotations.iloc[i]["text"]] for i in range(len(dataset))
    ]
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), num_samples=len(weights), replacement=True
    )


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    scheduler,
    epoch,
    epochs,
    mixup_alpha=0.0,
    use_onecycle=False,
    ema=None,
    grad_clip=None,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
    for batch_idx, (X, lengths, y) in enumerate(loop):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if mixup_alpha > 0.0:
            X, y_a, y_b, lam = mixup_data(X, y, alpha=mixup_alpha, device=device)
            with torch.amp.autocast(device.type):
                outputs = model(X)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                    outputs, y_b
                )
        else:
            with torch.amp.autocast(device.type):
                outputs = model(X)
                loss = criterion(outputs, y)

        scaler.scale(loss).backward()

        # градиентный клиппинг
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # scheduler step (для OneCycleLR требуется вызывать per-batch)
        if use_onecycle and scheduler is not None:
            scheduler.step()

        # EMA update
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        # если mixup, нельзя считать accuracy напрямую; использусть y (approx) — но для simplicity:
        if mixup_alpha > 0.0:
            # оценим по y_a с порогом lam (приближенно)
            acc_batch = (
                (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float())
                .mean()
                .item()
            )
            correct += acc_batch * y.size(0)
        else:
            correct += (preds == y).sum().item()
        total += y.size(0)
        loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device, epoch, epochs, ema=None):
    # если есть ema, используем её для предсказаний (копируем параметры временно)
    if ema is not None:
        eval_model = ema.ema
    else:
        eval_model = model
    eval_model.eval()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
    with torch.no_grad():
        for X, lengths, y in loop:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = eval_model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def save_checkpoint(state, path):
    torch.save(state, path)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataSet", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-bs", "--batchSize", type=int, default=256)
    parser.add_argument("-o", "--out", type=str, default="model_tcn.pth")
    parser.add_argument("--cache", type=str, default="cache_npy")
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 6) - 2))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched",type=str,choices=["onecycle", "plateau"],default="onecycle",help="onecycle = fast schedule, plateau = ReduceLROnPlateau",)
    parser.add_argument("--mixup", action="store_true", help="Use mixup augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--focal", action="store_true", help="Use focal loss instead of CE")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Clip grad norm (None to disable)")
    parser.add_argument("--early_stop",type=int,default=20,help="Early stopping patience (0 to disable)",)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_prefetch", type=int, default=2)
    args = parser.parse_args()

    annotations_path = os.path.join(args.dataSet, "NotWords.csv")
    train_set = GestureDataset(annotations_path, args.dataSet, train=True, augment=True, cache_dir=args.cache, max_len=args.max_len) #Обучение
    val_set = GestureDataset(annotations_path, args.dataSet, train=False, augment=False, cache_dir=args.cache, max_len=args.max_len) #Тесты

    sampler = make_sampler(train_set)
    train_loader = DataLoader(train_set,
        batch_size=args.batchSize,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.batch_prefetch,
    )
    val_loader = DataLoader(
        val_set,
        64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    torch.set_num_threads(4)

    sample_X, _, _ = next(iter(train_loader))
    input_size = sample_X.shape[2]
    num_classes = len(train_set.labels)

    model = TCNModel(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=[128, 256, 256],
        dropout=args.dropout,
    ).to(device)

    # attach init args for EMA cloning fallback
    try:
        model._init_args = {
            "input_size": input_size,
            "num_classes": num_classes,
            "num_channels": [128, 256, 256],
            "dropout": args.dropout,
        }
    except Exception:
        pass

    # Loss
    if args.focal:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = None
    use_onecycle = False
    if args.sched == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10 if args.lr < 1e-3 else args.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        use_onecycle = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
        )

    scaler = torch.amp.GradScaler(device.type)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_val_loss = float("inf")
    best_epoch = 0
    log_path = os.path.join(
        os.path.dirname(args.out) if os.path.dirname(args.out) else ".", "train_log.csv"
    )
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,time\n")

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            scheduler if use_onecycle else None,
            epoch,
            args.epochs,
            mixup_alpha=(args.mixup_alpha if args.mixup else 0.0),
            use_onecycle=use_onecycle,
            ema=ema,
            grad_clip=(args.grad_clip if args.grad_clip > 0 else None),
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device, epoch, args.epochs, ema=ema
        )
        elapsed = time.time() - start

        # scheduler step for ReduceLROnPlateau
        if not use_onecycle:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s"
        )

        # лог
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{elapsed:.2f}\n"
            )

        # Сохранение лучшей модели (и EMA state если есть)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "labels": train_set.labels,
            }
            if ema is not None:
                state["ema_state"] = ema.state_dict()
            save_checkpoint(state, args.out)
            print("✅ Saved best model ->", args.out)
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,time\n")
            with open(log_path, "a") as f:
                f.write(
                    f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{elapsed:.2f}\n"
                )
        # Early stopping
        if args.early_stop > 0 and (epoch + 1 - best_epoch) >= args.early_stop:
            print(
                f"Early stopping: no improvement in {args.early_stop} epochs (best epoch {best_epoch})."
            )
            break

    print("Training finished. Best val_loss:", best_val_loss)


if __name__ == "__main__":
    main()
