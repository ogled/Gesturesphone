import os, time, torch, torch.nn as nn, torch.optim as optim, pandas as pd, numpy as np, argparse
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch_optimizer import RAdam, Lookahead
from tqdm import tqdm
from pathlib import Path

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.chdir(Path(__file__).resolve().parent)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def angle_between_vectors(a, b, eps=1e-6):
    dot = (a * b).sum(dim=-1)
    na, nb = torch.norm(a, dim=-1), torch.norm(b, dim=-1)
    cos = torch.clamp(dot / (na * nb + eps), -1.0, 1.0)
    return torch.acos(cos)


# ----------------------- Dataset -----------------------
class GestureDataset(Dataset):
    def __init__(
        self,
        annotations_path,
        data_dir,
        train=True,
        augment=False,
        max_len=None,
        frame_dropout=0.1,
        noise_std=0.02,
        scale_jitter=0.03
    ):
        self.annotations = pd.read_csv(
            annotations_path, sep=None, engine="python", on_bad_lines="skip"
        )
        self.annotations = self.annotations[
            self.annotations["train"] == train
        ].reset_index(drop=True)

        self.data_dir = data_dir
        self.augment = augment
        self.max_len = max_len

        self.frame_dropout = frame_dropout
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter

        self.labels = sorted(self.annotations["text"].unique())
        self.label2idx = {c: i for i, c in enumerate(self.labels)}

    def __len__(self):
        return len(self.annotations)

    def _load_tensor(self, csv_path):
        npz_path = os.path.splitext(csv_path)[0] + ".npz"
        data = np.load(npz_path)["data"].astype(np.float32)  # (T,126)
        return torch.from_numpy(data).view(-1, 42, 3)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        label = torch.tensor(
            self.label2idx[row["text"]], dtype=torch.long
        )

        csv_path = os.path.join(
            self.data_dir,
            "train" if row["train"] else "test",
            f"{row['attachment_id']}.csv",
        )

        # ------ Load raw joints ------
        X = self._load_tensor(csv_path)  # (T,42,3)
        T = X.shape[0]

        # ------ Data augmentation (ONLY coords) ------
        if self.augment:
            if torch.rand(1) < 0.3:
                drop_idx = torch.randperm(T)[: int(T * self.frame_dropout)]
                X[drop_idx] = 0

            if torch.rand(1) < 0.5:
                X = X + torch.randn_like(X) * self.noise_std

            if torch.rand(1) < 0.5:
                scale = 1.0 + torch.randn(1).item() * self.scale_jitter
                X = X * scale

            if torch.rand(1) < 0.3:
                shift = torch.randint(0, T, (1,)).item()
                X = torch.roll(X, shifts=shift, dims=0)

        # ------ Feature engineering ------
        J_left = X[:, :21]
        J_right = X[:, 21:]

        def norm_hand(J):
            wrist = J[:, 0:1]
            Jc = J - wrist
            palm = Jc[:, 9]
            scale = torch.norm(palm, dim=1, keepdim=True) + 1e-6
            return Jc / scale.view(-1, 1, 1)

        J_left = norm_hand(J_left)
        J_right = norm_hand(J_right)

        coords = torch.cat([J_left, J_right], dim=1).view(T, -1)

        velocity = torch.zeros_like(coords)
        velocity[1:] = coords[1:] - coords[:-1]

        motion_energy = torch.norm(velocity, dim=1, keepdim=True)
        motion_energy = (motion_energy - motion_energy.mean()) / (
            motion_energy.std() + 1e-6
        )

        left_present = (J_left.abs().sum(dim=(1, 2)) > 0).float()
        right_present = (J_right.abs().sum(dim=(1, 2)) > 0).float()
        hands_count = ((left_present + right_present) / 2.0).unsqueeze(1)

        feats = torch.cat(
            [hands_count, coords, velocity, motion_energy], dim=1
        )

        # ------ Padding / Truncation (FINAL STEP) ------
        if self.max_len is not None:
            if feats.size(0) > self.max_len:
                feats = feats[: self.max_len]
            else:
                pad = torch.zeros(
                    self.max_len - feats.size(0), feats.size(1)
                )
                feats = torch.cat([feats, pad], dim=0)

        return feats, label



def collate_fn(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])
    padded = pad_sequence(seqs, batch_first=True)
    return padded, torch.tensor([len(s) for s in seqs]), labels


# ----------------------- Модель -----------------------
class SimpleTCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, 5, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=8, dilation=4),
            nn.ReLU()
        )
        self.attn = nn.Sequential(
            nn.Conv1d(128, 1, 1),
            nn.Softmax(dim=2)
        )
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.conv(x)
        feat = feat + torch.randn_like(feat) * 0.01  # легкий шум
        w = self.attn(feat)
        out = (feat * w).sum(dim=2)
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)


# ----------------------- Обучение -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, desc="train", leave=False)
    for X, _, y in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, desc="val", leave=False)
    with torch.no_grad():
        for X, _, y in loop:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataSet", default="Datasets/CreatedDS/")
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("-bs", "--batchSize", type=int, default=128)
    parser.add_argument("-o", "--out", type=str, default="Models/model.pth")
    parser.add_argument("--max_len", type=int, default=14)
    parser.add_argument("--early_stop", type=int, default=40)
    args = parser.parse_args()

    annotations_path = os.path.join(args.dataSet, "annotations.csv")
    train_set = GestureDataset(
        annotations_path, args.dataSet, train=True, augment=True, max_len=args.max_len
    )
    val_set = GestureDataset(
        annotations_path, args.dataSet, train=False, augment=False, max_len=args.max_len
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batchSize, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_set, 64, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    sample_X, _, _ = next(iter(train_loader))
    model = SimpleTCN(sample_X.shape[2], len(train_set.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    base_optim = RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = Lookahead(base_optim, k=5, alpha=0.5)   
    
    best_val_loss, best_acc, best_epoch = float("inf"), 0, 0
    log_path = "train_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,time\n")
    epochsWithoutBestModel = 0
    for epoch in range(args.epochs):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{args.epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{tr_loss:.4f},{tr_acc:.4f},{val_loss:.4f},{val_acc:.4f},{elapsed:.2f}\n"
            )

        if val_loss < best_val_loss:
            best_val_loss, best_acc, best_epoch = val_loss, val_acc, epoch + 1
            epochsWithoutBestModel = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state,
                    "val_loss": val_loss,
                    "labels": train_set.labels,
                },
                args.out,
            )
            print("✅ Saved best model ->", args.out)
        else:
            epochsWithoutBestModel += 1
            if epochsWithoutBestModel >= args.early_stop:
                print(f"Early stop. Best epoch {best_epoch}. Best val_acc {best_acc}")
                break

    print("Training finished. Best val_acc:", best_acc)


if __name__ == "__main__":
    main()
