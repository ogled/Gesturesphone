import os, time, torch, torch.nn as nn, torch.optim as optim, pandas as pd, numpy as np, argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_optimizer import RAdam
from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.chdir(Path(__file__).resolve().parent)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ----------------------- Dataset -----------------------
class GestureDataset(Dataset):
    def __init__(
        self,
        annotations_path,
        data_dir,
        train=True,
        augment=False,
        max_len=None,
        frame_dropout=0.06,
        noise_std=0.02,
        scale_jitter=0.03
    ):
        self.annotations = pd.read_csv(annotations_path, sep=None, engine="python")
        self.annotations = self.annotations[self.annotations["train"] == train].reset_index(drop=True)

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

    def _load_tensor(self, attachment_id, train):
        path = os.path.join(
            self.data_dir,
            "train" if train else "test",
            f"{attachment_id}.pt"
        )
        return torch.load(path, map_location="cpu")

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        label = torch.tensor(self.label2idx[row["text"]], dtype=torch.long)

        X = self._load_tensor(row["attachment_id"], row["train"])
        T = X.shape[0]

        if self.augment:
            if torch.rand(1) < 0.3:
                drop = torch.randperm(T)[: int(T * self.frame_dropout)]
                X[drop] = 0
            if torch.rand(1) < 0.5:
                X = X + torch.randn_like(X) * self.noise_std
            if torch.rand(1) < 0.5:
                X = X * (1.0 + torch.randn(1).item() * self.scale_jitter)
            

        J_left = X[:, 0]
        J_right = X[:, 1]

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
        motion_energy = (motion_energy - motion_energy.mean()) / (motion_energy.std() + 1e-6)

        left_present = (J_left.abs().sum(dim=(1,2)) > 0).float()
        right_present = (J_right.abs().sum(dim=(1,2)) > 0).float()
        hands_count = ((left_present + right_present) / 2.0).unsqueeze(1)

        feats = torch.cat([hands_count, coords, velocity, motion_energy], dim=1)

        if self.max_len is not None:
            if feats.size(0) > self.max_len:
                feats = feats[:self.max_len]
            else:
                feats = torch.cat([feats, torch.zeros(self.max_len - feats.size(0), feats.size(1))], dim=0)

        return feats, label

def collate_fn(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])
    padded = pad_sequence(seqs, batch_first=True)
    return padded, torch.tensor([len(s) for s in seqs]), labels

def class_preserving_mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y

    B = x.size(0)
    perm = torch.randperm(B, device=x.device)

    same_class = y == y[perm]
    if same_class.sum() == 0:
        return x, y

    lam = np.random.beta(alpha, alpha)
    x[same_class] = lam * x[same_class] + (1 - lam) * x[perm][same_class]
    return x, y

# ----------------------- SupCon -----------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()

# ----------------------- Модель -----------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=34.0, m=0.37):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        theta = torch.acos(torch.clamp(cosine, -1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = cosine * (1 - one_hot) + target * one_hot
        return logits * self.s
    
class SimpleTCN(nn.Module):
    def __init__(self, input_size, num_classes, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 256, 5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, 5, padding=4, dilation=2),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, emb_dim, 5, padding=8, dilation=4),
            nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Dropout(0.3)
        )

        self.arcface = ArcMarginProduct(emb_dim, num_classes)

    def forward(self, x, lengths=None, labels=None, return_embedding=False):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)

        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            x = x.mean(1)

        if return_embedding:
            return x

        return self.arcface(x, labels)

# ----------------------- Обучение -----------------------
def train_one_epoch(model, loader, criterion, supcon, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    scaler = torch.cuda.amp.GradScaler()

    for X, lengths, y in tqdm(loader, leave=False):
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)

        X, y = class_preserving_mixup(X, y, alpha=0.4)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            emb = model(X, lengths, return_embedding=True)
            logits = model(X, lengths, labels=y)

            loss = criterion(logits, y) + 0.2 * supcon(emb, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, desc="val", leave=False)

    with torch.no_grad():
        for X, lengths, y in loop:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            logits = model(X, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            loop.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")

    return total_loss / len(loader), correct / total

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataSet", default="Datasets/CreatedDS/")
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("-bs", "--batchSize", type=int, default=64)
    parser.add_argument("-o", "--out", type=str, default="Models/model.pth")
    parser.add_argument("--max_len", type=int, default=14)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--use_knn", action="store_true")
    args = parser.parse_args()

    ensure_dir("Models")
    ensure_dir("Logs")
    set_seed(42)

    annotations_path = os.path.join(args.dataSet, "annotations.csv")
    train_set = GestureDataset(annotations_path, args.dataSet, train=True, augment=True, max_len=args.max_len)
    val_set = GestureDataset(annotations_path, args.dataSet, train=False, augment=False, max_len=args.max_len)

    print(f"\nTotal classes: {len(train_set.labels)}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batchSize, shuffle=True, num_workers=6,
                              pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batchSize, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sample_X, _, _ = next(iter(train_loader))
    input_size = sample_X.shape[2]
    num_classes = len(train_set.labels)

    print(f"Input size: {input_size}")
    print(f"Num classes: {num_classes}")

    model = SimpleTCN(input_size, num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")

    supCon = SupConLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss, best_acc, best_epoch = float("inf"), 0, 0
    log_path = "Logs/train_log.csv"
    f = open(log_path, "w")
    f.write("epoch,train_loss,train_acc,val_loss,val_acc,time,lr\n")
    epochsWithoutBestModel = 0

    print("\n=== Starting Training ===")
    for epoch in range(args.epochs):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, supCon ,optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} | Time: {elapsed:.1f}s | LR: {current_lr:.2e}")

        f.write(f"{epoch+1},{tr_loss:.4f},{tr_acc:.4f},{val_loss:.4f},{val_acc:.4f},{elapsed:.2f},{current_lr:.6f}\n")

        epochsWithoutBestModel += 1
        if best_acc <= val_acc:
            best_val_loss, best_acc, best_epoch = val_loss, val_acc, epoch + 1
            epochsWithoutBestModel = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "labels": train_set.labels,
                "input_size": input_size,
                "num_classes": num_classes
            }, args.out)
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")

        if epochsWithoutBestModel >= args.early_stop:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            print(f"   Best epoch: {best_epoch} with val_acc: {best_acc:.4f}")
            break
    f.close()

    print("\n" + "="*80)
    print("Training finished!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best val_acc: {best_acc:.4f}")

    checkpoint = torch.load(args.out)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ----------------------- Final Eval + KNN -----------------------
    all_embeddings, all_labels = [], []
    final_correct, final_total = 0, 0
    with torch.no_grad():
        for X, lengths, y in val_loader:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            emb = model(X, lengths, return_embedding=True)
            all_embeddings.append(emb.cpu())
            all_labels.append(y.cpu())
            out = model(X, lengths)
            final_correct += (out.argmax(1) == y).sum().item()
            final_total += y.size(0)

    final_acc = final_correct / final_total
    print(f"Final validation accuracy: {final_acc:.4f}")

    # KNN на эмбеддингах
    print("Running KNN on embeddings...")
    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(all_embeddings, all_labels)
    knn_acc = knn.score(all_embeddings, all_labels)
    print(f"KNN accuracy on validation embeddings: {knn_acc:.4f}")

if __name__ == "__main__":
    main()
