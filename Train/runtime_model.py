import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Model(nn.Module):
    def __init__(self, input_size, num_classes, emb_dim=512):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv1d(input_size, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.mid_branch = nn.Sequential(
            nn.Conv1d(input_size, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, 5, padding=4, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.global_branch = nn.Sequential(
            nn.Conv1d(input_size, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, 7, padding=6, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(256 * 3, emb_dim, 1),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.arcface = ArcMarginProduct(emb_dim, num_classes)

    def forward(self, x, lengths=None, labels=None, return_embedding=False):
        x = x.permute(0, 2, 1)
        local_feat = self.local_branch(x)
        mid_feat = self.mid_branch(x)
        global_feat = self.global_branch(x)
        combined = torch.cat([local_feat, mid_feat, global_feat], dim=1)
        x = self.fusion(combined).squeeze(-1)
        if return_embedding:
            return x
        return self.arcface(x, labels)


class InferenceModel(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def forward(self, x):
        emb = self.model(x, return_embedding=True)
        return self.model.arcface(emb)

