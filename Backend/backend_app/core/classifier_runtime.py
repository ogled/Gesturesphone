from __future__ import annotations

import json
import os
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class RuntimeMetadata:
    labels: List[str]
    num_classes: int
    input_size: int
    sequence_length: int
    feature_contract_version: str


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def load_runtime_metadata(path: Path) -> RuntimeMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeMetadata(
        labels=list(payload["labels"]),
        num_classes=int(payload["num_classes"]),
        input_size=int(payload["input_size"]),
        sequence_length=int(payload["sequence_length"]),
        feature_contract_version=str(payload["feature_contract_version"]),
    )


class ClassifierRuntime:
    def __init__(self, metadata: RuntimeMetadata):
        self.metadata = metadata

    @property
    def labels(self) -> List[str]:
        return self.metadata.labels

    def predict_proba(self, sequence_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OnnxClassifierRuntime(ClassifierRuntime):
    def __init__(self, model_path: Path, metadata: RuntimeMetadata):
        import onnxruntime as ort

        super().__init__(metadata)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, os.cpu_count() // 2)
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_proba(self, sequence_batch: np.ndarray) -> np.ndarray:
        logits = self.session.run(
            [self.output_name], {self.input_name: sequence_batch.astype(np.float32)}
        )[0]
        probs = _softmax(logits)
        return probs[0]


class TorchClassifierRuntime(ClassifierRuntime):
    def __init__(self, checkpoint_path: Path):
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        F = importlib.import_module("torch.nn.functional")

        state = torch.load(checkpoint_path, map_location="cpu")
        metadata = RuntimeMetadata(
            labels=list(state["labels"]),
            num_classes=int(state["num_classes"]),
            input_size=int(state["input_size"]),
            sequence_length=14,
            feature_contract_version="v1",
        )
        super().__init__(metadata)

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

            def forward(self, x, labels=None, return_embedding=False):
                x = x.permute(0, 2, 1)
                local_feat = self.local_branch(x)
                mid_feat = self.mid_branch(x)
                global_feat = self.global_branch(x)
                combined = torch.cat([local_feat, mid_feat, global_feat], dim=1)
                x = self.fusion(combined).squeeze(-1)
                if return_embedding:
                    return x
                return self.arcface(x, labels)

        torch.set_num_threads(max(1, os.cpu_count() // 2))
        self.torch = torch
        self.model = Model(self.metadata.input_size, self.metadata.num_classes)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def predict_proba(self, sequence_batch: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            sequence_tensor = self.torch.from_numpy(sequence_batch.astype(np.float32))
            emb = self.model(sequence_tensor, return_embedding=True)
            logits = self.model.arcface(emb)
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()
        return probs[0]
