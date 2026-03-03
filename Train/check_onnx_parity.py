import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from runtime_model import InferenceModel, Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--onnx-model", default="model.onnx")
    parser.add_argument("--golden", default="golden/golden_sequences.npz")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--input-size", type=int, default=410)
    parser.add_argument("--min-top1", type=float, default=0.98)
    parser.add_argument("--max-median-conf-diff", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def load_sequences(args, train_dir: Path) -> np.ndarray:
    golden_path = (train_dir / args.golden).resolve()
    if golden_path.exists():
        payload = np.load(golden_path)
        sequences = payload["sequences"].astype(np.float32)
        return sequences

    rng = np.random.default_rng(args.seed)
    return rng.normal(
        0, 1, size=(args.samples, args.sequence_length, args.input_size)
    ).astype(np.float32)


def run_torch(checkpoint_path: Path, sequences: np.ndarray) -> np.ndarray:
    state = torch.load(checkpoint_path, map_location="cpu")
    model = Model(int(state["input_size"]), int(state["num_classes"]))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    infer_model = InferenceModel(model).eval()
    with torch.no_grad():
        logits = infer_model(torch.from_numpy(sequences)).cpu().numpy()
    return softmax(logits)


def run_onnx(onnx_path: Path, sequences: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: sequences.astype(np.float32)})[0]
    return softmax(logits)


def main():
    args = parse_args()
    train_dir = Path(__file__).resolve().parent
    checkpoint_path = (train_dir / args.checkpoint).resolve()
    onnx_path = (train_dir / args.onnx_model).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    sequences = load_sequences(args, train_dir)
    torch_probs = run_torch(checkpoint_path, sequences)
    onnx_probs = run_onnx(onnx_path, sequences)

    torch_top1 = np.argmax(torch_probs, axis=1)
    onnx_top1 = np.argmax(onnx_probs, axis=1)
    top1_match = float(np.mean(torch_top1 == onnx_top1))

    torch_conf = np.max(torch_probs, axis=1)
    onnx_conf = np.max(onnx_probs, axis=1)
    median_conf_diff = float(np.median(np.abs(torch_conf - onnx_conf)))

    print(f"Samples: {sequences.shape[0]}")
    print(f"Top-1 match: {top1_match:.4f}")
    print(f"Median confidence abs diff: {median_conf_diff:.4f}")

    if top1_match < args.min_top1 or median_conf_diff > args.max_median_conf_diff:
        raise SystemExit(
            "Parity check failed: "
            f"top1={top1_match:.4f} (min {args.min_top1:.4f}), "
            f"median_conf_diff={median_conf_diff:.4f} (max {args.max_median_conf_diff:.4f})"
        )

    print("Parity check passed.")


if __name__ == "__main__":
    main()

