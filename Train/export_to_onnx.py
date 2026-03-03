import argparse
import json
from pathlib import Path

import torch

from runtime_model import InferenceModel, Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--onnx-out", default="model.onnx")
    parser.add_argument("--meta-out", default="model.runtime.json")
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--contract-version", default="v1")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = Path(__file__).resolve().parent

    checkpoint_path = (train_dir / args.checkpoint).resolve()
    onnx_path = (train_dir / args.onnx_out).resolve()
    meta_path = (train_dir / args.meta_out).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    model = Model(int(state["input_size"]), int(state["num_classes"]))
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    infer_model = InferenceModel(model).eval()
    dummy = torch.zeros(
        (1, args.sequence_length, int(state["input_size"])), dtype=torch.float32
    )

    torch.onnx.export(
        infer_model,
        dummy,
        str(onnx_path),
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        opset_version=args.opset,
    )

    metadata = {
        "labels": list(state["labels"]),
        "num_classes": int(state["num_classes"]),
        "input_size": int(state["input_size"]),
        "sequence_length": int(args.sequence_length),
        "feature_contract_version": str(args.contract_version),
        "source_checkpoint": str(checkpoint_path),
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"ONNX exported to: {onnx_path}")
    print(f"Runtime metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

