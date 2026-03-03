import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="golden/golden_sequences.npz")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--input-size", type=int, default=410)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_fixed_length(sequence: np.ndarray, length: int) -> np.ndarray:
    if sequence.shape[0] >= length:
        return sequence[:length]
    pad = np.zeros((length - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, pad], axis=0)


def main():
    args = parse_args()
    train_dir = Path(__file__).resolve().parent
    input_dir = (train_dir / args.input_dir).resolve()
    output_path = (train_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rng = np.random.default_rng(args.seed)
    candidates = []
    for path in input_dir.rglob("*.npy"):
        try:
            arr = np.load(path)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] != args.input_size or arr.shape[0] < 1:
            continue
        candidates.append((path, arr.astype(np.float32)))

    if not candidates:
        raise RuntimeError(
            f"No valid .npy sequences found in {input_dir} with shape [T, {args.input_size}]"
        )

    take = min(args.sample_size, len(candidates))
    idx = rng.choice(len(candidates), size=take, replace=False)

    sequences = []
    source_files = []
    for i in idx:
        path, seq = candidates[int(i)]
        sequences.append(to_fixed_length(seq, args.sequence_length))
        source_files.append(str(path))

    payload = np.stack(sequences, axis=0).astype(np.float32)
    np.savez_compressed(output_path, sequences=payload, files=np.array(source_files))
    print(f"Golden set saved: {output_path}")
    print(f"Sequences: {payload.shape[0]}, shape per item: {payload.shape[1:]}")


if __name__ == "__main__":
    main()

