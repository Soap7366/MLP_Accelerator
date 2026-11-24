"""Rebuild and evaluate the supplied FP32 MLP on signed-int8 COE samples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Names match the original checkpoint exactly.
        self.fc1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


def read_coe(path):
    text = path.read_text(encoding="utf-8-sig")
    radix = re.search(r"memory_initialization_radix\s*=\s*(\d+)\s*;", text, re.I)
    vector = re.search(r"memory_initialization_vector\s*=\s*(.*?)\s*;?\s*$", text, re.I | re.S)
    if not radix or int(radix.group(1)) != 2 or not vector:
        raise ValueError("Expected a radix-2 COE file")
    tokens = re.split(r"[,\s]+", vector.group(1).strip())
    if not tokens or any(not re.fullmatch(r"[01]{8}", token) for token in tokens):
        raise ValueError("Every COE value must be exactly 8 binary digits")
    values = np.array([int(token, 2) for token in tokens], dtype=np.int16)
    values[values >= 128] -= 256  # Decode two's-complement int8.
    if values.size % 512:
        raise ValueError(f"{values.size} input values cannot form 512-element samples")
    return values.reshape(-1, 512).astype(np.float32)


def read_labels(path):
    tokens = [v for v in re.split(r"[,;\s]+", path.read_text(encoding="utf-8-sig").strip()) if v]
    if not tokens or any(v not in {"0", "1"} for v in tokens):
        raise ValueError("Labels must be 0 or 1")
    return np.array(tokens, dtype=np.int64)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "原始模型文件/2-FCNet1-485-0.945.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "输入数据/sel_data.coe")
    parser.add_argument("--labels", type=Path, default=ROOT / "输入数据标签/sel_labels.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor state_dict")
    if any(v.dtype != torch.float32 for v in state.values()):
        raise ValueError("Checkpoint is not entirely FP32")
    model = FCNet().float().eval()
    model.load_state_dict(state, strict=True)
    x = read_coe(args.data)
    labels = read_labels(args.labels)
    if len(x) != len(labels):
        raise ValueError(f"Sample/label count mismatch: {len(x)} vs {len(labels)}")

    with torch.inference_mode():
        logits = model(torch.from_numpy(x)).numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite model output")
    predictions = logits.argmax(axis=1)

    # Independent NumPy evaluation of z = W @ x + b, with ReLU after layers 1 and 2.
    reference = x.copy()
    for layer in ("fc1.0", "fc2.0", "fc3"):
        reference = reference @ state[f"{layer}.weight"].numpy().T + state[f"{layer}.bias"].numpy()
        if layer != "fc3":
            reference = np.maximum(reference, 0)
    np.testing.assert_allclose(logits, reference, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(predictions, reference.argmax(axis=1))

    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    correct = int((predictions == labels).sum())
    report = {
        "architecture": "512 -> Linear(128) -> ReLU -> Linear(32) -> ReLU -> Linear(2)",
        "parameter_dtype": "float32",
        "parameter_count": sum(v.numel() for v in model.parameters()),
        "input_encoding": "signed int8 two's complement, cast to float32; no preprocessing",
        "sample_layout": "512 consecutive COE values per sample; labels in file order",
        "decision": "argmax of two logits; exact ties select class 0",
        "sample_count": len(labels),
        "correct_count": correct,
        "accuracy": correct / len(labels),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "per_class": {
            str(c): {"count": int((labels == c).sum()), "correct": int(confusion[c, c]),
                     "recall": float(confusion[c, c] / (labels == c).sum()) if (labels == c).any() else None}
            for c in (0, 1)
        },
        "misclassified_sample_indices_1_based": (np.flatnonzero(predictions != labels) + 1).tolist(),
        "numpy_verification": {"passed": True, "max_absolute_error": float(np.max(np.abs(logits - reference)))},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "source_sha256": {str(p.resolve()): sha256(p) for p in (args.weights, args.data, args.labels)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index_1_based", "true_label", "predicted_label", "logit_0", "logit_1", "correct"])
        for i, (label, pred, scores) in enumerate(zip(labels, predictions, logits), start=1):
            writer.writerow([i, int(label), int(pred), float(scores[0]), float(scores[1]), int(label == pred)])
    print(f"FP32 MLP: {report['parameter_count']} parameters")
    print(f"Accuracy: {correct}/{len(labels)} = {report['accuracy']:.6%}")
    print(f"Confusion matrix (rows=true, columns=predicted):\n{confusion}")
    print(f"Misclassified samples (1-based): {report['misclassified_sample_indices_1_based']}")
    print(f"NumPy cross-check passed; max absolute error: {report['numpy_verification']['max_absolute_error']:.8g}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
