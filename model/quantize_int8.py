"""Calibrate/export signed W8A8 MLP and compare it with the FP32 checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from evaluate_model import ROOT, FCNet, read_coe, read_labels, sha256
from int8_model import Int8MLP, requantize


def fixed_multiplier(ratio):
    mantissa, exponent = np.frexp(ratio)
    multiplier = np.rint(mantissa * (1 << 31)).astype(np.int64)
    overflow = multiplier == (1 << 31)
    multiplier[overflow] //= 2
    exponent[overflow] += 1
    shift = (31 - exponent).astype(np.int64)
    if np.any((shift < 1) | (shift > 62)):
        raise ValueError("Scale ratio outside supported fixed-point range")
    return multiplier.astype(np.int32), shift


def quantize(state, calibration):
    arrays = {"input_scale": np.array(1.0), "zero_point": np.array(0, dtype=np.int32)}
    a = calibration
    input_scale = 1.0
    for i, name in enumerate(("fc1.0", "fc2.0", "fc3"), 1):
        w = state[name + ".weight"].numpy().astype(np.float64)
        b = state[name + ".bias"].numpy().astype(np.float64)
        a = a @ w.T + b
        if i < 3:
            a = np.maximum(a, 0)
        output_scale = max(float(np.abs(a).max()) / 127, 1e-12)
        weight_scale = np.maximum(np.abs(w).max(axis=1) / 127, 1e-12)
        qw = np.clip(np.rint(w / weight_scale[:, None]), -127, 127).astype(np.int8)
        qb = np.rint(b / (input_scale * weight_scale))
        if np.any(np.abs(qb) > np.iinfo(np.int32).max):
            raise OverflowError("Bias does not fit INT32")
        multiplier, shift = fixed_multiplier(input_scale * weight_scale / output_scale)
        for suffix, value in {"weight": qw, "bias": qb.astype(np.int32),
                              "weight_scale": weight_scale, "output_scale": np.array(output_scale),
                              "multiplier": multiplier, "shift": shift}.items():
            arrays[f"fc{i}_{suffix}"] = value
        input_scale = output_scale
    return Int8MLP(arrays)


def metrics(pred, labels):
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (labels, pred), 1)
    return {"correct": int((pred == labels).sum()), "total": len(labels),
            "accuracy": float((pred == labels).mean()), "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "misclassified_indices_1_based": (np.flatnonzero(pred != labels) + 1).tolist()}


def verify(model, x, traces):
    # Independent int64 MAC reference and Python arbitrary-precision requantization.
    a = x.astype(np.int64)
    for i, trace in enumerate(traces, 1):
        p = f"fc{i}_"
        acc = a @ model.arrays[p + "weight"].astype(np.int64).T + model.arrays[p + "bias"]
        np.testing.assert_array_equal(acc, trace["accumulator"])
        expected = np.empty_like(acc)
        for row in range(len(acc)):
            for col in range(acc.shape[1]):
                value = int(acc[row, col]) * int(model.arrays[p + "multiplier"][col])
                shift = int(model.arrays[p + "shift"][col])
                rounded = (abs(value) + (1 << (shift - 1))) // (1 << shift)
                expected[row, col] = rounded if value >= 0 else -rounded
        if i < 3:
            expected = np.maximum(expected, 0)
        a = np.clip(expected, -127, 127)
        np.testing.assert_array_equal(a, trace["activation"])
    # Exercise negative/positive half-way rounding including zero.
    np.testing.assert_array_equal(requantize(np.array([[-3, -1, 0, 1, 3]], dtype=np.int32),
                                           np.ones(5, dtype=np.int32), np.ones(5, dtype=np.int64)),
                                  [[-2, -1, 0, 1, 2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "原始模型文件/2-FCNet1-485-0.945.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "输入数据/sel_data.coe")
    parser.add_argument("--labels", type=Path, default=ROOT / "输入数据标签/sel_labels.txt")
    parser.add_argument("--calibration-data", type=Path, help="Separate representative COE file; defaults to evaluation data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "int8_results")
    args = parser.parse_args()
    calibration_path = args.calibration_data or args.data
    x, labels = read_coe(args.data), read_labels(args.labels)
    calibration = read_coe(calibration_path)
    if len(x) != len(labels):
        raise ValueError("Sample/label count mismatch")
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Expected finite FP32 weights")
    fp_model = FCNet().eval()
    fp_model.load_state_dict(state, strict=True)
    with torch.inference_mode():
        fp_logits = fp_model(torch.from_numpy(x)).numpy()
    model = quantize(state, calibration)
    q_logits, traces = model.forward(x.astype(np.int8))
    verify(model, x, traces)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model_int8.npz"
    np.savez(model_path, **model.arrays)
    reloaded = Int8MLP.load(model_path)
    np.testing.assert_array_equal(reloaded.forward(x.astype(np.int8))[0], q_logits)
    fp_pred, q_pred = fp_logits.argmax(axis=1), q_logits.argmax(axis=1)
    restored_logits = q_logits.astype(np.float64) * float(model.arrays["fc3_output_scale"])
    report = {
        "scheme": "Static PTQ W8A8: symmetric per-output-channel weights, per-tensor activations; all zero points 0",
        "integer_arithmetic": "int8 operands, int32 bias/MAC, int64 multiplier product and shift; no float operations in forward",
        "calibration": {"samples": len(calibration), "method": "FP forward absolute maximum / 127, without labels",
                        "same_file_as_evaluation": sha256(calibration_path) == sha256(args.data)},
        "fp32": metrics(fp_pred, labels), "int8": metrics(q_pred, labels),
        "accuracy_change_percentage_points": float(((q_pred == labels).mean() - (fp_pred == labels).mean()) * 100),
        "prediction_agreement": float((q_pred == fp_pred).mean()),
        "changed_predictions_indices_1_based": (np.flatnonzero(q_pred != fp_pred) + 1).tolist(),
        "output_ties": int((q_logits[:, 0] == q_logits[:, 1]).sum()),
        "dequantized_logits_mae": float(np.abs(restored_logits - fp_logits).mean()),
        "dequantized_logits_max_absolute_error": float(np.abs(restored_logits - fp_logits).max()),
        "layers": [{"layer": i, "output_scale": float(model.arrays[f"fc{i}_output_scale"]),
                    "clipped_values": t["clipped_count"], "activation_values": t["activation"].size,
                    "accumulator_max_abs": int(np.abs(t["accumulator"].astype(np.int64)).max())}
                   for i, t in enumerate(traces, 1)],
        "storage": {"fp32_weights_and_bias_bytes": sum(v.numel() * v.element_size() for v in state.values()),
                    "int8_weights_and_int32_bias_bytes": sum(v.nbytes for k, v in model.arrays.items() if k.endswith(("_weight", "_bias"))),
                    "all_exported_arrays_bytes": sum(v.nbytes for v in model.arrays.values()),
                    "npz_file_bytes": model_path.stat().st_size},
        "verification": "Passed independent int64 MAC, Python integer requantization, rounding boundaries and export/reload equality",
        "sources_sha256": {str(p.resolve()): sha256(p) for p in (args.weights, args.data, args.labels, calibration_path)},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_1_based", "true_label", "fp32_prediction", "int8_prediction", "fp32_logit0", "fp32_logit1", "int8_logit0", "int8_logit1", "dequant_logit0", "dequant_logit1", "int8_correct"])
        for n in range(len(labels)):
            writer.writerow([n + 1, labels[n], fp_pred[n], q_pred[n], *fp_logits[n], *q_logits[n], *restored_logits[n], int(q_pred[n] == labels[n])])
    print(json.dumps({k: report[k] for k in ("fp32", "int8", "prediction_agreement", "layers", "storage", "verification")}, indent=2))


if __name__ == "__main__":
    main()
