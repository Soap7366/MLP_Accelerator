"""Portable W8A8 inference: int32 MAC, int64 fixed-point requantization."""
from pathlib import Path
import numpy as np


def requantize(acc, multiplier, shift):
    """Round to nearest, ties away from zero; multiplier and shift per channel."""
    product = acc.astype(np.int64) * multiplier.astype(np.int64)
    offset = np.left_shift(np.int64(1), shift - 1)
    magnitude = (np.abs(product) + offset) >> shift
    return np.where(product < 0, -magnitude, magnitude)


class Int8MLP:
    def __init__(self, arrays):
        self.arrays = arrays

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({key: archive[key].copy() for key in archive.files})

    def forward(self, x):
        if x.dtype != np.int8 or x.ndim != 2 or x.shape[1] != 512:
            raise ValueError("Expected signed int8 input with shape (N, 512)")
        traces = []
        a = x
        for i in range(1, 4):
            p = f"fc{i}_"
            w, b = self.arrays[p + "weight"], self.arrays[p + "bias"]
            # Check worst-case absolute partial sum before int32 execution.
            bound = np.abs(w.astype(np.int64)).sum(axis=1) * (128 if i == 1 else 127) + np.abs(b.astype(np.int64))
            if np.any(bound > np.iinfo(np.int32).max):
                raise OverflowError("INT32 accumulator could overflow")
            acc = a.astype(np.int32) @ w.astype(np.int32).T + b
            q = requantize(acc, self.arrays[p + "multiplier"], self.arrays[p + "shift"])
            if i < 3:
                q = np.maximum(q, 0)
            clipped = (q < -127) | (q > 127)
            a = np.clip(q, -127, 127).astype(np.int8)
            traces.append({"accumulator": acc, "activation": a.copy(), "clipped_count": int(clipped.sum())})
        return a, traces

    def predict(self, x):
        # Both output channels share one scale and zero point, so compare directly.
        return self.forward(x)[0].argmax(axis=1)
