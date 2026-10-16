"""Shared helpers: seeding and run -> sample -> label mapping."""
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN_TO_SAMPLE = ROOT / "run_to_sample.json"
SAMPLE_TO_LABEL = ROOT / "sample_to_label.json"
LABEL_TO_INT = {"N": 0, "Y": 1}  # N = control, Y = T2D


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_run_table() -> list[dict]:
    """Return [{run, sample, label}] for every run that has a label."""
    run_to_sample = json.loads(RUN_TO_SAMPLE.read_text())
    sample_to_label = json.loads(SAMPLE_TO_LABEL.read_text())
    return [
        {"run": run, "sample": sample, "label": sample_to_label[sample]}
        for run, sample in sorted(run_to_sample.items())
        if sample in sample_to_label
    ]
