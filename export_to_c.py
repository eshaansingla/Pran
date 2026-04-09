"""
export_to_c.py
==============
Convert the trained XGBoost binary classifier (6 features, 2 classes)
to a C header file for ESP32 firmware deployment.

This replaces the stale xgboost_model.h which was for a 3-class, 11-feature model.

Usage:
    python export_to_c.py
    python export_to_c.py --model models/xgboost_binary.pkl.gz --output firmware/esp32_icp_monitor/xgboost_model.h
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
]

N_FEATURES = 6


def load_booster(model_path: Path):
    """Load XGBoost Booster from pkl or pkl.gz."""
    if model_path.suffix == ".gz":
        with gzip.open(model_path, "rb") as fh:
            return pickle.load(fh)
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def dump_trees_json(bst) -> list[dict]:
    """Extract tree structures from the Booster."""
    dump = bst.get_dump(dump_format="json")
    return [json.loads(t) for t in dump]


def _tree_to_c(node: dict, indent: int = 1) -> str:
    """Recursively convert a tree node to C if-else code."""
    pad = "    " * indent
    if "leaf" in node:
        return f"{pad}tv = {node['leaf']:.6f}f;\n"

    split_feat = int(node["split"].replace("f", ""))
    split_val  = float(node["split_condition"])
    yes_child  = node["children"][0]
    no_child   = node["children"][1]

    code  = f"{pad}if (features[{split_feat}] < {split_val:.6f}f) {{\n"
    code += _tree_to_c(yes_child, indent + 1)
    code += f"{pad}}} else {{\n"
    code += _tree_to_c(no_child, indent + 1)
    code += f"{pad}}}\n"
    return code


def generate_header(bst, output_path: Path) -> None:
    """Generate C header file from XGBoost Booster."""
    trees = dump_trees_json(bst)
    n_trees = len(trees)

    # Load metadata for threshold
    meta_path = Path("models/binary_meta.json")
    threshold = 0.5
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        threshold = meta.get("prob_threshold", 0.5)

    lines = [
        "#ifndef XGBOOST_MODEL_H",
        "#define XGBOOST_MODEL_H",
        "",
        "/*",
        " * Auto-generated XGBoost model for ICP binary classification.",
        f" * Trees: {n_trees}",
        f" * Features ({N_FEATURES}):",
    ]
    for i, name in enumerate(FEATURE_NAMES):
        lines.append(f" *   [{i:2d}] {name}")
    lines += [
        " *",
        " * Classes: 0=Normal (<15 mmHg), 1=Abnormal (>=15 mmHg)",
        f" * Decision threshold: {threshold:.4f}",
        " * DO NOT EDIT -- regenerate via: python export_to_c.py",
        " */",
        "",
        "#include <stdint.h>",
        "#include <math.h>",
        "",
        f"static inline float predict_xgboost_prob(const float features[{N_FEATURES}]) {{",
        "    float logit = 0.0f;",
        "    float tv;  /* temporary tree value */",
        "",
    ]

    for i, tree in enumerate(trees):
        lines.append(f"    /* Tree {i} */")
        lines.append(_tree_to_c(tree, indent=1).rstrip())
        lines.append("    logit += tv;")
        lines.append("")

    lines += [
        "    /* Sigmoid activation */",
        "    float prob = 1.0f / (1.0f + expf(-logit));",
        "    return prob;",
        "}",
        "",
        f"static inline uint8_t predict_xgboost(const float features[{N_FEATURES}]) {{",
        f"    return predict_xgboost_prob(features) >= {threshold:.4f}f ? 1 : 0;",
        "}",
        "",
        "#endif  /* XGBOOST_MODEL_H */",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output_path} ({n_trees} trees, {N_FEATURES} features, binary classification)")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Export XGBoost binary model to C header for ESP32",
    )
    parser.add_argument("--model",  type=Path,
                        default=Path("models/xgboost_binary.pkl.gz"))
    parser.add_argument("--output", type=Path,
                        default=Path("firmware/esp32_icp_monitor/xgboost_model.h"))
    args = parser.parse_args()

    bst = load_booster(args.model)
    generate_header(bst, args.output)


if __name__ == "__main__":
    main()
