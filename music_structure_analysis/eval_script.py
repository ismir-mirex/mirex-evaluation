#!/usr/bin/env python3
"""
Music Structure Analysis (MSA) Evaluation Script
For MIREX 2025 Reproduction

Source: https://github.com/ldzhangyx/music-structure-analysis-eval/blob/main/eval.py

Metrics:
- ACC: Frame-level accuracy (hop size = 0.2s)
- HR.5: Hit Rate F-measure with 0.5s tolerance (boundary detection)
- HR3: Hit Rate F-measure with 3.0s tolerance (boundary detection)

Usage:
    python eval.py ground_truth.jsonl predictions.jsonl
    python eval.py ground_truth.jsonl predictions.jsonl --split test
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------------
# 1. Label Conversion Logic
# ----------------------------------------------------------------------------

LABEL_MAPPING = [
    ("silence", "silence"), ("pre-chorus", "verse"),
    ("prechorus", "verse"), ("refrain", "chorus"),
    ("chorus", "chorus"), ("theme", "chorus"),
    ("stutter", "chorus"), ("verse", "verse"),
    ("rap", "verse"), ("section", "verse"),
    ("slow", "verse"), ("build", "verse"),
    ("dialog", "verse"), ("intro", "intro"),
    ("fadein", "intro"), ("opening", "intro"),
    ("bridge", "bridge"), ("trans", "bridge"),
    ("out", "outro"), ("coda", "outro"),
    ("ending", "outro"), ("break", "inst"),
    ("inst", "inst"), ("interlude", "inst"),
    ("impro", "inst"), ("solo", "inst")
]


def convert_label(label: str) -> str:
    """
    Converts a raw label string into one of the 7 canonical classes.
    (intro, verse, chorus, bridge, outro, inst, silence)
    """
    if label == "end":
        return "end"

    lower_label = label.lower()

    for substring, canonical in LABEL_MAPPING:
        if substring in lower_label:
            return canonical

    return "other"


# ----------------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# ----------------------------------------------------------------------------

def load_jsonl(file_path: str, split: Optional[str] = None) -> Dict:
    """
    Loads ground truth data from a .jsonl file into a dictionary keyed by data_id.

    Args:
        file_path: Path to the ground truth .jsonl file
        split: Optional split filter ("train", "val", "test"). If None, loads all.
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if split is None or item.get('split') == split:
                data[item['data_id']] = item['msa_info']
    return data


def load_jsonl_predict(file_path: str) -> Dict:
    """
    Loads prediction data from a .jsonl or .json file.

    Supports two formats:
    1. JSONL format (same as ground truth):
       {"data_id": "track_name", "msa_info": [[0.0, "intro"], [10.5, "verse"], ...]}

    2. Legacy JSON format:
       [{"id": "name.wav", "result": [[[start, end], label], ...]}, ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    data = {}

    # Try JSONL format first (same as ground truth)
    if content.startswith('{'):
        for line in content.split('\n'):
            if line.strip():
                item = json.loads(line)
                if 'data_id' in item and 'msa_info' in item:
                    data[item['data_id']] = item['msa_info']
        if data:
            return data

    # Try legacy JSON array format
    try:
        id_structure_dict = json.loads(content)
        if isinstance(id_structure_dict, list):
            for item in id_structure_dict:
                name = ".".join(item['id'].split(".")[:-1])
                res_list = item['result']

                res_list_new = []
                for seg in res_list:
                    res_list_new.append([seg[0][0], seg[1]])

                res_list_new.append([seg[0][1], "end"])
                data[name] = res_list_new
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return data


def preprocess_msa(msa_info: List) -> Tuple[List, float]:
    """
    Preprocesses msa_info into a [start, end, label] format and returns total duration.
    Applies the label conversion function during this process.
    """
    processed_segments = []
    msa_info = sorted(msa_info, key=lambda x: x[0])

    total_duration = 0.0

    for i in range(len(msa_info)):
        start_time, raw_label = msa_info[i]
        label = convert_label(raw_label)

        if label == "end":
            if i > 0:
                total_duration = start_time
            continue

        if i + 1 < len(msa_info):
            end_time = msa_info[i + 1][0]
        else:
            continue

        processed_segments.append([start_time, end_time, label])

    if not total_duration and processed_segments:
        total_duration = processed_segments[-1][1] if processed_segments else 0.0

    return processed_segments, total_duration


# ----------------------------------------------------------------------------
# 3. Evaluation Metrics
# ----------------------------------------------------------------------------

def calculate_accuracy(gt_data: Dict, pred_data: Dict, frame_hop: float = 0.2) -> float:
    """
    Calculates frame-level accuracy (ACC).

    Args:
        gt_data: Ground truth data dictionary
        pred_data: Prediction data dictionary
        frame_hop: Frame hop size in seconds (default: 0.2s)

    Returns:
        Frame-level accuracy score
    """
    total_frames = 0
    correct_frames = 0

    common_ids = set(gt_data.keys()) & set(pred_data.keys())

    for data_id in common_ids:
        gt_msa, gt_duration = preprocess_msa(gt_data[data_id])
        pred_msa, _ = preprocess_msa(pred_data[data_id])

        if gt_duration == 0:
            continue

        frame_times = np.arange(0, gt_duration, frame_hop)
        total_frames += len(frame_times)

        gt_ptr, pred_ptr = 0, 0
        for t in frame_times:
            while gt_ptr < len(gt_msa) and t >= gt_msa[gt_ptr][1]:
                gt_ptr += 1
            gt_label = gt_msa[gt_ptr][2] if gt_ptr < len(gt_msa) else None

            while pred_ptr < len(pred_msa) and t >= pred_msa[pred_ptr][1]:
                pred_ptr += 1
            pred_label = pred_msa[pred_ptr][2] if pred_ptr < len(pred_msa) else None

            if gt_label is not None and gt_label == pred_label and gt_label != 'other':
                correct_frames += 1

    return correct_frames / total_frames if total_frames > 0 else 0.0


def calculate_hit_rate(gt_data: Dict, pred_data: Dict, tolerance: float) -> Tuple[float, float, float]:
    """
    Calculates boundary detection Hit Rate (Precision, Recall, F-measure).

    A predicted boundary is a 'hit' if it falls within `tolerance` seconds
    of any ground truth boundary. Labels are NOT considered - this is
    purely boundary detection evaluation.

    Args:
        gt_data: Ground truth data dictionary
        pred_data: Prediction data dictionary
        tolerance: Tolerance window in seconds (e.g., 0.5 or 3.0)

    Returns:
        (precision, recall, f_measure)
    """
    total_hits = 0
    total_gt_boundaries = 0
    total_pred_boundaries = 0

    common_ids = set(gt_data.keys()) & set(pred_data.keys())

    for data_id in common_ids:
        gt_msa, _ = preprocess_msa(gt_data[data_id])
        pred_msa, _ = preprocess_msa(pred_data[data_id])

        if not gt_msa or not pred_msa:
            continue

        # Extract boundaries (start of each segment + end of last segment)
        gt_boundaries = [seg[0] for seg in gt_msa] + [gt_msa[-1][1]]
        pred_boundaries = [seg[0] for seg in pred_msa] + [pred_msa[-1][1]]

        total_gt_boundaries += len(gt_boundaries)
        total_pred_boundaries += len(pred_boundaries)

        # Count hits: for each GT boundary, check if any pred boundary is within tolerance
        for gt_b in gt_boundaries:
            for pred_b in pred_boundaries:
                if abs(gt_b - pred_b) <= tolerance:
                    total_hits += 1
                    break

    precision = total_hits / total_pred_boundaries if total_pred_boundaries > 0 else 0.0
    recall = total_hits / total_gt_boundaries if total_gt_boundaries > 0 else 0.0

    if precision + recall == 0:
        f_measure = 0.0
    else:
        f_measure = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f_measure


# ----------------------------------------------------------------------------
# 4. Main Evaluation Interface
# ----------------------------------------------------------------------------

def evaluate(gt_file: str, pred_file: str, split: Optional[str] = "test", verbose: bool = True) -> Dict:
    """
    Main evaluation function.

    Args:
        gt_file: Path to ground truth .jsonl file
        pred_file: Path to prediction .jsonl file
        split: Data split to evaluate ("train", "val", "test", or None for all)
        verbose: Whether to print results

    Returns:
        Dictionary containing all metric scores
    """
    gt_data = load_jsonl(gt_file, split=split)
    pred_data = load_jsonl_predict(pred_file)

    common_ids = set(gt_data.keys()) & set(pred_data.keys())

    if not common_ids:
        raise ValueError("No common data_ids found between ground truth and predictions.")

    if verbose:
        print(f"Loaded {len(gt_data)} ground truth entries (split={split})")
        print(f"Loaded {len(pred_data)} prediction entries")
        print(f"Evaluating on {len(common_ids)} common entries\n")

    # Calculate metrics
    accuracy = calculate_accuracy(gt_data, pred_data)
    p_05, r_05, f_05 = calculate_hit_rate(gt_data, pred_data, tolerance=0.5)
    p_3, r_3, f_3 = calculate_hit_rate(gt_data, pred_data, tolerance=3.0)

    results = {
        "ACC": accuracy,
        "HR.5_P": p_05,
        "HR.5_R": r_05,
        "HR.5_F": f_05,
        "HR3_P": p_3,
        "HR3_R": r_3,
        "HR3_F": f_3,
        "num_samples": len(common_ids)
    }

    if verbose:
        print("=" * 50)
        print("          MSA Evaluation Results")
        print("=" * 50)
        print(f"Frame-level Accuracy (ACC @ 0.2s): {accuracy:.4f}")
        print("-" * 50)
        print(f"Hit Rate (Tolerance=0.5s):")
        print(f"  Precision:  {p_05:.4f}")
        print(f"  Recall:     {r_05:.4f}")
        print(f"  F-measure:  {f_05:.4f}")
        print("-" * 50)
        print(f"Hit Rate (Tolerance=3.0s):")
        print(f"  Precision:  {p_3:.4f}")
        print(f"  Recall:     {r_3:.4f}")
        print(f"  F-measure:  {f_3:.4f}")
        print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Music Structure Analysis (MSA) predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test split (default)
  python eval.py harmonixset.corrected.20250821.jsonl predictions.jsonl

  # Evaluate on all splits
  python eval.py harmonixset.corrected.20250821.jsonl predictions.jsonl --split all

  # Output results as JSON
  python eval.py harmonixset.corrected.20250821.jsonl predictions.jsonl --json
        """
    )
    parser.add_argument("ground_truth_file", type=str,
                        help="Path to the ground truth .jsonl file")
    parser.add_argument("prediction_file", type=str,
                        help="Path to the prediction .jsonl file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Data split to evaluate (default: test)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    split = None if args.split == "all" else args.split

    results = evaluate(args.ground_truth_file, args.prediction_file, split=split, verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()