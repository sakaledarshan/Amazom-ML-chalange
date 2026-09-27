"""
Model training and threshold optimization module for Business Entity Resolution.
Trains an XGBoost classifier on candidate pairs and selects the decision threshold
that maximizes the macro-averaged F_0.5 score.
"""

from collections import defaultdict
import json
import os
import random
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import xgboost as xgb

from src import config
from src.blocking import BusinessRecord, CandidateBlocker
from src.evaluate import compute_macro_f05
from src.features import FEATURE_NAMES, extract_pair_features


def load_ground_truth(path: str, max_records: int = 50000) -> Dict[str, Set[str]]:
    """Loads ground truth mapping: source1_id -> set of true matching IDs."""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if i >= max_records:
                break
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matched = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
            gt[s1_id] = matched
    return gt


def load_source_records(path: str, allowed_ids: Optional[Set[str]] = None, max_records: Optional[int] = None) -> Dict[str, BusinessRecord]:
    """Loads records from a source TSV file."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if allowed_ids is None or eid in allowed_ids:
                    records[eid] = BusinessRecord(eid, name, addr, country)
    return records


def build_training_dataset(
    s1_records: Dict[str, BusinessRecord],
    target_records: Dict[str, BusinessRecord],
    ground_truth: Dict[str, Set[str]],
    blocker: CandidateBlocker,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates candidate pairs and extracts feature vectors + labels for training.
    """
    X_list = []
    y_list = []

    for s1_id, s1_rec in s1_records.items():
        true_matches = ground_truth.get(s1_id, set())
        candidates = blocker.generate_candidates_for_query(s1_rec)

        # Include candidate pairs
        for cand_id in candidates:
            if cand_id in target_records:
                cand_rec = target_records[cand_id]
                feats = extract_pair_features(
                    s1_rec.norm_name,
                    s1_rec.norm_addr,
                    s1_rec.postal_code,
                    s1_rec.numbers,
                    cand_rec.norm_name,
                    cand_rec.norm_addr,
                    cand_rec.postal_code,
                    cand_rec.numbers,
                    cand_id,
                )
                label = 1 if cand_id in true_matches else 0
                X_list.append(feats)
                y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def optimize_f05_threshold(
    model: xgb.XGBClassifier,
    val_s1: Dict[str, BusinessRecord],
    val_targets: Dict[str, BusinessRecord],
    val_gt: Dict[str, Set[str]],
    blocker: CandidateBlocker,
) -> Tuple[float, float]:
    """
    Evaluates different probability thresholds to find the threshold
    that maximizes the official macro-averaged F_0.5 score on validation data.
    """
    # Precompute candidate features for all validation entities
    eval_data = []
    for s1_id, s1_rec in val_s1.items():
        cands = blocker.generate_candidates_for_query(s1_rec)
        cand_pairs = []
        for cid in cands:
            if cid in val_targets:
                crec = val_targets[cid]
                f = extract_pair_features(
                    s1_rec.norm_name,
                    s1_rec.norm_addr,
                    s1_rec.postal_code,
                    s1_rec.numbers,
                    crec.norm_name,
                    crec.norm_addr,
                    crec.postal_code,
                    crec.numbers,
                    cid,
                )
                cand_pairs.append((cid, f))
        eval_data.append((s1_id, cand_pairs))

    # Score all candidate pairs with model
    all_features = []
    index_map = []
    for s1_idx, (_, cand_pairs) in enumerate(eval_data):
        for cid, f in cand_pairs:
            all_features.append(f)
            index_map.append((s1_idx, cid))

    if not all_features:
        return 0.70, 0.0

    probs = model.predict_proba(np.array(all_features, dtype=np.float32))[:, 1]

    # Group scores by S1 entity
    s1_scores = defaultdict(list)
    for (s1_idx, cid), prob in zip(index_map, probs):
        s1_id = eval_data[s1_idx][0]
        s1_scores[s1_id].append((cid, prob))

    best_thresh = 0.70
    best_score = -1.0

    # Search thresholds from 0.40 to 0.95
    for thresh in np.arange(0.40, 0.95, 0.05):
        preds = {}
        for s1_id in val_gt.keys():
            matched = {cid for cid, p in s1_scores.get(s1_id, []) if p >= thresh}
            preds[s1_id] = matched

        res = compute_macro_f05(preds, val_gt)
        macro_f05 = res["macro_f05"]
        if macro_f05 > best_score:
            best_score = macro_f05
            best_thresh = float(thresh)

    return best_thresh, best_score


def train_model(sample_size: int = 5000):
    """
    Trains an XGBoost model on a representative validation slice of the training data.
    """
    print(f"Loading sample of ground truth (size={sample_size})...")
    gt = load_ground_truth(str(config.TRAIN_GROUND_TRUTH), max_records=sample_size * 2)

    all_s1_ids = list(gt.keys())
    random.seed(config.RANDOM_SEED)
    random.shuffle(all_s1_ids)

    train_ids = set(all_s1_ids[:sample_size])
    val_ids = set(all_s1_ids[sample_size : sample_size + 1500])

    print("Loading Source 1 records...")
    train_s1 = load_source_records(str(config.TRAIN_SOURCE1), allowed_ids=train_ids)
    val_s1 = load_source_records(str(config.TRAIN_SOURCE1), allowed_ids=val_ids)

    # Collect needed target IDs for indexing
    needed_target_ids = set()
    for s1_id in list(train_ids) + list(val_ids):
        needed_target_ids.update(gt.get(s1_id, set()))

    print(f"Loading needed target records (S2 & S3)...")
    s2_records = load_source_records(str(config.TRAIN_SOURCE2), allowed_ids=needed_target_ids, max_records=50000)
    s3_records = load_source_records(str(config.TRAIN_SOURCE3), allowed_ids=needed_target_ids, max_records=50000)

    all_targets = {**s2_records, **s3_records}

    print("Indexing target records in blocker...")
    blocker = CandidateBlocker(max_candidates_per_entity=15)
    blocker.index_target_records(list(all_targets.values()))

    print("Extracting training features and labels...")
    X_train, y_train = build_training_dataset(train_s1, all_targets, gt, blocker)
    print(f"Training dataset: {len(X_train)} candidate pairs (Positives: {np.sum(y_train)}, Negatives: {len(y_train) - np.sum(y_train)})")

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    print("Optimizing decision threshold for Macro F_0.5...")
    val_gt = {k: gt[k] for k in val_ids}
    best_thresh, best_f05 = optimize_f05_threshold(model, val_s1, all_targets, val_gt, blocker)
    print(f"Optimal Threshold: {best_thresh:.2f} with Validation Macro F_0.5: {best_f05:.4f}")

    # Save model and parameters
    model_path = config.MODELS_DIR / "matcher_xgb.json"
    model.save_model(str(model_path))

    thresh_path = config.MODELS_DIR / "threshold.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thresh, "best_val_f05": best_f05, "feature_names": FEATURE_NAMES}, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Threshold saved to {thresh_path}")


if __name__ == "__main__":
    train_model(sample_size=3000)
