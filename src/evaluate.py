"""
Evaluation module computing the official Macro-averaged F_0.5 score
per the Amazon ML Challenge 2026 specifications.
"""

from typing import Dict, Iterable, Set


def compute_entity_f05(pred_set: Set[str], gt_set: Set[str]) -> float:
    """
    Computes F_0.5 for a single Source 1 entity.
    Formula:
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Special handling:
        - True singleton (gt is empty):
            scores 1.0 if pred is empty, 0.0 otherwise.
        - Non-singleton (gt non-empty):
            scores 0.0 if pred is empty or TP is 0.
    """
    is_gt_singleton = len(gt_set) == 0
    is_pred_singleton = len(pred_set) == 0

    if is_gt_singleton:
        return 1.0 if is_pred_singleton else 0.0

    if is_pred_singleton:
        return 0.0

    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    
    Returns:
        dict containing 'macro_f05', 'singleton_accuracy', 'non_singleton_f05', etc.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0}

    total_f05 = 0.0
    singleton_correct = 0
    total_singletons = 0
    non_singleton_f05_sum = 0.0
    total_non_singletons = 0

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, gt_set)
        total_f05 += score

        if len(gt_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            total_non_singletons += 1
            non_singleton_f05_sum += score

    macro_f05 = total_f05 / total_entities
    singleton_acc = (singleton_correct / total_singletons) if total_singletons > 0 else 1.0
    non_singleton_f05 = (non_singleton_f05_sum / total_non_singletons) if total_non_singletons > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "total_entities": total_entities,
        "total_singletons": total_singletons,
        "total_non_singletons": total_non_singletons,
    }


if __name__ == "__main__":
    # Test with official problem statement example:
    # Model predicts: [S2-00047, S2-00193, S3-00812]
    # Ground truth:   [S2-00047, S3-00812]
    # P = 2/3, R = 1.0 => F_0.5 = 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    gt = {"S2-00047", "S3-00812"}
    score = compute_entity_f05(pred, gt)
    print(f"Example Score: {score:.3f} (Expected: ~0.714)")
    assert abs(score - 0.714285) < 1e-4, f"Mismatch: {score}"
    print("Unit test passed successfully!")
