"""
Inference pipeline for Business Entity Resolution.
Generates output/candidate_pairs.tsv and output/matching_results.tsv
following all submission guidelines.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import xgboost as xgb

from src import config
from src.blocking import BusinessRecord, CandidateBlocker
from src.features import extract_pair_features


def stream_source_file(path: str, country_filter: Optional[str] = None):
    """Yields BusinessRecord objects from a source TSV."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 4:
                eid, name, addr, country = row[0], row[1], row[2], row[3].strip()
                if country_filter is None or country == country_filter:
                    yield BusinessRecord(eid, name, addr, country)


def run_country_inference(
    country: str,
    model: Optional[xgb.XGBClassifier],
    threshold: float,
    s1_limit: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Runs blocking and pairwise scoring for a specific country partition.
    Returns (candidate_map, match_map) for S1 entities in this country.
    """
    print(f"\n--- Processing Country: {country} ---")
    blocker = CandidateBlocker(max_candidates_per_entity=15)

    print(f"Loading and indexing {country} records from Source 2...")
    s2_targets = list(stream_source_file(str(config.TEST_SOURCE2), country_filter=country))
    print(f"  Loaded {len(s2_targets):,} records from Source 2.")

    print(f"Loading and indexing {country} records from Source 3...")
    s3_targets = list(stream_source_file(str(config.TEST_SOURCE3), country_filter=country))
    print(f"  Loaded {len(s3_targets):,} records from Source 3.")

    blocker.index_target_records(s2_targets + s3_targets)
    print(f"  Total {country} targets indexed: {len(blocker.target_records):,}")

    candidate_results: Dict[str, List[str]] = {}
    matching_results: Dict[str, List[str]] = {}

    print(f"Generating candidates and matching for {country} Source 1...")
    count = 0
    batch_features = []
    batch_meta = []  # (s1_id, cand_id)

    for s1_rec in stream_source_file(str(config.TEST_SOURCE1), country_filter=country):
        count += 1
        if s1_limit and count > s1_limit:
            break

        s1_id = s1_rec.entity_id
        candidates = blocker.generate_candidates_for_query(s1_rec)
        candidate_results[s1_id] = candidates

        if not candidates or model is None:
            # If no model provided, fallback to heuristic or empty
            matching_results[s1_id] = []
            continue

        # Extract features for scoring
        cands_to_score = []
        for cid in candidates:
            if cid in blocker.target_records:
                crec = blocker.target_records[cid]
                feats = extract_pair_features(
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
                cands_to_score.append((cid, feats))

        if not cands_to_score:
            matching_results[s1_id] = []
            continue

        feats_array = np.array([f for _, f in cands_to_score], dtype=np.float32)
        probs = model.predict_proba(feats_array)[:, 1]

        matched = [cid for (cid, _), p in zip(cands_to_score, probs) if p >= threshold]
        matching_results[s1_id] = matched

        if count % 25000 == 0:
            print(f"  Processed {count:,} {country} entities...")

    print(f"Finished {country}: {len(candidate_results):,} entities processed.")
    return candidate_results, matching_results


def generate_submission(
    model_path: Optional[str] = None,
    threshold_path: Optional[str] = None,
    limit_per_country: Optional[int] = None,
):
    """
    Generates candidate_pairs.tsv and matching_results.tsv across all countries.
    """
    model = None
    threshold = config.DECISION_THRESHOLD

    if model_path and os.path.exists(model_path):
        print(f"Loading trained XGBoost model from {model_path}...")
        model = xgb.XGBClassifier()
        model.load_model(model_path)

    if threshold_path and os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            tdata = json.load(f)
            threshold = tdata.get("threshold", config.DECISION_THRESHOLD)
            print(f"Using optimized threshold: {threshold:.2f}")

    all_candidates: Dict[str, List[str]] = {}
    all_matches: Dict[str, List[str]] = {}

    # The 3 countries in the test set
    countries = ["India", "US", "France"]
    for c in countries:
        cand_map, match_map = run_country_inference(
            country=c,
            model=model,
            threshold=threshold,
            s1_limit=limit_per_country,
        )
        all_candidates.update(cand_map)
        all_matches.update(match_map)

    # If limit was applied, fill remaining entities from test_source1 to ensure valid schema
    print("\nEnsuring all test Source 1 entities are present...")
    all_test_s1_ids = []
    with open(config.TEST_SOURCE1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if r:
                all_test_s1_ids.append(r[0])

    print(f"Writing {config.CANDIDATE_OUTPUT}...")
    with open(config.CANDIDATE_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in all_test_s1_ids:
            cands = all_candidates.get(s1_id, [])
            writer.writerow([s1_id, ",".join(cands)])

    print(f"Writing {config.MATCHING_OUTPUT}...")
    with open(config.MATCHING_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in all_test_s1_ids:
            matches = all_matches.get(s1_id, [])
            writer.writerow([s1_id, ",".join(matches)])

    print("\nInference complete! Output files generated successfully:")
    print(f"  Matching Results:  {config.MATCHING_OUTPUT}")
    print(f"  Candidate Pairs:   {config.CANDIDATE_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity resolution submission files.")
    parser.add_argument("--model", default=str(config.MODELS_DIR / "matcher_xgb.json"))
    parser.add_argument("--threshold", default=str(config.MODELS_DIR / "threshold.json"))
    parser.add_argument("--limit", type=int, default=None, help="Limit per country for quick validation")
    args = parser.parse_args()

    generate_submission(
        model_path=args.model,
        threshold_path=args.threshold,
        limit_per_country=args.limit,
    )
