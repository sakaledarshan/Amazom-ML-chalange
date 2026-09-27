"""
Feature engineering module for pairwise business entity resolution.
Uses rapidfuzz for C++ accelerated string distance calculations.
"""

from typing import Any, Dict, List, Set
from rapidfuzz import distance, fuzz

FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_first_token_match",
    "name_exact_match",
    "name_length_ratio",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_num_overlap_count",
    "addr_num_jaccard",
    "postal_code_match",
    "is_source_2",
    "is_source_3",
]


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def extract_pair_features(
    name1: str,
    addr1: str,
    postal1: str,
    nums1: Set[str],
    name2: str,
    addr2: str,
    postal2: str,
    nums2: Set[str],
    id2: str,
) -> List[float]:
    """
    Extracts a dense numerical feature vector for a candidate pair (record1, record2).
    """
    # Name features
    n_ratio = fuzz.ratio(name1, name2) / 100.0
    n_partial = fuzz.partial_ratio(name1, name2) / 100.0
    n_token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
    n_token_set = fuzz.token_set_ratio(name1, name2) / 100.0
    n_jw = distance.JaroWinkler.similarity(name1, name2)

    toks1 = name1.split()
    toks2 = name2.split()
    first_match = 1.0 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0.0
    exact_match = 1.0 if name1 == name2 and len(name1) > 0 else 0.0

    l1, l2 = len(name1), len(name2)
    len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0

    # Address features
    a_ratio = fuzz.ratio(addr1, addr2) / 100.0
    a_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    a_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0

    addr_toks1 = set(addr1.split())
    addr_toks2 = set(addr2.split())
    a_jaccard = jaccard_similarity(addr_toks1, addr_toks2)

    num_overlap = float(len(nums1 & nums2))
    num_jaccard = jaccard_similarity(nums1, nums2)

    # Postal code match
    if postal1 and postal2:
        postal_match = 1.0 if postal1 == postal2 else 0.0
    elif not postal1 and not postal2:
        postal_match = 0.5  # Neutral / missing in both
    else:
        postal_match = 0.2  # Missing in one

    # Source indicators
    is_s2 = 1.0 if id2.startswith("S2-") else 0.0
    is_s3 = 1.0 if id2.startswith("S3-") else 0.0

    return [
        n_ratio,
        n_partial,
        n_token_sort,
        n_token_set,
        n_jw,
        first_match,
        exact_match,
        len_ratio,
        a_ratio,
        a_token_sort,
        a_token_set,
        a_jaccard,
        num_overlap,
        num_jaccard,
        postal_match,
        is_s2,
        is_s3,
    ]


if __name__ == "__main__":
    v = extract_pair_features(
        name1="walmart supercenter",
        addr1="123 main st bentonville ar 72712",
        postal1="72712",
        nums1={"123", "72712"},
        name2="walmart inc",
        addr2="123 main street suite a bentonville 72712",
        postal2="72712",
        nums2={"123", "72712"},
        id2="S2-00123",
    )
    print("Extracted feature vector length:", len(v))
    for name, val in zip(FEATURE_NAMES, v):
        print(f"  {name:25s}: {val:.4f}")
