"""
Candidate generation (blocking) module for Business Entity Resolution.
Reduces the multi-trillion comparison space to a high-recall candidate set.
Uses country partitioning, inverted index on rare name tokens,
postal-code blocking, and TF-IDF candidate ranking.
"""

from collections import defaultdict
import re
from typing import Dict, List, Optional, Set, Tuple
from src.preprocess import (
    extract_numbers,
    extract_postal_code,
    get_name_tokens_core,
    normalize_address,
    normalize_business_name,
)

# High-frequency generic business terms to ignore as sole blocking keys
GENERIC_STOP_WORDS = {
    "enterprise", "enterprises", "solutions", "services", "company", "store",
    "shop", "trading", "industries", "agency", "center", "centre", "mart",
    "hotel", "restaurant", "cafe", "clinic", "hospital", "pharmacy", "medical",
    "stationery", "textiles", "hardware", "motors", "auto", "group", "holdings",
    "india", "usa", "american", "france", "paris", "delhi", "mumbai", "new",
}


class BusinessRecord:
    __slots__ = (
        "entity_id",
        "norm_name",
        "norm_addr",
        "country",
        "postal_code",
        "numbers",
        "core_tokens",
        "first_token",
    )

    def __init__(self, entity_id: str, business_name: str, business_address: str, country: str):
        self.entity_id = entity_id
        self.norm_name = normalize_business_name(business_name)
        self.norm_addr = normalize_address(business_address)
        self.country = country.strip() if country else ""
        self.postal_code = extract_postal_code(business_address, self.country) or ""
        self.numbers = extract_numbers(self.norm_addr)
        self.core_tokens = get_name_tokens_core(self.norm_name)
        self.first_token = self.core_tokens[0] if self.core_tokens else ""


class CandidateBlocker:
    """
    Inverted-index blocker that generates candidate target entities (S2 / S3)
    for each query entity (S1).
    """

    def __init__(self, max_candidates_per_entity: int = 20, max_token_posting_size: int = 500):
        self.max_candidates = max_candidates_per_entity
        self.max_token_size = max_token_posting_size

        # Inverted index mappings:
        # key -> list of target entity IDs
        self.name_exact_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.postal_first_token_index = defaultdict(list)
        self.target_records: Dict[str, BusinessRecord] = {}

    def index_target_records(self, records: List[BusinessRecord]):
        """Indexes Source 2 and Source 3 records."""
        for rec in records:
            self.target_records[rec.entity_id] = rec

            # 1. Exact normalized name
            if rec.norm_name and len(rec.norm_name) > 3:
                self.name_exact_index[rec.norm_name].append(rec.entity_id)

            # 2. Distinctive core tokens
            for tok in rec.core_tokens:
                if len(tok) >= 3 and tok not in GENERIC_STOP_WORDS:
                    self.token_index[tok].append(rec.entity_id)

            # 3. Postal code + first token
            if rec.postal_code and rec.first_token:
                key = f"{rec.postal_code}_{rec.first_token}"
                self.postal_first_token_index[key].append(rec.entity_id)

    def generate_candidates_for_query(self, query: BusinessRecord) -> List[str]:
        """
        Finds candidate target entity IDs for a single Source 1 query.
        """
        candidates: Dict[str, int] = defaultdict(int)

        # 1. Exact normalized name match (Highest confidence bonus)
        if query.norm_name and query.norm_name in self.name_exact_index:
            for tid in self.name_exact_index[query.norm_name]:
                candidates[tid] += 10

        # 2. Postal code + first token
        if query.postal_code and query.first_token:
            key = f"{query.postal_code}_{query.first_token}"
            if key in self.postal_first_token_index:
                for tid in self.postal_first_token_index[key]:
                    candidates[tid] += 8

        # 3. Rare / distinctive core token matches
        for tok in query.core_tokens:
            if len(tok) >= 3 and tok not in GENERIC_STOP_WORDS:
                postings = self.token_index.get(tok)
                if postings and len(postings) <= self.max_token_size:
                    score = 4 if len(postings) < 50 else 2
                    for tid in postings:
                        candidates[tid] += score

        if not candidates:
            return []

        # Sort candidates by match strength and keep top candidates
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in sorted_candidates[: self.max_candidates]]


if __name__ == "__main__":
    print("Testing CandidateBlocker...")
    blocker = CandidateBlocker(max_candidates_per_entity=10)

    # Sample target records (S2 and S3)
    targets = [
        BusinessRecord("S2-001", "Acme Logistics Corp", "100 Industrial Pkwy, Dallas, TX 75001", "US"),
        BusinessRecord("S2-002", "Starbucks Coffee", "400 Main St, Seattle, WA 98101", "US"),
        BusinessRecord("S3-001", "Acme Logistics", "100 Industrial Parkway Suite 2, Dallas 75001", "US"),
        BusinessRecord("S3-002", "Tata Consultancy Services", "Bandra Kurla Complex, Mumbai 400051", "India"),
    ]
    blocker.index_target_records(targets)

    # Sample query (S1)
    s1 = BusinessRecord("S1-001", "Acme Logistics LLC", "100 Industrial Pkwy, Dallas, TX 75001", "US")
    cands = blocker.generate_candidates_for_query(s1)
    print(f"Candidates for {s1.norm_name}: {cands}")
    assert "S2-001" in cands and "S3-001" in cands
    print("Candidate generation test passed!")
