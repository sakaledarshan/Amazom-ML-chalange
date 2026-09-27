"""
Configuration parameters for the Business Entity Resolution Pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training file paths
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test file paths
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Pipeline Parameters
RANDOM_SEED = 42

# Blocking configuration
MAX_CANDIDATES_PER_SOURCE = 15  # Max candidates to retrieve per source file
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (2, 4)  # Char n-grams for typo tolerance

# Classification & Metric Parameters
F_BETA = 0.5
DECISION_THRESHOLD = 0.70  # Conservative threshold to protect precision (F_0.5)
