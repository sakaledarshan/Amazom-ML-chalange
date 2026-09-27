# Business Entity Resolution — Amazon ML Challenge 2026

An industrial-scale Entity Resolution (ER) solution designed to link noisy business records from 3 independent data sources (`Source 1`, `Source 2`, `Source 3`) to a deduplicated reference source (`Source 1`), evaluated using the macro-averaged $F_{0.5}$ metric.

---

## 1. Architecture Overview

The pipeline employs a two-stage approach optimized for recall, precision, and CPU execution speed:

1. **Stage 1: Country Partitioning & Inverted Index Blocking (`src/blocking.py`)**
   - **Zero-loss country partitioning:** Records are strictly partitioned into US, India, and France sub-spaces (0 cross-country matches observed in ground truth).
   - **Inverted token indexing:** Multilingual core name tokens (with generic company stopwords filtered) index candidate targets.
   - **Locality blocking:** Combines postal code matching with first name token hashes to capture local business branches.
   - Generates high-recall candidates recorded in `output/candidate_pairs.tsv`.

2. **Stage 2: Pairwise Feature Engineering & Gradient Boosted Classification (`src/features.py`, `src/train.py`)**
   - Uses C++ accelerated RapidFuzz for high-throughput string metrics: Token Sort Ratio, Token Set Ratio, Jaro-Winkler similarity, Partial Ratio.
   - Address and locality overlap: House/street number set intersection, postal code consistency, address token Jaccard similarity.
   - **Threshold Optimization for $F_{0.5}$:** Because the competition metric penalizes False Positives 2x more than False Negatives ($F_{0.5}$), a conservative threshold is tuned on validation entities. Singletons (true non-matches) receive a full 1.0 score when predicted empty.

---

## 2. Project Layout

```text
├── output/
│   ├── matching_results.tsv      # Scored leaderboard submission
│   └── candidate_pairs.tsv       # Blocking candidate set
├── src/
│   ├── config.py                 # Paths and hyperparameters
│   ├── preprocess.py             # Normalization & address parsers (US, India, France)
│   ├── blocking.py               # Inverted-index candidate generation
│   ├── features.py               # Pairwise feature vector extraction
│   ├── evaluate.py               # Official Macro F_0.5 metric evaluation
│   ├── train.py                  # XGBoost model training and threshold tuning
│   └── predict.py                # End-to-end streaming batch inference
├── package_submission.py         # Submissions validation and zip packager
├── requirements.txt              # Pinned environment dependencies
└── README.md                     # Pipeline documentation
```

---

## 3. End-to-End Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Matching Model
Train the pairwise classifier and compute the optimal decision threshold for $F_{0.5}$:
```bash
python -m src.train
```

### Step 3: Generate Predictions & Candidate Pairs
Run streaming inference across all test countries:
```bash
python -m src.predict
```
*(For quick testing or dry runs, pass `--limit 1000`)*.

### Step 4: Validate and Package Submission
Run the official competition validator and generate the submission zip:
```bash
python package_submission.py --team "YourTeamName"
```
This produces `YourTeamName_submission.zip` ready for upload.
