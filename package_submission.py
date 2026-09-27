"""
Packaging script for ML Challenge 2026.
Creates the compliant <team_name>_submission.zip package.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SRC_DIR = BASE_DIR / "src"
VALIDATOR = BASE_DIR / "student_resource" / "utils" / "validate_submission.py"
DOC_TEMPLATE = BASE_DIR / "student_resource" / "Documentation_template.md"


def validate_outputs():
    """Runs the official submission validator."""
    print("Running official validation script...")
    matching_tsv = OUTPUT_DIR / "matching_results.tsv"
    candidate_tsv = OUTPUT_DIR / "candidate_pairs.tsv"
    test_dir = BASE_DIR / "student_resource" / "dataset" / "test"

    if not matching_tsv.exists():
        print(f"ERROR: {matching_tsv} does not exist. Run predict.py first.")
        return False

    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--matching", str(matching_tsv),
        "--candidate", str(candidate_tsv),
        "--test-dir", str(test_dir),
    ]

    res = subprocess.run(cmd)
    return res.returncode == 0


def create_submission_zip(team_name: str = "MyTeam"):
    """Creates the submission zip archive adhering to the specified folder layout."""
    if not validate_outputs():
        print("\nWARNING: Outputs did not pass validation! Aborting zip packaging.")
        return

    zip_filename = f"{team_name}_submission.zip"
    zip_path = BASE_DIR / zip_filename
    print(f"\nBuilding submission package: {zip_filename}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. output/
        z.write(OUTPUT_DIR / "matching_results.tsv", "output/matching_results.tsv")
        if (OUTPUT_DIR / "candidate_pairs.tsv").exists():
            z.write(OUTPUT_DIR / "candidate_pairs.tsv", "output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        for root, _, files in os.walk(SRC_DIR):
            for file in files:
                if file.endswith(".py"):
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(BASE_DIR)
                    z.write(full_p, f"code/business_entity_resolution/{rel_p}")

        readme_p = BASE_DIR / "README.md"
        if readme_p.exists():
            z.write(readme_p, "code/business_entity_resolution/README.md")

        req_p = BASE_DIR / "requirements.txt"
        if req_p.exists():
            z.write(req_p, "code/business_entity_resolution/requirements.txt")

        # 3. Documentation_template.md
        doc_p = BASE_DIR / "Documentation_template.md"
        if not doc_p.exists() and DOC_TEMPLATE.exists():
            shutil.copy(DOC_TEMPLATE, doc_p)
        if doc_p.exists():
            z.write(doc_p, "Documentation_template.md")

    print(f"Package successfully created: {zip_path}")
    print(f"Package size: {zip_path.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip.")
    parser.add_argument("--team", default="AmazonML_Team", help="Your team name")
    args = parser.parse_args()

    create_submission_zip(args.team)
