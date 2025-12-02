import sys
import os
import pandas as pd

REQUIRED_COLUMNS = ['ID', 'Target', 'Target_Accuracy']
ALLOWED_TARGETS = {'free flowing','light delay','moderate delay','heavy delay'}

if len(sys.argv) < 2:
    print("Usage: python scripts/validate_submission_strict.py <path-to-submission.csv>")
    sys.exit(1)

path = sys.argv[1]
if not os.path.exists(path):
    print(f"❌ File not found: {path}")
    sys.exit(2)

# Check header exactness
with open(path, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split(',')

if header != REQUIRED_COLUMNS:
    print(f"❌ Invalid header. Expected {REQUIRED_COLUMNS}, got {header}")
    sys.exit(3)

# Read data
try:
    df = pd.read_csv(path)
except Exception as e:
    print(f"❌ Could not read CSV: {e}")
    sys.exit(4)

# Check shape vs SampleSubmission
sample_path = os.path.join(os.path.dirname(__file__), '..', 'SampleSubmission.csv')
sample_path = os.path.abspath(sample_path)
if not os.path.exists(sample_path):
    print("⚠️ SampleSubmission.csv not found; skipping row count check")
else:
    sample = pd.read_csv(sample_path)
    if len(df) != len(sample):
        print(f"❌ Row count mismatch. Expected {len(sample)}, got {len(df)}")
        sys.exit(5)

# Check ID set matches sample (if available)
if 'sample' in locals():
    sample_ids = set(sample['ID'])
    df_ids = set(df['ID'])
    missing = sample_ids - df_ids
    extra = df_ids - sample_ids
    if missing:
        print(f"❌ Missing IDs: {len(missing)}")
        sys.exit(6)
    if extra:
        print(f"❌ Extra IDs: {len(extra)}")
        sys.exit(7)

# Check target values
if not set(df['Target'].astype(str).str.lower().unique()).issubset(ALLOWED_TARGETS):
    print(f"❌ Invalid values found in Target column")
    print(f"Allowed: {sorted(ALLOWED_TARGETS)}")
    sys.exit(8)

# Check target_accuracy values
if not set(df['Target_Accuracy'].astype(str).str.lower().unique()).issubset(ALLOWED_TARGETS):
    print(f"❌ Invalid values found in Target_Accuracy column")
    print(f"Allowed: {sorted(ALLOWED_TARGETS)}")
    sys.exit(9)

print("✓ Submission is valid: header, rows, IDs, and label values are OK")
