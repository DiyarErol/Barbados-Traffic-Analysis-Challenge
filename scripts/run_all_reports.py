import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)

scripts = [
    os.path.join(ROOT, 'scripts', 'submission_indexer.py'),
    os.path.join(ROOT, 'scripts', 'data_profile.py'),
]

ok = True
for s in scripts:
    print(f"\n>>> Running {os.path.basename(s)}")
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        ok = False
        print(f"Script failed: {s}")

sys.exit(0 if ok else 1)
