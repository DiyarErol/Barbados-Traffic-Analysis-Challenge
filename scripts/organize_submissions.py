import os
import glob
import shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
SRC_PATTERN = os.path.join(ROOT, 'submission*.csv')
DEST_DIR = os.path.join(ROOT, 'submissions')

os.makedirs(DEST_DIR, exist_ok=True)

count = 0
for path in sorted(glob.glob(SRC_PATTERN)):
    name = os.path.basename(path)
    dest = os.path.join(DEST_DIR, name)
    try:
        if not os.path.exists(dest):
            shutil.copyfile(path, dest)
            count += 1
    except Exception as e:
        print(f"⚠️ Copy failed for {name}: {e}")

print(f"Copied {count} submissions into {DEST_DIR}")
