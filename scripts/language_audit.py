import os
import re
import chardet
import csv
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
REPORTS_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Text file extensions to consider
TEXT_EXTS = {
    '.py','.md','.txt','.csv','.json','.ipynb','.yml','.yaml','.ini','.cfg','.log'
}

# Common Turkish keywords to flag
TURKISH_KEYWORDS = [
    'merhaba','kullanım','çalıştırma','özellik','hedef','doğruluk','yüzde','sinyal','zaman','geliştirme',
    'analiz','rapor','izleme','doğrulama','dosya','klasör','ağır','hafif','orta','gecikme','üretildi','kaydedildi',
    'sayısı','satır','başlık','kural','uyum','gözlem','özet','yapı','pipeline','oluşturuldu','çalıştı','çalıştırıldı'
]

# Simple ASCII and Turkish letter detection
TURKISH_CHARS = set('çğıöşüÇĞİÖŞÜ')

results = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    # Skip virtual/env, cache, binary folders
    skip_dirs = {'__pycache__','venv','.git','.azure','videos','.ipynb_checkpoints'}
    if any(sd in dirpath for sd in skip_dirs):
        continue
    for fname in filenames:
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(dirpath, fname)
        if ext not in TEXT_EXTS:
            # Skip obvious binaries
            continue
        try:
            with open(fpath, 'rb') as fb:
                raw = fb.read()
            enc = chardet.detect(raw).get('encoding') or 'utf-8'
            text = raw.decode(enc, errors='ignore')
        except Exception as e:
            results.append({'file': os.path.relpath(fpath, ROOT), 'status':'read_error', 'detail':str(e), 'has_turkish_chars':False, 'turkish_hits':0})
            continue
        has_turkish_chars = any(c in TURKISH_CHARS for c in text)
        hits = 0
        low = text.lower()
        for kw in TURKISH_KEYWORDS:
            hits += low.count(kw)
        status = 'ok'
        if has_turkish_chars or hits>0:
            status = 'needs_english'
        results.append({'file': os.path.relpath(fpath, ROOT), 'status':status, 'detail':'', 'has_turkish_chars':has_turkish_chars, 'turkish_hits':hits})

# Write CSV report
csv_path = os.path.join(REPORTS_DIR, f'language_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file','status','has_turkish_chars','turkish_hits','detail'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# Write summary
summary = []
summary.append(f"# Language Audit\n\nGenerated: {datetime.now().isoformat()}\n")
summary.append(f"- Files scanned: {len(results)}\n")
needs = [r for r in results if r['status']=='needs_english']
summary.append(f"- Files flagged (needs English): {len(needs)}\n")
summary.append("\n## Flagged Files\n")
for r in needs[:50]:
    summary.append(f"- {r['file']} | turkish_chars={r['has_turkish_chars']} | hits={r['turkish_hits']}")

md_path = os.path.join(REPORTS_DIR, f'language_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary))

print(f"✓ Language audit complete: {csv_path}")
print(f"✓ Summary: {md_path}")
