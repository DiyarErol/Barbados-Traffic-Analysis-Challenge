import pandas as pd

print('='*60)
print('SUBMISSION VALIDATION')
print('='*60)

# Load files
sample = pd.read_csv('SampleSubmission.csv')
submission = pd.read_csv('submission.csv')
test_input = pd.read_csv('TestInputSegments.csv')

print(f'\n📊 File Sizes:')
print(f'  Sample Submission: {len(sample):,} rows')
print(f'  Our Submission: {len(submission):,} rows')
print(f'  Test Input: {len(test_input):,} rows')

# Check missing IDs
sample_ids = set(sample['ID'])
submission_ids = set(submission['ID'])
missing_ids = sample_ids - submission_ids

print(f'\n🔍 ID Check:')
print(f'  IDs in Sample: {len(sample_ids):,}')
print(f'  IDs in Submission: {len(submission_ids):,}')
print(f'  Missing IDs: {len(missing_ids):,}')

if missing_ids:
    print(f'\n⚠️ First 20 Missing IDs:')
    for i, mid in enumerate(list(missing_ids)[:20], 1):
        print(f'  {i}. {mid}')
    
    # Analyze missing IDs
    print(f'\n📋 cycle_phase distribution for missing IDs:')
    missing_df = test_input[test_input['ID_enter'].isin(missing_ids) | test_input['ID_exit'].isin(missing_ids)]
    if len(missing_df) > 0:
        print(missing_df['cycle_phase'].value_counts())
    else:
        # Check if missing IDs exist in test_input at all
        all_test_ids = set(test_input['ID_enter']).union(set(test_input['ID_exit']))
        truly_missing = missing_ids - all_test_ids
        print(f'\n❌ IDs not present in Test file: {len(truly_missing):,}')
        if truly_missing:
            print(f'\nFirst 10:')
            for i, mid in enumerate(list(truly_missing)[:10], 1):
                print(f'  {i}. {mid}')

# Check extra IDs
extra_ids = submission_ids - sample_ids
if extra_ids:
    print(f'\n⚠️ Extra IDs (present in submission but not in sample): {len(extra_ids):,}')
    for i, eid in enumerate(list(extra_ids)[:10], 1):
        print(f'  {i}. {eid}')

print('\n' + '='*60)
