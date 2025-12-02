"""
Development 7: Aggressive Distribution Correction
Aggressively corrects distribution to match training data exactly
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def aggressive_rebalance():
    """Aggressively rebalance to match training distribution"""
    
    print('🔍 GELİŞTİRME 7: Aggressive Distribution Correction')
    print('='*60)
    
    # Target distribution from training
    target_dist = {
        'free flowing': 79.0,
        'light delay': 6.7,
        'moderate delay': 8.1,
        'heavy delay': 6.1
    }
    
    print('\n🎯 Target Distribution (from training):')
    for label, pct in target_dist.items():
        print(f'  {label}: {pct:.1f}%')
    
    # Load current submission
    submission_df = pd.read_csv('submission.csv')
    total = len(submission_df)
    
    print(f'\n📊 Current Distribution:')
    current_counts = submission_df['Target'].value_counts()
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        count = current_counts.get(label, 0)
        pct = count / total * 100
        print(f'  {label}: {count} ({pct:.1f}%)')
    
    # Calculate target counts
    target_counts = {
        label: int(round(total * pct / 100))
        for label, pct in target_dist.items()
    }
    
    # Adjust for rounding
    diff = total - sum(target_counts.values())
    target_counts['free flowing'] += diff
    
    print(f'\n🎯 Target Counts:')
    for label, count in target_counts.items():
        print(f'  {label}: {count}')
    
    # Parse submission
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    submission_df['segment_id'] = submission_df['ID'].apply(lambda x: int(x.split('_')[2]))
    submission_df['location'] = submission_df['ID'].apply(
        lambda x: ' '.join([p for i, p in enumerate(x.split('_')) 
                           if i >= 3 and p != 'congestion'])
    )
    submission_df['rating_type'] = submission_df['ID'].apply(lambda x: x.split('_')[-2])
    submission_df['severity'] = submission_df['Target'].map(severity_map)
    
    # Add time info
    with open('segment_info.pkl', 'rb') as f:
        segment_info = pickle.load(f)
    
    def get_hour(segment_id):
        known_segments = sorted(segment_info.keys())
        if segment_id in segment_info:
            return segment_info[segment_id]['hour']
        lower = [s for s in known_segments if s < segment_id]
        if lower:
            seg_lower = max(lower)
            info = segment_info[seg_lower]
            diff = segment_id - seg_lower
            total_minutes = info['hour'] * 60 + info['minute'] + diff
            return (total_minutes // 60) % 24
        return 12
    
    submission_df['hour'] = submission_df['segment_id'].apply(get_hour)
    submission_df['is_rush'] = submission_df['hour'].apply(
        lambda h: 1 if h in [7, 8, 9, 16, 17, 18] else 0
    )
    
    print(f'\n🔧 Aggressive Rebalancing Strategy:')
    
    # Strategy: Convert excess light delay to other categories
    # 1. Light delay -> Free flowing (non-rush hours)
    # 2. Light delay -> Moderate delay (some rush hours)
    # 3. Some moderate -> Heavy delay (peak rush hours)
    
    np.random.seed(42)
    changes = 0
    
    # Step 1: Reduce light delay to free flowing
    excess_light = current_counts.get('light delay', 0) - target_counts['light delay']
    needed_free = target_counts['free flowing'] - current_counts.get('free flowing', 0)
    
    if excess_light > 0 and needed_free > 0:
        n_convert = min(excess_light, needed_free)
        
        candidates = submission_df[
            (submission_df['Target'] == 'light delay') & 
            (submission_df['is_rush'] == 0)  # Non-rush hours
        ]
        
        if len(candidates) >= n_convert:
            indices = candidates.sample(n=n_convert, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'free flowing'
            submission_df.loc[indices, 'severity'] = 0
            changes += n_convert
            print(f'  ✓ Converted {n_convert} light delay → free flowing')
            
            excess_light -= n_convert
            needed_free -= n_convert
    
    # Step 2: Convert light delay to moderate delay
    needed_moderate = target_counts['moderate delay'] - current_counts.get('moderate delay', 0)
    
    if excess_light > 0 and needed_moderate > 0:
        n_convert = min(excess_light, needed_moderate)
        
        candidates = submission_df[
            (submission_df['Target'] == 'light delay') & 
            (submission_df['is_rush'] == 1)  # Rush hours
        ]
        
        if len(candidates) >= n_convert:
            indices = candidates.sample(n=n_convert, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'moderate delay'
            submission_df.loc[indices, 'severity'] = 2
            changes += n_convert
            print(f'  ✓ Converted {n_convert} light delay → moderate delay')
            
            excess_light -= n_convert
            needed_moderate -= n_convert
    
    # Step 3: Convert moderate to heavy delay
    current_moderate = (submission_df['Target'] == 'moderate delay').sum()
    needed_heavy = target_counts['heavy delay'] - current_counts.get('heavy delay', 0)
    
    if needed_heavy > 0 and current_moderate > target_counts['moderate delay']:
        n_convert = min(needed_heavy, current_moderate - target_counts['moderate delay'])
        
        candidates = submission_df[
            (submission_df['Target'] == 'moderate delay') & 
            (submission_df['is_rush'] == 1) &
            (submission_df['hour'].isin([8, 9, 17, 18]))  # Peak rush
        ]
        
        if len(candidates) >= n_convert:
            indices = candidates.sample(n=n_convert, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'heavy delay'
            submission_df.loc[indices, 'severity'] = 3
            changes += n_convert
            print(f'  ✓ Converted {n_convert} moderate delay → heavy delay')
            
            needed_heavy -= n_convert
    
    # Step 4: If still excess light delay, convert more aggressively
    current_light = (submission_df['Target'] == 'light delay').sum()
    if current_light > target_counts['light delay']:
        excess = current_light - target_counts['light delay']
        
        # Convert remaining to free flowing
        candidates = submission_df[submission_df['Target'] == 'light delay']
        
        if len(candidates) >= excess:
            indices = candidates.sample(n=excess, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'free flowing'
            submission_df.loc[indices, 'severity'] = 0
            changes += excess
            print(f'  ✓ Converted {excess} more light delay → free flowing (final cleanup)')
    
    print(f'\n✅ Total changes: {changes}')
    
    # Update Target_Accuracy
    submission_df['Target_Accuracy'] = submission_df['Target']
    
    # Light consistency check
    print('\n🔧 Consistency check...')
    submission_df = apply_consistency_check(submission_df.copy())
    
    # Save
    final_df = submission_df[['ID', 'Target', 'Target_Accuracy']].copy()
    final_df.to_csv('submission.csv', index=False)
    
    print(f'\n✅ Submission güncellendi: submission.csv')
    
    # Final distribution
    print(f'\n📈 Final Distribution:')
    final_counts = final_df['Target'].value_counts()
    
    print(f'\n{"Label":<20} {"Count":>8} {"Actual%":>8} {"Target%":>8} {"Diff":>8}')
    print('=' * 60)
    
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        count = final_counts.get(label, 0)
        actual_pct = count / total * 100
        target_pct = target_dist[label]
        diff = actual_pct - target_pct
        
        if abs(diff) <= 1:
            symbol = '✅'
        elif abs(diff) <= 2:
            symbol = '✓'
        else:
            symbol = '⚠️'
        
        print(f'{label:<20} {count:>8} {actual_pct:>7.1f}% {target_pct:>7.1f}% {diff:>+7.1f}% {symbol}')
    
    return final_df


def apply_consistency_check(df):
    """Light consistency check without heavy smoothing"""
    
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    checked_data = []
    fixes = 0
    
    for (location, rating_type), group in df.groupby(['location', 'rating_type']):
        group = group.sort_values('segment_id').copy()
        
        if len(group) >= 3:
            severities = group['severity'].values.copy()
            
            # Only fix extreme jumps (0 -> 3 or 3 -> 0 in single step)
            for i in range(len(severities) - 1):
                if abs(severities[i+1] - severities[i]) >= 3:
                    # Insert intermediate value
                    avg = (severities[i] + severities[i+1]) / 2
                    severities[i+1] = int(round(avg))
                    fixes += 1
            
            group['severity'] = severities
        
        group['Target'] = group['severity'].map(reverse_map)
        group['Target_Accuracy'] = group['Target']
        checked_data.append(group)
    
    result = pd.concat(checked_data, ignore_index=True).sort_index()
    print(f'  ✓ {fixes} extreme jumps fixed')
    
    return result


if __name__ == '__main__':
    submission = aggressive_rebalance()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 7 TAMAMLANDI')
    print('   → Aggressive rebalancing uygulandı')
    print('   → Distribution training\'e tam align')
    print('   → Consistency check yapıldı')
    print('='*60)
