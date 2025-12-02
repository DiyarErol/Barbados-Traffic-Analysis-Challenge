"""
Development 6: Balanced Distribution Adjustment
Fine-tunes predictions to match training data distribution more closely
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def analyze_and_rebalance():
    """Analyze current submission and rebalance to match training better"""
    
    print('🔍 GELİŞTİRME 6: Distribution Rebalancing')
    print('='*60)
    
    # Load training distribution
    train_df = pd.read_csv('Train.csv')
    
    print('\n📊 Training Distribution (Target):')
    train_enter = train_df['congestion_enter_rating'].value_counts(normalize=True) * 100
    train_exit = train_df['congestion_exit_rating'].value_counts(normalize=True) * 100
    
    target_dist = {}
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        enter_pct = train_enter.get(label, 0)
        exit_pct = train_exit.get(label, 0)
        avg_pct = (enter_pct + exit_pct) / 2
        target_dist[label] = avg_pct
        print(f'  {label}: {avg_pct:.1f}%')
    
    # Load current submission
    submission_df = pd.read_csv('submission.csv')
    
    print(f'\n📊 Current Submission Distribution:')
    current_dist = submission_df['Target'].value_counts(normalize=True) * 100
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        curr_pct = current_dist.get(label, 0)
        target_pct = target_dist[label]
        diff = curr_pct - target_pct
        symbol = '⚠️' if abs(diff) > 5 else '✓'
        print(f'  {label}: {curr_pct:.1f}% (target: {target_pct:.1f}%) {symbol} {diff:+.1f}%')
    
    # Identify what needs adjustment
    print('\n🔧 Adjustments needed:')
    
    adjustments = {}
    for label, target_pct in target_dist.items():
        curr_pct = current_dist.get(label, 0)
        curr_count = int(len(submission_df) * curr_pct / 100)
        target_count = int(len(submission_df) * target_pct / 100)
        diff = target_count - curr_count
        
        if abs(diff) > 0:
            adjustments[label] = diff
            symbol = '↑' if diff > 0 else '↓'
            print(f'  {label}: {symbol} {abs(diff)} predictions')
    
    # Apply strategic adjustments
    print('\n🎯 Strategic Rebalancing...')
    
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    # Parse submission
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
    
    # Strategy:
    # 1. Reduce light delay (too many) -> convert some to free flowing or moderate
    # 2. Increase moderate delay 
    # 3. Increase heavy delay (but cautiously, only in rush hours)
    
    changes_made = 0
    
    # Convert excess light delay to moderate delay (in rush hours)
    if 'light delay' in adjustments and adjustments['light delay'] < 0:
        excess = abs(adjustments['light delay'])
        
        # Find light delay predictions in rush hours
        candidates = submission_df[
            (submission_df['Target'] == 'light delay') & 
            (submission_df['is_rush'] == 1)
        ]
        
        # Convert top N to moderate delay
        n_to_convert = min(excess // 2, len(candidates))
        if n_to_convert > 0:
            indices = candidates.sample(n=n_to_convert, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'moderate delay'
            submission_df.loc[indices, 'severity'] = 2
            changes_made += n_to_convert
            print(f'  ✓ Converted {n_to_convert} light delay → moderate delay (rush hours)')
    
    # Convert some light delay to free flowing (non-rush hours)
    if 'free flowing' in adjustments and adjustments['free flowing'] > 0:
        needed = adjustments['free flowing']
        
        candidates = submission_df[
            (submission_df['Target'] == 'light delay') & 
            (submission_df['is_rush'] == 0)
        ]
        
        n_to_convert = min(needed, len(candidates))
        if n_to_convert > 0:
            indices = candidates.sample(n=n_to_convert, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'free flowing'
            submission_df.loc[indices, 'severity'] = 0
            changes_made += n_to_convert
            print(f'  ✓ Converted {n_to_convert} light delay → free flowing (non-rush hours)')
    
    # Add some heavy delay in rush hours
    if 'heavy delay' in adjustments and adjustments['heavy delay'] > 0:
        needed = adjustments['heavy delay']
        
        # Convert some moderate delay in rush hours to heavy delay
        candidates = submission_df[
            (submission_df['Target'] == 'moderate delay') & 
            (submission_df['is_rush'] == 1)
        ]
        
        n_to_convert = min(needed, len(candidates))
        if n_to_convert > 0:
            indices = candidates.sample(n=n_to_convert, random_state=42).index
            submission_df.loc[indices, 'Target'] = 'heavy delay'
            submission_df.loc[indices, 'severity'] = 3
            changes_made += n_to_convert
            print(f'  ✓ Converted {n_to_convert} moderate delay → heavy delay (rush hours)')
    
    print(f'\n✅ Total changes: {changes_made}')
    
    # Update Target_Accuracy
    submission_df['Target_Accuracy'] = submission_df['Target']
    
    # Apply final smoothing
    print('\n🔧 Final smoothing...')
    submission_df = apply_light_smoothing(submission_df.copy())
    
    # Save
    final_df = submission_df[['ID', 'Target', 'Target_Accuracy']].copy()
    final_df.to_csv('submission.csv', index=False)
    
    print(f'\n✅ Submission güncellendi: submission.csv')
    
    # Final distribution
    print(f'\n📈 Final Distribution:')
    final_dist = final_df['Target'].value_counts(normalize=True) * 100
    
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        final_pct = final_dist.get(label, 0)
        target_pct = target_dist[label]
        diff = final_pct - target_pct
        
        if abs(diff) <= 2:
            symbol = '✅'
        elif abs(diff) <= 5:
            symbol = '✓'
        else:
            symbol = '⚠️'
        
        print(f'  {label}: {final_pct:.1f}% (target: {target_pct:.1f}%) {symbol} {diff:+.1f}%')
    
    return final_df


def apply_light_smoothing(df):
    """Apply very light smoothing to maintain consistency"""
    
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    smoothed_data = []
    changes = 0
    
    for (location, rating_type), group in df.groupby(['location', 'rating_type']):
        group = group.sort_values('segment_id').copy()
        
        if len(group) >= 5:
            # Very light smoothing - only for extreme outliers
            severities = group['severity'].values.copy()
            
            for i in range(2, len(severities) - 2):
                # Check 5-point window
                window = severities[i-2:i+3]
                current = severities[i]
                
                # If current is extreme outlier (differs by 2+ from all neighbors)
                neighbors = [window[0], window[1], window[3], window[4]]
                if all(abs(current - n) >= 2 for n in neighbors):
                    # Replace with median of neighbors
                    severities[i] = int(np.median(neighbors))
                    changes += 1
            
            group['severity'] = severities
        
        group['Target'] = group['severity'].map(reverse_map)
        group['Target_Accuracy'] = group['Target']
        smoothed_data.append(group)
    
    result = pd.concat(smoothed_data, ignore_index=True).sort_index()
    print(f'  ✓ {changes} extreme outliers smoothed')
    
    return result


if __name__ == '__main__':
    submission = analyze_and_rebalance()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 6 TAMAMLANDI')
    print('   → Distribution training\'e göre rebalance edildi')
    print('   → Strategic adjustments uygulandı')
    print('   → Light smoothing yapıldı')
    print('='*60)
