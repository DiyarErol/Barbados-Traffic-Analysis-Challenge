"""
Development 4: Temporal Smoothing & Consistency
Ensures consecutive segments have consistent predictions
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def apply_temporal_smoothing(submission_df, window_size=3):
    """
    Apply temporal smoothing to consecutive segments at same location
    Consecutive segments should have similar congestion levels
    """
    
    print('🔍 GELİŞTİRME 4: Temporal Smoothing & Consistency')
    print('='*60)
    
    print(f'\n📊 Smoothing öncesi dağılım:')
    dist_before = submission_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist_before.items():
        pct = count / len(submission_df) * 100
        print(f'  {label}: {count:,} ({pct:.1f}%)')
    
    # Parse IDs and organize by location + segment
    submission_df['segment_id'] = submission_df['ID'].apply(
        lambda x: int(x.split('_')[2])
    )
    
    submission_df['location'] = submission_df['ID'].apply(
        lambda x: ' '.join([p for i, p in enumerate(x.split('_')) 
                           if i >= 3 and p != 'congestion'])
    )
    
    submission_df['rating_type'] = submission_df['ID'].apply(
        lambda x: x.split('_')[-2]
    )
    
    # Congestion severity mapping
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    
    reverse_map = {v: k for k, v in severity_map.items()}
    
    submission_df['severity'] = submission_df['Target'].map(severity_map)
    
    # Group by location and rating type
    smoothed_data = []
    changes_made = 0
    
    for (location, rating_type), group in submission_df.groupby(['location', 'rating_type']):
        # Sort by segment_id
        group = group.sort_values('segment_id').copy()
        
        if len(group) < window_size:
            # Too few segments, keep original
            smoothed_data.append(group)
            continue
        
        # Apply rolling median for smoothing
        original_severity = group['severity'].values.copy()
        smoothed_severity = group['severity'].rolling(
            window=window_size, 
            center=True, 
            min_periods=1
        ).median().round().astype(int).values
        
        # Count changes
        changes_made += (original_severity != smoothed_severity).sum()
        
        # Update severity
        group['severity'] = smoothed_severity
        group['Target'] = group['severity'].map(reverse_map)
        group['Target_Accuracy'] = group['Target']
        
        smoothed_data.append(group)
    
    # Combine back
    smoothed_df = pd.concat(smoothed_data, ignore_index=True)
    
    # Sort by original order (ID)
    smoothed_df = smoothed_df.sort_index()
    
    print(f'\n✅ Temporal smoothing uygulandı (window={window_size})')
    print(f'📝 {changes_made} prediction değiştirildi ({changes_made/len(submission_df)*100:.1f}%)')
    
    print(f'\n📊 Smoothing sonrası dağılım:')
    dist_after = smoothed_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist_after.items():
        pct = count / len(smoothed_df) * 100
        change = count - dist_before.get(label, 0)
        symbol = '↑' if change > 0 else '↓' if change < 0 else '='
        print(f'  {label}: {count:,} ({pct:.1f}%) {symbol}{abs(change)}')
    
    # Return only needed columns
    return smoothed_df[['ID', 'Target', 'Target_Accuracy']]


def apply_anomaly_detection(submission_df):
    """
    Detect and fix anomalies - sudden spikes in congestion that don't make sense
    """
    
    print('\n🔍 Anomaly Detection...')
    
    # Parse data
    submission_df['segment_id'] = submission_df['ID'].apply(
        lambda x: int(x.split('_')[2])
    )
    
    submission_df['location'] = submission_df['ID'].apply(
        lambda x: ' '.join([p for i, p in enumerate(x.split('_')) 
                           if i >= 3 and p != 'congestion'])
    )
    
    submission_df['rating_type'] = submission_df['ID'].apply(
        lambda x: x.split('_')[-2]
    )
    
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    
    reverse_map = {v: k for k, v in severity_map.items()}
    submission_df['severity'] = submission_df['Target'].map(severity_map)
    
    # Detect anomalies
    fixed_data = []
    anomalies_fixed = 0
    
    for (location, rating_type), group in submission_df.groupby(['location', 'rating_type']):
        group = group.sort_values('segment_id').copy()
        
        if len(group) < 3:
            fixed_data.append(group)
            continue
        
        # Check for isolated spikes
        severities = group['severity'].values
        
        for i in range(1, len(severities) - 1):
            prev_sev = severities[i-1]
            curr_sev = severities[i]
            next_sev = severities[i+1]
            
            # If current is much higher than neighbors, it's an anomaly
            if curr_sev >= prev_sev + 2 and curr_sev >= next_sev + 2:
                # Fix: use average of neighbors
                new_sev = int(round((prev_sev + next_sev) / 2))
                severities[i] = new_sev
                anomalies_fixed += 1
        
        group['severity'] = severities
        group['Target'] = group['severity'].map(reverse_map)
        group['Target_Accuracy'] = group['Target']
        
        fixed_data.append(group)
    
    fixed_df = pd.concat(fixed_data, ignore_index=True).sort_index()
    
    print(f'✅ {anomalies_fixed} anomaly düzeltildi')
    
    return fixed_df[['ID', 'Target', 'Target_Accuracy']]


def generate_improved_submission_v4():
    """Generate submission with temporal smoothing and consistency"""
    
    print('\n' + '='*60)
    print('📝 SUBMISSION V4: Temporal Smoothing + Anomaly Fix')
    print('='*60)
    
    # Load previous submission
    submission_df = pd.read_csv('submission.csv')
    print(f'\n✓ V3 submission yüklendi: {len(submission_df):,} satır')
    
    # Apply temporal smoothing
    smoothed_df = apply_temporal_smoothing(submission_df.copy(), window_size=3)
    
    # Apply anomaly detection
    final_df = apply_anomaly_detection(smoothed_df.copy())
    
    # Save
    final_df.to_csv('submission.csv', index=False)
    
    print(f'\n✅ Submission güncellendi: submission.csv')
    print(f'📊 Toplam: {len(final_df):,} satır')
    
    # Final distribution
    print(f'\n📈 Final Tahmin Dağılımı:')
    dist = final_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(final_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    # Validation
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = set(sample_df['ID'])
    submitted_ids = set(final_df['ID'])
    missing = required_ids - submitted_ids
    
    if len(missing) > 0:
        print(f'\n⚠️ UYARI: {len(missing)} eksik ID var!')
    else:
        print(f'\n✅ Validation: Tüm ID\'ler mevcut (0 eksik)')
    
    return final_df


if __name__ == '__main__':
    submission = generate_improved_submission_v4()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 4 TAMAMLANDI')
    print('   → Temporal smoothing uygulandı')
    print('   → Anomaly detection yapıldı')
    print('   → Consecutive segments daha tutarlı')
    print('='*60)
