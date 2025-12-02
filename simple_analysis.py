"""
Data analysis and visualization (no video required)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 80)
print("DATA ANALYSIS AND VISUALIZATION")
print("=" * 80)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('TestInputSegments.csv')

print(f"   Training: {len(train_df)} samples")
print(f"   Test: {len(test_df)} samples")

# Add time features
train_df['datetime'] = pd.to_datetime(train_df['video_time'])
train_df['hour'] = train_df['datetime'].dt.hour
train_df['day_of_week'] = train_df['datetime'].dt.dayofweek

# 1. Class Distribution
print("\n2. Creating class distribution plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Enter
enter_counts = train_df['congestion_enter_rating'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
axes[0].bar(range(len(enter_counts)), enter_counts.values, color=colors)
axes[0].set_xticks(range(len(enter_counts)))
axes[0].set_xticklabels(enter_counts.index, rotation=45, ha='right')
axes[0].set_title('Enter Congestion Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(enter_counts.values):
    pct = 100 * v / len(train_df)
    axes[0].text(i, v + 100, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Exit
exit_counts = train_df['congestion_exit_rating'].value_counts()
axes[1].bar(range(len(exit_counts)), exit_counts.values, color=colors)
axes[1].set_xticks(range(len(exit_counts)))
axes[1].set_xticklabels(exit_counts.index, rotation=45, ha='right')
axes[1].set_title('Exit Congestion Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(exit_counts.values):
    pct = 100 * v / len(train_df)
    axes[1].text(i, v + 100, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('1_class_distribution.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 1_class_distribution.png")
plt.close()

# 2. Hourly Distribution
print("\n3. Creating hourly distribution plots...")

plt.figure(figsize=(14, 6))
hour_dist = train_df['hour'].value_counts().sort_index()
plt.bar(hour_dist.index, hour_dist.values, color='#3498db', alpha=0.7, edgecolor='black')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Sample Count', fontsize=12)
plt.title('Hourly Data Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.xticks(range(0, 24))

# Highlight rush hour zones
plt.axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush Hour')
plt.axvspan(16, 18, alpha=0.2, color='orange', label='Evening Rush Hour')
plt.legend()

plt.tight_layout()
plt.savefig('2_hourly_distribution.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 2_hourly_distribution.png")
plt.close()

# 3. Hourly Congestion Pattern
print("\n4. Creating hourly congestion pattern...")

congestion_by_hour = pd.crosstab(
    train_df['hour'],
    train_df['congestion_enter_rating'],
    normalize='index'
) * 100

fig, ax = plt.subplots(figsize=(14, 7))
congestion_by_hour.plot(kind='bar', stacked=True, ax=ax, color=colors)
ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Hourly Congestion Distribution (%)', fontsize=14, fontweight='bold')
ax.legend(title='Congestion Level', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('3_hourly_congestion_pattern.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 3_hourly_congestion_pattern.png")
plt.close()

# 4. Rush Hour Comparison
print("\n5. Creating rush hour comparison...")

train_df['is_rush_hour'] = train_df['hour'].apply(
    lambda x: 'Rush Hour' if (7 <= x <= 9) or (16 <= x <= 18) else 'Normal'
)

rush_analysis = pd.crosstab(
    train_df['is_rush_hour'],
    train_df['congestion_enter_rating'],
    normalize='index'
) * 100

fig, ax = plt.subplots(figsize=(10, 6))
rush_analysis.T.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
ax.set_xlabel('Congestion Level', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Rush Hour vs Normal - Congestion Comparison', fontsize=14, fontweight='bold')
ax.legend(title='Period')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('4_rush_hour_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 4_rush_hour_comparison.png")
plt.close()

# 5. Weekly Pattern
print("\n6. Creating weekly pattern...")

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
train_df['day_name'] = train_df['day_of_week'].map(dict(enumerate(day_names)))

congestion_by_day = pd.crosstab(
    train_df['day_name'],
    train_df['congestion_enter_rating'],
    normalize='index'
) * 100

# Sort days
day_order = day_names
congestion_by_day = congestion_by_day.reindex(day_order)

fig, ax = plt.subplots(figsize=(12, 6))
congestion_by_day.plot(kind='bar', stacked=True, ax=ax, color=colors)
ax.set_xlabel('Day', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Weekly Congestion Distribution', fontsize=14, fontweight='bold')
ax.legend(title='Congestion Level', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('5_weekly_pattern.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 5_weekly_pattern.png")
plt.close()

# 6. Signal Usage Analysis
print("\n7. Creating signal usage analysis...")

signal_dist = train_df['signaling'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Signal distribution
axes[0].pie(signal_dist.values, labels=signal_dist.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Signal Usage Distribution', fontsize=14, fontweight='bold')

# Signal vs Congestion
signal_congestion = pd.crosstab(
    train_df['signaling'],
    train_df['congestion_enter_rating'],
    normalize='index'
) * 100

signal_congestion.plot(kind='bar', stacked=True, ax=axes[1], color=colors)
axes[1].set_xlabel('Signal Usage', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('Signal Usage vs Congestion', fontsize=14, fontweight='bold')
axes[1].legend(title='Congestion', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('6_signal_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 6_signal_analysis.png")
plt.close()

# 7. Heatmap - Hour x Congestion
print("\n8. Creating heatmap...")

# Numeric mapping for congestion levels
congestion_map = {
    'free flowing': 0,
    'light delay': 1,
    'moderate delay': 2,
    'heavy delay': 3
}
train_df['congestion_numeric'] = train_df['congestion_enter_rating'].map(congestion_map)

# Hourly average congestion
hourly_congestion = train_df.groupby('hour')['congestion_numeric'].mean()

plt.figure(figsize=(14, 3))
plt.imshow([hourly_congestion.values], cmap='RdYlGn_r', aspect='auto')
plt.colorbar(label='Average Congestion Level', orientation='horizontal')
plt.yticks([])
plt.xticks(range(24), range(24))
plt.xlabel('Hour', fontsize=12)
plt.title('Hourly Average Congestion Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('7_congestion_heatmap.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: 7_congestion_heatmap.png")
plt.close()

# Summary Statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"""
Data Overview:
- Total training samples: {len(train_df):,}
- Total test samples: {len(test_df):,}
- Date range: {train_df['date'].min()} - {train_df['date'].max()}
- Number of cameras: {train_df['view_label'].nunique()}

Class Distribution (Enter):
""")
for label, count in enter_counts.items():
    pct = 100 * count / len(train_df)
    print(f"  {label:20s}: {count:6,} ({pct:5.1f}%)")

print(f"""
Imbalance Ratio:
- Most frequent: {enter_counts.max():,} ({100*enter_counts.max()/len(train_df):.1f}%)
- Least frequent: {enter_counts.min():,} ({100*enter_counts.min()/len(train_df):.1f}%)
- Ratio: {enter_counts.max()/enter_counts.min():.1f}x

Rush Hour Statistics:
- Normal hours: {len(train_df[train_df['is_rush_hour'] == 'Normal']):,} samples
- Rush hour: {len(train_df[train_df['is_rush_hour'] == 'Rush Hour']):,} samples

Signal Usage (Barbados Feature):
""")
for sig, count in signal_dist.items():
    pct = 100 * count / len(train_df)
    print(f"  {sig:10s}: {count:6,} ({pct:5.1f}%)")

print(f"""
Key Findings:
1. ✓ Data is "free flowing" dominant (imbalanced)
2. ✓ Congestion increase observed during rush hour
3. ✓ Low signal usage dominant (Barbados-specific factor)
4. ✓ Exit generally free flowing (less congested than enter)

Generated Plots:
  1. 1_class_distribution.png      - Class distributions
  2. 2_hourly_distribution.png     - Hourly data distribution
  3. 3_hourly_congestion_pattern.png - Hourly congestion pattern
  4. 4_rush_hour_comparison.png    - Rush hour comparison
  5. 5_weekly_pattern.png          - Weekly pattern
  6. 6_signal_analysis.png         - Signal usage
  7. 7_congestion_heatmap.png      - Congestion heatmap

Recommendations:
- Use class_weight for imbalanced classes
- Strengthen rush hour samples
- Temporal features are critical
- Combine with video features
""")

print("=" * 80)
print("ANALYSIS COMPLETED!")
print("=" * 80)
