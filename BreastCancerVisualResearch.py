import pandas as pd
import matplotlib.pyplot as plt

# Read and clean data
df = pd.read_csv('data.csv')
df_clean = df.dropna()

# Get counts
diagnosis_counts = df_clean['diagnosis'].value_counts()

# Create simple bar plot
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4']
bars = plt.bar(['Malignant (M)', 'Benign (B)'],
               [diagnosis_counts.get('M', 0), diagnosis_counts.get('B', 0)],
               color=colors, edgecolor='black', linewidth=2, alpha=0.8)

plt.title('Cancer Diagnosis Distribution', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of Cases', fontsize=13, fontweight='bold')
plt.xlabel('Diagnosis Type', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add percentage labels
total = sum(diagnosis_counts.values)
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height/total)*100
    plt.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{percentage:.1f}%',
            ha='center', va='center', fontweight='bold', fontsize=11, color='white')

plt.tight_layout()
plt.savefig('diagnosis_simple.png', dpi=300, bbox_inches='tight')
plt.show()