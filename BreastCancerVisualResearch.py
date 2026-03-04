import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("=" * 80)
print("BREAST CANCER DIAGNOSIS DATASET - ANALYSIS & PREPROCESSING")
print("=" * 80)

# Load the dataset
try:
    df = pd.read_csv('data.csv')
    print("\n✓ Dataset loaded successfully!")
except FileNotFoundError:
    print("ERROR: data.csv not found in current directory")
    exit()

# Display basic information
print("\n" + "=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n📋 First few rows:")
print(df.head())

print("\n📋 Last few rows:")
print(df.tail())

print("\n📋 Dataset Info:")
print(df.info())

# ============================================================================
# SECTION 2: MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("2. MISSING VALUES ANALYSIS")
print("=" * 80)

missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nMissing Values Count:")
print(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values found!")

print("\nMissing Values Percentage:")
print(missing_percent[missing_percent > 0] if missing_percent.sum() > 0 else "No missing values!")

# Remove rows with missing values
df_clean = df.dropna()
print(f"\n✓ Rows before cleaning: {df.shape[0]}")
print(f"✓ Rows after cleaning: {df_clean.shape[0]}")
print(f"✓ Rows removed: {df.shape[0] - df_clean.shape[0]}")

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS - DIAGNOSIS COLUMN
# ============================================================================

print("\n" + "=" * 80)
print("3. DIAGNOSIS COLUMN ANALYSIS")
print("=" * 80)

diagnosis_counts = df_clean['diagnosis'].value_counts()
diagnosis_percent = (df_clean['diagnosis'].value_counts() / len(df_clean)) * 100

print("\nDiagnosis Distribution:")
print(f"  M (Malignant): {diagnosis_counts.get('M', 0)} ({diagnosis_percent.get('M', 0):.2f}%)")
print(f"  B (Benign): {diagnosis_counts.get('B', 0)} ({diagnosis_percent.get('B', 0):.2f}%)")

print("\n📊 Complete Value Counts:")
print(diagnosis_counts)

# ============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS - AREA_MEAN COLUMN
# ============================================================================

print("\n" + "=" * 80)
print("4. AREA_MEAN COLUMN ANALYSIS")
print("=" * 80)

area_mean_stats = df_clean['area_mean'].describe()
print("\nArea_Mean Statistics:")
print(area_mean_stats)

print("\nArea_Mean by Diagnosis:")
print(df_clean.groupby('diagnosis')['area_mean'].describe())

# ============================================================================
# SECTION 5: GENERAL DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("5. DESCRIPTIVE STATISTICS - ALL NUMERICAL FEATURES")
print("=" * 80)

print("\n📊 Complete Statistical Summary:")
print(df_clean.describe())

# ============================================================================
# SECTION 6: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("6. DATA PREPROCESSING & ENCODING")
print("=" * 80)

# Create a copy for preprocessing
df_processed = df_clean.copy()

# Encode diagnosis column (M=1, B=0)
print("\n🔄 Step 1: Categorical Encoding")
print(f"  Original diagnosis values: {df_processed['diagnosis'].unique()}")

diagnosis_encoding = {'M': 1, 'B': 0}
df_processed['diagnosis'] = df_processed['diagnosis'].map(diagnosis_encoding)

print(f"  Encoded diagnosis values: {df_processed['diagnosis'].unique()}")
print(f"  M (Malignant) → 1")
print(f"  B (Benign) → 0")

# Separate features and target
print("\n🔄 Step 2: Feature Separation")
X = df_processed.drop(['id', 'diagnosis'], axis=1)
y = df_processed['diagnosis']

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Features: {list(X.columns[:5])}... (and {len(X.columns) - 5} more)")

# Feature Scaling/Normalization
print("\n🔄 Step 3: Feature Scaling (StandardScaler)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"  ✓ Features normalized to mean=0, std=1")
print(f"  Sample means (should be ~0): {X_scaled_df.mean().head().round(4).values}")
print(f"  Sample stds (should be ~1): {X_scaled_df.std().head().round(4).values}")

# Create final processed dataset
df_final = pd.concat([
    pd.DataFrame({'id': df_clean['id'].values, 'diagnosis': y.values}),
    X_scaled_df
], axis=1)

print("\n✓ Preprocessing Complete!")
print(f"  Final dataset shape: {df_final.shape}")

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("7. GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# 1. Pair Plot
print("\n🎨 Creating Pair Plot (showing first 6 features)...")
features_for_pairplot = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                         'area_mean', 'smoothness_mean']
plt.figure(figsize=(14, 10))
sns.pairplot(df_clean[features_for_pairplot], hue='diagnosis',
             diag_kind='hist', palette={'M': '#e74c3c', 'B': '#3498db'})
plt.suptitle('Pair Plot: Key Features by Diagnosis', y=1.00, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('01_pairplot.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved as '01_pairplot.png'")
plt.close()

# 2. Correlation Heatmap
print("\n🎨 Creating Correlation Heatmap...")
plt.figure(figsize=(16, 12))
correlation_matrix = df_clean.drop(['id', 'diagnosis'], axis=1).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap - All Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved as '02_correlation_heatmap.png'")
plt.close()

# 3. Box Plots by Diagnosis
print("\n🎨 Creating Box Plots...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
features_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
                    'concave points_mean', 'symmetry_mean']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    sns.boxplot(data=df_clean, x='diagnosis', y=feature,
                palette={'M': '#e74c3c', 'B': '#3498db'}, ax=ax)
    ax.set_title(f'{feature} by Diagnosis', fontweight='bold')
    ax.set_xlabel('Diagnosis (M: Malignant, B: Benign)')
    ax.set_ylabel(feature)

plt.suptitle('Feature Distributions by Diagnosis Type',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('03_boxplots.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved as '03_boxplots.png'")
plt.close()

# 4. Distribution Plots
print("\n🎨 Creating Distribution Plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Diagnosis distribution
df_clean['diagnosis'].value_counts().plot(kind='bar', ax=axes[0],
                                          color=['#e74c3c', '#3498db'])
axes[0].set_title('Diagnosis Distribution', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Diagnosis (M: Malignant, B: Benign)')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

# Area_mean distribution
axes[1].hist(df_clean[df_clean['diagnosis'] == 'M']['area_mean'],
             bins=30, alpha=0.6, label='Malignant (M)', color='#e74c3c')
axes[1].hist(df_clean[df_clean['diagnosis'] == 'B']['area_mean'],
             bins=30, alpha=0.6, label='Benign (B)', color='#3498db')
axes[1].set_title('Area_Mean Distribution by Diagnosis', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Area Mean')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.suptitle('Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('04_distribution_plots.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved as '04_distribution_plots.png'")
plt.close()

# ============================================================================
# SECTION 8: SAVE PROCESSED DATA
# ============================================================================

print("\n" + "=" * 80)
print("8. SAVING PROCESSED DATA")
print("=" * 80)

# Save scaled data with diagnosis
df_final.to_csv('data_processed_scaled.csv', index=False)
print("\n✓ Saved preprocessed (scaled) data to 'data_processed_scaled.csv'")

# Save encoded but not scaled
df_processed.to_csv('data_processed_encoded.csv', index=False)
print("✓ Saved preprocessed (encoded only) data to 'data_processed_encoded.csv'")

# ============================================================================
# SECTION 9: SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("9. SUMMARY REPORT")
print("=" * 80)

print(f"""
📊 DATA PROCESSING SUMMARY
{'=' * 80}

1. ORIGINAL DATASET
   - Rows: {df.shape[0]}
   - Columns: {df.shape[1]}
   - Missing values: {df.isnull().sum().sum()}

2. AFTER CLEANING
   - Rows: {df_clean.shape[0]}
   - Rows removed: {df.shape[0] - df_clean.shape[0]}

3. DIAGNOSIS BREAKDOWN
   - Malignant (M): {diagnosis_counts.get('M', 0)} ({diagnosis_percent.get('M', 0):.2f}%)
   - Benign (B): {diagnosis_counts.get('B', 0)} ({diagnosis_percent.get('B', 0):.2f}%)

4. AREA_MEAN STATISTICS
   - Min: {df_clean['area_mean'].min():.2f}
   - Max: {df_clean['area_mean'].max():.2f}
   - Mean: {df_clean['area_mean'].mean():.2f}
   - Median: {df_clean['area_mean'].median():.2f}
   - Std Dev: {df_clean['area_mean'].std():.2f}

5. PREPROCESSING APPLIED
   ✓ Missing values handled
   ✓ Categorical encoding (M→1, B→0)
   ✓ Feature scaling (StandardScaler)
   ✓ {X.shape[1]} features normalized

6. OUTPUT FILES GENERATED
   ✓ data_processed_scaled.csv (scaled features)
   ✓ data_processed_encoded.csv (encoded only)
   ✓ 01_pairplot.png
   ✓ 02_correlation_heatmap.png
   ✓ 03_boxplots.png
   ✓ 04_distribution_plots.png

{'=' * 80}
""")

print("\n✅ ANALYSIS COMPLETE!\n")

# Display sample of processed data
print("Sample of Processed Data (Scaled):")
print(df_final.head(10))
