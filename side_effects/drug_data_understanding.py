import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#1A1F3A', '#2D3250', '#D4A574', '#C85C3C', '#5B8FA3', '#E8B968']
GRADIENT = ['#1A1F3A', '#232945', '#2D3250', '#3C4565', '#5B8FA3', '#7BA3B5', '#9BB8C7', '#C85C3C', '#D4A574', '#E8B968']

plt.style.use('dark_background')

print("="*100)
print("PHARMACEUTICAL DRUG SAFETY DATABASE: DATA UNDERSTANDING AND CLEANING REPORT")
print("="*100)
print("\nData Source: drugs_side_effects_drugs_com.csv")
print("Analysis Date:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
print("="*100)

df_raw = pd.read_csv('drugs_side_effects_drugs_com.csv')

print("\n" + "="*100)
print("SECTION 1: INITIAL DATA INSPECTION")
print("="*100)

print(f"\nDataset Dimensions:")
print(f"  Rows (Records): {df_raw.shape[0]:,}")
print(f"  Columns (Features): {df_raw.shape[1]}")
print(f"  Total Data Points: {df_raw.shape[0] * df_raw.shape[1]:,}")
print(f"  Memory Usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n{'Column Name':<40} {'Data Type':<15} {'Non-Null Count':<15} {'Null Count':<12} {'Null %'}")
print("-" * 100)
for col in df_raw.columns:
    dtype = str(df_raw[col].dtype)
    non_null = df_raw[col].notna().sum()
    null_count = df_raw[col].isna().sum()
    null_pct = (null_count / len(df_raw)) * 100
    print(f"{col:<40} {dtype:<15} {non_null:<15,} {null_count:<12,} {null_pct:>6.2f}%")

print("\n" + "-" * 100)
print("FIRST 5 RECORDS (RAW DATA)")
print("-" * 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.width', None)
print(df_raw.head())

print("\n" + "-" * 100)
print("LAST 5 RECORDS (RAW DATA)")
print("-" * 100)
print(df_raw.tail())

print("\n" + "-" * 100)
print("RANDOM SAMPLE OF 5 RECORDS")
print("-" * 100)
print(df_raw.sample(5, random_state=42))

print("\n" + "="*100)
print("SECTION 2: DATA TYPE ANALYSIS")
print("="*100)

print("\nData Type Distribution:")
dtype_counts = df_raw.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

print("\nNumeric Columns:")
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    for col in numeric_cols:
        print(f"  - {col}")
else:
    print("  None found")

print("\nObject (Text) Columns:")
object_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    print(f"  - {col}")

print("\n" + "-" * 100)
print("NUMERIC COLUMN STATISTICS")
print("-" * 100)
if len(numeric_cols) > 0:
    print(df_raw[numeric_cols].describe())
else:
    print("No numeric columns in dataset")

print("\n" + "="*100)
print("SECTION 3: MISSING DATA ANALYSIS")
print("="*100)

missing_data = pd.DataFrame({
    'Column': df_raw.columns,
    'Missing_Count': df_raw.isnull().sum(),
    'Missing_Percentage': (df_raw.isnull().sum() / len(df_raw)) * 100,
    'Non_Missing': df_raw.notna().sum(),
    'Data_Type': df_raw.dtypes
})
missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)
missing_data = missing_data[missing_data['Missing_Count'] > 0]

print(f"\nColumns with Missing Data: {len(missing_data)} of {len(df_raw.columns)}")
print("\n" + "-" * 100)
print(f"{'Column':<40} {'Missing':<12} {'%':<10} {'Non-Missing':<15} {'Type'}")
print("-" * 100)
for _, row in missing_data.iterrows():
    print(f"{row['Column']:<40} {row['Missing_Count']:<12.0f} {row['Missing_Percentage']:<10.2f} {row['Non_Missing']:<15.0f} {row['Data_Type']}")

if len(missing_data) == 0:
    print("\nNo missing data detected in any columns.")

print("\n" + "-" * 100)
print("MISSING DATA PATTERNS")
print("-" * 100)

high_missing = missing_data[missing_data['Missing_Percentage'] > 50]
if len(high_missing) > 0:
    print(f"\nHIGH MISSING (>50%):")
    for col in high_missing['Column']:
        pct = high_missing[high_missing['Column'] == col]['Missing_Percentage'].values[0]
        print(f"  - {col}: {pct:.1f}% missing")
else:
    print("\nNo columns with >50% missing data")

moderate_missing = missing_data[(missing_data['Missing_Percentage'] > 10) & (missing_data['Missing_Percentage'] <= 50)]
if len(moderate_missing) > 0:
    print(f"\nMODERATE MISSING (10-50%):")
    for col in moderate_missing['Column']:
        pct = moderate_missing[moderate_missing['Column'] == col]['Missing_Percentage'].values[0]
        print(f"  - {col}: {pct:.1f}% missing")
else:
    print("\nNo columns with 10-50% missing data")

low_missing = missing_data[(missing_data['Missing_Percentage'] > 0) & (missing_data['Missing_Percentage'] <= 10)]
if len(low_missing) > 0:
    print(f"\nLOW MISSING (<10%):")
    for col in low_missing['Column']:
        pct = low_missing[low_missing['Column'] == col]['Missing_Percentage'].values[0]
        print(f"  - {col}: {pct:.1f}% missing")
else:
    print("\nNo columns with <10% missing data")

print("\n" + "="*100)
print("SECTION 4: UNIQUE VALUE ANALYSIS")
print("="*100)

print(f"\n{'Column':<40} {'Unique Values':<20} {'Most Common Value':<30} {'Frequency'}")
print("-" * 100)
for col in df_raw.columns:
    unique_count = df_raw[col].nunique()
    if df_raw[col].notna().sum() > 0:
        most_common = df_raw[col].value_counts().index[0] if len(df_raw[col].value_counts()) > 0 else 'N/A'
        most_common_str = str(most_common)[:30]
        freq = df_raw[col].value_counts().values[0] if len(df_raw[col].value_counts()) > 0 else 0
    else:
        most_common_str = 'All Missing'
        freq = 0
    print(f"{col:<40} {unique_count:<20,} {most_common_str:<30} {freq:,}")

print("\n" + "-" * 100)
print("CARDINALITY ANALYSIS")
print("-" * 100)

high_cardinality = []
low_cardinality = []
for col in df_raw.columns:
    unique_count = df_raw[col].nunique()
    total_count = len(df_raw)
    cardinality_ratio = unique_count / total_count
    
    if cardinality_ratio > 0.9:
        high_cardinality.append((col, unique_count, cardinality_ratio))
    elif cardinality_ratio < 0.05:
        low_cardinality.append((col, unique_count, cardinality_ratio))

print("\nHIGH CARDINALITY COLUMNS (>90% unique):")
if high_cardinality:
    for col, unique, ratio in high_cardinality:
        print(f"  - {col}: {unique:,} unique values ({ratio*100:.1f}% cardinality)")
else:
    print("  None")

print("\nLOW CARDINALITY COLUMNS (<5% unique):")
if low_cardinality:
    for col, unique, ratio in low_cardinality:
        print(f"  - {col}: {unique:,} unique values ({ratio*100:.1f}% cardinality)")
else:
    print("  None")

print("\n" + "="*100)
print("SECTION 5: CATEGORICAL VARIABLE DISTRIBUTIONS")
print("="*100)

categorical_candidates = [col for col in df_raw.columns 
                         if df_raw[col].dtype == 'object' and df_raw[col].nunique() < 50]

for col in categorical_candidates[:8]:
    print(f"\n{col.upper()} - Value Counts:")
    print("-" * 80)
    value_counts = df_raw[col].value_counts()
    for value, count in value_counts.head(15).items():
        pct = (count / len(df_raw)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {str(value)[:50]:<50} {count:>8,} ({pct:>5.1f}%) {bar}")
    if len(value_counts) > 15:
        print(f"  ... and {len(value_counts) - 15} more values")

print("\n" + "="*100)
print("SECTION 6: TEXT FIELD ANALYSIS")
print("="*100)

text_columns = ['side_effects', 'medical_condition_description', 'related_drugs']

for col in text_columns:
    if col in df_raw.columns:
        print(f"\n{col.upper()}:")
        print("-" * 80)
        non_null = df_raw[col].notna()
        if non_null.sum() > 0:
            lengths = df_raw[col].str.len()
            print(f"  Non-null entries: {non_null.sum():,}")
            print(f"  Average length: {lengths.mean():.0f} characters")
            print(f"  Median length: {lengths.median():.0f} characters")
            print(f"  Min length: {lengths.min():.0f} characters")
            print(f"  Max length: {lengths.max():.0f} characters")
            print(f"  Total characters: {lengths.sum():,.0f}")
            
            print(f"\n  Sample (first 200 chars):")
            sample_text = df_raw[col].dropna().iloc[0]
            print(f"    '{sample_text[:200]}...'")
        else:
            print(f"  All values are missing")

print("\n" + "="*100)
print("SECTION 7: DATA QUALITY ISSUES")
print("="*100)

print("\nDuplicate Analysis:")
duplicate_rows = df_raw.duplicated().sum()
print(f"  Duplicate rows: {duplicate_rows:,}")
if duplicate_rows > 0:
    print(f"  Percentage: {(duplicate_rows / len(df_raw)) * 100:.2f}%")

print("\nDuplicate Drug Names:")
if 'drug_name' in df_raw.columns:
    duplicate_drugs = df_raw['drug_name'].duplicated().sum()
    print(f"  Duplicate drug names: {duplicate_drugs:,}")
    if duplicate_drugs > 0:
        print(f"\n  Examples of duplicate drugs:")
        dup_drugs = df_raw[df_raw['drug_name'].duplicated(keep=False)]['drug_name'].value_counts().head(10)
        for drug, count in dup_drugs.items():
            print(f"    - {drug}: appears {count} times")

print("\nEmpty String Detection:")
for col in df_raw.select_dtypes(include=['object']).columns:
    empty_strings = (df_raw[col] == '').sum()
    if empty_strings > 0:
        print(f"  {col}: {empty_strings:,} empty strings")

print("\nWhitespace Issues:")
for col in df_raw.select_dtypes(include=['object']).columns[:5]:
    leading_space = df_raw[col].str.startswith(' ', na=False).sum()
    trailing_space = df_raw[col].str.endswith(' ', na=False).sum()
    if leading_space > 0 or trailing_space > 0:
        print(f"  {col}: {leading_space:,} leading, {trailing_space:,} trailing spaces")

print("\n" + "="*100)
print("SECTION 8: DATA CLEANING RECOMMENDATIONS")
print("="*100)

print("\nRECOMMENDED CLEANING ACTIONS:")

actions = []

if 'rating' in df_raw.columns and df_raw['rating'].isna().sum() > 0:
    pct = (df_raw['rating'].isna().sum() / len(df_raw)) * 100
    actions.append(f"1. Rating column has {pct:.1f}% missing - consider imputation or subset analysis")

high_missing_cols = missing_data[missing_data['Missing_Percentage'] > 50]['Column'].tolist()
if high_missing_cols:
    actions.append(f"2. High missing columns ({len(high_missing_cols)}): Consider dropping or targeted imputation")
    for col in high_missing_cols[:3]:
        pct = missing_data[missing_data['Column'] == col]['Missing_Percentage'].values[0]
        actions.append(f"   - {col} ({pct:.1f}% missing)")

if duplicate_rows > 0:
    actions.append(f"3. Remove {duplicate_rows:,} duplicate rows")

if 'side_effects' in df_raw.columns and df_raw['side_effects'].isna().sum() > 0:
    pct = (df_raw['side_effects'].isna().sum() / len(df_raw)) * 100
    actions.append(f"4. Side effects missing in {pct:.1f}% - critical for analysis, may need exclusion")

for col in ['pregnancy_category', 'rx_otc']:
    if col in df_raw.columns and df_raw[col].isna().sum() > 0:
        pct = (df_raw[col].isna().sum() / len(df_raw)) * 100
        actions.append(f"5. {col} has {pct:.1f}% missing - fill with 'Unknown' or 'N/A'")

if len(actions) > 0:
    for action in actions:
        print(f"  {action}")
else:
    print("  No major data quality issues detected")

print("\n" + "="*100)
print("SECTION 9: DATA CLEANING EXECUTION")
print("="*100)

df_clean = df_raw.copy()

print("\nCleaning Step 1: Remove duplicate rows")
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
print(f"  Removed {before - after:,} duplicate rows")
print(f"  Remaining: {after:,} rows")

print("\nCleaning Step 2: Standardize text fields (strip whitespace)")
text_cols = df_clean.select_dtypes(include=['object']).columns
for col in text_cols:
    df_clean[col] = df_clean[col].str.strip()
print(f"  Cleaned {len(text_cols)} text columns")

print("\nCleaning Step 3: Handle empty strings")
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].replace('', np.nan)
empty_converted = 0
for col in text_cols:
    empty_converted += (df_raw[col] == '').sum()
print(f"  Converted {empty_converted:,} empty strings to NaN")

print("\nCleaning Step 4: Create clean numeric rating field")
if 'rating' in df_clean.columns:
    df_clean['rating_clean'] = pd.to_numeric(df_clean['rating'], errors='coerce')
    invalid = df_clean['rating'].notna() & df_clean['rating_clean'].isna()
    print(f"  Created rating_clean column")
    print(f"  Invalid ratings converted to NaN: {invalid.sum()}")

print("\nCleaning Step 5: Create binary indicators")
if 'rating' in df_clean.columns:
    df_clean['has_rating'] = df_clean['rating'].notna().astype(int)
if 'side_effects' in df_clean.columns:
    df_clean['has_side_effects'] = df_clean['side_effects'].notna().astype(int)
if 'pregnancy_category' in df_clean.columns:
    df_clean['has_pregnancy_data'] = df_clean['pregnancy_category'].notna().astype(int)
print(f"  Created 3 binary indicator columns")

print("\nCleaning Step 6: Extract activity percentage")
if 'activity' in df_clean.columns:
    df_clean['activity_numeric'] = df_clean['activity'].str.rstrip('%').astype(float, errors='ignore')
    print(f"  Extracted numeric activity values")

print("\n" + "-" * 100)
print("CLEANED DATA SUMMARY")
print("-" * 100)
print(f"Original rows: {len(df_raw):,}")
print(f"Cleaned rows: {len(df_clean):,}")
print(f"Rows removed: {len(df_raw) - len(df_clean):,}")
print(f"Original columns: {len(df_raw.columns)}")
print(f"New columns: {len(df_clean.columns)}")
print(f"Columns added: {len(df_clean.columns) - len(df_raw.columns)}")

print("\n" + "-" * 100)
print("FIRST 3 RECORDS (CLEANED DATA)")
print("-" * 100)
print(df_clean[['drug_name', 'medical_condition', 'rating', 'rx_otc', 
               'pregnancy_category', 'has_rating', 'has_side_effects']].head(3))

print("\n" + "="*100)
print("SECTION 10: FINAL DATA PROFILE")
print("="*100)

print(f"\nFinal Dataset Statistics:")
print(f"  Total Records: {len(df_clean):,}")
print(f"  Total Columns: {len(df_clean.columns)}")
print(f"  Complete Records (no nulls): {df_clean.notna().all(axis=1).sum():,}")
print(f"  Completeness Rate: {(df_clean.notna().all(axis=1).sum() / len(df_clean)) * 100:.2f}%")

print(f"\nKey Analytical Subsets:")
if 'has_rating' in df_clean.columns:
    rated = df_clean[df_clean['has_rating'] == 1]
    print(f"  Records with ratings: {len(rated):,} ({len(rated)/len(df_clean)*100:.1f}%)")
if 'has_side_effects' in df_clean.columns:
    with_effects = df_clean[df_clean['has_side_effects'] == 1]
    print(f"  Records with side effects: {len(with_effects):,} ({len(with_effects)/len(df_clean)*100:.1f}%)")
if 'has_pregnancy_data' in df_clean.columns:
    with_preg = df_clean[df_clean['has_pregnancy_data'] == 1]
    print(f"  Records with pregnancy data: {len(with_preg):,} ({len(with_preg)/len(df_clean)*100:.1f}%)")

print("\n" + "="*100)
print("DATA UNDERSTANDING COMPLETE")
print("="*100)

df_clean.to_csv('drugs_side_effects_CLEANED.csv', index=False)
print(f"\nCleaned dataset saved to: drugs_side_effects_CLEANED.csv")
print(f"Ready for downstream analysis")

fig = plt.figure(figsize=(28, 24))
fig.patch.set_facecolor('#0D1117')
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
missing_plot_data = missing_data.sort_values('Missing_Percentage', ascending=True).tail(15)
bars = ax1.barh(range(len(missing_plot_data)), missing_plot_data['Missing_Percentage'],
                color=[COLORS[2] if x < 25 else COLORS[3] if x < 50 else COLORS[4] for x in missing_plot_data['Missing_Percentage']],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax1.set_yticks(range(len(missing_plot_data)))
ax1.set_yticklabels(missing_plot_data['Column'], fontsize=10, color=COLORS[5], weight='bold')
ax1.set_xlabel('Missing Data Percentage', fontsize=13, color=COLORS[5], weight='bold')
ax1.set_title('Missing Data Profile\nTop 15 Columns by Missing Percentage', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax1.axvline(x=50, color=COLORS[5], linestyle='--', linewidth=2.5, alpha=0.5, label='50% Threshold')
ax1.set_facecolor('#141B25')
ax1.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax1.legend(fontsize=11, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
ax1.tick_params(colors=COLORS[5], labelsize=9, width=2)
for i, val in enumerate(missing_plot_data['Missing_Percentage']):
    ax1.text(val, i, f' {val:.1f}%', va='center', ha='left', color=COLORS[5], fontsize=9, weight='bold')

ax2 = fig.add_subplot(gs[0, 1])
completeness_categories = ['Complete\n(No Nulls)', 'Partial\n(Some Nulls)', 'Critical Cols\nComplete']
complete_rows = df_clean.notna().all(axis=1).sum()
partial_rows = len(df_clean) - complete_rows
critical_cols = ['drug_name', 'medical_condition']
critical_complete = df_clean[critical_cols].notna().all(axis=1).sum()
values = [complete_rows, partial_rows, critical_complete]
colors_comp = [COLORS[2], COLORS[3], COLORS[4]]
bars = ax2.bar(range(len(values)), values, color=colors_comp, 
              edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax2.set_xticks(range(len(values)))
ax2.set_xticklabels(completeness_categories, fontsize=11, color=COLORS[5], weight='bold')
ax2.set_ylabel('Number of Records', fontsize=13, color=COLORS[5], weight='bold')
ax2.set_title('Data Completeness Analysis\nRecord-Level Quality Assessment', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax2.set_facecolor('#141B25')
ax2.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax2.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, val in enumerate(values):
    pct = (val / len(df_clean)) * 100
    ax2.text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', 
            color=COLORS[5], fontsize=10, weight='bold')

ax3 = fig.add_subplot(gs[0, 2])
cardinality_data = pd.DataFrame({
    'Column': df_clean.columns,
    'Unique': [df_clean[col].nunique() for col in df_clean.columns],
    'Total': len(df_clean)
})
cardinality_data['Ratio'] = cardinality_data['Unique'] / cardinality_data['Total']
cardinality_data = cardinality_data.sort_values('Unique', ascending=False).head(15)
bars = ax3.barh(range(len(cardinality_data)), cardinality_data['Unique'],
                color=[GRADIENT[i % len(GRADIENT)] for i in range(len(cardinality_data))],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax3.set_yticks(range(len(cardinality_data)))
ax3.set_yticklabels(cardinality_data['Column'], fontsize=9, color=COLORS[5], weight='bold')
ax3.set_xlabel('Unique Values', fontsize=13, color=COLORS[5], weight='bold')
ax3.set_xscale('log')
ax3.set_title('Cardinality Analysis\nUnique Values per Column (Log Scale)', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax3.set_facecolor('#141B25')
ax3.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax3.tick_params(colors=COLORS[5], labelsize=9, width=2)

ax4 = fig.add_subplot(gs[1, 0])
if 'side_effects' in df_clean.columns:
    lengths = df_clean['side_effects'].str.len().dropna()
    ax4.hist(lengths, bins=50, color=COLORS[2], edgecolor=COLORS[5], 
            linewidth=1.5, alpha=0.9)
    ax4.axvline(x=lengths.mean(), color=COLORS[3], linestyle='--', linewidth=3, 
               alpha=0.8, label=f'Mean: {lengths.mean():.0f}')
    ax4.axvline(x=lengths.median(), color=COLORS[4], linestyle='--', linewidth=3, 
               alpha=0.8, label=f'Median: {lengths.median():.0f}')
    ax4.set_xlabel('Character Length', fontsize=13, color=COLORS[5], weight='bold')
    ax4.set_ylabel('Frequency', fontsize=13, color=COLORS[5], weight='bold')
    ax4.set_title('Side Effects Text Length Distribution\nCharacter Count Analysis', 
                  fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax4.set_facecolor('#141B25')
    ax4.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
    ax4.legend(fontsize=11, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
    ax4.tick_params(colors=COLORS[5], labelsize=11, width=2)

ax5 = fig.add_subplot(gs[1, 1])
if 'rating' in df_clean.columns:
    rating_quality = pd.DataFrame({
        'Category': ['Has Rating', 'Missing Rating', 'Has Reviews', 'Missing Reviews'],
        'Count': [
            df_clean['rating'].notna().sum(),
            df_clean['rating'].isna().sum(),
            df_clean['no_of_reviews'].notna().sum() if 'no_of_reviews' in df_clean.columns else 0,
            df_clean['no_of_reviews'].isna().sum() if 'no_of_reviews' in df_clean.columns else len(df_clean)
        ]
    })
    bars = ax5.bar(range(len(rating_quality)), rating_quality['Count'],
                   color=[COLORS[2], COLORS[3], COLORS[4], COLORS[5]],
                   edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
    ax5.set_xticks(range(len(rating_quality)))
    ax5.set_xticklabels(rating_quality['Category'], fontsize=11, color=COLORS[5], 
                       weight='bold', rotation=20, ha='right')
    ax5.set_ylabel('Number of Records', fontsize=13, color=COLORS[5], weight='bold')
    ax5.set_title('Rating Data Quality\nAvailability of Rating Information', 
                  fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax5.set_facecolor('#141B25')
    ax5.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
    ax5.tick_params(colors=COLORS[5], labelsize=11, width=2)
    for i, val in enumerate(rating_quality['Count']):
        pct = (val / len(df_clean)) * 100
        ax5.text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', 
                color=COLORS[5], fontsize=9, weight='bold')

ax6 = fig.add_subplot(gs[1, 2])
dtypes = df_clean.dtypes.value_counts()
colors_dtype = [COLORS[2], COLORS[3], COLORS[4]][:len(dtypes)]
wedges, texts, autotexts = ax6.pie(dtypes.values, labels=[str(x) for x in dtypes.index], 
                                     autopct='%1.1f%%', colors=colors_dtype, startangle=90,
                                     textprops={'fontsize': 12, 'color': COLORS[5], 'weight': 'bold'},
                                     wedgeprops={'edgecolor': COLORS[5], 'linewidth': 2.5, 'alpha': 0.9})
for autotext in autotexts:
    autotext.set_color('#0D1117')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')
ax6.set_title('Data Type Distribution\nColumn Type Breakdown', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')

ax7 = fig.add_subplot(gs[2, :])
if 'medical_condition' in df_clean.columns:
    condition_quality = df_clean.groupby('medical_condition').agg({
        'drug_name': 'count',
        'rating': lambda x: x.notna().sum(),
        'side_effects': lambda x: x.notna().sum(),
        'pregnancy_category': lambda x: x.notna().sum()
    }).reset_index()
    condition_quality.columns = ['Condition', 'Total_Drugs', 'Has_Rating', 'Has_Side_Effects', 'Has_Pregnancy']
    condition_quality = condition_quality.sort_values('Total_Drugs', ascending=False).head(15)
    
    x = np.arange(len(condition_quality))
    width = 0.2
    ax7.bar(x - width*1.5, condition_quality['Total_Drugs'], width, label='Total Drugs',
            color=COLORS[2], edgecolor=COLORS[5], linewidth=1.5, alpha=0.9)
    ax7.bar(x - width*0.5, condition_quality['Has_Rating'], width, label='Has Rating',
            color=COLORS[3], edgecolor=COLORS[5], linewidth=1.5, alpha=0.9)
    ax7.bar(x + width*0.5, condition_quality['Has_Side_Effects'], width, label='Has Side Effects',
            color=COLORS[4], edgecolor=COLORS[5], linewidth=1.5, alpha=0.9)
    ax7.bar(x + width*1.5, condition_quality['Has_Pregnancy'], width, label='Has Pregnancy Data',
            color=COLORS[5], edgecolor=COLORS[5], linewidth=1.5, alpha=0.9)
    
    ax7.set_xticks(x)
    ax7.set_xticklabels(condition_quality['Condition'], fontsize=10, color=COLORS[5], 
                       weight='bold', rotation=45, ha='right')
    ax7.set_ylabel('Number of Drugs', fontsize=13, color=COLORS[5], weight='bold')
    ax7.set_title('Data Completeness by Medical Condition\nTop 15 Conditions - Feature Availability', 
                  fontsize=16, color=COLORS[5], pad=20, weight='bold')
    ax7.set_facecolor('#141B25')
    ax7.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
    ax7.legend(fontsize=11, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2], ncol=4)
    ax7.tick_params(colors=COLORS[5], labelsize=10, width=2)

ax8 = fig.add_subplot(gs[3, 0])
quality_score_components = {
    'Has Drug Name': (df_clean['drug_name'].notna().sum() / len(df_clean)) * 100,
    'Has Condition': (df_clean['medical_condition'].notna().sum() / len(df_clean)) * 100,
    'Has Side Effects': (df_clean['side_effects'].notna().sum() / len(df_clean)) * 100,
    'Has Rating': (df_clean['rating'].notna().sum() / len(df_clean)) * 100,
    'Has Rx Status': (df_clean['rx_otc'].notna().sum() / len(df_clean)) * 100,
    'Has Pregnancy': (df_clean['pregnancy_category'].notna().sum() / len(df_clean)) * 100
}
bars = ax8.barh(range(len(quality_score_components)), list(quality_score_components.values()),
                color=[COLORS[2] if v >= 80 else COLORS[3] if v >= 50 else COLORS[4] 
                      for v in quality_score_components.values()],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax8.set_yticks(range(len(quality_score_components)))
ax8.set_yticklabels(list(quality_score_components.keys()), fontsize=11, color=COLORS[5], weight='bold')
ax8.set_xlabel('Completeness Percentage', fontsize=13, color=COLORS[5], weight='bold')
ax8.set_xlim(0, 100)
ax8.axvline(x=80, color=COLORS[5], linestyle='--', linewidth=2.5, alpha=0.5, label='80% Target')
ax8.set_title('Feature Completeness Scores\nData Availability by Key Field', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax8.set_facecolor('#141B25')
ax8.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax8.legend(fontsize=11, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
ax8.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, val in enumerate(quality_score_components.values()):
    ax8.text(val, i, f' {val:.1f}%', va='center', ha='left', color=COLORS[5], fontsize=10, weight='bold')

ax9 = fig.add_subplot(gs[3, 1:])
cleaning_impact = pd.DataFrame({
    'Metric': ['Total Rows', 'Complete Rows', 'Rows with Rating', 'Rows with Side Effects',
              'Unique Drugs', 'Unique Conditions'],
    'Before': [
        len(df_raw),
        df_raw.notna().all(axis=1).sum(),
        df_raw['rating'].notna().sum() if 'rating' in df_raw.columns else 0,
        df_raw['side_effects'].notna().sum() if 'side_effects' in df_raw.columns else 0,
        df_raw['drug_name'].nunique() if 'drug_name' in df_raw.columns else 0,
        df_raw['medical_condition'].nunique() if 'medical_condition' in df_raw.columns else 0
    ],
    'After': [
        len(df_clean),
        df_clean.notna().all(axis=1).sum(),
        df_clean['rating'].notna().sum() if 'rating' in df_clean.columns else 0,
        df_clean['side_effects'].notna().sum() if 'side_effects' in df_clean.columns else 0,
        df_clean['drug_name'].nunique() if 'drug_name' in df_clean.columns else 0,
        df_clean['medical_condition'].nunique() if 'medical_condition' in df_clean.columns else 0
    ]
})
x = np.arange(len(cleaning_impact))
width = 0.35
bars1 = ax9.bar(x - width/2, cleaning_impact['Before'], width, label='Before Cleaning',
                color=COLORS[3], edgecolor=COLORS[5], linewidth=2, alpha=0.9)
bars2 = ax9.bar(x + width/2, cleaning_impact['After'], width, label='After Cleaning',
                color=COLORS[2], edgecolor=COLORS[5], linewidth=2, alpha=0.9)
ax9.set_xticks(x)
ax9.set_xticklabels(cleaning_impact['Metric'], fontsize=11, color=COLORS[5], 
                   weight='bold', rotation=30, ha='right')
ax9.set_ylabel('Count', fontsize=13, color=COLORS[5], weight='bold')
ax9.set_title('Data Cleaning Impact Assessment\nBefore vs After Comparison', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax9.set_facecolor('#141B25')
ax9.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax9.legend(fontsize=11, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
ax9.tick_params(colors=COLORS[5], labelsize=10, width=2)

plt.savefig('drug_data_understanding_cleaning.png', dpi=300, 
           facecolor='#0D1117', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: drug_data_understanding_cleaning.png")
print("\n" + "="*100)
print("ANALYSIS COMPLETE - DATASET READY FOR MODELING")
print("="*100)
