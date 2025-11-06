import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#1A1F3A', '#2D3250', '#D4A574', '#C85C3C', '#5B8FA3', '#E8B968']
GRADIENT = ['#1A1F3A', '#232945', '#2D3250', '#3C4565', '#5B8FA3', '#7BA3B5', '#9BB8C7', '#C85C3C', '#D4A574', '#E8B968']
SAVE_PATH = './'

plt.style.use('dark_background')

df = pd.read_csv('drugs_side_effects_drugs_com.csv')

print("Pharmaceutical Drug Safety Analysis: Foundation Framework\n")
print(f"Dataset: {len(df)} unique drug records across {df['medical_condition'].nunique()} medical conditions")
print(f"Coverage: {df['generic_name'].nunique()} generic medications with {df['drug_classes'].nunique()} drug classifications")
print(f"Review Corpus: {df['no_of_reviews'].sum():.0f} patient reviews informing {len(df[df['rating'].notna()])} drug ratings\n")

print("Data Quality Assessment:")
print(f"  Complete Drug Information: {len(df[df['side_effects'].notna()])} records ({len(df[df['side_effects'].notna()])/len(df)*100:.1f}%)")
print(f"  Drugs with User Ratings: {len(df[df['rating'].notna()])} records ({len(df[df['rating'].notna()])/len(df)*100:.1f}%)")
print(f"  Prescription Classification: {len(df[df['rx_otc'].notna()])} records ({len(df[df['rx_otc'].notna()])/len(df)*100:.1f}%)")
print(f"  Pregnancy Safety Data: {len(df[df['pregnancy_category'].notna()])} records ({len(df[df['pregnancy_category'].notna()])/len(df)*100:.1f}%)\n")

df_rated = df[df['rating'].notna()].copy()
print(f"Analytical Subset: {len(df_rated)} drugs with complete rating data")
print(f"Average Drug Rating: {df_rated['rating'].mean():.2f}/10.0 (SD: {df_rated['rating'].std():.2f})")
print(f"Median Reviews per Drug: {df_rated['no_of_reviews'].median():.0f} reviews")
print(f"Rating Distribution Skewness: {df_rated['rating'].skew():.3f} (negative skew indicates ceiling effect)")

fig, axes = plt.subplots(3, 3, figsize=(28, 24))
fig.patch.set_facecolor('#0D1117')

ax = axes[0, 0]
rating_counts = df_rated['rating'].value_counts().sort_index()
bars = ax.bar(rating_counts.index, rating_counts.values, 
              color=COLORS[2], edgecolor=COLORS[5], linewidth=2.5, alpha=0.9, width=0.7)
ax.set_xlabel('Patient Rating Score', fontsize=14, color=COLORS[5], weight='bold')
ax.set_ylabel('Number of Drugs', fontsize=14, color=COLORS[5], weight='bold')
ax.set_title('Drug Efficacy Rating Distribution\nPatient-Reported Satisfaction Scores', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.axvline(x=df_rated['rating'].mean(), color=COLORS[3], linestyle='--', linewidth=3, alpha=0.8, label=f'Mean: {df_rated["rating"].mean():.1f}')
ax.axvline(x=df_rated['rating'].median(), color=COLORS[4], linestyle='--', linewidth=3, alpha=0.8, label=f'Median: {df_rated["rating"].median():.1f}')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax.legend(fontsize=12, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2], loc='upper left')
ax.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (idx, val) in enumerate(zip(rating_counts.index, rating_counts.values)):
    if val > 10:
        ax.text(idx, val, f'{val}', ha='center', va='bottom', color=COLORS[5], fontsize=10, weight='bold')

ax = axes[0, 1]
condition_counts = df['medical_condition'].value_counts().head(15)
bars = ax.barh(range(len(condition_counts)), condition_counts.values,
               color=[GRADIENT[i % len(GRADIENT)] for i in range(len(condition_counts))],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax.set_yticks(range(len(condition_counts)))
ax.set_yticklabels(condition_counts.index, fontsize=11, color=COLORS[5], weight='bold')
ax.set_xlabel('Number of Available Drugs', fontsize=14, color=COLORS[5], weight='bold')
ax.set_title('Drug Availability by Medical Condition\nTop 15 Conditions by Drug Count', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (bar, val) in enumerate(zip(bars, condition_counts.values)):
    ax.text(val, i, f'  {val}', va='center', ha='left', color=COLORS[5], fontsize=10, weight='bold')

ax = axes[0, 2]
rx_counts = df['rx_otc'].value_counts()
colors_pie = [COLORS[2], COLORS[3], COLORS[4]]
wedges, texts, autotexts = ax.pie(rx_counts.values, labels=rx_counts.index, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90, textprops={'fontsize': 14, 'color': COLORS[5], 'weight': 'bold'},
                                    wedgeprops={'edgecolor': COLORS[5], 'linewidth': 2.5, 'alpha': 0.9})
for autotext in autotexts:
    autotext.set_color('#0D1117')
    autotext.set_fontsize(13)
    autotext.set_weight('bold')
ax.set_title('Prescription Status Distribution\nRegulatory Classification of Medications', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')

ax = axes[1, 0]
preg_cat = df[df['pregnancy_category'] != 'N']['pregnancy_category'].value_counts().sort_index()
bars = ax.bar(range(len(preg_cat)), preg_cat.values,
              color=[GRADIENT[i*2] for i in range(len(preg_cat))],
              edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax.set_xticks(range(len(preg_cat)))
ax.set_xticklabels(preg_cat.index, fontsize=13, color=COLORS[5], weight='bold')
ax.set_ylabel('Number of Drugs', fontsize=14, color=COLORS[5], weight='bold')
ax.set_xlabel('FDA Pregnancy Category', fontsize=14, color=COLORS[5], weight='bold')
ax.set_title('Pregnancy Safety Classification\nFDA Risk Categories (Excluding N/A)', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=11, width=2)
safety_labels = {'A': 'Safest', 'B': 'Safe', 'C': 'Caution', 'D': 'Risk', 'X': 'Contraindicated'}
for i, (bar, cat, val) in enumerate(zip(bars, preg_cat.index, preg_cat.values)):
    ax.text(i, val, f'{val}\n{safety_labels.get(cat, "")}', ha='center', va='bottom', 
           color=COLORS[5], fontsize=10, weight='bold')

ax = axes[1, 1]
top_conditions_rated = df_rated.groupby('medical_condition')['rating'].agg(['mean', 'count'])
top_conditions_rated = top_conditions_rated[top_conditions_rated['count'] >= 10].sort_values('mean', ascending=False).head(15)
bars = ax.barh(range(len(top_conditions_rated)), top_conditions_rated['mean'],
               color=[COLORS[2] if x >= 7 else COLORS[3] for x in top_conditions_rated['mean']],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax.set_yticks(range(len(top_conditions_rated)))
ax.set_yticklabels(top_conditions_rated.index, fontsize=10, color=COLORS[5], weight='bold')
ax.set_xlabel('Average Patient Rating', fontsize=14, color=COLORS[5], weight='bold')
ax.set_xlim(0, 10)
ax.axvline(x=df_rated['rating'].mean(), color=COLORS[4], linestyle='--', linewidth=3, alpha=0.7)
ax.set_title('Treatment Satisfaction by Condition\nMean Ratings (min 10 reviews)', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, (val, count) in enumerate(zip(top_conditions_rated['mean'], top_conditions_rated['count'])):
    ax.text(val, i, f' {val:.1f} (n={count})', va='center', ha='left', 
           color=COLORS[5], fontsize=9, weight='bold')

ax = axes[1, 2]
review_bins = [0, 10, 50, 100, 500, 1000, df_rated['no_of_reviews'].max()]
review_labels = ['1-10', '11-50', '51-100', '101-500', '501-1000', '1000+']
df_rated['review_bin'] = pd.cut(df_rated['no_of_reviews'], bins=review_bins, labels=review_labels)
review_dist = df_rated['review_bin'].value_counts().sort_index()
bars = ax.bar(range(len(review_dist)), review_dist.values,
              color=[GRADIENT[i+2] for i in range(len(review_dist))],
              edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax.set_xticks(range(len(review_dist)))
ax.set_xticklabels(review_labels, fontsize=11, color=COLORS[5], weight='bold', rotation=30, ha='right')
ax.set_ylabel('Number of Drugs', fontsize=14, color=COLORS[5], weight='bold')
ax.set_xlabel('Review Count Range', fontsize=14, color=COLORS[5], weight='bold')
ax.set_title('Review Volume Distribution\nPatient Engagement by Drug', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (bar, val) in enumerate(zip(bars, review_dist.values)):
    ax.text(i, val, f'{val}', ha='center', va='bottom', color=COLORS[5], fontsize=10, weight='bold')

ax = axes[2, 0]
rating_review_corr = df_rated.groupby('review_bin')['rating'].mean()
bars = ax.bar(range(len(rating_review_corr)), rating_review_corr.values,
              color=COLORS[2], edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax.set_xticks(range(len(rating_review_corr)))
ax.set_xticklabels(rating_review_corr.index, fontsize=11, color=COLORS[5], weight='bold', rotation=30, ha='right')
ax.set_ylabel('Average Rating', fontsize=14, color=COLORS[5], weight='bold')
ax.set_xlabel('Review Volume Category', fontsize=14, color=COLORS[5], weight='bold')
ax.axhline(y=df_rated['rating'].mean(), color=COLORS[3], linestyle='--', linewidth=3, alpha=0.7)
ax.set_ylim(0, 10)
ax.set_title('Rating Reliability by Review Volume\nConsensus Effect Analysis', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, val in enumerate(rating_review_corr.values):
    ax.text(i, val, f'{val:.2f}', ha='center', va='bottom', color=COLORS[5], fontsize=10, weight='bold')

ax = axes[2, 1]
top_classes = df['drug_classes'].dropna().str.split(',').explode().str.strip().value_counts().head(12)
bars = ax.barh(range(len(top_classes)), top_classes.values,
               color=[GRADIENT[i % len(GRADIENT)] for i in range(len(top_classes))],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax.set_yticks(range(len(top_classes)))
ax.set_yticklabels(top_classes.index, fontsize=10, color=COLORS[5], weight='bold')
ax.set_xlabel('Number of Drugs', fontsize=14, color=COLORS[5], weight='bold')
ax.set_title('Most Common Drug Classifications\nPharmacological Categories', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, (bar, val) in enumerate(zip(bars, top_classes.values)):
    ax.text(val, i, f'  {val}', va='center', ha='left', color=COLORS[5], fontsize=9, weight='bold')

ax = axes[2, 2]
rx_ratings = df_rated.groupby('rx_otc')['rating'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(rx_ratings))
bars = ax.bar(x_pos, rx_ratings['mean'], yerr=rx_ratings['std']/2,
              color=[COLORS[2], COLORS[3], COLORS[4]],
              edgecolor=COLORS[5], linewidth=2.5, alpha=0.9, capsize=8, error_kw={'linewidth': 2.5})
ax.set_xticks(x_pos)
ax.set_xticklabels(rx_ratings.index, fontsize=13, color=COLORS[5], weight='bold')
ax.set_ylabel('Average Rating', fontsize=14, color=COLORS[5], weight='bold')
ax.set_xlabel('Prescription Status', fontsize=14, color=COLORS[5], weight='bold')
ax.set_ylim(0, 10)
ax.axhline(y=df_rated['rating'].mean(), color=COLORS[5], linestyle='--', linewidth=2.5, alpha=0.5)
ax.set_title('Efficacy by Prescription Classification\nMean Ratings with Standard Deviation', 
             fontsize=17, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#141B25')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (mean_val, count) in enumerate(zip(rx_ratings['mean'], rx_ratings['count'])):
    ax.text(i, mean_val, f'{mean_val:.2f}\n(n={count})', ha='center', va='bottom', 
           color=COLORS[5], fontsize=10, weight='bold')

plt.tight_layout(pad=3)
plt.savefig(f'{SAVE_PATH}drug_safety_foundation_analysis.png', dpi=300, 
           facecolor='#0D1117', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: drug_safety_foundation_analysis.png")

print("\n" + "="*80)
print("Statistical Summary: Key Findings")
print("="*80)

print(f"\nDrug Rating Patterns:")
print(f"  Highly Rated (8-10): {len(df_rated[df_rated['rating'] >= 8])} drugs ({len(df_rated[df_rated['rating'] >= 8])/len(df_rated)*100:.1f}%)")
print(f"  Moderately Rated (5-7.9): {len(df_rated[(df_rated['rating'] >= 5) & (df_rated['rating'] < 8)])} drugs ({len(df_rated[(df_rated['rating'] >= 5) & (df_rated['rating'] < 8)])/len(df_rated)*100:.1f}%)")
print(f"  Poorly Rated (0-4.9): {len(df_rated[df_rated['rating'] < 5])} drugs ({len(df_rated[df_rated['rating'] < 5])/len(df_rated)*100:.1f}%)")

print(f"\nCondition Coverage:")
most_drugs = condition_counts.index[0]
least_common = df['medical_condition'].value_counts().tail(1).index[0]
print(f"  Highest Drug Availability: {most_drugs} ({condition_counts.iloc[0]} drugs)")
print(f"  Therapeutic Gaps: {least_common} ({df['medical_condition'].value_counts().tail(1).values[0]} drugs)")

print(f"\nPrescription Accessibility:")
print(f"  Prescription-Only: {rx_counts.get('Rx', 0)} drugs ({rx_counts.get('Rx', 0)/len(df)*100:.1f}%)")
print(f"  Over-the-Counter: {rx_counts.get('OTC', 0)} drugs ({rx_counts.get('OTC', 0)/len(df)*100:.1f}%)")
print(f"  Dual Availability: {rx_counts.get('Rx/OTC', 0)} drugs ({rx_counts.get('Rx/OTC', 0)/len(df)*100:.1f}%)")

print(f"\nPregnancy Safety Profile:")
print(f"  Category A (Safest): {preg_cat.get('A', 0)} drugs")
print(f"  Category B (Generally Safe): {preg_cat.get('B', 0)} drugs")
print(f"  Category C (Caution): {preg_cat.get('C', 0)} drugs")
print(f"  Category D (Proven Risk): {preg_cat.get('D', 0)} drugs")
print(f"  Category X (Contraindicated): {preg_cat.get('X', 0)} drugs")
print(f"  High-Risk Categories (D+X): {preg_cat.get('D', 0) + preg_cat.get('X', 0)} drugs ({(preg_cat.get('D', 0) + preg_cat.get('X', 0))/len(df[df['pregnancy_category'] != 'N'])*100:.1f}% of classified drugs)")

print(f"\nReview Engagement Insights:")
median_reviews = df_rated['no_of_reviews'].median()
high_engagement = len(df_rated[df_rated['no_of_reviews'] > median_reviews])
print(f"  Median Review Count: {median_reviews:.0f} patient reviews per drug")
print(f"  High-Engagement Drugs: {high_engagement} drugs (>{median_reviews:.0f} reviews)")
print(f"  Total Patient Feedback: {df_rated['no_of_reviews'].sum():.0f} reviews across {len(df_rated)} rated drugs")

print(f"\nRating-Review Correlation:")
from scipy.stats import spearmanr
corr, pval = spearmanr(df_rated['rating'], df_rated['no_of_reviews'])
print(f"  Spearman Correlation: {corr:.3f} (p={pval:.4f})")
if abs(corr) < 0.1:
    print(f"  Interpretation: Negligible relationship between review volume and rating")
elif pval < 0.05:
    print(f"  Interpretation: Statistically significant {'positive' if corr > 0 else 'negative'} association")

print("\n" + "="*80)
print("Foundation Analysis Complete")
print("="*80)
print("This analysis establishes baseline understanding of drug safety landscape,")
print("regulatory classifications, and patient satisfaction patterns across major")
print("medical conditions. Subsequent frameworks build on these insights with")
print("advanced statistical modeling and predictive analytics.")
