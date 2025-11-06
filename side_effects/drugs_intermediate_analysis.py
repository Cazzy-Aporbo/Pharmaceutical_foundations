import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#1A1F3A', '#2D3250', '#D4A574', '#C85C3C', '#5B8FA3', '#E8B968']
GRADIENT = ['#1A1F3A', '#232945', '#2D3250', '#3C4565', '#5B8FA3', '#7BA3B5', '#9BB8C7', '#C85C3C', '#D4A574', '#E8B968']
SAVE_PATH = './'

plt.style.use('dark_background')

df = pd.read_csv('drugs_side_effects_drugs_com.csv')

print("Pharmaceutical Drug Safety Analysis: Intermediate Framework\n")
print("="*80)
print("Text Mining and Statistical Modeling of Drug Side Effects")
print("="*80 + "\n")

df_complete = df[df['side_effects'].notna()].copy()
print(f"Analyzing {len(df_complete)} drugs with documented side effect profiles")
print(f"Side effect text corpus: {df_complete['side_effects'].str.len().sum():,.0f} characters")
print(f"Average side effect description: {df_complete['side_effects'].str.len().mean():.0f} characters per drug\n")

def extract_side_effects(text):
    if pd.isna(text):
        return []
    text = text.lower()
    symptoms = []
    keywords = ['pain', 'nausea', 'vomiting', 'diarrhea', 'headache', 'dizziness', 'rash', 'itching', 
                'fever', 'swelling', 'bleeding', 'breathing', 'fatigue', 'weakness', 'drowsiness',
                'insomnia', 'anxiety', 'depression', 'constipation', 'cough', 'infection', 'weight',
                'appetite', 'stomach', 'vision', 'hearing', 'muscle', 'joint', 'liver', 'kidney',
                'heart', 'blood pressure', 'chest', 'throat', 'skin', 'hives', 'redness', 'burning']
    for keyword in keywords:
        if keyword in text:
            symptoms.append(keyword)
    return symptoms

df_complete['side_effect_list'] = df_complete['side_effects'].apply(extract_side_effects)
df_complete['side_effect_count'] = df_complete['side_effect_list'].apply(len)
df_complete['side_effect_severity'] = df_complete['side_effects'].str.lower().str.count('severe|serious|emergency')
df_complete['mention_death'] = df_complete['side_effects'].str.lower().str.contains('death|fatal|life-threatening', na=False).astype(int)

print("Side Effect Extraction Complete:")
print(f"  Unique symptoms identified across drugs: {len(set([s for sublist in df_complete['side_effect_list'] for s in sublist]))}")
print(f"  Average symptoms per drug: {df_complete['side_effect_count'].mean():.1f}")
print(f"  Drugs with severe warnings: {df_complete[df_complete['side_effect_severity'] > 0].shape[0]} ({df_complete[df_complete['side_effect_severity'] > 0].shape[0]/len(df_complete)*100:.1f}%)")
print(f"  Drugs with life-threatening warnings: {df_complete['mention_death'].sum()} ({df_complete['mention_death'].sum()/len(df_complete)*100:.1f}%)\n")

all_symptoms = [symptom for symptoms in df_complete['side_effect_list'] for symptom in symptoms]
symptom_freq = Counter(all_symptoms)

df_rated = df_complete[df_complete['rating'].notna()].copy()
print(f"Rating Analysis Subset: {len(df_rated)} drugs with both side effects and ratings")
print(f"Correlation Analysis:")
corr_symptoms_rating = stats.spearmanr(df_rated['side_effect_count'], df_rated['rating'])
print(f"  Side Effect Count vs Rating: r={corr_symptoms_rating[0]:.3f}, p={corr_symptoms_rating[1]:.4f}")
corr_severity_rating = stats.spearmanr(df_rated['side_effect_severity'], df_rated['rating'])
print(f"  Severity Mentions vs Rating: r={corr_severity_rating[0]:.3f}, p={corr_severity_rating[1]:.4f}")

drug_class_exploded = df.copy()
drug_class_exploded['drug_classes'] = drug_class_exploded['drug_classes'].str.split(',')
drug_class_exploded = drug_class_exploded.explode('drug_classes')
drug_class_exploded['drug_classes'] = drug_class_exploded['drug_classes'].str.strip()

class_ratings = drug_class_exploded[drug_class_exploded['rating'].notna()].groupby('drug_classes').agg({
    'rating': ['mean', 'std', 'count'],
    'no_of_reviews': 'sum'
}).reset_index()
class_ratings.columns = ['drug_class', 'mean_rating', 'std_rating', 'count', 'total_reviews']
class_ratings = class_ratings[class_ratings['count'] >= 5].sort_values('mean_rating', ascending=False)

print(f"\nDrug Class Performance Analysis:")
print(f"  Total drug classes analyzed: {len(class_ratings)}")
print(f"  Best performing class: {class_ratings.iloc[0]['drug_class']} (rating: {class_ratings.iloc[0]['mean_rating']:.2f})")
print(f"  Worst performing class: {class_ratings.iloc[-1]['drug_class']} (rating: {class_ratings.iloc[-1]['mean_rating']:.2f})")
print(f"  Rating range across classes: {class_ratings['mean_rating'].max() - class_ratings['mean_rating'].min():.2f} points")

fig = plt.figure(figsize=(32, 24))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
fig.patch.set_facecolor('#0D1117')

ax1 = fig.add_subplot(gs[0, 0])
top_symptoms = dict(symptom_freq.most_common(18))
bars = ax1.barh(range(len(top_symptoms)), list(top_symptoms.values()),
                color=[GRADIENT[i % len(GRADIENT)] for i in range(len(top_symptoms))],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax1.set_yticks(range(len(top_symptoms)))
ax1.set_yticklabels(list(top_symptoms.keys()), fontsize=11, color=COLORS[5], weight='bold')
ax1.set_xlabel('Frequency Across Drug Profiles', fontsize=13, color=COLORS[5], weight='bold')
ax1.set_title('Most Common Side Effects\nFrequency Analysis Across All Drugs', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax1.set_facecolor('#141B25')
ax1.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax1.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, (bar, val) in enumerate(zip(bars, top_symptoms.values())):
    ax1.text(val, i, f'  {val}', va='center', ha='left', color=COLORS[5], fontsize=9, weight='bold')

ax2 = fig.add_subplot(gs[0, 1])
severity_dist = df_complete['side_effect_severity'].value_counts().sort_index()
severity_labels = ['No Warning', '1 Warning', '2 Warnings', '3+ Warnings']
severity_grouped = [
    len(df_complete[df_complete['side_effect_severity'] == 0]),
    len(df_complete[df_complete['side_effect_severity'] == 1]),
    len(df_complete[df_complete['side_effect_severity'] == 2]),
    len(df_complete[df_complete['side_effect_severity'] >= 3])
]
bars = ax2.bar(range(len(severity_grouped)), severity_grouped,
               color=[COLORS[2], COLORS[3], COLORS[4], COLORS[5]],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax2.set_xticks(range(len(severity_grouped)))
ax2.set_xticklabels(severity_labels, fontsize=11, color=COLORS[5], weight='bold', rotation=20, ha='right')
ax2.set_ylabel('Number of Drugs', fontsize=13, color=COLORS[5], weight='bold')
ax2.set_title('Severity Warning Distribution\nSerious/Severe Adverse Event Mentions', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax2.set_facecolor('#141B25')
ax2.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax2.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (bar, val) in enumerate(zip(bars, severity_grouped)):
    pct = val / len(df_complete) * 100
    ax2.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', 
            color=COLORS[5], fontsize=10, weight='bold')

ax3 = fig.add_subplot(gs[0, 2])
symptom_bins = pd.cut(df_complete['side_effect_count'], bins=[0, 5, 10, 15, 20, 50], 
                      labels=['1-5', '6-10', '11-15', '16-20', '21+'])
symptom_dist = symptom_bins.value_counts().sort_index()
bars = ax3.bar(range(len(symptom_dist)), symptom_dist.values,
               color=[GRADIENT[i+2] for i in range(len(symptom_dist))],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax3.set_xticks(range(len(symptom_dist)))
ax3.set_xticklabels(symptom_dist.index, fontsize=12, color=COLORS[5], weight='bold')
ax3.set_ylabel('Number of Drugs', fontsize=13, color=COLORS[5], weight='bold')
ax3.set_xlabel('Side Effect Count Category', fontsize=13, color=COLORS[5], weight='bold')
ax3.set_title('Side Effect Profile Complexity\nSymptom Count Distribution', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax3.set_facecolor('#141B25')
ax3.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax3.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (bar, val) in enumerate(zip(bars, symptom_dist.values)):
    ax3.text(i, val, f'{val}', ha='center', va='bottom', color=COLORS[5], fontsize=10, weight='bold')

ax4 = fig.add_subplot(gs[1, 0])
top_classes_chart = class_ratings.head(15)
bars = ax4.barh(range(len(top_classes_chart)), top_classes_chart['mean_rating'],
                xerr=top_classes_chart['std_rating']/2,
                color=[COLORS[2] if x >= 7.5 else COLORS[3] for x in top_classes_chart['mean_rating']],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9, capsize=6, 
                error_kw={'linewidth': 2})
ax4.set_yticks(range(len(top_classes_chart)))
ax4.set_yticklabels(top_classes_chart['drug_class'], fontsize=10, color=COLORS[5], weight='bold')
ax4.set_xlabel('Mean Rating (with SD)', fontsize=13, color=COLORS[5], weight='bold')
ax4.set_xlim(0, 10)
ax4.axvline(x=df_rated['rating'].mean(), color=COLORS[4], linestyle='--', linewidth=3, alpha=0.7, 
           label=f'Overall Mean: {df_rated["rating"].mean():.2f}')
ax4.set_title('Highest-Rated Drug Classes\nTop 15 by Mean Patient Satisfaction (n≥5)', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax4.set_facecolor('#141B25')
ax4.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax4.legend(fontsize=11, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
ax4.tick_params(colors=COLORS[5], labelsize=9, width=2)
for i, (val, count) in enumerate(zip(top_classes_chart['mean_rating'], top_classes_chart['count'])):
    ax4.text(val, i, f' {val:.2f} (n={count})', va='center', ha='left', 
            color=COLORS[5], fontsize=8, weight='bold')

ax5 = fig.add_subplot(gs[1, 1])
symptom_rating_data = df_rated.groupby(pd.cut(df_rated['side_effect_count'], bins=[0, 5, 10, 15, 20, 50]))['rating'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(symptom_rating_data))
bars = ax5.bar(x_pos, symptom_rating_data['mean'], 
               yerr=symptom_rating_data['std']/np.sqrt(symptom_rating_data['count']),
               color=COLORS[2], edgecolor=COLORS[5], linewidth=2.5, alpha=0.9,
               capsize=8, error_kw={'linewidth': 2.5})
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['1-5', '6-10', '11-15', '16-20', '21+'], fontsize=11, color=COLORS[5], weight='bold')
ax5.set_ylabel('Mean Rating', fontsize=13, color=COLORS[5], weight='bold')
ax5.set_xlabel('Number of Side Effects', fontsize=13, color=COLORS[5], weight='bold')
ax5.axhline(y=df_rated['rating'].mean(), color=COLORS[3], linestyle='--', linewidth=3, alpha=0.7)
ax5.set_ylim(0, 10)
ax5.set_title('Side Effect Burden vs Patient Satisfaction\nMean Ratings by Symptom Count (±SEM)', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax5.set_facecolor('#141B25')
ax5.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax5.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (val, count) in enumerate(zip(symptom_rating_data['mean'], symptom_rating_data['count'])):
    ax5.text(i, val, f'{val:.2f}\n(n={count})', ha='center', va='bottom', 
            color=COLORS[5], fontsize=9, weight='bold')

ax6 = fig.add_subplot(gs[1, 2])
condition_symptom = df_complete.groupby('medical_condition')['side_effect_count'].agg(['mean', 'std'])
condition_symptom = condition_symptom.sort_values('mean', ascending=False).head(15)
bars = ax6.barh(range(len(condition_symptom)), condition_symptom['mean'],
                xerr=condition_symptom['std']/2,
                color=[GRADIENT[i % len(GRADIENT)] for i in range(len(condition_symptom))],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9, capsize=5,
                error_kw={'linewidth': 2})
ax6.set_yticks(range(len(condition_symptom)))
ax6.set_yticklabels(condition_symptom.index, fontsize=10, color=COLORS[5], weight='bold')
ax6.set_xlabel('Mean Side Effect Count', fontsize=13, color=COLORS[5], weight='bold')
ax6.axvline(x=df_complete['side_effect_count'].mean(), color=COLORS[4], linestyle='--', 
           linewidth=3, alpha=0.7)
ax6.set_title('Side Effect Burden by Condition\nAverage Symptom Count per Drug (with SD)', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax6.set_facecolor('#141B25')
ax6.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax6.tick_params(colors=COLORS[5], labelsize=9, width=2)

ax7 = fig.add_subplot(gs[2, 0])
preg_symptom = df_complete[df_complete['pregnancy_category'] != 'N'].groupby('pregnancy_category')['side_effect_count'].agg(['mean', 'std', 'count'])
preg_symptom = preg_symptom.sort_index()
bars = ax7.bar(range(len(preg_symptom)), preg_symptom['mean'],
               yerr=preg_symptom['std']/2,
               color=[COLORS[2], COLORS[2], COLORS[3], COLORS[4], COLORS[5]],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9, capsize=7,
               error_kw={'linewidth': 2.5})
ax7.set_xticks(range(len(preg_symptom)))
ax7.set_xticklabels(preg_symptom.index, fontsize=13, color=COLORS[5], weight='bold')
ax7.set_ylabel('Mean Side Effect Count', fontsize=13, color=COLORS[5], weight='bold')
ax7.set_xlabel('FDA Pregnancy Category', fontsize=13, color=COLORS[5], weight='bold')
ax7.set_title('Side Effect Profile by Pregnancy Safety\nSymptom Burden Across FDA Categories', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax7.set_facecolor('#141B25')
ax7.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax7.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (val, count) in enumerate(zip(preg_symptom['mean'], preg_symptom['count'])):
    ax7.text(i, val, f'{val:.1f}\n(n={count})', ha='center', va='bottom', 
            color=COLORS[5], fontsize=10, weight='bold')

ax8 = fig.add_subplot(gs[2, 1])
rx_symptom = df_complete.groupby('rx_otc')['side_effect_count'].agg(['mean', 'std', 'count'])
bars = ax8.bar(range(len(rx_symptom)), rx_symptom['mean'],
               yerr=rx_symptom['std']/2,
               color=[COLORS[2], COLORS[3], COLORS[4]],
               edgecolor=COLORS[5], linewidth=2.5, alpha=0.9, capsize=8,
               error_kw={'linewidth': 2.5})
ax8.set_xticks(range(len(rx_symptom)))
ax8.set_xticklabels(rx_symptom.index, fontsize=13, color=COLORS[5], weight='bold')
ax8.set_ylabel('Mean Side Effect Count', fontsize=13, color=COLORS[5], weight='bold')
ax8.set_xlabel('Prescription Classification', fontsize=13, color=COLORS[5], weight='bold')
ax8.axhline(y=df_complete['side_effect_count'].mean(), color=COLORS[5], linestyle='--', 
           linewidth=2.5, alpha=0.5)
ax8.set_title('Side Effects by Regulatory Status\nPrescription vs OTC Safety Profiles', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax8.set_facecolor('#141B25')
ax8.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax8.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (val, count) in enumerate(zip(rx_symptom['mean'], rx_symptom['count'])):
    ax8.text(i, val, f'{val:.1f}\n(n={count})', ha='center', va='bottom', 
            color=COLORS[5], fontsize=10, weight='bold')

ax9 = fig.add_subplot(gs[2, 2])
life_threat_conditions = df_complete[df_complete['mention_death'] == 1].groupby('medical_condition').size().sort_values(ascending=False).head(12)
bars = ax9.barh(range(len(life_threat_conditions)), life_threat_conditions.values,
                color=COLORS[3], edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax9.set_yticks(range(len(life_threat_conditions)))
ax9.set_yticklabels(life_threat_conditions.index, fontsize=10, color=COLORS[5], weight='bold')
ax9.set_xlabel('Number of Drugs with Life-Threatening Warnings', fontsize=13, color=COLORS[5], weight='bold')
ax9.set_title('Severe Risk Warnings by Condition\nDrugs Mentioning Death/Fatal/Life-Threatening', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax9.set_facecolor('#141B25')
ax9.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax9.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, (bar, val) in enumerate(zip(bars, life_threat_conditions.values)):
    total_in_condition = len(df_complete[df_complete['medical_condition'] == life_threat_conditions.index[i]])
    pct = val / total_in_condition * 100
    ax9.text(val, i, f'  {val} ({pct:.0f}%)', va='center', ha='left', 
            color=COLORS[5], fontsize=9, weight='bold')

plt.savefig(f'{SAVE_PATH}drug_safety_intermediate_analysis.png', dpi=300, 
           facecolor='#0D1117', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: drug_safety_intermediate_analysis.png")

print("\n" + "="*80)
print("Advanced Statistical Insights")
print("="*80)

print(f"\nSymptom Frequency Analysis:")
top_5_symptoms = list(symptom_freq.most_common(5))
for i, (symptom, count) in enumerate(top_5_symptoms, 1):
    pct = count / len(df_complete) * 100
    print(f"  {i}. {symptom.title()}: {count} drugs ({pct:.1f}%)")

print(f"\nSeverity Stratification:")
severe_drugs = len(df_complete[df_complete['side_effect_severity'] > 0])
critical_drugs = len(df_complete[df_complete['mention_death'] == 1])
print(f"  Drugs with severity warnings: {severe_drugs} ({severe_drugs/len(df_complete)*100:.1f}%)")
print(f"  Drugs with life-threatening warnings: {critical_drugs} ({critical_drugs/len(df_complete)*100:.1f}%)")

print(f"\nRating-Symptom Relationship:")
if corr_symptoms_rating[1] < 0.05:
    direction = "negative" if corr_symptoms_rating[0] < 0 else "positive"
    print(f"  Statistically significant {direction} correlation (p={corr_symptoms_rating[1]:.4f})")
    if corr_symptoms_rating[0] < 0:
        print(f"  Interpretation: More side effects associated with lower patient satisfaction")
else:
    print(f"  No significant correlation between symptom count and rating (p={corr_symptoms_rating[1]:.4f})")

print(f"\nDrug Class Performance:")
print(f"  Best class: {class_ratings.iloc[0]['drug_class']}")
print(f"    Mean rating: {class_ratings.iloc[0]['mean_rating']:.2f}/10")
print(f"    Sample size: {class_ratings.iloc[0]['count']} drugs")
print(f"  Worst class: {class_ratings.iloc[-1]['drug_class']}")
print(f"    Mean rating: {class_ratings.iloc[-1]['mean_rating']:.2f}/10")
print(f"    Sample size: {class_ratings.iloc[-1]['count']} drugs")

print(f"\nPrescription vs OTC Safety:")
rx_only_symptoms = rx_symptom.loc['Rx', 'mean']
otc_symptoms = rx_symptom.loc['OTC', 'mean']
print(f"  Rx-only average symptoms: {rx_only_symptoms:.1f}")
print(f"  OTC average symptoms: {otc_symptoms:.1f}")
print(f"  Difference: {abs(rx_only_symptoms - otc_symptoms):.1f} symptoms")

from scipy.stats import ttest_ind
rx_only_data = df_complete[df_complete['rx_otc'] == 'Rx']['side_effect_count']
otc_data = df_complete[df_complete['rx_otc'] == 'OTC']['side_effect_count']
tstat, pval = ttest_ind(rx_only_data, otc_data)
if pval < 0.05:
    print(f"  Statistically significant difference (t={tstat:.2f}, p={pval:.4f})")
else:
    print(f"  No significant difference (t={tstat:.2f}, p={pval:.4f})")

print("\n" + "="*80)
print("Intermediate Analysis Complete")
print("="*80)
print("This framework extends foundation insights through text mining of side effect")
print("profiles, statistical testing of drug class performance, and correlation analysis")
print("between safety burden and patient outcomes. Next level implements predictive")
print("modeling and advanced natural language processing.")
