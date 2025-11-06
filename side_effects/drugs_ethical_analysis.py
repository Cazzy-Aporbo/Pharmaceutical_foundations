import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#1A1F3A', '#2D3250', '#D4A574', '#C85C3C', '#5B8FA3', '#E8B968']
GRADIENT = ['#1A1F3A', '#232945', '#2D3250', '#3C4565', '#5B8FA3', '#7BA3B5', '#9BB8C7', '#C85C3C', '#D4A574', '#E8B968']
SAVE_PATH = './'

plt.style.use('dark_background')

df = pd.read_csv('drugs_side_effects_drugs_com.csv')

print("Pharmaceutical Drug Safety Analysis: Ethical Framework\n")
print("="*80)
print("Safety Profiling, Pregnancy Risk Assessment, and Pharmacovigilance")
print("="*80 + "\n")

print("Phase 1: Pregnancy Safety Classification\n")

df_preg = df[df['pregnancy_category'].isin(['A', 'B', 'C', 'D', 'X'])].copy()

def categorize_risk(cat):
    if cat in ['A', 'B']:
        return 'Low Risk'
    elif cat == 'C':
        return 'Moderate Risk'
    elif cat in ['D', 'X']:
        return 'High Risk'
    return 'Unknown'

df_preg['risk_category'] = df_preg['pregnancy_category'].apply(categorize_risk)

print(f"Pregnancy Safety Data: {len(df_preg)} drugs with FDA classifications")
print(f"  Category A (Controlled studies show no risk): {len(df_preg[df_preg['pregnancy_category'] == 'A'])}")
print(f"  Category B (No evidence of risk): {len(df_preg[df_preg['pregnancy_category'] == 'B'])}")
print(f"  Category C (Risk cannot be ruled out): {len(df_preg[df_preg['pregnancy_category'] == 'C'])}")
print(f"  Category D (Positive evidence of risk): {len(df_preg[df_preg['pregnancy_category'] == 'D'])}")
print(f"  Category X (Contraindicated in pregnancy): {len(df_preg[df_preg['pregnancy_category'] == 'X'])}\n")

risk_by_condition = df_preg.groupby('medical_condition')['risk_category'].value_counts().unstack(fill_value=0)
risk_by_condition['total'] = risk_by_condition.sum(axis=1)
risk_by_condition['high_risk_pct'] = risk_by_condition['High Risk'] / risk_by_condition['total'] * 100
risk_by_condition = risk_by_condition.sort_values('high_risk_pct', ascending=False)

print("Conditions with Highest Pregnancy Risk:")
for condition in risk_by_condition.head(8).index:
    hr_count = risk_by_condition.loc[condition, 'High Risk']
    total = risk_by_condition.loc[condition, 'total']
    pct = risk_by_condition.loc[condition, 'high_risk_pct']
    print(f"  {condition}: {hr_count}/{total} drugs ({pct:.1f}%) high risk")

print("\n\nPhase 2: Severe Adverse Event Profiling\n")

def extract_severe_events(text):
    if pd.isna(text):
        return {
            'death': 0, 'hospitalization': 0, 'disability': 0, 
            'liver_damage': 0, 'kidney_damage': 0, 'heart_attack': 0,
            'stroke': 0, 'bleeding': 0, 'infection': 0, 'seizure': 0
        }
    text_lower = text.lower()
    return {
        'death': int('death' in text_lower or 'fatal' in text_lower or 'life-threatening' in text_lower),
        'hospitalization': int('hospital' in text_lower or 'admission' in text_lower),
        'disability': int('disability' in text_lower or 'permanent' in text_lower or 'irreversible' in text_lower),
        'liver_damage': int('liver damage' in text_lower or 'hepatotoxicity' in text_lower or 'liver failure' in text_lower),
        'kidney_damage': int('kidney' in text_lower or 'renal' in text_lower),
        'heart_attack': int('heart attack' in text_lower or 'myocardial infarction' in text_lower or 'cardiac arrest' in text_lower),
        'stroke': int('stroke' in text_lower),
        'bleeding': int('bleeding' in text_lower or 'hemorrhage' in text_lower),
        'infection': int('infection' in text_lower or 'sepsis' in text_lower),
        'seizure': int('seizure' in text_lower or 'convulsion' in text_lower)
    }

severe_events_data = df['side_effects'].apply(extract_severe_events).apply(pd.Series)
df_safety = pd.concat([df, severe_events_data], axis=1)

print("Severe Adverse Event Detection:")
for event in ['death', 'hospitalization', 'liver_damage', 'heart_attack', 'bleeding']:
    count = df_safety[event].sum()
    pct = count / len(df_safety) * 100
    print(f"  {event.replace('_', ' ').title()}: {count} drugs ({pct:.1f}%)")

df_safety['severe_event_count'] = df_safety[['death', 'hospitalization', 'disability', 'liver_damage', 
                                              'kidney_damage', 'heart_attack', 'stroke', 'bleeding', 
                                              'infection', 'seizure']].sum(axis=1)

print(f"\nAggregate Risk Profile:")
print(f"  No severe warnings: {len(df_safety[df_safety['severe_event_count'] == 0])} drugs ({len(df_safety[df_safety['severe_event_count'] == 0])/len(df_safety)*100:.1f}%)")
print(f"  1-2 severe warnings: {len(df_safety[(df_safety['severe_event_count'] >= 1) & (df_safety['severe_event_count'] <= 2)])} drugs ({len(df_safety[(df_safety['severe_event_count'] >= 1) & (df_safety['severe_event_count'] <= 2)])/len(df_safety)*100:.1f}%)")
print(f"  3+ severe warnings: {len(df_safety[df_safety['severe_event_count'] >= 3])} drugs ({len(df_safety[df_safety['severe_event_count'] >= 3])/len(df_safety)*100:.1f}%)")

print("\n\nPhase 3: Prescription Access Equity Analysis\n")

rx_condition_counts = df.groupby(['medical_condition', 'rx_otc']).size().unstack(fill_value=0)
rx_condition_counts['total'] = rx_condition_counts.sum(axis=1)
rx_condition_counts['rx_only_pct'] = (rx_condition_counts.get('Rx', 0) / rx_condition_counts['total'] * 100)

conditions_high_barrier = rx_condition_counts[rx_condition_counts['rx_only_pct'] > 80].sort_values('rx_only_pct', ascending=False)
conditions_accessible = rx_condition_counts[rx_condition_counts.get('OTC', 0) > 10].sort_values('OTC', ascending=False)

print("Prescription Access Patterns:")
print(f"\nConditions with Limited OTC Access (>80% Rx-only):")
for condition in conditions_high_barrier.head(6).index:
    rx_pct = conditions_high_barrier.loc[condition, 'rx_only_pct']
    total = conditions_high_barrier.loc[condition, 'total']
    print(f"  {condition}: {rx_pct:.1f}% Rx-only (n={total})")

print(f"\nConditions with Good OTC Availability:")
for condition in conditions_accessible.head(6).index:
    otc_count = conditions_accessible.loc[condition, 'OTC']
    total = conditions_accessible.loc[condition, 'total']
    pct = otc_count / total * 100
    print(f"  {condition}: {otc_count} OTC options ({pct:.1f}%)")

print("\n\nPhase 4: Drug Class Safety Analysis\n")

df_class_exploded = df.copy()
df_class_exploded['drug_classes'] = df_class_exploded['drug_classes'].str.split(',')
df_class_exploded = df_class_exploded.explode('drug_classes')
df_class_exploded['drug_classes'] = df_class_exploded['drug_classes'].str.strip()

class_safety = df_class_exploded.merge(df_safety[['drug_name', 'severe_event_count']], on='drug_name', how='left')
class_risk = class_safety.groupby('drug_classes').agg({
    'severe_event_count': ['mean', 'count'],
    'rating': 'mean'
}).reset_index()
class_risk.columns = ['drug_class', 'mean_severe_events', 'drug_count', 'mean_rating']
class_risk = class_risk[class_risk['drug_count'] >= 5].sort_values('mean_severe_events', ascending=False)

print("Drug Classes by Safety Profile (n≥5):")
print("\nHighest Risk Classes:")
for idx, row in class_risk.head(6).iterrows():
    print(f"  {row['drug_class']}: {row['mean_severe_events']:.2f} avg warnings (n={row['drug_count']}, rating={row['mean_rating']:.2f})")

print("\nSafest Classes:")
for idx, row in class_risk.tail(6).iterrows():
    print(f"  {row['drug_class']}: {row['mean_severe_events']:.2f} avg warnings (n={row['drug_count']}, rating={row['mean_rating']:.2f})")

fig = plt.figure(figsize=(32, 28))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
fig.patch.set_facecolor('#0D1117')

ax1 = fig.add_subplot(gs[0, 0])
preg_dist = df_preg['pregnancy_category'].value_counts().sort_index()
colors_preg = [COLORS[2], COLORS[2], COLORS[3], COLORS[4], COLORS[5]]
bars = ax1.bar(range(len(preg_dist)), preg_dist.values,
               color=colors_preg, edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax1.set_xticks(range(len(preg_dist)))
ax1.set_xticklabels(preg_dist.index, fontsize=13, color=COLORS[5], weight='bold')
ax1.set_ylabel('Number of Drugs', fontsize=13, color=COLORS[5], weight='bold')
ax1.set_xlabel('FDA Pregnancy Category', fontsize=13, color=COLORS[5], weight='bold')
ax1.set_title('Pregnancy Safety Distribution\nFDA Risk Classification Across All Drugs', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax1.set_facecolor('#141B25')
ax1.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax1.tick_params(colors=COLORS[5], labelsize=11, width=2)
labels_detailed = {'A': 'Safest\nControlled Studies', 'B': 'Safe\nNo Evidence of Risk', 
                  'C': 'Caution\nRisk Not Ruled Out', 'D': 'Risk\nPositive Evidence', 
                  'X': 'Contraindicated\nFetal Abnormalities'}
for i, (cat, val) in enumerate(zip(preg_dist.index, preg_dist.values)):
    pct = val / len(df_preg) * 100
    ax1.text(i, val, f'{val}\n({pct:.1f}%)\n{labels_detailed[cat]}', ha='center', va='bottom', 
            color=COLORS[5], fontsize=9, weight='bold')

ax2 = fig.add_subplot(gs[0, 1])
top_high_risk = risk_by_condition.head(12)
bars = ax2.barh(range(len(top_high_risk)), top_high_risk['high_risk_pct'],
                color=[COLORS[3] if x > 50 else COLORS[2] for x in top_high_risk['high_risk_pct']],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax2.set_yticks(range(len(top_high_risk)))
ax2.set_yticklabels(top_high_risk.index, fontsize=10, color=COLORS[5], weight='bold')
ax2.set_xlabel('Percentage High-Risk Drugs (Cat D+X)', fontsize=13, color=COLORS[5], weight='bold')
ax2.axvline(x=50, color=COLORS[5], linestyle='--', linewidth=2.5, alpha=0.5)
ax2.set_title('Pregnancy Risk by Medical Condition\nPercent Category D or X Drugs', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax2.set_facecolor('#141B25')
ax2.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax2.tick_params(colors=COLORS[5], labelsize=9, width=2)
for i, (val, total) in enumerate(zip(top_high_risk['high_risk_pct'], top_high_risk['High Risk'])):
    ax2.text(val, i, f' {val:.0f}% (n={total})', va='center', ha='left', 
            color=COLORS[5], fontsize=8, weight='bold')

ax3 = fig.add_subplot(gs[0, 2])
severe_event_types = ['death', 'hospitalization', 'liver_damage', 'heart_attack', 'bleeding', 
                      'kidney_damage', 'stroke', 'infection', 'seizure', 'disability']
severe_counts = [df_safety[event].sum() for event in severe_event_types]
bars = ax3.barh(range(len(severe_event_types)), severe_counts,
                color=[GRADIENT[i % len(GRADIENT)] for i in range(len(severe_event_types))],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax3.set_yticks(range(len(severe_event_types)))
ax3.set_yticklabels([e.replace('_', ' ').title() for e in severe_event_types], 
                    fontsize=10, color=COLORS[5], weight='bold')
ax3.set_xlabel('Number of Drugs with Warning', fontsize=13, color=COLORS[5], weight='bold')
ax3.set_title('Severe Adverse Event Frequency\nMajor Safety Warnings Across Drug Database', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax3.set_facecolor('#141B25')
ax3.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax3.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, val in enumerate(severe_counts):
    pct = val / len(df_safety) * 100
    ax3.text(val, i, f'  {val} ({pct:.1f}%)', va='center', ha='left', 
            color=COLORS[5], fontsize=8, weight='bold')

ax4 = fig.add_subplot(gs[1, 0])
severe_bins = pd.cut(df_safety['severe_event_count'], bins=[-0.1, 0, 1, 2, 3, 10], 
                     labels=['None', '1 Warning', '2 Warnings', '3 Warnings', '4+ Warnings'])
severe_dist = severe_bins.value_counts().sort_index()
colors_severity = [COLORS[2], COLORS[2], COLORS[3], COLORS[4], COLORS[5]]
bars = ax4.bar(range(len(severe_dist)), severe_dist.values,
               color=colors_severity, edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax4.set_xticks(range(len(severe_dist)))
ax4.set_xticklabels(severe_dist.index, fontsize=11, color=COLORS[5], weight='bold', rotation=20, ha='right')
ax4.set_ylabel('Number of Drugs', fontsize=13, color=COLORS[5], weight='bold')
ax4.set_title('Severe Warning Burden Distribution\nAggregate Major Adverse Events per Drug', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax4.set_facecolor('#141B25')
ax4.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax4.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, val in enumerate(severe_dist.values):
    pct = val / len(df_safety) * 100
    ax4.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', 
            color=COLORS[5], fontsize=10, weight='bold')

ax5 = fig.add_subplot(gs[1, 1])
rx_access_conditions = rx_condition_counts.sort_values('rx_only_pct', ascending=False).head(12)
bars = ax5.barh(range(len(rx_access_conditions)), rx_access_conditions['rx_only_pct'],
                color=[COLORS[3] if x > 90 else COLORS[2] for x in rx_access_conditions['rx_only_pct']],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax5.set_yticks(range(len(rx_access_conditions)))
ax5.set_yticklabels(rx_access_conditions.index, fontsize=10, color=COLORS[5], weight='bold')
ax5.set_xlabel('Percentage Prescription-Only', fontsize=13, color=COLORS[5], weight='bold')
ax5.axvline(x=90, color=COLORS[5], linestyle='--', linewidth=2.5, alpha=0.5, label='90% Threshold')
ax5.set_title('Healthcare Access Barriers\nConditions with Limited OTC Options', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax5.set_facecolor('#141B25')
ax5.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax5.legend(fontsize=10, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
ax5.tick_params(colors=COLORS[5], labelsize=9, width=2)

ax6 = fig.add_subplot(gs[1, 2])
conditions_otc_best = conditions_accessible.head(12)
bars = ax6.barh(range(len(conditions_otc_best)), conditions_otc_best['OTC'],
                color=COLORS[2], edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax6.set_yticks(range(len(conditions_otc_best)))
ax6.set_yticklabels(conditions_otc_best.index, fontsize=10, color=COLORS[5], weight='bold')
ax6.set_xlabel('Number of OTC Options', fontsize=13, color=COLORS[5], weight='bold')
ax6.set_title('Over-the-Counter Accessibility\nConditions with Most OTC Drug Options', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax6.set_facecolor('#141B25')
ax6.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax6.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, (val, total) in enumerate(zip(conditions_otc_best['OTC'], conditions_otc_best['total'])):
    pct = val / total * 100
    ax6.text(val, i, f'  {val} ({pct:.0f}%)', va='center', ha='left', 
            color=COLORS[5], fontsize=8, weight='bold')

ax7 = fig.add_subplot(gs[2, 0])
top_risky_classes = class_risk.head(15)
bars = ax7.barh(range(len(top_risky_classes)), top_risky_classes['mean_severe_events'],
                color=[COLORS[3] if x > 2 else COLORS[2] for x in top_risky_classes['mean_severe_events']],
                edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax7.set_yticks(range(len(top_risky_classes)))
ax7.set_yticklabels(top_risky_classes['drug_class'], fontsize=9, color=COLORS[5], weight='bold')
ax7.set_xlabel('Mean Severe Warning Count', fontsize=13, color=COLORS[5], weight='bold')
ax7.set_title('Highest-Risk Drug Classes\nAverage Severe Adverse Events (n≥5)', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax7.set_facecolor('#141B25')
ax7.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax7.tick_params(colors=COLORS[5], labelsize=9, width=2)
for i, (val, count) in enumerate(zip(top_risky_classes['mean_severe_events'], top_risky_classes['drug_count'])):
    ax7.text(val, i, f' {val:.2f} (n={count})', va='center', ha='left', 
            color=COLORS[5], fontsize=8, weight='bold')

ax8 = fig.add_subplot(gs[2, 1])
df_rated_safety = df_safety[df_safety['rating'].notna()].copy()
severe_rating_groups = df_rated_safety.groupby(pd.cut(df_rated_safety['severe_event_count'], 
                                                       bins=[-0.1, 0, 1, 2, 5]))['rating'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(severe_rating_groups))
bars = ax8.bar(x_pos, severe_rating_groups['mean'],
               yerr=severe_rating_groups['std']/np.sqrt(severe_rating_groups['count']),
               color=COLORS[2], edgecolor=COLORS[5], linewidth=2.5, alpha=0.9,
               capsize=8, error_kw={'linewidth': 2.5})
ax8.set_xticks(x_pos)
ax8.set_xticklabels(['None', '1', '2', '3+'], fontsize=12, color=COLORS[5], weight='bold')
ax8.set_ylabel('Mean Rating', fontsize=13, color=COLORS[5], weight='bold')
ax8.set_xlabel('Number of Severe Warnings', fontsize=13, color=COLORS[5], weight='bold')
ax8.axhline(y=df_rated_safety['rating'].mean(), color=COLORS[3], linestyle='--', linewidth=3, alpha=0.7)
ax8.set_ylim(0, 10)
ax8.set_title('Patient Satisfaction vs Safety Profile\nRating by Severe Warning Count (±SEM)', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax8.set_facecolor('#141B25')
ax8.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax8.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (val, count) in enumerate(zip(severe_rating_groups['mean'], severe_rating_groups['count'])):
    ax8.text(i, val, f'{val:.2f}\n(n={count})', ha='center', va='bottom', 
            color=COLORS[5], fontsize=9, weight='bold')

ax9 = fig.add_subplot(gs[2, 2])
preg_severe = df_preg[df_preg['pregnancy_category'].isin(['A', 'B', 'C', 'D', 'X'])].copy()
preg_severe = preg_severe.merge(df_safety[['drug_name', 'severe_event_count']], on='drug_name', how='left')
preg_severe_grouped = preg_severe.groupby('pregnancy_category')['severe_event_count'].agg(['mean', 'std', 'count'])
bars = ax9.bar(range(len(preg_severe_grouped)), preg_severe_grouped['mean'],
               yerr=preg_severe_grouped['std']/2,
               color=colors_preg, edgecolor=COLORS[5], linewidth=2.5, alpha=0.9,
               capsize=7, error_kw={'linewidth': 2.5})
ax9.set_xticks(range(len(preg_severe_grouped)))
ax9.set_xticklabels(preg_severe_grouped.index, fontsize=13, color=COLORS[5], weight='bold')
ax9.set_ylabel('Mean Severe Warning Count', fontsize=13, color=COLORS[5], weight='bold')
ax9.set_xlabel('FDA Pregnancy Category', fontsize=13, color=COLORS[5], weight='bold')
ax9.set_title('Pregnancy Risk vs General Safety Profile\nSevere Warnings by FDA Category', 
              fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax9.set_facecolor('#141B25')
ax9.grid(True, alpha=0.15, color=COLORS[4], axis='y', linewidth=1.5)
ax9.tick_params(colors=COLORS[5], labelsize=11, width=2)
for i, (val, count) in enumerate(zip(preg_severe_grouped['mean'], preg_severe_grouped['count'])):
    ax9.text(i, val, f'{val:.2f}\n(n={count})', ha='center', va='bottom', 
            color=COLORS[5], fontsize=9, weight='bold')

ax10 = fig.add_subplot(gs[3, :])
condition_severe = df_safety.groupby('medical_condition')['severe_event_count'].agg(['mean', 'count'])
condition_severe = condition_severe[condition_severe['count'] >= 20].sort_values('mean', ascending=False).head(18)
bars = ax10.barh(range(len(condition_severe)), condition_severe['mean'],
                 color=[GRADIENT[i % len(GRADIENT)] for i in range(len(condition_severe))],
                 edgecolor=COLORS[5], linewidth=2.5, alpha=0.9)
ax10.set_yticks(range(len(condition_severe)))
ax10.set_yticklabels(condition_severe.index, fontsize=11, color=COLORS[5], weight='bold')
ax10.set_xlabel('Mean Severe Warning Count per Drug', fontsize=13, color=COLORS[5], weight='bold')
ax10.axvline(x=df_safety['severe_event_count'].mean(), color=COLORS[4], linestyle='--', 
            linewidth=3, alpha=0.7, label=f'Overall Mean: {df_safety["severe_event_count"].mean():.2f}')
ax10.set_title('Safety Profile by Medical Condition\nAverage Severe Adverse Event Warnings (n≥20 drugs)', 
               fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax10.set_facecolor('#141B25')
ax10.grid(True, alpha=0.15, color=COLORS[4], axis='x', linewidth=1.5)
ax10.legend(fontsize=12, framealpha=0.95, facecolor='#0D1117', edgecolor=COLORS[2])
ax10.tick_params(colors=COLORS[5], labelsize=10, width=2)
for i, (val, count) in enumerate(zip(condition_severe['mean'], condition_severe['count'])):
    ax10.text(val, i, f' {val:.2f} (n={count})', va='center', ha='left', 
             color=COLORS[5], fontsize=9, weight='bold')

plt.savefig(f'{SAVE_PATH}drug_safety_ethical_analysis.png', dpi=300, 
           facecolor='#0D1117', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: drug_safety_ethical_analysis.png")

print("\n" + "="*80)
print("Ethical Considerations and Policy Implications")
print("="*80)

print(f"\nPregnancy Safety Concerns:")
high_risk_preg = len(df_preg[df_preg['pregnancy_category'].isin(['D', 'X'])])
total_classified = len(df_preg)
print(f"  High-risk drugs (D+X): {high_risk_preg} of {total_classified} ({high_risk_preg/total_classified*100:.1f}%)")
print(f"  Safest drugs (A+B): {len(df_preg[df_preg['pregnancy_category'].isin(['A', 'B'])])} ({len(df_preg[df_preg['pregnancy_category'].isin(['A', 'B'])])/total_classified*100:.1f}%)")
print(f"  Unclear risk (C): {len(df_preg[df_preg['pregnancy_category'] == 'C'])} ({len(df_preg[df_preg['pregnancy_category'] == 'C'])/total_classified*100:.1f}%)")

print(f"\nSevere Adverse Event Prevalence:")
life_threat = df_safety['death'].sum()
organ_damage = df_safety[['liver_damage', 'kidney_damage']].sum().sum()
cardiovascular = df_safety[['heart_attack', 'stroke']].sum().sum()
print(f"  Life-threatening warnings: {life_threat} drugs ({life_threat/len(df_safety)*100:.1f}%)")
print(f"  Organ damage warnings: {organ_damage} drugs ({organ_damage/len(df_safety)*100:.1f}%)")
print(f"  Cardiovascular events: {cardiovascular} drugs ({cardiovascular/len(df_safety)*100:.1f}%)")

print(f"\nHealthcare Access Equity:")
high_barrier_conditions = len(conditions_high_barrier)
print(f"  Conditions with >80% Rx-only: {high_barrier_conditions}")
print(f"  Total OTC drugs available: {rx_condition_counts['OTC'].sum():.0f}")
print(f"  Average OTC options per condition: {rx_condition_counts['OTC'].mean():.1f}")

print(f"\nDrug Class Risk Stratification:")
high_risk_classes = len(class_risk[class_risk['mean_severe_events'] > 2])
print(f"  High-risk drug classes (>2 avg warnings): {high_risk_classes} of {len(class_risk)}")
print(f"  Safest class: {class_risk.iloc[-1]['drug_class']} ({class_risk.iloc[-1]['mean_severe_events']:.2f} avg warnings)")
print(f"  Riskiest class: {class_risk.iloc[0]['drug_class']} ({class_risk.iloc[0]['mean_severe_events']:.2f} avg warnings)")

print(f"\nRating-Safety Correlation:")
from scipy.stats import spearmanr
rated_with_severe = df_rated_safety[['rating', 'severe_event_count']].dropna()
corr, pval = spearmanr(rated_with_severe['rating'], rated_with_severe['severe_event_count'])
print(f"  Spearman correlation: {corr:.3f} (p={pval:.4f})")
if pval < 0.05:
    if corr < 0:
        print(f"  Interpretation: Higher severe warnings associated with lower ratings")
    else:
        print(f"  Interpretation: Unexpected positive correlation requires investigation")
else:
    print(f"  Interpretation: No significant relationship between safety warnings and ratings")

print("\n" + "="*80)
print("Ethical Framework Complete")
print("="*80)
print("This analysis identifies critical safety patterns, pregnancy risks, access barriers,")
print("and regulatory considerations essential for responsible pharmacovigilance and patient")
print("safety advocacy. These insights inform clinical decision-making, policy development,")
print("and risk communication strategies in pharmaceutical care.")
