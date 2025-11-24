import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--condition_number', type=int, choices=range(1, 7), default=1,
                    help='Condition number to evaluate (1-6)')
args = parser.parse_args()

condition_number = int(args.condition_number)

# --- Step 1: Read data ---
df_re = pd.read_csv(f'{condition_number}_re.csv')
df_unre = pd.read_csv(f'{condition_number}_unre.csv')

df_re['condition_type']   = 'reversed'    
df_unre['condition_type'] = 'non_reverse' 

df = pd.concat([df_re, df_unre], ignore_index=True)

# Convert hd_mask_pos from string to integer (extract digits)
df['hd_mask_pos'] = df['hd_mask_pos'].str.extract(r'(\d+)').astype(int)

# --- Step 2: Compute the measure of dataset diversity ---
# 1) Extract the residue at the masked position
df['aa_at_mask'] = df.apply(lambda row: row['hd_original'][row['hd_mask_pos']], axis=1)

# 2) Group by hd_mask_pos and count unique amino acids
diversity_df = (
    df.groupby('hd_mask_pos')['aa_at_mask']
      .nunique()
      .reset_index(name='diversity')
)

# --- Step 3: Melt the DataFrame to long format for NLL ---
nll_cols = [col for col in df.columns if 'nll' in col]
df_nll = df.melt(
    id_vars=['hd_original', 'hd_mask_count', 'hd_mask_pos', 'condition_type'], 
    value_vars=nll_cols,
    var_name='model', 
    value_name='nll'
)
df_nll['model'] = df_nll['model'].str.extract(r'(condition\d+)')

# --- Step 4: Melt the DataFrame for accuracy ---
acc_cols = [col for col in df.columns if 'acc' in col]
df_acc = df.melt(
    id_vars=['hd_original', 'hd_mask_count', 'hd_mask_pos', 'condition_type'], 
    value_vars=acc_cols,
    var_name='model', 
    value_name='accuracy'
)
df_acc['model'] = df_acc['model'].str.extract(r'(condition\d+)')

# --- Step 5: Merge NLL and accuracy DataFrames ---
df_long = pd.merge(
    df_nll, 
    df_acc, 
    on=['hd_original', 'hd_mask_count', 'hd_mask_pos', 'condition_type', 'model']
)

# --- Step 6: Compute average nll and accuracy per mask position, model, and condition_type ---
avg_perf = (
    df_long.groupby(['hd_mask_pos', 'model', 'condition_type'])
           .agg({'nll': 'mean', 'accuracy': 'mean'})
           .reset_index()
)

# --- Step 7: Merge avg_perf with diversity ---
# We'll merge so each row has the average accuracy + the diversity at that position
merged_df = pd.merge(avg_perf, diversity_df, on='hd_mask_pos', how='left')

# --- Step 8: Plotting with a secondary y-axis for diversity ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary axis: Accuracy
sns.lineplot(
    data=merged_df, 
    x='hd_mask_pos', 
    y='accuracy', 
    hue='condition_type',  
    style='condition_type',
    markers=True, 
    dashes=False,
    ax=ax1
)

ax1.set_title(f'Model Accuracy Across Mask Positions (Condition {condition_number})')
ax1.set_xlabel('Mask Position')
ax1.set_ylabel('Average Accuracy')

# Create a second y-axis for the diversity
ax2 = ax1.twinx()
sns.lineplot(
    data=diversity_df, 
    x='hd_mask_pos', 
    y='diversity', 
    color='black', 
    marker='o',
    ax=ax2,
    legend=False  # We'll manually handle the legend
)
ax2.set_ylabel('Diversity (Unique AAs)')

# Combine legends from ax1 and ax2
lines1, labels1 = ax1.get_legend_handles_labels()
# Manually create a handle/label for the black line (diversity)
line_diversity = ax2.lines[0]
line_diversity.set_label('diversity')

lines2 = [line_diversity]
labels2 = [line_diversity.get_label()]

all_handles = lines1 + lines2
all_labels = labels1 + labels2

# Place legend at center right
ax1.legend(all_handles, all_labels, loc='center right')

plt.tight_layout()
plt.savefig(f'{condition_number}_accuracy_diversity_plot.png')
plt.show()

# --- Step 9: Compute Spearman correlation between accuracy and diversity ---
print("\n=== Spearman Correlation ===")

# 9a) Overall correlation (all condition_types combined)
rho_all, p_all = spearmanr(merged_df['accuracy'], merged_df['diversity'])
print(f"Overall Spearman's R: {rho_all:.3f}, p-value: {p_all:.3e}")

# 9b) By condition_type
for cond_type in merged_df['condition_type'].unique():
    subset = merged_df[merged_df['condition_type'] == cond_type]
    rho, p_val = spearmanr(subset['accuracy'], subset['diversity'])
    print(f"Condition: {cond_type}")
    print(f"  Spearman's R: {rho:.3f}, p-value: {p_val:.3e}")

# --- Optional: Print out the aggregated data for debugging ---
print("\nData for 'non_reverse':")
print(merged_df[merged_df['condition_type'] == 'non_reverse'])

print("\nData for 'reversed':")
print(merged_df[merged_df['condition_type'] == 'reversed'])
