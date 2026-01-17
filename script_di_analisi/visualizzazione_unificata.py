import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Define paths
base_dir = Path(__file__).parent
testbigdata_dir = base_dir / "TestBigData"
originale_dir = base_dir / "Risultato codice originale"
output_dir = base_dir / "confronto_risultati"

# Test configurations
testbigdata_tests = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "TestBuono"]
originale_tests = ["Test1", "Test2"]

def load_test_data(test_dir, test_name):
    """Load test results from CSV file"""
    csv_path = test_dir / test_name / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None

# Create comprehensive single plot
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main plot: All F1 curves together
ax_main = fig.add_subplot(gs[0:2, :])

# Color schemes
new_colors = plt.cm.Greens(np.linspace(0.4, 0.9, 7))
orig_colors = plt.cm.Reds(np.linspace(0.6, 0.9, 2))

# Plot new code tests
print("Plotting new code tests...")
for idx, test in enumerate(testbigdata_tests):
    df = load_test_data(testbigdata_dir, test)
    if df is not None:
        label = f"New_{test}"
        ax_main.plot(df['epoch'], df['test_f1'], 
                    label=label, 
                    color=new_colors[idx], 
                    linewidth=2.5,
                    alpha=0.8)
        
        # Mark best point
        best_idx = df['test_f1'].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_f1 = df.loc[best_idx, 'test_f1']
        ax_main.scatter(best_epoch, best_f1, color=new_colors[idx], 
                       s=150, marker='*', edgecolors='black', linewidth=1.5, zorder=5)

# Plot original code tests
print("Plotting original code tests...")
for idx, test in enumerate(originale_tests):
    df = load_test_data(originale_dir, test)
    if df is not None:
        label = f"Original_{test}"
        ax_main.plot(df['epoch'], df['test_f1'], 
                    label=label, 
                    color=orig_colors[idx], 
                    linewidth=3,
                    alpha=0.9,
                    linestyle='--')
        
        # Mark best point
        best_idx = df['test_f1'].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_f1 = df.loc[best_idx, 'test_f1']
        ax_main.scatter(best_epoch, best_f1, color=orig_colors[idx], 
                       s=200, marker='D', edgecolors='black', linewidth=2, zorder=5)

ax_main.set_xlabel('Epoch', fontweight='bold', fontsize=14)
ax_main.set_ylabel('Test F1 Score', fontweight='bold', fontsize=14)
ax_main.set_title('Confronto Completo: Tutte le Configurazioni Test\n(* = Best per Nuovo Codice, ◆ = Best per Codice Originale)', 
                  fontweight='bold', fontsize=16, pad=20)
ax_main.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.9)
ax_main.grid(alpha=0.3, linestyle=':', linewidth=1)
ax_main.set_ylim([0, 100])

# Add horizontal line at 85% (good performance threshold)
ax_main.axhline(y=85, color='gold', linestyle=':', linewidth=2, alpha=0.5, label='Soglia 85%')

# Bottom left: Best F1 comparison
ax_bl = fig.add_subplot(gs[2, 0])
summary_df = pd.read_csv(output_dir / "comparison_summary.csv")

configs = summary_df['Configuration'].str.replace('New_', 'N_').str.replace('Original_', 'O_')
f1_scores = summary_df['best_test_f1']
colors_bar = ['#2ecc71' if 'new' in t else '#e74c3c' for t in summary_df['Type']]

bars = ax_bl.barh(configs, f1_scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
ax_bl.set_xlabel('Best Test F1 Score', fontweight='bold', fontsize=11)
ax_bl.set_title('Miglior F1 per Configurazione', fontweight='bold', fontsize=12)
ax_bl.grid(axis='x', alpha=0.3)

# Add values
for i, (bar, val) in enumerate(zip(bars, f1_scores)):
    ax_bl.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
              f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

# Bottom middle: Convergence speed
ax_bm = fig.add_subplot(gs[2, 1])
epochs_to_best = summary_df['epochs_to_best']

bars = ax_bm.barh(configs, epochs_to_best, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
ax_bm.set_xlabel('Epoche per Raggiungere Best F1', fontweight='bold', fontsize=11)
ax_bm.set_title('Velocità di Convergenza', fontweight='bold', fontsize=12)
ax_bm.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, epochs_to_best)):
    ax_bm.text(val + 1, bar.get_y() + bar.get_height()/2, 
              f'{int(val)}', va='center', fontsize=9, fontweight='bold')

# Bottom right: Stability
ax_br = fig.add_subplot(gs[2, 2])
stability = summary_df['stability']

bars = ax_br.barh(configs, stability, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
ax_br.set_xlabel('Deviazione Standard (ultime 10 epoche)', fontweight='bold', fontsize=11)
ax_br.set_title('Stabilità (meno è meglio)', fontweight='bold', fontsize=12)
ax_br.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, stability)):
    ax_br.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
              f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

plt.savefig(output_dir / "confronto_completo_unificato.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved unified comparison to {output_dir / 'confronto_completo_unificato.png'}")
plt.close()

print("\n" + "="*80)
print("VISUALIZZAZIONE UNIFICATA COMPLETATA")
print("="*80)
