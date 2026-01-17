import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define paths
base_dir = Path(__file__).parent
testbigdata_dir = base_dir / "TestBigData"
originale_dir = base_dir / "Risultato codice originale"
output_dir = base_dir / "confronto_risultati"
output_dir.mkdir(exist_ok=True)

# Test configurations
testbigdata_tests = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "TestBuono"]
originale_tests = ["Test1", "Test2"]

def load_test_data(test_dir, test_name):
    """Load test results from CSV file"""
    csv_path = test_dir / test_name / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Extract configuration parameters from combination name
        if 'combination' in df.columns and len(df) > 0:
            config = df['combination'].iloc[0]
            params = extract_params(config)
            return df, params
    return None, None

def extract_params(config_str):
    """Extract parameters from configuration string"""
    params = {}
    parts = config_str.split('_')
    for i in range(0, len(parts)-1, 2):
        if i+1 < len(parts):
            params[parts[i]] = parts[i+1]
    return params

def get_best_metrics(df):
    """Extract best and final metrics from results"""
    if df is None or len(df) == 0:
        return None
    
    metrics = {
        'best_test_f1': df['test_f1'].max(),
        'best_test_acc': df['test_accuracy'].max(),
        'final_test_f1': df['test_f1'].iloc[-1],
        'final_test_acc': df['test_accuracy'].iloc[-1],
        'best_train_f1': df['train_f1'].max(),
        'best_train_acc': df['train_accuracy'].max(),
        'final_train_f1': df['train_f1'].iloc[-1],
        'final_train_acc': df['train_accuracy'].iloc[-1],
        'min_test_loss': df['test_loss'].min(),
        'final_test_loss': df['test_loss'].iloc[-1],
        'epochs_to_best': df.loc[df['test_f1'].idxmax(), 'epoch'],
        'total_epochs': len(df),
        'convergence_speed': df.loc[df['test_f1'].idxmax(), 'epoch'] / len(df) if len(df) > 0 else 0,
        'stability': df['test_f1'].tail(10).std() if len(df) >= 10 else df['test_f1'].std()
    }
    return metrics

# Collect all data
all_results = {}

print("Loading TestBigData results...")
for test in testbigdata_tests:
    df, params = load_test_data(testbigdata_dir, test)
    if df is not None:
        metrics = get_best_metrics(df)
        all_results[f"New_{test}"] = {
            'df': df,
            'params': params,
            'metrics': metrics,
            'type': 'new'
        }
        print(f"  ✓ {test}: {len(df)} epochs")

print("\nLoading Risultato codice originale results...")
for test in originale_tests:
    df, params = load_test_data(originale_dir, test)
    if df is not None:
        metrics = get_best_metrics(df)
        all_results[f"Original_{test}"] = {
            'df': df,
            'params': params,
            'metrics': metrics,
            'type': 'original'
        }
        print(f"  ✓ {test}: {len(df)} epochs")

# Create comparison DataFrame
comparison_data = []
for name, data in all_results.items():
    if data['metrics']:
        row = {
            'Configuration': name,
            'Type': data['type'],
            **data['params'],
            **data['metrics']
        }
        comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)
print(f"\n✓ Saved comparison summary to {output_dir / 'comparison_summary.csv'}")

# ============================================================================
# VISUALIZATION 1: Best Performance Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Confronto Performance: Codice Nuovo vs Originale', fontsize=16, fontweight='bold')

# F1 Score Comparison
ax = axes[0, 0]
x_pos = np.arange(len(comparison_df))
colors = ['#2ecc71' if t == 'new' else '#e74c3c' for t in comparison_df['Type']]
bars = ax.bar(x_pos, comparison_df['best_test_f1'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Configurazione', fontweight='bold')
ax.set_ylabel('Best Test F1 Score', fontweight='bold')
ax.set_title('Miglior F1 Score Raggiunto')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['Configuration'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.legend([plt.Rectangle((0,0),1,1, fc='#2ecc71'), plt.Rectangle((0,0),1,1, fc='#e74c3c')], 
          ['Nuovo Codice', 'Codice Originale'])

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, comparison_df['best_test_f1'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Accuracy Comparison
ax = axes[0, 1]
bars = ax.bar(x_pos, comparison_df['best_test_acc'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Configurazione', fontweight='bold')
ax.set_ylabel('Best Test Accuracy', fontweight='bold')
ax.set_title('Miglior Accuracy Raggiunta')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['Configuration'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, comparison_df['best_test_acc'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Convergence Speed
ax = axes[1, 0]
bars = ax.bar(x_pos, comparison_df['epochs_to_best'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Configurazione', fontweight='bold')
ax.set_ylabel('Epoche per Raggiungere il Best', fontweight='bold')
ax.set_title('Velocità di Convergenza (meno è meglio)')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['Configuration'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, comparison_df['epochs_to_best'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Stability
ax = axes[1, 1]
bars = ax.bar(x_pos, comparison_df['stability'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Configurazione', fontweight='bold')
ax.set_ylabel('Deviazione Standard (ultime 10 epoche)', fontweight='bold')
ax.set_title('Stabilità del Modello (meno è meglio)')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['Configuration'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, comparison_df['stability'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "comparison_overview.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved overview comparison to {output_dir / 'comparison_overview.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 2: Learning Curves - All Tests
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Learning Curves: Tutti i Test', fontsize=16, fontweight='bold')

for idx, (name, data) in enumerate(all_results.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    df = data['df']
    color = '#2ecc71' if data['type'] == 'new' else '#e74c3c'
    
    ax.plot(df['epoch'], df['test_f1'], label='Test F1', color=color, linewidth=2)
    ax.plot(df['epoch'], df['train_f1'], label='Train F1', color=color, linewidth=2, alpha=0.5, linestyle='--')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title(f'{name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mark best epoch
    best_epoch = df.loc[df['test_f1'].idxmax(), 'epoch']
    best_f1 = df['test_f1'].max()
    ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.5)
    ax.text(best_epoch, best_f1, f'  Best: {best_f1:.2f}', 
            fontsize=9, va='bottom', ha='left', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "all_learning_curves.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved all learning curves to {output_dir / 'all_learning_curves.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 3: Direct Comparison - New vs Original
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Confronto Diretto: Nuovo vs Originale (Test F1 Score)', fontsize=16, fontweight='bold')

# Plot each comparison
comparisons = [
    ("Test1", axes[0, 0]),
    ("Test2", axes[0, 1]),
]

for test_name, ax in comparisons:
    new_key = f"New_{test_name}"
    orig_key = f"Original_{test_name}"
    
    if new_key in all_results and orig_key in all_results:
        new_df = all_results[new_key]['df']
        orig_df = all_results[orig_key]['df']
        
        ax.plot(new_df['epoch'], new_df['test_f1'], label='Nuovo Codice (Active Prefixes)', 
                color='#2ecc71', linewidth=2.5)
        ax.plot(orig_df['epoch'], orig_df['test_f1'], label='Codice Originale', 
                color='#e74c3c', linewidth=2.5)
        
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Test F1 Score', fontweight='bold', fontsize=12)
        ax.set_title(f'{test_name}', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # Add best scores
        new_best = new_df['test_f1'].max()
        orig_best = orig_df['test_f1'].max()
        improvement = ((new_best - orig_best) / orig_best * 100) if orig_best > 0 else 0
        
        textstr = f'Nuovo Best: {new_best:.2f}\nOriginale Best: {orig_best:.2f}\nMiglioramento: {improvement:+.2f}%'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Summary statistics
ax = axes[1, 0]
ax.axis('off')
summary_text = "STATISTICHE COMPARATIVE\n" + "="*50 + "\n\n"

new_tests = [k for k in all_results.keys() if k.startswith('New_')]
orig_tests = [k for k in all_results.keys() if k.startswith('Original_')]

summary_text += f"Numero Test Nuovo Codice: {len(new_tests)}\n"
summary_text += f"Numero Test Codice Originale: {len(orig_tests)}\n\n"

new_avg_f1 = np.mean([all_results[k]['metrics']['best_test_f1'] for k in new_tests])
orig_avg_f1 = np.mean([all_results[k]['metrics']['best_test_f1'] for k in orig_tests])

summary_text += f"F1 Medio Nuovo: {new_avg_f1:.2f}\n"
summary_text += f"F1 Medio Originale: {orig_avg_f1:.2f}\n"
summary_text += f"Differenza: {new_avg_f1 - orig_avg_f1:+.2f}\n\n"

new_avg_acc = np.mean([all_results[k]['metrics']['best_test_acc'] for k in new_tests])
orig_avg_acc = np.mean([all_results[k]['metrics']['best_test_acc'] for k in orig_tests])

summary_text += f"Accuracy Media Nuovo: {new_avg_acc:.2f}\n"
summary_text += f"Accuracy Media Originale: {orig_avg_acc:.2f}\n"
summary_text += f"Differenza: {new_avg_acc - orig_avg_acc:+.2f}\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Best configurations
ax = axes[1, 1]
ax.axis('off')
best_new = max(new_tests, key=lambda k: all_results[k]['metrics']['best_test_f1'])
best_orig = max(orig_tests, key=lambda k: all_results[k]['metrics']['best_test_f1'])

best_text = "MIGLIORI CONFIGURAZIONI\n" + "="*50 + "\n\n"
best_text += f"MIGLIOR NUOVO:\n{best_new}\n"
best_text += f"F1: {all_results[best_new]['metrics']['best_test_f1']:.2f}\n"
best_text += f"Acc: {all_results[best_new]['metrics']['best_test_acc']:.2f}\n"
best_text += f"Epoche: {all_results[best_new]['metrics']['epochs_to_best']:.0f}\n\n"

best_text += f"MIGLIOR ORIGINALE:\n{best_orig}\n"
best_text += f"F1: {all_results[best_orig]['metrics']['best_test_f1']:.2f}\n"
best_text += f"Acc: {all_results[best_orig]['metrics']['best_test_acc']:.2f}\n"
best_text += f"Epoche: {all_results[best_orig]['metrics']['epochs_to_best']:.0f}\n"

ax.text(0.1, 0.9, best_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / "direct_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved direct comparison to {output_dir / 'direct_comparison.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 4: Parameter Impact Analysis
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Analisi Impatto Parametri', fontsize=16, fontweight='bold')

# Extract parameter variations for new code tests
new_df = comparison_df[comparison_df['Type'] == 'new'].copy()

params_to_analyze = ['k', 'numlayers', 'numneurons', 'lr', 'batchsize', 'epochs']
param_labels = ['K (neighbors)', 'Num Layers', 'Num Neurons', 'Learning Rate', 'Batch Size', 'Total Epochs']

for idx, (param, label) in enumerate(zip(params_to_analyze, param_labels)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if param in new_df.columns:
        # Convert to numeric if possible
        try:
            new_df[param] = pd.to_numeric(new_df[param])
            x_vals = new_df[param]
            y_vals = new_df['best_test_f1']
            
            ax.scatter(x_vals, y_vals, s=200, alpha=0.6, c='#2ecc71', edgecolors='black', linewidth=2)
            
            # Add labels for each point
            for i, config in enumerate(new_df['Configuration']):
                ax.annotate(config.replace('New_', ''), (x_vals.iloc[i], y_vals.iloc[i]), 
                           fontsize=8, ha='right', va='bottom')
            
            ax.set_xlabel(label, fontweight='bold', fontsize=11)
            ax.set_ylabel('Best Test F1 Score', fontweight='bold', fontsize=11)
            ax.set_title(f'Impatto di {label}', fontweight='bold')
            ax.grid(alpha=0.3)
        except:
            ax.text(0.5, 0.5, f'Dati non disponibili\nper {label}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.axis('off')

plt.tight_layout()
plt.savefig(output_dir / "parameter_impact.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved parameter impact analysis to {output_dir / 'parameter_impact.png'}")
plt.close()

# ============================================================================
# Generate detailed report
# ============================================================================
print("\n" + "="*80)
print("ANALISI COMPLETATA")
print("="*80)
print(f"\nFile generati in: {output_dir}")
print("  - comparison_summary.csv: Tabella riassuntiva di tutti i test")
print("  - comparison_overview.png: Panoramica confronto performance")
print("  - all_learning_curves.png: Curve di apprendimento di tutti i test")
print("  - direct_comparison.png: Confronto diretto nuovo vs originale")
print("  - parameter_impact.png: Analisi impatto parametri")
print("\n" + "="*80)
