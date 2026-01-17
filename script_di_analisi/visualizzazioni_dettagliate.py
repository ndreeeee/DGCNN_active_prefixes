"""
Script per generare visualizzazioni dettagliate e separate per il confronto
tra TestBigData e Risultato codice originale.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurazione stile
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Paths
base_dir = Path(__file__).parent
testbigdata_dir = base_dir / "TestBigData"
originale_dir = base_dir / "Risultato codice originale"
output_dir = base_dir / "confronto_risultati" / "visualizzazioni_separate"
output_dir.mkdir(parents=True, exist_ok=True)

# Configurazioni test
testbigdata_tests = ["Test1", "Test2", "Test3", "Test4", "Test5", "Test6", "TestBuono"]
originale_tests = ["Test1", "Test2"]

# Colori personalizzati
COLORS_NEW = {
    'Test1': '#27ae60',
    'Test2': '#2ecc71', 
    'Test3': '#1abc9c',
    'Test4': '#16a085',
    'Test5': '#3498db',
    'Test6': '#2980b9',
    'TestBuono': '#9b59b6'
}
COLORS_ORIG = {
    'Test1': '#e74c3c',
    'Test2': '#c0392b'
}

def load_all_data():
    """Carica tutti i dati dei test"""
    all_data = {}
    
    # Carica test nuovo codice
    for test in testbigdata_tests:
        csv_path = testbigdata_dir / test / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            config = df['combination'].iloc[0] if 'combination' in df.columns else ""
            all_data[f"Nuovo_{test}"] = {
                'df': df,
                'config': config,
                'type': 'nuovo',
                'color': COLORS_NEW.get(test, '#2ecc71')
            }
            print(f"  ✓ Nuovo_{test}: {len(df)} epoche")
    
    # Carica test codice originale
    for test in originale_tests:
        csv_path = originale_dir / test / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            config = df['combination'].iloc[0] if 'combination' in df.columns else ""
            all_data[f"Originale_{test}"] = {
                'df': df,
                'config': config,
                'type': 'originale',
                'color': COLORS_ORIG.get(test, '#e74c3c')
            }
            print(f"  ✓ Originale_{test}: {len(df)} epoche")
    
    return all_data

def extract_params(config_str):
    """Estrae i parametri dalla stringa di configurazione"""
    params = {}
    if not config_str:
        return params
    parts = config_str.split('_')
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        val = parts[i + 1]
        if key in ['epochs', 'batchsize', 'k', 'numlayers', 'numneurons', 'lr', 'seed']:
            params[key] = val
            i += 2
        else:
            i += 1
    return params

def get_metrics(df):
    """Calcola metriche per un dataframe"""
    if df is None or len(df) == 0:
        return None
    return {
        'best_test_f1': df['test_f1'].max(),
        'best_test_acc': df['test_accuracy'].max(),
        'final_test_f1': df['test_f1'].iloc[-1],
        'final_test_acc': df['test_accuracy'].iloc[-1],
        'best_epoch': int(df.loc[df['test_f1'].idxmax(), 'epoch']),
        'total_epochs': len(df),
        'stability': df['test_f1'].tail(20).std() if len(df) >= 20 else df['test_f1'].std(),
        'degradation': df['test_f1'].max() - df['test_f1'].iloc[-1]
    }

print("\n" + "="*70)
print("CARICAMENTO DATI")
print("="*70)
all_data = load_all_data()

# ============================================================================
# 1. GRAFICO: Confronto F1 Score - Solo Nuovo Codice
# ============================================================================
print("\n[1/10] Generando: F1 Score - Nuovo Codice...")
fig, ax = plt.subplots(figsize=(14, 8))

for name, data in all_data.items():
    if data['type'] == 'nuovo':
        df = data['df']
        label = name.replace('Nuovo_', '')
        ax.plot(df['epoch'], df['test_f1'], 
                label=f"{label} (max: {df['test_f1'].max():.1f}%)",
                color=data['color'], linewidth=2.5, alpha=0.85)
        
        # Segna il punto migliore
        best_idx = df['test_f1'].idxmax()
        ax.scatter(df.loc[best_idx, 'epoch'], df.loc[best_idx, 'test_f1'],
                  color=data['color'], s=120, zorder=5, edgecolors='black', linewidth=1.5)

ax.set_xlabel('Epoca', fontweight='bold')
ax.set_ylabel('Test F1 Score (%)', fontweight='bold')
ax.set_title('Nuovo Codice (Active Prefixes): Evoluzione F1 Score per Test', 
             fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.95)
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3)

# Zona di buone performance
ax.axhspan(85, 100, alpha=0.1, color='green', label='_nolegend_')
ax.text(5, 92, 'Zona Performance Ottimale (>85%)', fontsize=10, alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / "01_f1_nuovo_codice.png", dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 2. GRAFICO: Confronto F1 Score - Solo Codice Originale  
# ============================================================================
print("[2/10] Generando: F1 Score - Codice Originale...")
fig, ax = plt.subplots(figsize=(14, 8))

for name, data in all_data.items():
    if data['type'] == 'originale':
        df = data['df']
        label = name.replace('Originale_', '')
        ax.plot(df['epoch'], df['test_f1'], 
                label=f"{label} (max: {df['test_f1'].max():.1f}%)",
                color=data['color'], linewidth=3, alpha=0.9)
        
        best_idx = df['test_f1'].idxmax()
        ax.scatter(df.loc[best_idx, 'epoch'], df.loc[best_idx, 'test_f1'],
                  color=data['color'], s=150, zorder=5, edgecolors='black', linewidth=2)

ax.set_xlabel('Epoca', fontweight='bold')
ax.set_ylabel('Test F1 Score (%)', fontweight='bold')
ax.set_title('Codice Originale: Evoluzione F1 Score per Test', 
             fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.95)
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3)
ax.axhspan(85, 100, alpha=0.1, color='green')
ax.text(5, 92, 'Zona Performance Ottimale (>85%)', fontsize=10, alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / "02_f1_codice_originale.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 3. GRAFICO: Confronto Diretto Migliori Configurazioni
# ============================================================================
print("[3/10] Generando: Confronto Migliori Configurazioni...")
fig, ax = plt.subplots(figsize=(14, 8))

# Migliore nuovo: TestBuono
# Migliore originale: Test2
best_new = all_data['Nuovo_TestBuono']['df']
best_orig = all_data['Originale_Test2']['df']

ax.plot(best_new['epoch'], best_new['test_f1'], 
        label=f"Nuovo: TestBuono (F1 max: {best_new['test_f1'].max():.2f}%)",
        color='#27ae60', linewidth=3)
ax.plot(best_orig['epoch'], best_orig['test_f1'],
        label=f"Originale: Test2 (F1 max: {best_orig['test_f1'].max():.2f}%)",
        color='#e74c3c', linewidth=3, linestyle='--')

# Aree sotto le curve per visualizzare meglio
ax.fill_between(best_new['epoch'], best_new['test_f1'], alpha=0.15, color='#27ae60')
ax.fill_between(best_orig['epoch'], best_orig['test_f1'], alpha=0.15, color='#e74c3c')

ax.set_xlabel('Epoca', fontweight='bold')
ax.set_ylabel('Test F1 Score (%)', fontweight='bold')
ax.set_title('Confronto Diretto: Miglior Configurazione Nuovo vs Originale', 
             fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3)

# Annotazioni
ax.annotate(f'Max: {best_new["test_f1"].max():.2f}%', 
            xy=(best_new.loc[best_new['test_f1'].idxmax(), 'epoch'], best_new['test_f1'].max()),
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60'))

plt.tight_layout()
plt.savefig(output_dir / "03_confronto_migliori.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 4. GRAFICO: Bar Chart - Best F1 Score per Configurazione
# ============================================================================
print("[4/10] Generando: Bar Chart Best F1...")
fig, ax = plt.subplots(figsize=(14, 8))

names = []
values = []
colors = []
types = []

for name, data in all_data.items():
    metrics = get_metrics(data['df'])
    if metrics:
        names.append(name.replace('Nuovo_', 'N: ').replace('Originale_', 'O: '))
        values.append(metrics['best_test_f1'])
        colors.append('#27ae60' if data['type'] == 'nuovo' else '#e74c3c')
        types.append(data['type'])

# Ordina per valore
sorted_indices = np.argsort(values)[::-1]
names = [names[i] for i in sorted_indices]
values = [values[i] for i in sorted_indices]
colors = [colors[i] for i in sorted_indices]

bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Aggiungi valori
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 0.5, i, f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('Best Test F1 Score (%)', fontweight='bold')
ax.set_title('Classifica: Best F1 Score per Configurazione', fontweight='bold', pad=15)
ax.set_xlim([0, 100])
ax.grid(axis='x', alpha=0.3)

# Legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#27ae60', label='Nuovo Codice'),
                   Patch(facecolor='#e74c3c', label='Codice Originale')]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)

plt.tight_layout()
plt.savefig(output_dir / "04_classifica_f1.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 5. GRAFICO: Velocità di Convergenza
# ============================================================================
print("[5/10] Generando: Velocità di Convergenza...")
fig, ax = plt.subplots(figsize=(14, 8))

conv_data = []
for name, data in all_data.items():
    metrics = get_metrics(data['df'])
    if metrics:
        conv_data.append({
            'name': name.replace('Nuovo_', 'N: ').replace('Originale_', 'O: '),
            'epochs_to_best': metrics['best_epoch'],
            'color': '#27ae60' if data['type'] == 'nuovo' else '#e74c3c',
            'type': data['type']
        })

conv_df = pd.DataFrame(conv_data)
conv_df = conv_df.sort_values('epochs_to_best')

bars = ax.barh(range(len(conv_df)), conv_df['epochs_to_best'], 
               color=conv_df['color'], alpha=0.8, edgecolor='black', linewidth=1.2)

for i, (bar, val) in enumerate(zip(bars, conv_df['epochs_to_best'])):
    ax.text(val + 1, i, f'{int(val)} ep.', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(range(len(conv_df)))
ax.set_yticklabels(conv_df['name'], fontsize=11)
ax.set_xlabel('Epoche per Raggiungere Best F1', fontweight='bold')
ax.set_title('Velocità di Convergenza (meno epoche = più veloce)', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)

plt.tight_layout()
plt.savefig(output_dir / "05_velocita_convergenza.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 6. GRAFICO: Stabilità (Standard Deviation)
# ============================================================================
print("[6/10] Generando: Analisi Stabilità...")
fig, ax = plt.subplots(figsize=(14, 8))

stab_data = []
for name, data in all_data.items():
    metrics = get_metrics(data['df'])
    if metrics:
        stab_data.append({
            'name': name.replace('Nuovo_', 'N: ').replace('Originale_', 'O: '),
            'stability': metrics['stability'],
            'color': '#27ae60' if data['type'] == 'nuovo' else '#e74c3c',
            'type': data['type']
        })

stab_df = pd.DataFrame(stab_data)
stab_df = stab_df.sort_values('stability')

bars = ax.barh(range(len(stab_df)), stab_df['stability'], 
               color=stab_df['color'], alpha=0.8, edgecolor='black', linewidth=1.2)

for i, (bar, val) in enumerate(zip(bars, stab_df['stability'])):
    ax.text(val + 0.2, i, f'{val:.2f}', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(range(len(stab_df)))
ax.set_yticklabels(stab_df['name'], fontsize=11)
ax.set_xlabel('Deviazione Standard F1 (ultime 20 epoche)', fontweight='bold')
ax.set_title('Stabilità del Training (valore basso = più stabile)', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)

plt.tight_layout()
plt.savefig(output_dir / "06_stabilita.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 7. GRAFICO: Degradazione (Best - Final)
# ============================================================================
print("[7/10] Generando: Analisi Degradazione...")
fig, ax = plt.subplots(figsize=(14, 8))

deg_data = []
for name, data in all_data.items():
    metrics = get_metrics(data['df'])
    if metrics:
        deg_data.append({
            'name': name.replace('Nuovo_', 'N: ').replace('Originale_', 'O: '),
            'degradation': metrics['degradation'],
            'color': '#27ae60' if data['type'] == 'nuovo' else '#e74c3c',
            'type': data['type']
        })

deg_df = pd.DataFrame(deg_data)
deg_df = deg_df.sort_values('degradation', ascending=False)

# Colori diversi per degradazione alta/bassa
bar_colors = []
for _, row in deg_df.iterrows():
    if row['degradation'] > 50:  # Degradazione catastrofica
        bar_colors.append('#ff6b6b')
    elif row['degradation'] > 10:
        bar_colors.append('#ffa502')
    else:
        bar_colors.append('#2ed573')

bars = ax.barh(range(len(deg_df)), deg_df['degradation'], 
               color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)

for i, (bar, val, name) in enumerate(zip(bars, deg_df['degradation'], deg_df['name'])):
    label = f'{val:.1f}%'
    if val > 50:
        label += ' ⚠️ COLLASSO'
    ax.text(val + 0.5, i, label, va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(deg_df)))
ax.set_yticklabels(deg_df['name'], fontsize=11)
ax.set_xlabel('Degradazione F1: (Best - Final) in punti percentuali', fontweight='bold')
ax.set_title('Degradazione Performance (valore basso = migliore)', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Legenda personalizzata
from matplotlib.patches import Patch
deg_legend = [Patch(facecolor='#2ed573', label='Bassa (<10%)'),
              Patch(facecolor='#ffa502', label='Media (10-50%)'),
              Patch(facecolor='#ff6b6b', label='Alta/Collasso (>50%)')]
ax.legend(handles=deg_legend, loc='lower right', framealpha=0.95, title='Livello Degradazione')

plt.tight_layout()
plt.savefig(output_dir / "07_degradazione.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 8. GRAFICO: Scatter - F1 vs Stabilità
# ============================================================================
print("[8/10] Generando: Scatter F1 vs Stabilità...")
fig, ax = plt.subplots(figsize=(12, 8))

for name, data in all_data.items():
    metrics = get_metrics(data['df'])
    if metrics:
        marker = 'o' if data['type'] == 'nuovo' else 's'
        size = 250 if data['type'] == 'nuovo' else 300
        ax.scatter(metrics['best_test_f1'], metrics['stability'],
                  s=size, c=data['color'], marker=marker, 
                  alpha=0.8, edgecolors='black', linewidth=2,
                  label=name.replace('Nuovo_', 'N: ').replace('Originale_', 'O: '))

ax.set_xlabel('Best Test F1 Score (%)', fontweight='bold')
ax.set_ylabel('Stabilità (Dev. Std.) - più basso = meglio', fontweight='bold')
ax.set_title('Trade-off: Performance vs Stabilità', fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3)

# Quadranti
ax.axvline(x=85, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=5, color='gray', linestyle=':', alpha=0.5)
ax.text(87, 0.5, 'IDEALE\n(alto F1, bassa std)', fontsize=9, ha='center', alpha=0.7, color='green')
ax.text(75, 0.5, 'STABILE\n(F1 medio, bassa std)', fontsize=9, ha='center', alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / "08_f1_vs_stabilita.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 9. GRAFICO: Heatmap Parametri vs Performance
# ============================================================================
print("[9/10] Generando: Heatmap Parametri...")
fig, ax = plt.subplots(figsize=(14, 10))

# Prepara dati per heatmap
param_data = []
for name, data in all_data.items():
    params = extract_params(data['config'])
    metrics = get_metrics(data['df'])
    if params and metrics:
        param_data.append({
            'Test': name.replace('Nuovo_', 'N:').replace('Originale_', 'O:'),
            'K': int(params.get('k', 0)),
            'Layers': int(params.get('numlayers', 0)),
            'Neurons': int(params.get('numneurons', 0)),
            'LR': float(params.get('lr', 0)),
            'Best_F1': metrics['best_test_f1'],
            'Stability': metrics['stability']
        })

param_df = pd.DataFrame(param_data)

# Crea matrice per heatmap
if len(param_df) > 0:
    # Normalizza le colonne per la heatmap
    heatmap_cols = ['K', 'Layers', 'Neurons', 'LR', 'Best_F1', 'Stability']
    heatmap_data = param_df[heatmap_cols].copy()
    
    # Normalizza ogni colonna tra 0 e 1
    for col in heatmap_cols:
        min_val = heatmap_data[col].min()
        max_val = heatmap_data[col].max()
        if max_val > min_val:
            heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
        else:
            heatmap_data[col] = 0.5
    
    # Per Stability, inverti (così basso = buono = verde)
    heatmap_data['Stability'] = 1 - heatmap_data['Stability']
    
    # Crea heatmap
    sns.heatmap(heatmap_data.T, annot=param_df[heatmap_cols].T.round(2), 
                fmt='', cmap='RdYlGn', ax=ax,
                xticklabels=param_df['Test'], yticklabels=heatmap_cols,
                cbar_kws={'label': 'Score Normalizzato (0-1)'},
                linewidths=0.5, linecolor='white')
    
    ax.set_title('Heatmap: Parametri e Performance per Configurazione\n(Verde = Meglio, Rosso = Peggio)',
                 fontweight='bold', pad=15)
    ax.set_xlabel('Configurazione Test', fontweight='bold')
    ax.set_ylabel('Parametro/Metrica', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "09_heatmap_parametri.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# 10. GRAFICO: Riepilogo Finale Dashboard
# ============================================================================
print("[10/10] Generando: Dashboard Riepilogo...")
fig = plt.figure(figsize=(20, 14))

# Layout: 3 righe x 3 colonne
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

# --- 10.1: Top 3 Configurazioni ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
top3_text = """
🏆 TOP 3 CONFIGURAZIONI

🥇 GOLD: Nuovo_TestBuono
   F1: 88.02% | Acc: 90.42%
   Stabilità: 0.38 | LR: 0.001
   
🥈 SILVER: Originale_Test2
   F1: 87.27% | Acc: 90.55%
   Stabilità: 0.52 | LR: 0.0005
   
🥉 BRONZE: Nuovo_Test3
   F1: 87.19% | Acc: 90.37%
   Stabilità: 0.44 | LR: 0.005
"""
ax1.text(0.1, 0.95, top3_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

# --- 10.2: Statistiche Comparative ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

new_f1s = [get_metrics(d['df'])['best_test_f1'] for d in all_data.values() if d['type'] == 'nuovo']
orig_f1s = [get_metrics(d['df'])['best_test_f1'] for d in all_data.values() if d['type'] == 'originale']
new_stab = [get_metrics(d['df'])['stability'] for d in all_data.values() if d['type'] == 'nuovo']
orig_stab = [get_metrics(d['df'])['stability'] for d in all_data.values() if d['type'] == 'originale']

stats_text = f"""
📊 STATISTICHE COMPARATIVE

NUOVO CODICE (7 test):
  F1 medio: {np.mean(new_f1s):.2f}%
  F1 max:   {np.max(new_f1s):.2f}%
  F1 min:   {np.min(new_f1s):.2f}%
  Stab. media: {np.mean(new_stab):.2f}

CODICE ORIGINALE (2 test):
  F1 medio: {np.mean(orig_f1s):.2f}%
  F1 max:   {np.max(orig_f1s):.2f}%
  Stab. media: {np.mean(orig_stab):.2f}

Δ F1 (max): +{np.max(new_f1s) - np.max(orig_f1s):.2f}%
"""
ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#4caf50'))

# --- 10.3: Configurazioni da Evitare ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
avoid_text = """
⚠️ CONFIGURAZIONI DA EVITARE

❌ Nuovo_Test2 (LR=0.01)
   Collasso: F1 86→4%
   
❌ Nuovo_Test4 (LR=0.01)
   Collasso: F1 87→4%
   
❌ Nuovo_Test1 (LR=0.0005, k=40)
   Overfitting: F1 72→64%
   
PROBLEMA COMUNE:
Learning Rate troppo alto (0.01)
causa instabilità e collasso.
"""
ax3.text(0.1, 0.95, avoid_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#f44336'))

# --- 10.4: Learning Curves (tutti insieme) ---
ax4 = fig.add_subplot(gs[1, :])
for name, data in all_data.items():
    df = data['df']
    linestyle = '-' if data['type'] == 'nuovo' else '--'
    linewidth = 2 if data['type'] == 'nuovo' else 2.5
    ax4.plot(df['epoch'], df['test_f1'], 
             label=name.replace('Nuovo_', 'N:').replace('Originale_', 'O:'),
             color=data['color'], linewidth=linewidth, linestyle=linestyle, alpha=0.8)

ax4.set_xlabel('Epoca', fontweight='bold')
ax4.set_ylabel('Test F1 Score (%)', fontweight='bold')
ax4.set_title('Evoluzione F1 Score: Tutti i Test a Confronto', fontweight='bold')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
ax4.set_ylim([0, 100])
ax4.grid(True, alpha=0.3)
ax4.axhline(y=85, color='gold', linestyle=':', linewidth=2, alpha=0.7)

# --- 10.5: Bar Chart Comparativo ---
ax5 = fig.add_subplot(gs[2, 0:2])
x = np.arange(len(all_data))
width = 0.35

names_short = [n.replace('Nuovo_', 'N:').replace('Originale_', 'O:') for n in all_data.keys()]
best_f1s = [get_metrics(d['df'])['best_test_f1'] for d in all_data.values()]
final_f1s = [get_metrics(d['df'])['final_test_f1'] for d in all_data.values()]
colors = ['#27ae60' if d['type'] == 'nuovo' else '#e74c3c' for d in all_data.values()]

bars1 = ax5.bar(x - width/2, best_f1s, width, label='Best F1', color=colors, alpha=0.9, edgecolor='black')
bars2 = ax5.bar(x + width/2, final_f1s, width, label='Final F1', color=colors, alpha=0.5, edgecolor='black', hatch='//')

ax5.set_ylabel('F1 Score (%)', fontweight='bold')
ax5.set_title('Confronto Best vs Final F1 per Configurazione', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(names_short, rotation=45, ha='right')
ax5.legend()
ax5.set_ylim([0, 100])
ax5.grid(axis='y', alpha=0.3)

# --- 10.6: Raccomandazioni ---
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
rec_text = """
📋 RACCOMANDAZIONI

PER PRODUZIONE:
→ Originale_Test2
  (Affidabile, stabile)

PER MAX PERFORMANCE:
→ Nuovo_TestBuono
  (F1 più alto: 88.02%)

PER TRAINING VELOCE:
→ Nuovo_Test3
  (Solo 50 epoche)

LEARNING RATE OTTIMALE:
• Nuovo: 0.001 ✅
• Originale: 0.0005 ✅
• Da evitare: 0.01 ❌
"""
ax6.text(0.1, 0.95, rec_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e3f2fd', edgecolor='#2196f3'))

plt.suptitle('DASHBOARD RIEPILOGO: Confronto Test DGCNN\nNuovo Codice (Active Prefixes) vs Codice Originale',
             fontsize=16, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(output_dir / "10_dashboard_riepilogo.png", dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================================
# COMPLETAMENTO
# ============================================================================
print("\n" + "="*70)
print("VISUALIZZAZIONI GENERATE CON SUCCESSO!")
print("="*70)
print(f"\nFile salvati in: {output_dir}")
print("\nElenco visualizzazioni:")
for f in sorted(output_dir.glob("*.png")):
    print(f"  📊 {f.name}")
print("\n" + "="*70)
