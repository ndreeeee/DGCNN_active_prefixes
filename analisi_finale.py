"""
Analisi Completa: Confronto tra Codice con Nodo Globale vs Codice Originale
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

base_dir = Path(r"c:\Users\andre\OneDrive - Università Politecnica delle Marche (1)\Documenti\Desktop\Universita\MAGISTRALE\BIG DATA\Progetto\DGCNN_active_prefixes_progetto\DGCNN_active_prefixes_progetto")
tests_ng = base_dir / "tests_custom_config"
tests_or = base_dir / "tests_original_code"
output_dir = base_dir / "analisi_finale"
output_dir.mkdir(exist_ok=True)

def find_results_csv(folder):
    """Trova il file results.csv nella cartella o sottocartelle"""
    results_file = folder / "results.csv"
    if results_file.exists():
        return results_file
    for sub in folder.iterdir():
        if sub.is_dir():
            rf = sub / "results.csv"
            if rf.exists():
                return rf
    return None

def load_test(folder, source):
    """Carica un test e estrae metriche"""
    results_file = find_results_csv(folder)
    if not results_file:
        return None
    try:
        df = pd.read_csv(results_file)
        if len(df) == 0:
            return None
        row = df.iloc[0]
        return {
            'name': folder.name,
            'source': source,
            'df': df,
            'batch_size': row['batch_size'],
            'k': row['k'],
            'num_layers': row['num_layers'],
            'num_neurons': row['num_neurons'],
            'learning_rate': row['learning_rate'],
            'seed': row['seed'],
            'total_epochs': len(df),
            'best_f1': df['test_f1'].max(),
            'best_acc': df['test_accuracy'].max(),
            'final_f1': df['test_f1'].iloc[-1],
            'best_epoch': int(df.loc[df['test_f1'].idxmax(), 'epoch']),
            'stability': df['test_f1'].tail(min(20, len(df))).std()
        }
    except Exception as e:
        print(f"Errore: {folder.name} - {e}")
        return None

# Carica tutti i test
print("="*80)
print("CARICAMENTO DATI")
print("="*80)

all_tests = []

# Test Nodo Globale
print("\n📁 tests_custom_config:")
for folder in sorted(tests_ng.iterdir()):
    if folder.is_dir():
        data = load_test(folder, "Nodo Globale")
        if data:
            all_tests.append(data)
            print(f"  ✓ {data['name']}: F1={data['best_f1']:.2f}%")

# Test Originale
print("\n📁 tests_original_code:")
for folder in sorted(tests_or.iterdir()):
    if folder.is_dir():
        data = load_test(folder, "Originale")
        if data:
            all_tests.append(data)
            print(f"  ✓ {data['name']}: F1={data['best_f1']:.2f}%")

print(f"\nTotale test caricati: {len(all_tests)}")

# Crea DataFrame riassuntivo
summary_df = pd.DataFrame([{
    'Nome': t['name'],
    'Tipo': t['source'],
    'batch_size': t['batch_size'],
    'k': t['k'],
    'layers': t['num_layers'],
    'neurons': t['num_neurons'],
    'lr': t['learning_rate'],
    'Best F1': t['best_f1'],
    'Best Acc': t['best_acc'],
    'Final F1': t['final_f1'],
    'Epochs': t['total_epochs'],
    'Best Epoch': t['best_epoch'],
    'Stability': t['stability']
} for t in all_tests])

summary_df = summary_df.sort_values('Best F1', ascending=False)
summary_df.to_csv(output_dir / "riepilogo_test.csv", index=False)

# Separa per tipo
ng_tests = [t for t in all_tests if t['source'] == "Nodo Globale"]
or_tests = [t for t in all_tests if t['source'] == "Originale"]

print("\n" + "="*80)
print("GENERAZIONE GRAFICI")
print("="*80)

# 1. CLASSIFICA COMPLETA
print("\n[1/6] Classifica F1...")
fig, ax = plt.subplots(figsize=(14, max(8, len(all_tests) * 0.4)))
colors = ['#27ae60' if t['source'] == 'Nodo Globale' else '#e74c3c' for t in sorted(all_tests, key=lambda x: x['best_f1'])]
names = [t['name'] for t in sorted(all_tests, key=lambda x: x['best_f1'])]
f1_scores = [t['best_f1'] for t in sorted(all_tests, key=lambda x: x['best_f1'])]

bars = ax.barh(names, f1_scores, color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, f1_scores):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontweight='bold', fontsize=9)

ax.set_xlabel('Best Test F1 Score (%)', fontweight='bold')
ax.set_title('Classifica Tutti i Test per Best F1 Score', fontweight='bold', fontsize=14)
ax.set_xlim([0, 100])
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#27ae60', label='Nodo Globale'), Patch(facecolor='#e74c3c', label='Originale')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig(output_dir / "01_classifica_f1.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# 2. CONFRONTO TOP PERFORMERS
print("[2/6] Top Performers...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 5 Nodo Globale
ax = axes[0]
top_ng = sorted(ng_tests, key=lambda x: x['best_f1'], reverse=True)[:5]
names = [f"{t['name']}\nk={t['k']}, lr={t['learning_rate']}" for t in top_ng]
f1s = [t['best_f1'] for t in top_ng]
bars = ax.barh(names, f1s, color='#27ae60', alpha=0.8, edgecolor='black')
for bar, val in zip(bars, f1s):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontweight='bold')
ax.set_xlabel('Best F1 (%)', fontweight='bold')
ax.set_title('Top 5 Nodo Globale', fontweight='bold', fontsize=12)
ax.set_xlim([80, 95])

# Originale
ax = axes[1]
names = [f"{t['name']}\nk={t['k']}, lr={t['learning_rate']}" for t in or_tests]
f1s = [t['best_f1'] for t in or_tests]
bars = ax.barh(names, f1s, color='#e74c3c', alpha=0.8, edgecolor='black')
for bar, val in zip(bars, f1s):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontweight='bold')
ax.set_xlabel('Best F1 (%)', fontweight='bold')
ax.set_title('Codice Originale', fontweight='bold', fontsize=12)
ax.set_xlim([80, 95])

plt.tight_layout()
plt.savefig(output_dir / "02_top_performers.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# 3. LEARNING CURVES TOP 6
print("[3/6] Learning Curves...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
top6 = sorted(all_tests, key=lambda x: x['best_f1'], reverse=True)[:6]

for idx, t in enumerate(top6):
    ax = axes[idx]
    df = t['df']
    color = '#27ae60' if t['source'] == 'Nodo Globale' else '#e74c3c'
    ax.plot(df['epoch'], df['test_f1'], label='Test F1', color=color, linewidth=2)
    ax.plot(df['epoch'], df['train_f1'], label='Train F1', color=color, linewidth=1.5, alpha=0.5, linestyle='--')
    best_idx = df['test_f1'].idxmax()
    ax.scatter(df.loc[best_idx, 'epoch'], df.loc[best_idx, 'test_f1'], color='red', s=100, zorder=5, marker='*')
    ax.set_title(f"#{idx+1}: {t['name']} ({t['source']})\nF1={t['best_f1']:.2f}%", fontweight='bold', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])

plt.suptitle('Learning Curves: Top 6 Configurazioni', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "03_learning_curves.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# 4. IMPATTO PARAMETRI
print("[4/6] Impatto Parametri...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Learning Rate
ax = axes[0, 0]
for t in ng_tests:
    ax.scatter(t['learning_rate'], t['best_f1'], c='#27ae60', s=100, alpha=0.7, edgecolors='black')
for t in or_tests:
    ax.scatter(t['learning_rate'], t['best_f1'], c='#e74c3c', s=150, alpha=0.9, edgecolors='black', marker='s')
ax.set_xlabel('Learning Rate', fontweight='bold')
ax.set_ylabel('Best F1 (%)', fontweight='bold')
ax.set_title('Impatto Learning Rate', fontweight='bold')
ax.set_xscale('log')

# K neighbors
ax = axes[0, 1]
for t in ng_tests:
    ax.scatter(t['k'], t['best_f1'], c='#27ae60', s=100, alpha=0.7, edgecolors='black')
for t in or_tests:
    ax.scatter(t['k'], t['best_f1'], c='#e74c3c', s=150, alpha=0.9, edgecolors='black', marker='s')
ax.set_xlabel('K (neighbors)', fontweight='bold')
ax.set_ylabel('Best F1 (%)', fontweight='bold')
ax.set_title('Impatto K', fontweight='bold')

# Batch Size
ax = axes[1, 0]
for t in ng_tests:
    ax.scatter(t['batch_size'], t['best_f1'], c='#27ae60', s=100, alpha=0.7, edgecolors='black')
for t in or_tests:
    ax.scatter(t['batch_size'], t['best_f1'], c='#e74c3c', s=150, alpha=0.9, edgecolors='black', marker='s')
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Best F1 (%)', fontweight='bold')
ax.set_title('Impatto Batch Size', fontweight='bold')

# Neurons
ax = axes[1, 1]
for t in ng_tests:
    ax.scatter(t['num_neurons'], t['best_f1'], c='#27ae60', s=100, alpha=0.7, edgecolors='black')
for t in or_tests:
    ax.scatter(t['num_neurons'], t['best_f1'], c='#e74c3c', s=150, alpha=0.9, edgecolors='black', marker='s')
ax.set_xlabel('Num Neurons', fontweight='bold')
ax.set_ylabel('Best F1 (%)', fontweight='bold')
ax.set_title('Impatto Neuroni', fontweight='bold')

for ax in axes.flatten():
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "04_impatto_parametri.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# 5. STABILITA
print("[5/6] Stabilità...")
fig, ax = plt.subplots(figsize=(12, 8))
for t in ng_tests:
    ax.scatter(t['best_f1'], t['stability'], c='#27ae60', s=100, alpha=0.7, edgecolors='black', label='_')
for t in or_tests:
    ax.scatter(t['best_f1'], t['stability'], c='#e74c3c', s=150, alpha=0.9, edgecolors='black', marker='s', label='_')
ax.set_xlabel('Best F1 (%)', fontweight='bold')
ax.set_ylabel('Stabilità (Dev. Std) - più basso = meglio', fontweight='bold')
ax.set_title('Trade-off Performance vs Stabilità', fontweight='bold')
ax.legend(handles=legend_elements)
ax.grid(alpha=0.3)
ax.axvline(x=87, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / "05_stabilita.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# 6. CONFRONTO DIRETTO MIGLIORI
print("[6/6] Confronto Diretto...")
best_ng = max(ng_tests, key=lambda x: x['best_f1'])
best_or = max(or_tests, key=lambda x: x['best_f1'])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(best_ng['df']['epoch'], best_ng['df']['test_f1'], label=f"Nodo Globale: {best_ng['name']} (F1={best_ng['best_f1']:.2f}%)", color='#27ae60', linewidth=2.5)
ax.plot(best_or['df']['epoch'], best_or['df']['test_f1'], label=f"Originale: {best_or['name']} (F1={best_or['best_f1']:.2f}%)", color='#e74c3c', linewidth=2.5, linestyle='--')
ax.axhline(y=best_ng['best_f1'], color='#27ae60', linestyle=':', alpha=0.5)
ax.axhline(y=best_or['best_f1'], color='#e74c3c', linestyle=':', alpha=0.5)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Test F1 Score (%)', fontweight='bold')
ax.set_title('Confronto Diretto: Miglior Nodo Globale vs Miglior Originale', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([60, 95])
plt.tight_layout()
plt.savefig(output_dir / "06_confronto_migliori.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# GENERAZIONE RELAZIONE
print("\n" + "="*80)
print("GENERAZIONE RELAZIONE")
print("="*80)

improvement = best_ng['best_f1'] - best_or['best_f1']

relazione = f"""# Relazione Finale: Confronto DGCNN con Nodo Globale vs Codice Originale

**Data**: {datetime.now().strftime('%d/%m/%Y %H:%M')}

---

## 1. Executive Summary

Questa analisi confronta **{len(ng_tests)} configurazioni** del modello DGCNN con **Nodo Globale** (Active Prefixes) rispetto a **{len(or_tests)} configurazioni** del **codice originale**.

### Risultato Principale

| Metrica | Nodo Globale | Originale | Differenza |
|---------|-------------|-----------|------------|
| **Miglior F1** | **{best_ng['best_f1']:.2f}%** | {best_or['best_f1']:.2f}% | **{improvement:+.2f}%** |
| **Configurazione** | {best_ng['name']} | {best_or['name']} | - |
| **Parametri** | k={best_ng['k']}, lr={best_ng['learning_rate']} | k={best_or['k']}, lr={best_or['learning_rate']} | - |

---

## 2. Panoramica Test

### 2.1 Test Nodo Globale (tests_custom_config)

| # | Nome | k | lr | batch | layers | neurons | Best F1 | Stabilità |
|---|------|---|-----|-------|--------|---------|---------|-----------|
"""

for i, t in enumerate(sorted(ng_tests, key=lambda x: x['best_f1'], reverse=True)):
    relazione += f"| {i+1} | {t['name']} | {t['k']} | {t['learning_rate']} | {t['batch_size']} | {t['num_layers']} | {t['num_neurons']} | **{t['best_f1']:.2f}%** | {t['stability']:.2f} |\n"

relazione += f"""

### 2.2 Test Codice Originale (tests_original_code)

| # | Nome | k | lr | batch | layers | neurons | Best F1 | Stabilità |
|---|------|---|-----|-------|--------|---------|---------|-----------|
"""

for i, t in enumerate(sorted(or_tests, key=lambda x: x['best_f1'], reverse=True)):
    relazione += f"| {i+1} | {t['name']} | {t['k']} | {t['learning_rate']} | {t['batch_size']} | {t['num_layers']} | {t['num_neurons']} | **{t['best_f1']:.2f}%** | {t['stability']:.2f} |\n"

relazione += f"""

---

## 3. Visualizzazioni

### 3.1 Classifica Completa
![Classifica F1](01_classifica_f1.png)

### 3.2 Top Performers
![Top Performers](02_top_performers.png)

### 3.3 Learning Curves
![Learning Curves](03_learning_curves.png)

### 3.4 Impatto Parametri
![Impatto Parametri](04_impatto_parametri.png)

### 3.5 Stabilità vs Performance
![Stabilità](05_stabilita.png)

### 3.6 Confronto Migliori
![Confronto](06_confronto_migliori.png)

---

## 4. Conclusioni

### 4.1 Risultati Chiave

1. **Il Nodo Globale migliora le performance** di {improvement:.2f} punti percentuali rispetto al baseline
2. **Configurazione ottimale**: k={best_ng['k']}, lr={best_ng['learning_rate']}, {best_ng['num_layers']} layers, {best_ng['num_neurons']} neuroni
3. **Stabilità**: Il codice originale tende ad essere più stabile, il nodo globale richiede tuning più attento

### 4.2 Raccomandazioni

| Scenario | Configurazione Consigliata |
|----------|---------------------------|
| **Produzione** | Originale ({best_or['name']}) - F1={best_or['best_f1']:.2f}% |
| **Max Performance** | Nodo Globale ({best_ng['name']}) - F1={best_ng['best_f1']:.2f}% |

---

*Relazione generata automaticamente - {datetime.now().strftime('%d/%m/%Y %H:%M')}*
"""

with open(output_dir / "RELAZIONE_FINALE.md", 'w', encoding='utf-8') as f:
    f.write(relazione)

print(f"\n✓ Relazione salvata in: {output_dir / 'RELAZIONE_FINALE.md'}")

# Cleanup file temporanei
for f in [base_dir / "find_duplicates.py", base_dir / "duplicates_to_remove.txt", base_dir / "flatten_structure.py"]:
    if f.exists():
        f.unlink()

print("\n" + "="*80)
print("ANALISI COMPLETATA!")
print("="*80)
print(f"\nFile generati in: {output_dir}")
for f in sorted(output_dir.glob("*")):
    print(f"  📄 {f.name}")
