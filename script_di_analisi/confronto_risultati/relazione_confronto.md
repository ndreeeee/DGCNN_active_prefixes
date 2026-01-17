# Relazione: Analisi Comparativa dei Modelli DGCNN
## Confronto tra Nuovo Codice (Active Prefixes) e Codice Originale

---

## 1. Introduzione

### 1.1 Obiettivo dello Studio
Questa relazione presenta un'analisi comparativa approfondita tra due implementazioni del modello DGCNN (Dynamic Graph Convolutional Neural Network) per la predizione di processi:

1. **Nuovo Codice**: Implementazione con meccanismo di "Active Prefixes" che connette prefissi attivi attraverso un nodo globale per fornire informazioni contestuali alla predizione.

2. **Codice Originale**: Implementazione baseline senza il meccanismo di active prefixes.

### 1.2 Dataset e Contesto
I test sono stati condotti sul dataset **RequestForPayments** utilizzato per la predizione dell'attività successiva (next activity prediction) in processi di business.

### 1.3 Configurazioni Testate
Sono state analizzate **9 configurazioni** totali:
- **7 configurazioni** per il nuovo codice (TestBigData: Test1-6, TestBuono)
- **2 configurazioni** per il codice originale (Risultato codice originale: Test1-2)

---

## 2. Metodologia

### 2.1 Metriche di Valutazione
Le configurazioni sono state valutate secondo le seguenti metriche:

| Metrica | Descrizione | Obiettivo |
|---------|-------------|-----------|
| **Best Test F1** | Massimo F1 score raggiunto sul test set | Massimizzare |
| **Best Test Accuracy** | Massima accuracy raggiunta sul test set | Massimizzare |
| **Final Test F1** | F1 score all'ultima epoca | Massimizzare |
| **Stability** | Deviazione standard F1 nelle ultime 20 epoche | Minimizzare |
| **Degradation** | Differenza tra Best F1 e Final F1 | Minimizzare |
| **Convergence Speed** | Numero di epoche per raggiungere il best F1 | Dipende dal contesto |

### 2.2 Parametri Analizzati
I parametri chiave considerati nell'analisi sono:
- **Learning Rate (LR)**: 0.0005, 0.001, 0.005, 0.01
- **K (neighbors)**: 30, 40, 50, 70
- **Numero di layers**: 5, 7
- **Numero di neuroni**: 64, 128
- **Batch size**: 64 (costante)
- **Epoche totali**: 50, 100, 300

---

## 3. Risultati Sperimentali

### 3.1 Riepilogo delle Configurazioni

| Configurazione | Tipo | Epochs | K | Layers | Neurons | LR | Best F1 | Final F1 | Stability |
|----------------|------|--------|---|--------|---------|-----|---------|----------|-----------|
| Nuovo_TestBuono | Nuovo | 300 | 40 | 5 | 64 | 0.001 | **88.02** | 86.44 | 0.38 |
| Originale_Test2 | Orig | 300 | 30 | 5 | 64 | 0.0005 | 87.27 | 85.36 | **0.52** |
| Originale_Test1 | Orig | 100 | 30 | 5 | 64 | 0.0005 | 87.27 | 85.93 | 0.70 |
| Nuovo_Test5 | Nuovo | 50 | 70 | 7 | 128 | 0.005 | 87.24 | 86.70 | 8.83 |
| Nuovo_Test3 | Nuovo | 50 | 50 | 7 | 128 | 0.005 | 87.19 | 86.50 | 0.44 |
| Nuovo_Test6 | Nuovo | 300 | 40 | 5 | 64 | 0.0005 | 87.84 | 87.33 | 0.17 |
| Nuovo_Test4 | Nuovo | 50 | 70 | 7 | 128 | 0.01 | 86.87 | 4.10 | 28.15 |
| Nuovo_Test2 | Nuovo | 50 | 50 | 7 | 128 | 0.01 | 86.21 | 4.11 | 0.004 |
| Nuovo_Test1 | Nuovo | 300 | 40 | 5 | 64 | 0.0005 | 72.27 | 64.35 | 0.79 |

### 3.2 Analisi delle Migliori Configurazioni

#### 3.2.1 Miglior Configurazione Nuovo Codice: TestBuono

La configurazione **TestBuono** ha ottenuto il miglior F1 score assoluto tra tutte le configurazioni testate:

- **F1 Score**: 88.02% (raggiunto all'epoca 74)
- **Accuracy**: 90.42%
- **Stabilità**: 0.38 (eccellente)
- **Degradazione**: 1.58 punti percentuali

**Parametri ottimali identificati**:
- Learning rate: 0.001 (doppio rispetto al codice originale)
- K = 40 neighbors
- Architettura: 5 layers, 64 neuroni

**Interpretazione**: Il learning rate leggermente più alto (0.001 vs 0.0005) accelera la convergenza verso un minimo locale migliore, mantenendo comunque una buona stabilità. L'architettura compatta (5 layers, 64 neurons) previene l'overfitting.

#### 3.2.2 Miglior Configurazione Codice Originale: Test2

La configurazione **Originale_Test2** rappresenta la baseline più robusta:

- **F1 Score**: 87.27% (raggiunto all'epoca 38)
- **Accuracy**: 90.55%
- **Stabilità**: 0.52 (buona)
- **Degradazione**: 1.91 punti percentuali

**Parametri**:
- Learning rate: 0.0005 (conservativo)
- K = 30 neighbors
- Architettura: 5 layers, 64 neuroni

**Interpretazione**: Il codice originale con parametri conservativi mostra una convergenza più veloce (38 epoche) e stabilità eccellente, rendendolo adatto per ambienti di produzione dove l'affidabilità è prioritaria.

### 3.3 Confronto Diretto: Nuovo vs Originale

| Aspetto | Nuovo (TestBuono) | Originale (Test2) | Vincitore |
|---------|-------------------|-------------------|-----------|
| Best F1 | 88.02% | 87.27% | **Nuovo (+0.75%)** |
| Best Accuracy | 90.42% | 90.55% | Originale (+0.13%) |
| Stabilità (std) | 0.38 | 0.52 | Nuovo |
| Convergenza | 74 epoche | 38 epoche | **Originale** |
| Semplicità | 7 test, più tuning | 2 test, robusto | Originale |

**Insight chiave**: Il nuovo codice con active prefixes raggiunge un F1 score superiore dello 0.75%, dimostrando che il meccanismo di connessione dei prefissi attivi fornisce informazioni utili per la predizione. Tuttavia, richiede un tuning più attento dei parametri.

---

## 4. Analisi degli Insuccessi

### 4.1 Collasso del Training (Test2 e Test4)

Due configurazioni del nuovo codice hanno mostrato un comportamento patologico definito "collasso catastrofico":

**Nuovo_Test2** (LR=0.01, k=50):
- Best F1 raggiunto: 86.21% all'epoca 3
- Final F1: 4.11% (collasso quasi totale)
- Causa: Learning rate eccessivo (0.01)

**Nuovo_Test4** (LR=0.01, k=70):
- Best F1 raggiunto: 86.87% all'epoca 14
- Final F1: 4.10% (collasso quasi totale)
- Causa: Learning rate eccessivo (0.01)

**Analisi del fenomeno**: Con learning rate = 0.01, il modello converge inizialmente verso una buona soluzione, ma successivamente "salta fuori" dal minimo locale, divergendo verso performance casuali. Questo è un problema tipico di learning rate troppo alti che causano oscillazioni eccessive nei pesi della rete.

**Raccomandazione**: Il learning rate 0.01 è categoricamente sconsigliato per questo tipo di problema.

### 4.2 Overfitting Severo (Test1)

La configurazione **Nuovo_Test1** ha mostrato un pattern di overfitting pronunciato:

- Best F1: 72.27% raggiunto molto presto (epoca 6)
- Final F1: 64.35% dopo 107 epoche
- Degradazione: 7.92 punti percentuali (11% del valore iniziale)

**Analisi**: Nonostante i parametri siano identici a Test6 (che performa molto meglio), Test1 mostra un comportamento anomalo. Questo potrebbe indicare:
1. Differenze nell'inizializzazione dei pesi
2. Differenze nei dati di training/test split
3. Instabilità numerica durante il training

**Nota**: Questa discrepanza richiede un'investigazione approfondita, poiché configurazioni identiche dovrebbero produrre risultati simili.

---

## 5. Analisi dell'Impatto dei Parametri

### 5.1 Learning Rate

Il learning rate è emerso come il parametro più critico per le performance del modello:

| Learning Rate | N° Config. | F1 Medio | Stabilità Media | Esito |
|---------------|------------|----------|-----------------|-------|
| 0.0005 | 4 | 83.65% | 0.55 | ✅ Stabile, convergenza lenta |
| 0.001 | 1 | 88.02% | 0.38 | ✅ **Ottimale** |
| 0.005 | 2 | 87.22% | 4.64 | ⚠️ Variabile |
| 0.01 | 2 | 86.54% | 14.08 | ❌ Instabile, collasso |

**Conclusione**: Il learning rate **0.001** rappresenta il punto ottimale per il nuovo codice, bilanciando velocità di convergenza e stabilità.

### 5.2 Numero di Neighbors (K)

| K | Tipo | F1 Medio | Osservazioni |
|---|------|----------|--------------|
| 30 | Originale | 87.27% | Ottimale per codice originale |
| 40 | Nuovo | 82.71% | Variabile (dipende da LR) |
| 50 | Nuovo | 86.70% | Buono con LR=0.005 |
| 70 | Nuovo | 87.06% | Buono con LR=0.005 |

**Conclusione**: Il nuovo codice beneficia di valori di K più alti (50-70), probabilmente perché il meccanismo di active prefixes richiede un grafo più denso per propagare efficacemente le informazioni.

### 5.3 Architettura (Layers e Neurons)

| Architettura | N° Config. | F1 Medio | Pro | Contro |
|--------------|------------|----------|-----|--------|
| 5 layers, 64 neurons | 6 | 84.65% | Stabile, veloce | Capacità limitata |
| 7 layers, 128 neurons | 4 | 86.87% | Più espressiva | Richiede LR attento |

**Conclusione**: L'architettura più complessa (7L, 128N) ha potenziale superiore ma richiede un learning rate appropriato (0.005). L'architettura semplice (5L, 64N) è più robusta e consigliata per il deployment.

---

## 6. Discussione

### 6.1 Efficacia del Meccanismo Active Prefixes

L'analisi dei risultati suggerisce che il meccanismo di active prefixes **offre un vantaggio marginale ma misurabile**:

- **Miglioramento F1**: +0.75% rispetto alla migliore configurazione originale
- **Trade-off**: Richiede tuning più attento, stabilità potenzialmente inferiore

Il meccanismo funziona connettendo prefissi che sono "attivi" (cioè non ancora completati) al momento della predizione attraverso un nodo globale. Questo fornisce al modello informazioni sul contesto parallelo delle istanze di processo in corso.

### 6.2 Limitazioni dello Studio

1. **Numero limitato di configurazioni originali**: Solo 2 test per il codice originale vs 7 per il nuovo
2. **Stesso dataset**: I risultati potrebbero non generalizzare ad altri dataset
3. **Variabilità inspiegata**: Discrepanza tra Test1 e Test6 (stessi parametri, risultati diversi)
4. **Seed fisso**: Tutti i test usano seed=42, limitando la valutazione della variabilità statistica

### 6.3 Robustezza vs Performance

L'analisi evidenzia un trade-off fondamentale:

- **Codice originale**: Più robusto, meno sensibile ai parametri, performance leggermente inferiori
- **Nuovo codice**: Potenziale superiore, ma richiede tuning accurato e validazione estesa

Per ambienti di **produzione** dove l'affidabilità è prioritaria, il codice originale rimane la scelta raccomandata.

Per **ricerca** o contesti dove il massimo F1 è l'obiettivo primario, il nuovo codice con la configurazione TestBuono rappresenta la scelta migliore.

---

## 7. Conclusioni e Raccomandazioni

### 7.1 Configurazioni Raccomandate

| Contesto | Configurazione | Parametri | F1 Atteso |
|----------|----------------|-----------|-----------|
| **Produzione** | Originale_Test2 | LR=0.0005, K=30, 5L, 64N | 87.27% |
| **Max Performance** | Nuovo_TestBuono | LR=0.001, K=40, 5L, 64N | 88.02% |
| **Training Veloce** | Nuovo_Test3 | LR=0.005, K=50, 7L, 128N | 87.19% |

### 7.2 Best Practices Identificate

1. **Learning Rate**: Usare 0.001 per nuovo codice, 0.0005 per originale. Mai usare 0.01.
2. **Architettura**: Preferire 5 layers, 64 neurons per stabilità; 7 layers, 128 neurons solo con LR=0.005.
3. **Early Stopping**: Implementare early stopping per evitare degradazione post-convergenza.
4. **Validazione**: Eseguire multiple run con seed diversi per valutare la variabilità.

### 7.3 Sviluppi Futuri

1. **Learning Rate Scheduling**: Implementare decay del learning rate per stabilizzare il training
2. **Ensemble**: Combinare predizioni di configurazioni multiple per migliorare robustezza
3. **Analisi Active Prefixes**: Studiare il numero ottimale di prefissi attivi da considerare
4. **Cross-validation**: Valutare le configurazioni con k-fold cross-validation

---

## 8. Appendice: Visualizzazioni Generate

Le seguenti visualizzazioni sono state generate e salvate in `confronto_risultati/visualizzazioni_separate/`:

1. **01_f1_nuovo_codice.png** - Evoluzione F1 per tutti i test del nuovo codice
2. **02_f1_codice_originale.png** - Evoluzione F1 per i test del codice originale
3. **03_confronto_migliori.png** - Confronto diretto TestBuono vs Originale_Test2
4. **04_classifica_f1.png** - Classifica delle configurazioni per Best F1
5. **05_velocita_convergenza.png** - Epoche necessarie per raggiungere il best F1
6. **06_stabilita.png** - Deviazione standard nelle ultime 20 epoche
7. **07_degradazione.png** - Differenza tra Best F1 e Final F1
8. **08_f1_vs_stabilita.png** - Scatter plot performance vs stabilità
9. **09_heatmap_parametri.png** - Heatmap parametri e metriche
10. **10_dashboard_riepilogo.png** - Dashboard riassuntivo completo

---

*Relazione generata il: 17 Gennaio 2026*
*Dataset: RequestForPayments*
*Framework: DGCNN con meccanismo Active Prefixes*
