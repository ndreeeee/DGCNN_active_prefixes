# DGCNN - Next Activity Prediction

Progetto universitario realizzato per il corso di **BIG Data Analytics and Machine Learning** del corso di Laurea Magistrale in **Ingegneria Informatica e dell'Automazione** presso l'**Università Politecnica delle Marche (UNIVPM)**.

## Autori
- Andrea Altieri
- Andrea Flaiani

## Descrizione del Progetto

Questo repository contiene un'applicazione per prevedere l'attività futura all'interno di un processo (Predictive Process Monitoring). Per far questo, il sistema unisce le tecniche di **Process Mining** con reti neurali avanzate chiamate **Deep Graph Convolutional Neural Networks (DGCNN)**.

In parole semplici, il sistema prende i file di log aziendali (in formato `.xes`) e li trasforma in "grafi": insiemi di punti (le attività fatte) uniti da linee (l'ordine temporale in cui sono state fatte). Dopodiché, fa studiare questi grafi a un'intelligenza artificiale per farle imparare a prevedere quale sarà la mossa o l'attività successiva in un processo ancora in corso.

## Caratteristiche Principali

1. **Gestione del File e dei Dati (Algoritmo BIG)**: Legge i dati dal file di log e corregge automaticamente eventuali errori nelle registrazioni delle singole sequenze. Il risultato è la creazione di grafi esatti, ripuliti dal rumore.
2. **Supporto alle Esecuzioni in Parallelo**: Nel mondo reale, molte attività succedono in parallelo. Per questo motivo, lo strumento collega i vari casi attivi in quel momento creando un "Super-Grafo". Tramite l'uso di un nodo globale `G`, il sistema fa vedere all'Intelligenza Artificiale il contesto temporale completo, permettendo predizioni molto più precise e logiche.
3. **Rete Neurale Avanzata (DGCNN)**: Usa la libreria PyTorch per analizzare strutture geometriche che cambiano di volta in volta, elaborando migliaia di grafi complessi per impararne i pattern.

## Struttura del Codice

- `config.py`: File in cui impostare tutte le configurazioni (dove pescare i file di input, quante "epoche" far fare alla rete neurale, cartelle di output, ecc.).
- `BIG.py`: Lo script che trasforma il log sorgente in grafi (file con estensione `.g`).
- `TO_GRAPHS_ACTIVE_NODES_NORES_OPT.py`: File di collegamento che trasforma i grafi scritti al passo precedente nella forma matematica che PyTorch riesce a capire (tensori). Gestisce anche la creazione del "Super-Grafo" per i decorsi contemporanei.
- `DGCNN.py`: Contiene semplicemente il posizionamento e le specifiche matematiche dei vari layer della rete.
- `TRAINING.py`: Il vero motore del programma. Usa tutti gli script precedenti per addestrare l'IA, mettere alla prova i modelli su dati nascosti e fornire grafici sulle reali performance di previsione.
- `requirements.txt`: La lista di tutti i pacchetti esterni necessari da scaricare.

## Come Farlo Funzionare

**Requisiti e Installazione**  
Per prima cosa, assicurati di scaricare i moduli usati dallo script lanciando questo comando da riga di comando (es. PowerShell o Terminal):

```bash
pip install -r requirements.txt
```

**Esecuzione Passata a Passata:**

1. **Preparazione dei Dati:** Inserisci il nome del tuo file log dentro a `log_name.txt` nella cartella principale (senza scrivere `.xes`). Inserisci invece il file vero e proprio dentro alla cartella `input/xes/`. 
2. **Primo Step - Creazione dei Grafi:** Genera tutti i grafi dal file log eseguendo:
   ```bash
   python BIG.py
   ```
3. **Secondo Step - Trasformazione e Addestramento:** Una volta ultimata la fase dei grafi, avvia  l'apprendimento del modello lanciando:
   ```bash
   python TRAINING.py
   ```
   *Nota: lo script genererà in autonomia molteplici grafici finali sulle performance e sulle metriche usate (come F1-Score).*
