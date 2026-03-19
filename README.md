<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=DGCNN%20Next%20Activity%20Prediction&fontSize=50&animation=fadeIn" alt="Header Banner">
  <br>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Process_Mining-00599C?style=for-the-badge&logo=data-version-control&logoColor=white" alt="Process Mining">
</div>

<br>

Progetto universitario realizzato per il corso di **BIG Data Analytics and Machine Learning**, all'interno del corso di Laurea Magistrale in **Ingegneria Informatica e dell'Automazione** presso l'**Università Politecnica delle Marche (UNIVPM)**.

## Autori
- Andrea Altieri
- Andrea Flaiani

## Descrizione del progetto

Questo repository contiene un'applicazione orientata al **Predictive Process Monitoring**, che integra lo studio e l'analisi dei log degli eventi aziendali (**Process Mining**) con algoritmi basati sull'apprendimento profondo su grafi (**Deep Graph Convolutional Neural Networks - DGCNN**). L'obiettivo finale di quest'architettura è imparare dal passato per prevedere quale sarà la successiva operazione all'interno di un flusso in via di esecuzione.

Il sistema prende in analisi file log informatici (formato `.xes`), identificati dai sistemi aziendali, e li converte in strutture a grafo. In queste strutture, i nodi sono le attività specifiche, mentre gli archi definiscono una connessione e una direzione temporale. Studiando accuratamente queste sequenze topologiche, il modello intelligente estrae un pattern per le sue predizioni.

## Caratteristiche Principali

1. **Estrazione e pulizia dei dati (Algoritmo BIG)**: l'algoritmo di partenza legge le tracce dal file log, tentando di definire un modello ideale (come una Rete di Petri). Inoltre, mitiga eventuali errori o rumore causato da registrazioni scorrette (utilizzando logiche di *insertion* e *deletion repair*), garantendo la costruzione di grafi ordinati e puliti.
2. **Gestione dei processi paralleli**: nella realtà aziendale più casi vengono avviati simultaneamente. Questo applicativo non si limita al singolo caso, ma è capace di connettere istanze parallele attive in uno stesso lasso temporale. Lo fa attraverso dei "Super-Grafi" collegati da uno speciale nodo globale `G` che offre all'algoritmo visibilità del panorama operativo nella sua completezza (Context-Awareness).
3. **Deep Graph Convolutional Neural Network (DGCNN)**: lato Deep Learning, le strutture e i tensori passano attraverso l'architettura fornita da PyTorch Geometric (`SAGEConv` unitamente a procedimenti di `SortAggregation`). Questo sistema di pooling riesce efficacemente a processare grafi con ampiezze e layout differenti di volta in volta.

## Struttura della repository

- `config.py`: file di controllo che centralizza tutte le directory (input e dataset in uscita) e tutti gli iperparametri previsti per l'addestramento (epoche, valori del learning rate, k-neighbors, ecc.).
- `BIG.py`: lo script designato alla traduzione del log di input in grafi standard per le esecuzioni (`.g`).
- `TO_GRAPHS_ACTIVE_NODES_NORES_OPT.py`: modulo di pre-elaborazione che mappa i grafi `.g` creati in oggetti-tensore decifrabili da PyTorch, implementando anche la logica unificatrice dei Super-Grafi sui casi attivi.
- `DGCNN.py`: script della rete neurale vera e propria con relative specifiche topologiche (layer lineari, network e dropout preposti per generare output multiclasse).
- `TRAINING.py`: modulo che concretizza l'intero ciclo logico. Suddivide le occorrenze fra set di train e testing tutelando l'osservanza temporale cronologica, addestra il modello, genera le perdite (Loss) e logga l'esito avvalendosi di librerie per le metriche F1-Score e dell'Accuracy.
- `requirements.txt`: elenco delle librerie esterne raccomandate.

## Come iniziare

**Requisiti e installazione**  
Raccomandiamo di appoggiarsi ad un environment virtuale (venv o conda). Installa le librerie tramite riga di comando:

```bash
pip install -r requirements.txt
```

**Esecuzione passo per passo:**

1. **Preparazione dati sorgente:** il nome del dataset da usare deve essere scritto all'interno del file testuale `log_name.txt` sito nella directory principale (senza scrivere l'estensione `.xes`). Ad esempio, se il tuo log si chiama `processo_vendite.xes`, scriverai semplicemente `processo_vendite` dentro il `.txt`. Assicurati che il file `.xes` fisico sia posizionato nella cartella `input/xes/`. 
2. **Estrazione e generazione grafi:** genera il reticolo per i nodi, partendo dal log:
   ```bash
   python BIG.py
   ```
3. **Conversione dei nodi e creazione dei Super-Grafi:** esegui questo script appena i file `.g` si sono creati, così da trasformare la topologia in tensori di PyTorch associando dinamicamente i prefissi attivi.
   ```bash
   python TO_GRAPHS_ACTIVE_NODES_NORES_OPT.py
   ```
4. **Formazione del modello e risultati:** a generazione tensoriale avvenuta, si esegue il file deputato all'addestramento della rete neurale e all'estrazione terminale sui risultati analitici:
   ```bash
   python TRAINING.py
   ```
