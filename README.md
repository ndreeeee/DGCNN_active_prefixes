# DGCNN - Next Activity Prediction

Progetto universitario realizzato per il corso di **BIG Data Analytics and Machine Learning**, all'interno del corso di Laurea Magistrale in **Ingegneria Informatica e dell'Automazione** presso l'**Università Politecnica delle Marche (UNIVPM)**.

## Autori
- Andrea Altieri
- Andrea Flaiani

## Descrizione del Progetto

Questo repository contiene un'applicazione orientata al **Predictive Process Monitoring**, che integra lo studio e l'analisi dei log degli eventi aziendali (**Process Mining**) con algoritmi basati sull'apprendimento profondo su grafi (**Deep Graph Convolutional Neural Networks - DGCNN**). L'obiettivo finale di quest'architettura è imparare dal passato per prevedere quale sarà la successiva operazione all'interno di un flusso in via di esecuzione.

Il sistema prende in analisi file log informatici (formato `.xes`), identificati dai sistemi aziendali, e li converte in strutture a grafo. In queste strutture, i nodi sono le attività specifiche, mentre gli archi definiscono una connessione e una direzione temporale. Studiando accuratamente queste sequenze topologiche, il modello intelligente estrae un pattern causale per le sue predizioni esatte.

## Caratteristiche Principali

1. **Estrazione e Pulizia dei Dati (Algoritmo BIG)**: L'algoritmo di partenza legge le tracce dal file log, tentando di definire un modello ideale (come una Rete di Petri). Inoltre, mitiga eventuali errori o rumore causato da registrazioni scorrette (utilizzando logiche di *insertion* e *deletion repair*), garantendo la costruzione di grafi ordinati e puliti.
2. **Gestione dei Processi Paralleli**: Nella realtà aziendale più casi vengono avviati simultaneamente. Questo applicativo non si limita al singolo caso, ma è capace di connettere istanze parallele attive in uno stesso lasso temporale. Lo fa attraverso dei "Super-Grafi" collegati da uno speciale nodo globale `G` che offre all'algoritmo visibilità del panorama operativo nella sua completezza (Context-Awareness).
3. **Deep Graph Convolutional Neural Network (DGCNN)**: Lato Deep Learning, le strutture e i tensori passano attraverso l'architettura fornita da PyTorch Geometric (`SAGEConv` unitamente a procedimenti di `SortAggregation`). Questo sistema di pooling riesce efficacemente a processare grafi con ampiezze e layout differenti di volta in volta.

## Struttura della Repository

- `config.py`: File di controllo che centralizza tutte le directory (input e dataset in uscita) e tutti gli iperparametri previsti per l'addestramento (epoche, valori del learning rate, k-neighbors, ecc.).
- `BIG.py`: Lo script designato alla traduzione del log di input in grafi standard per le esecuzioni (`.g`).
- `TO_GRAPHS_ACTIVE_NODES_NORES_OPT.py`: Modulo di pre-elaborazione che mappa i grafi .g creati in oggetti-tensore decifrabili da PyTorch, implementando anche la logica unificatrice dei Super-Grafi sui casi attivi.
- `DGCNN.py`: Script della rete neurale vera e propria con relative specifiche topologiche (layer lineari, network e dropout preposti per generare output multiclasse).
- `TRAINING.py`: Modulo che concretizza l'intero ciclo logico. Suddivide le occorrenze fra set di train e testing tutelando l'osservanza temporale cronologica, addestra il modello, genera le perdite (Loss) e logga l'esito avvalendosi di librerie per le metriche F1-Score e dell'Accuracy.
- `requirements.txt`: Elenco delle librerie esterne raccomandate.

## Come iniziare

**Requisiti e Installazione**  
Raccomandiamo di appoggiarsi ad un environment virtuale (venv o conda). Installa le librerie tramite riga di comando:

```bash
pip install -r requirements.txt
```

*(Importante: l'architettura è supportata attivamente da `torch==1.13.1` e `torch-geometric==2.3.1`. Accertati che sia compatibile con la tua variante CUDA, qualora desiderassi addestrare la rete avvalendoti della corretta GPU accelerator).*

**Esecuzione Passata a Passata:**

1. **Preparazione Dati Sorgente:** Crea of modifica il file testuale `log_name.txt` all'interno della directory root per trascrivervi (senza l'estesione del formato) come dev'essere riconosciuto il nome del log file. Tale operazione implicherà il dover piazzare fisicamente questo file `.xes` esattamente posizionato nella cartella `input/xes/`. 
2. **Estrazione e Generazione Grafi:** Genera il reticolo per i nodi, partendo dal log:
   ```bash
   python BIG.py
   ```
3. **Tensori, Conversione e Formazione del Modello:** A generazione grafi avvenuta, si esegue il file deputato all'addestramento della rete neurale e all'estrazione terminale sui risultati analitici:
   ```bash
   python TRAINING.py
   ```
