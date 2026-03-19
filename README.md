# DGCNN Active Prefixes

Questo repository implementa un'applicazione avanzata di **Predictive Process Monitoring** (PPM) che combina discipline di **Process Mining** e **Deep Graph Learning**. L'obiettivo primario è prevedere la successiva attività di un processo aziendale (Next Activity Prediction), elaborando e trasformando le tracce formali estratte in grafi arricchiti da informazioni strutturali e temporali.

## Caratteristiche Principali

Il progetto si basa sulla traduzione delle tracce dei log degli eventi (`.xes`) in veri e propri "Instance Graph", permettendo alle reti convoluzionali di sfruttare topologia e contesto:

- **Estrazione dei Modelli e Conformity Tracking (Algoritmo BIG):** Il sistema tenta la scoperta induttiva di un modello normativo (sotto forma di Rete di Petri). Segue l'allineamento delle tracce a quest'ultima e una complessa procedura di pulizia via *insertion repair* e *deletion repair*. Tale meccanismo garantisce grafi delle istanze altamente robusti al rumore, i quali catturano le esatte e conformi sequenze causali e parallele tra le operazioni del log origiale.
- **Rappresentazione a Grafo Multiplo (Active Prefixes Context-awareness):** In ecosistemi complessi (soprattutto esposti ad esecuzioni parallele non sequenziali), il framework costruisce al volo dei "Super-Grafi". Questo approccio dinamico connette il prefisso di traccia attenzionato (Reference Prefix) con tutti gli altri "Prefissi Attivi" appartenenti a casi concomitanti ed eseguiti nel medesimo lasso di tempo. L'unione avviene tramite un nodo Globale convergente (chiamato `G`).
- **Deep Graph Convolutional Neural Network (DGCNN):**  La rete esibisce architetture convoluzionali spaziali modellate col framework GraphSAGE di Pytorch Geometric (`SAGEConv`). Essendo i grafi estratti variabili in volume e topologia, gli stack convolutivi delegano l'uniformità dei tensori originati ad uno strato di `SortAggregation` (Pooling spaziale a k-nodi).
- **Temporal Split & Valutazione Metriche:** Un generatore di layout visivi (alimentato da `networkx`) consente un'esplorazione step-by-step dei layer interattivi sui tensori. I target sono divisi coerentemente col tempo reale delle log (Temporal Split) e i log statistici finali descrivono variazioni di Loss, metriche multi-classe F1-Score e curve aggregate per la lunghezza d'istanza analizzata.

## Struttura della Codebase

Di seguito è riportata un'overview dei macro-moduli:

- `config.py`: Centralizza e manipola l'assoluta interezza delle configurazioni (path delle directory Input/Output, iperparametri e pesi per le fasi di tuning della neural net - e.g. epoch, seed, learning rate). Supporta l'istanziamento autogestito delle directory previste dal path.
- `BIG.py`: Contiene l'implementazione dell'algoritmo *BIG*. Interagendo ampiamente con le interfacce esposte da librerie come `pm4py`, scopre i modelli Petri, ne testa l'aderenza con le tracce iniziali e ne esegue allineamenti causali. Finalizza emettendo file ad estensione `.g` (istanze a grani individuali).
- `TO_GRAPHS_ACTIVE_NODES_NORES_OPT.py`: Motore pre-elaborativo preposto alla mappatura dei `.g` generati in autentici tensori (`Data` format di `torch_geometric`). Implementa logiche di merge e collision solving per unire i prefix attualizzati al Super-Grafo.
- `DGCNN.py`: Definizione architetturale del modulo di rete Pytorch multi-layer descrivente la `DGCNNSTATE`.
- `TRAINING.py`: Modulo predisposto per assorbire i subset (Train vs Test Dataset), attuando cicli di forward/backward sul device, producendo diagrammi e statistiche ad alto livello (es. confusion matrix, matrici di somiglianza strutturale sui set multi-layer e grafici loss-based).

## Dipendenze e Requisiti

Assicurati di disporre delle seguenti dipendenze principali prima dell'esecuzione:
- `torch` == 1.13.1  *(o versioni superiori compatibili col proprio hardware)*
- `torch-geometric` == 2.3.1
- `pm4py` == 2.2.16
- Pacchetti standard aggiuntivi annotati in `requirements.txt` (`matplotlib`, `networkx`, `pandas`, `scikit-learn`, ecc.).

Per installarle è possibile utilizzare il classico gestore pip:
```bash
pip install -r requirements.txt
```

## Come eseguire

1. **Configurazione:** Iniziate modificando parametri desiderati (come *num_epochs* o *patience*) situati all'interno della funzione `load()` in `config.py`. Creare la stringa di testo all'interno di `log_name.txt` allocato presso la root directory del progetto qualora vorreste targetizzare uno `.xes` specifico in `input/xes/`. 
2. **Inizializzazione Albero di Eventi:** Posizionate il vostro file Event Log all'interno della corretta sottocartella `input/xes/`. In assenza di informazioni puntuali `config.py` cercherà una root con file base (es. `testXes.xes`).
3. **Elaborazione delle Tracce in Graph Formats:**
   Eseguite ora lo script generativo per trasformare il Log in topologie archiviate temporaneamente nell'output preposto format `.g`.
   ```bash
   python BIG.py
   ```
4. **Apprendimento ed Analisi Modello DGCNN:**
   L'output ora incanalato è pronto a fungere da Dataset tensoriale; l'ultimo passaggio consisterà nel processare dinamicamente col convertitore i grafi estesi, per poi passarlo da script ai backend di PyTorch.
   ```bash
   python TRAINING.py
   ```
   *(Lo script consentirà -se desiderato e de-commentato ove descrittoci- sia la navigazione step by step dei prefissi esaminati su pop-up esterni interattivi, seguitamente l'esplicitazione formale delle feature extractate).*
