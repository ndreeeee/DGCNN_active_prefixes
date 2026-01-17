import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import InMemoryDataset, Dataset
from DGCNN import DGCNNSTATE as DGCNN
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, concat
import numpy as np
from itertools import product
from time import time
import random
import seaborn as sns
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import defaultdict

from os import makedirs
from os.path import join, exists
from datetime import datetime
from config import (load, NET_RESULTS_PATH, TARGET_PAR_FILE_PATH, TARGET_STD_FILE_PATH,
                    ATTRIBUTES_FILE_PATH, COMPLETE_IGS_NAME, DATASET_PATH)

args = load()
plt.ioff()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.ioff()


class TraceDataset(InMemoryDataset):
    def __init__(self):
        super(TraceDataset, self).__init__(DATASET_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f'{COMPLETE_IGS_NAME}_par.pt']


# dizionario: chiave=target, valore=indice progressivo (da 0)
def dict_target():
    target_std = {}
    i = 0
    with open(TARGET_STD_FILE_PATH, "r") as f:
        for lines in f.readlines():
            lines = lines[:-1]
            target_std[lines] = i
            i += 1
    # print('target std', target_std)

    target_par = {}
    # print('target par', target_par)
    return target_std, target_par


def pad_tensor(tensor, target_shape):
    pad_width = [(0, max(0, ts - s)) for s, ts in zip(tensor.shape, target_shape)]
    return np.pad(tensor, pad_width, mode='constant')


def check_similar_elements(tensor1, tensor2):
    # Flatten the tensors to 1D arrays
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()

    # Convert the arrays to sets to get unique elements
    set_tensor1 = set(flat_tensor1)
    set_tensor2 = set(flat_tensor2)

    # Find intersection of element sets
    similar_elements = set_tensor1.intersection(set_tensor2)

    return similar_elements


def save_similarity_matrix(data, row_headers, col_headers):
    # Find the maximum width for each column
    col_width = max(len(header) for header in col_headers) + 2
    data_width = max(len(f"{num:.2f}") for num in data.flatten()) + 2
    total_width = max(col_width, data_width)

    # Format the headers
    header_row = " " * total_width + "".join(f"{header:<{total_width}}" for header in col_headers)

    # Format the rows with data
    rows = []
    for header, row in zip(row_headers, data):
        row_str = f"{header:<{total_width}}" + "".join(f"{num:<{total_width}.2f}" for num in row)
        rows.append(row_str)

    # Add separators
    separator = "-" * total_width * (data.shape[1] + 1)

    # Combine everything into a single string
    table = header_row + "\n" + separator + "\n" + f"\n{separator}\n".join(rows) + "\n" + separator

    # Save to a text file
    with open('/path/to/your/file/matrix_table.txt', 'w') as f:
        f.write(table)


def plot_labels_line(labels_list, dot_indexes):
    unique_values = list(set(labels_list))

    labels_list = labels_list[8000:10000]
    dot_indexes = [idx - 8000 for idx in dot_indexes if 8000 <= idx < 10000]

    predefined_colors = [
        (1, 'red'),
        (2, 'green'),
        (3, 'blue'),
        (4, 'orange'),
        (5, 'purple'),
        (6, 'cyan'),
        (7, 'magenta'),
        (8, 'yellow'),
        (9, 'brown')
    ]

    # Ensure that there are enough colors for the unique values
    if len(unique_values) > len(predefined_colors):
        raise ValueError("Not enough predefined colors for the unique values in labels_list.")

    color_map = {value: color for value, color in predefined_colors if value in unique_values}
    # Generate the x values (indices)
    x = np.arange(len(labels_list))
    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plot each segment with the corresponding color

    for i in range(len(labels_list) - 1):
        plt.scatter(x[i:i + 2], labels_list[i:i + 2], color=color_map[labels_list[i]])

    for idx in dot_indexes:
        plt.plot(idx, labels_list[idx], marker='o', markersize=10, color='black')

    plt.legend()
    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Class')
    plt.title('Split by classes')
    plt.savefig('output/similarities/split_by_class_reducted.png')
    plt.close()
    
    
def temporal_split(G, train_ratio=0.67):
    """
    Divide il dataset in train e test rispettando l'ordine temporale.

    Args:
        G (list): lista di Data con attributo .start_timestamp
        train_ratio (float): percentuale da usare come training (0-1)

    Returns:
        (train, test): due liste di Data
    """
    # Ordina le tracce per timestamp di inizio
    sorted_data = sorted(G, key=lambda d: d.start_timestamp)

    # Calcola l'indice di cut
    split_idx = int(len(sorted_data) * train_ratio)

    # Split effettivo
    train = sorted_data[:split_idx]
    test = sorted_data[split_idx:]

    return train, test

def analyze_temporal_split(train, test):
    # --- 1. Controllo duplicati ---
    print(len(train))
    print(len(test))
    train_ids = set([d.start_timestamp for d in train])
    test_ids = set([d.start_timestamp for d in test])
    duplicates = train_ids.intersection(test_ids)
    if duplicates:
        print(f"Trovati {len(duplicates)} duplicati tra train e test: {duplicates}")
    else:
        print("Nessun duplicato tra train e test")

    # --- 2. Distribuzione temporale ---
    train_times = [d.start_timestamp for d in train]
    test_times = [d.start_timestamp for d in test]

    plt.figure(figsize=(12,4))
    plt.scatter(train_times, [1]*len(train_times), color='blue', label='Train', alpha=0.5)
    plt.scatter(test_times, [2]*len(test_times), color='red', label='Test', alpha=0.5)
    plt.yticks([1,2], ['Train','Test'])
    plt.xlabel('Timestamp')
    plt.title('Distribuzione temporale di Train e Test')
    plt.legend()
    plt.show()

    # --- 3. Percentuali per classe ---
    train_classes = [int(d.y) for d in train]
    test_classes = [int(d.y) for d in test]
    all_classes = set(train_classes + test_classes)
    print("\nPercentuale train/test per classe:")
    for cls in all_classes:
        total = train_classes.count(cls) + test_classes.count(cls)
        train_pct = train_classes.count(cls)/total*100
        test_pct = test_classes.count(cls)/total*100
        print(f"Classe {cls}: Train {train_pct:.2f}%, Test {test_pct:.2f}%")


"""
# la divisione fra train e test viene fatta proporzionalmente per ogni classe
def split_target_print(G, per):
    dict = {}
    tar_std, _ = dict_target()  # dizionario: chiave=attività (target), valore=codice progressivo
    for x in tar_std.keys():  # crea coppie chiave (codice progressivo) - lista vuota per ogni target
        dict[tar_std[x]] = []

    tensor_data = []
    tensor_labels = []
    indexes = []

    for x in G:
        dict[int(x.y[0])].append(x)
    for data in G:
        tensor_data.append(data.x)
    for l in G:
        tensor_labels.append(int(l.y))

    # aggiunge alla lista (valore) nel dizionario alla chiave (codice progressivo) il Data che descrive il grafo
    train = []
    test = []
    for x in dict.keys():
        target = x
        a = []
        a.extend(dict[x])  # inserisce la lista del dizionario con chiave x alla lista a

        # split effettivo
        l = int(len(a) / 100 * per)
        # si confrontano gli ultimi 10% elementi di l per il train e i primi per il test
        # percentage_to_compare = int(l*0.05)
        percentage_to_compare = 20
        atr = a[:l]
        if atr:
            index = next((i for i, tensor in enumerate(tensor_data) if torch.equal(tensor, atr[-1].x)), None)
            indexes.append(index)
        to_compare_train = list(reversed(atr[-percentage_to_compare:]))
        ate = a[l:]
        to_compare_test = ate[0:percentage_to_compare]
        if percentage_to_compare > len(to_compare_test):
            percentage_to_compare = len(to_compare_test)
            to_compare_train = list(reversed(atr[-percentage_to_compare:]))

        similarities = []
        common_elements = []
        distances = []
        similarities_train_test = []
        distances_train_test = []
        common_elements_train_test = []

        column_headers = [len(i.x) for i in to_compare_train]
        row_headers = [len(i.x) for i in to_compare_test]

        for i in range(0, len(to_compare_train)):
            if similarities:
                similarities_train_test.append(similarities)
                distances_train_test.append(distances)
                common_elements_train_test.append(common_elements)
            similarities = []
            distances = []
            common_elements = []
            for j in range(0, len(to_compare_test)):

                # Determine the target shape (maximum dimensions)
                test_tensor = to_compare_test[i].x
                train_tensor = to_compare_train[j].x
                test_tensor_position = next(
                    (i for i, tensor in enumerate(tensor_data) if torch.equal(tensor, test_tensor)), None)
                train_tensor_position = next(
                    (i for i, tensor in enumerate(tensor_data) if torch.equal(tensor, train_tensor)), None)

                set1 = set(map(tuple, test_tensor.tolist()))
                set2 = set(map(tuple, train_tensor.tolist()))

                # see if tensors are the same
                try:
                    is_equal = bool(torch.all(train_tensor.eq(test_tensor)))
                    if is_equal:
                        print('same tensors')
                except Exception:
                    pass

                # Find common rows
                common_rows = set1 & set2

                similarity = len(common_rows) / max(len(set1), (len(set2)))
                distance = max(train_tensor_position, test_tensor_position) - min(train_tensor_position,
                                                                                  test_tensor_position)
                common_element = len(common_rows)

                similarities.append(similarity)
                distances.append(distance)
                common_elements.append(common_element)

        similarities_train_test.append(similarities)
        distances_train_test.append(distances)
        common_elements_train_test.append(common_elements)
        if similarities_train_test != [[]]:
            plot_matrixes(similarities_train_test, 1, dict, False, False, target, a, False)
            # 'output/similarities/class'+str(target))
            plot_matrixes(common_elements_train_test, False, dict, column_headers, row_headers, target, a, False)
            # 'output/similarities/class'+str(target)+'_tensors_count')
        if distances_train_test != [[]]:
            plot_matrixes(distances_train_test, -1, dict, False, False, target, a, False)
            # 'output/distances/class'+str(target))

        train.extend(atr.copy())
        test.extend(ate.copy())
    # plot_labels_line(labels_list=tensor_labels, dot_indexes=indexes)
    return train, test


def split_target(G, per):
    dict = {}
    tar_std, _ = dict_target()  # dizionario: chiave=attività (target), valore=codice progressivo
    for x in tar_std.keys():  # crea coppie chiave (codice progressivo) - lista vuota per ogni target
        dict[tar_std[x]] = []

    for x in G:
        dict[int(x.y[0])].append(x)
        # aggiunge alla lista (valore) nel dizionario alla chiave (codice progressivo) il Data che descrive il grafo
    train = []
    test = []
    for x in dict.keys():
        a = []
        a.extend(dict[x])  # inserisce la lista del dizionario con chiave x alla lista a
        # split effettivo
        l = int(len(a) / 100 * per)
        atr = a[:l]
        ate = a[l:]
        train.extend(atr.copy())
        test.extend(ate.copy())
    return train, test


def split_target_val(G, n_folds, shuffle=True, random_state=42):
    # Initialize StratifiedKFold with specified number of splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    # Initialize lists to hold the train and test sets for each fold
    train_test_splits = []

    # Iterate through each fold
    for train_index, test_index in skf.split(G, list(G.y)):
        train_data = [G[i] for i in train_index]
        test_data = [G[i] for i in test_index]
        train_test_splits.append((train_data, test_data))

    return train_test_splits

"""
def plot_matrixes(data, similarity, dict, column_headers, row_headers, target, a, output):
    matrix = np.array(data)
    plt.figure(figsize=(12, 8))
    if similarity == 1:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='BuGn', linewidths=0.5, linecolor='gray', vmax=np.max(matrix),
                    robust=True, annot_kws={"fontsize": 8})
        plt.title('Matrix for label ' + str(target) + ', class distribution: ' +
                  str(len(a) / sum(len(lst) for lst in dict.values())))  # Add title
    elif similarity == -1:
        cmap = sns.color_palette("BuGn_r", as_cmap=True)
        sns.heatmap(matrix, annot=True, cmap=cmap, linewidths=0.5, linecolor='gray', vmax=np.max(matrix),
                    robust=True, annot_kws={"fontsize": 8})
        plt.title('Matrix distances for label ' + str(target) + ', class distribution: ' +
                  str(len(a) / sum(len(lst) for lst in dict.values())))  # Add title
    else:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='BuGn', linewidths=0.5, linecolor='gray',
                    vmax=np.max(matrix), robust=True, annot_kws={"fontsize": 8},
                    xticklabels=column_headers, yticklabels=row_headers)
        plt.title('Matrix for label ' + str(target) + ', class distribution: ' +
                  str(len(a) / sum(len(lst) for lst in dict.values())))
    plt.xlabel('Train data')  # Add x-axis label if needed
    plt.ylabel('Test data')  # Add y-axis label if needed
    if output:
        plt.savefig(output, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(reals, predictions, keyword, epoch, path):
    stacked = torch.stack((reals, predictions), dim=1)
    classes = []
    with open(ATTRIBUTES_FILE_PATH, 'r') as f:
        for lines in f.readlines():
            lines = lines[:-1]
            classes.append(lines)

    cmt = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1

    plt.figure(figsize=(15, 15))
    plt.imshow(cmt, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    for i, j in product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(j, i, format(cmt[i, j], 'd'), horizontalalignment="center",
                 color="white" if cmt[i, j] > cmt.max() / 2. else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(path, f'cm_{keyword}_{epoch}.png'))
    plt.close()


def calculate_metrics_by_prefixes(prefix_results_df, path):
    def calculate_metrics(df):
        df['y_true'] = df['y_true'].astype(int)
        df['y_pred'] = df['y_pred'].astype(int)
        accuracy = accuracy_score(df['y_true'], df['y_pred'])
        f1 = f1_score(df['y_true'], df['y_pred'], average='weighted')
        return Series({'Accuracy': accuracy, 'F1_Score': f1})

    grouped_results = prefix_results_df.groupby('prefix_len').apply(calculate_metrics).sort_values(
        by='prefix_len')
    plt.figure(figsize=(10, 6))
    grouped_samples = prefix_results_df.groupby('prefix_len').size().reset_index(name='Counts').sort_values(
        by='prefix_len')

    plt.plot(grouped_results.index, grouped_results['Accuracy'], label='Accuracy')  #, marker='o')
    plt.plot(grouped_results.index, grouped_results['F1_Score'], label='F1 Score')  #, marker='x')
    plt.plot(grouped_samples['prefix_len'], grouped_samples['Counts'] / grouped_samples['Counts'].max(), 'r--',
             label='#samples')
    # plt.xticks(range(int(np.min(prefix_results_df['prefix_len']))-1, int(np.max(prefix_results_df['prefix_len']))+1, 3), rotation=30)

    # plt.ylabel('Weighted F1-score')
    plt.xlabel('Prefix lengths')
    plt.title('Weighted F1-score varying prefix lengths')
    plt.legend()
    # Imposta lo sfondo bianco e rimuovi la griglia
    plt.gca().set_facecolor('white')
    plt.grid(False)
    plt.savefig(join(path, f'prefix_f1_acc_score.png'))
    plt.close()
    return grouped_results


def plot_comb_metrics(result_df, path):
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(result_df['train_loss'].tolist(), label='Loss train')
    plt.plot(result_df['test_loss'].tolist(), label='Loss test')
    plt.legend()
    plt.savefig(join(path, 'losses.png'))
    plt.close('all')

    plt.xlabel('Epochs')
    plt.ylabel('Weighted F1-score')
    plt.plot(result_df['train_f1'].tolist(), label='Weighted F1-score train')
    plt.plot(result_df['test_f1'].tolist(), label='Weighted F1-score test')
    plt.legend()
    plt.savefig(join(path, 'f1.png'))
    plt.close('all')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(result_df['train_accuracy'].tolist(), label='Accuracy train')
    plt.plot(result_df['test_accuracy'].tolist(), label='Accuracy test')
    plt.legend()
    plt.savefig(join(path, 'acc.png'))
    plt.close('all')


def plot_comb_prefix_metrics(result_df, path):
    def calculate_metrics(df):
        df['y_true'] = df['y_true'].astype(int)
        df['y_pred'] = df['y_pred'].astype(int)
        accuracy = accuracy_score(df['y_true'], df['y_pred'])
        f1 = f1_score(df['y_true'], df['y_pred'], average='weighted')
        return Series({'Accuracy': accuracy, 'F1_Score': f1})

    grouped_results = result_df.groupby('prefix_len').apply(calculate_metrics).sort_values(
        by='prefix_len')
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_results.index, grouped_results['Accuracy'], label='Accuracy', marker='o')
    plt.plot(grouped_results.index, grouped_results['F1_Score'], label='F1 Score', marker='x')

    plt.xticks(range(int(np.min(result_df['prefix_len'])) - 1, int(np.max(result_df['prefix_len'])) + 1, 1))

    plt.xlabel('Number of prefixes')
    plt.ylabel('Metrics')
    plt.title('Weighted F1 and Accuracy varying prefixes')
    plt.legend()
    # Imposta lo sfondo bianco e rimuovi la griglia
    plt.gca().set_facecolor('white')
    plt.grid(False)
    plt.savefig(join(path, f'prefix_f1_score_{epoch}.png'))
    plt.close()
    return grouped_results



def visualize_pyg_graph_interactive(data, title="Grafo"):
    """
    Converte un oggetto Data di torch_geometric in un grafo networkx e lo mostra a schermo.
    I nodi vengono etichettati con il loro indice numerico.
    """
    # Converte l'oggetto Data in un grafo networkx
    g = to_networkx(data, to_undirected=True)

    plt.figure(figsize=(12, 10))
    plt.title(f"{title}\nClasse: {data.y.item()} - Nodi: {data.num_nodes} - Archi: {data.num_edges}")
    
    # Usiamo un layout che tende a distribuire bene i nodi
    pos = nx.spring_layout(g, seed=42)
    
    # Disegniamo il grafo con etichette numeriche per i nodi
    nx.draw(g, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10)
    
    # Mostra il grafico in una nuova finestra
    plt.show()


def show_interactive_analysis(train_data, test_data):
    """
    Funzione master che permette di sfogliare TUTTI i grafi dei dataset
    in modo interattivo e poi mostra le analisi aggregate.
    """
    print("\n" + "="*50)
    print("INIZIO ANALISI INTERATTIVA DEL DATASET")
    print("="*50)
    print("Ora potrai ispezionare i grafi uno per uno.")

    # Attiva la modalità interattiva per mostrare subito i plot
    plt.ion()

    # --- 1. Sfoglia i grafi del TRAINING SET ---
    print(f"\n--- Sfoglio i {len(train_data)} grafi del TRAINING set ---")
    for i, data in enumerate(train_data):
        print(f"\n[VIS] Mostro il grafo di TRAINING {i+1}/{len(train_data)}...")
        visualize_pyg_graph_interactive(data, title=f"Grafo di Training #{i+1}")
        
        # Pausa e attesa dell'input dell'utente
        user_input = input(">>> Premi INVIO per il prossimo grafo, o digita 'q' e INVIO per saltare al set di test... ")
        if user_input.lower() == 'q':
            print("...Saltando i restanti grafi di training.")
            plt.close('all') # Chiude la finestra del grafo corrente
            break
        plt.close('all') # Chiude la finestra del grafo corrente prima di aprire la successiva

    # --- 2. Sfoglia i grafi del TEST SET ---
    print(f"\n--- Sfoglio i {len(test_data)} grafi del TEST set ---")
    for i, data in enumerate(test_data):
        print(f"\n[VIS] Mostro il grafo di TEST {i+1}/{len(test_data)}...")
        visualize_pyg_graph_interactive(data, title=f"Grafo di Test #{i+1}")
        
        # Pausa e attesa dell'input dell'utente
        user_input = input(">>> Premi INVIO per il prossimo grafo, o digita 'q' e INVIO per terminare l'ispezione... ")
        if user_input.lower() == 'q':
            print("...Saltando i restanti grafi di test.")
            plt.close('all')
            break
        plt.close('all')

    # --- 3. Mostra i grafici di analisi aggregata ---
    print("\n[ANALYSIS] Mostro i grafici di analisi aggregata (distribuzione temporale e dimensioni).")
    
    # Distribuzione temporale
    train_times = [d.start_timestamp for d in train_data]
    test_times = [d.start_timestamp for d in test_data]
    plt.figure("Distribuzione Temporale", figsize=(12, 4))
    plt.scatter(train_times, [1]*len(train_times), color='blue', label='Train', alpha=0.5)
    plt.scatter(test_times, [2]*len(test_times), color='red', label='Test', alpha=0.5)
    plt.yticks([1, 2], ['Train', 'Test'])
    plt.xlabel('Timestamp')
    plt.title('Distribuzione temporale di Train e Test')
    plt.legend()
    plt.show()

    # Distribuzione delle dimensioni
    train_node_counts = [d.num_nodes for d in train_data]
    test_node_counts = [d.num_nodes for d in test_data]
    plt.figure("Distribuzione Dimensioni Grafi", figsize=(12, 6))
    plt.hist(train_node_counts, bins=50, alpha=0.7, label='Training Set', color='blue')
    plt.hist(test_node_counts, bins=50, alpha=0.7, label='Test Set', color='red')
    plt.xlabel("Numero di Nodi per Grafo")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione delle Dimensioni dei Grafi")
    plt.legend()
    plt.show()
    
    # --- 4. Pausa finale prima del training ---
    print("\n" + "="*50)
    input(">>> Ispezione completata. CHIUDI TUTTE LE FINESTRE e premi INVIO per iniziare il TRAINING... <<<")
    print("="*50 + "\n")
    
    # Disattiva la modalità interattiva
    plt.ioff()
    plt.close('all')


if __name__ == '__main__':
    G = TraceDataset()
    dropout, patience, perc_split = args.dropout, args.patience, args.per
    criterion = torch.nn.CrossEntropyLoss()
    if args.kfold == 1:
        train, test = temporal_split(G, train_ratio=perc_split/100)
        random.shuffle(train)
        random.shuffle(test)
        #show_interactive_analysis(train, test)

    #else:
        #train_test_splits = split_target_val(G, args.kfold, shuffle=False, random_state=None)

    """"
    if args.grid_search:
        epochs = [300]
        k_values = [30]
        num_layers_values = [5, 7]
        lr_values = [0.0005, 0.00005, 0.001, 0.0001]
        batch_size_values = [64]
        num_neurons = [64]
        seeds = [42]
        n_of_comb = (len(seeds) * len(epochs) * len(k_values) * len(num_layers_values)
                     * len(lr_values) * len(batch_size_values) * len(num_neurons))
    """
    if args.grid_search:
        # Configurazione ottimizzata per RequestForPayments su PC locale
        # Basata sui best parameters noti: num_layers=7, lr=0.01
        # Ridotto a 4 combinazioni mirate per test rapidi
        
        epochs = [100]  # Ridotto a 50 per PC locale (~277s/epoca = ~3.8h per 4 combinazioni)
        # SUGGERIMENTO: Riduci anche patience a 15-20 in config.py per early stopping più aggressivo
        
        # 1. K values - testare intorno a valori medi (CRUCIALE per nodo globale)
        k_values = [40]  # 2 valori: medio e alto
        
        # 2. Num layers - fissato a 7 (best known per RfP)
        num_layers_values = [5]  # 1 valore: ottimale noto
        
        # 3. Learning rate - testare intorno a 0.01 (best known per RfP)
        lr_values = [0.001]  # 2 valori: ottimale noto e variante
        
        # 4. Num neurons - fissato a 128 (maggiore capacità per nodo globale)
        num_neurons = [128]  # 1 valore: capacità alta
        
        # Valori fissi
        batch_size_values = [128] 
        seeds = [42]
        
        n_of_comb = (len(seeds) * len(epochs) * len(k_values) * len(num_layers_values)
                    * len(lr_values) * len(batch_size_values) * len(num_neurons))
        
    else:
        batch_size_values = [args.batch_size]
        epochs = [args.num_epochs]
        k_values = [args.k]
        num_layers_values = [args.num_layers]
        lr_values = [args.learning_rate]
        num_neurons = [args.num_neurons]
        seeds = [args.seed]
        n_of_comb = 1

    actual_comb = 1
    current_timestamp = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'

    for batch_size in batch_size_values:
        for total_epochs in epochs:
            for seed in seeds:
                for k in k_values:
                    for num_layers in num_layers_values:
                        for num_neuron in num_neurons:
                            for lr in lr_values:
                                if args.kfold != 1:
                                    accuracies_train = []
                                    f1_scores_train = []
                                    accuracies_test = []
                                    f1_scores_test = []
                                for fold in range(args.kfold):
                                    if k < num_layers:
                                        if fold == 0:
                                            print(f'Skipping invalid configuration: k ({k}) < num_layers ({num_layers})')
                                            actual_comb += 1
                                        continue
                                    """if actual_comb == 4 or actual_comb == 5 or actual_comb == 6 or actual_comb == 7 or actual_comb == 8:
                                        actual_comb += 1
                                        continue"""
                                    if args.kfold != 1:
                                        try:
                                            train, test = train_test_splits[fold]
                                        except:
                                            print('out of bound')
                                    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
                                    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

                                    data_comb = f"epochs_{total_epochs}_batchsize_{batch_size}_k_{k}" \
                                                f"_numlayers_{num_layers}" \
                                                f"_numneurons_{num_neuron}_lr_{lr}_seed_{seed}"

                                    results_df = DataFrame(columns=['combination', 'best', 'seed', 'total_epochs',
                                                                    'epoch', 'batch_size', 'k', 'num_layers',
                                                                    'num_neurons', 'learning_rate', 'train_loss',
                                                                    'train_accuracy', 'train_f1', 'test_loss',
                                                                    'test_accuracy', 'test_f1'])
                                    """results_df = DataFrame(columns=['seed', 'combination', 'best', 'total_epochs', 'epoch',
                                                                    'batchsize', 'k', 'num_layers', 'num_neurons',
                                                                    'learning_rate', 'train_loss', 'train_accuracy',
                                                                    'train_f1', 'test_loss', 'test_accuracy', 'test_f1'])"""
                                    prefix_results_df = DataFrame(columns=['prefix_len', 'y_pred', 'y_true', 'epoch'])
                                    """prefix_results_df = DataFrame(columns=['prefix_len', 'active_cases', 
                                                                          'running_activities', 'y_pred', 'y_true',
                                                                          'epoch'])"""
                                    if args.kfold != 1:
                                        comb_path = join(NET_RESULTS_PATH, current_timestamp, data_comb, 'fold'+str(fold+1))
                                    else:
                                        comb_path = join(NET_RESULTS_PATH, current_timestamp, data_comb)

                                    if not exists(comb_path):
                                        makedirs(comb_path)

                                    if args.kfold == 1:
                                        print(f"\n\nStarting combination ({actual_comb}/{n_of_comb}):\n"
                                              f"-> SEED: {seed} | EPOCHS: {total_epochs} | PATIENCE: {patience}\n"
                                              f"-> NUM_NEURONS: {num_neuron} | K: {k} | NUM_LAYERS: {num_layers}\n"
                                              f"-> BATCH_SIZE: {batch_size} | LEARNING RATE: {lr}\n")
                                    else:
                                        print(f"\n\nStarting combination ({actual_comb}/{n_of_comb}), fold {fold+1}:\n"
                                              f"-> SEED: {seed} | EPOCHS: {total_epochs} | PATIENCE: {patience}\n"
                                              f"-> NUM_NEURONS: {num_neuron} | K: {k} | NUM_LAYERS: {num_layers}\n"
                                              f"-> BATCH_SIZE: {batch_size} | LEARNING RATE: {lr}\n")

                                    model = DGCNN(dataset=G, num_layers=num_layers, dropout=dropout, num_neurons=num_neuron,
                                                  k=k)
                                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                                    best_loss_test, trigger_times = 0, 0

                                    for epoch in range(total_epochs):
                                        print(f'\nStarting epoch: {epoch + 1}/{total_epochs}')
                                        start_time = time()

                                        epoch_path = join(comb_path, f'{epoch}')
                                        if not exists(epoch_path):
                                            makedirs(epoch_path)

                                        model.train(True)
                                        train_loss, correct_predictions = 0, 0
                                        predictions, reals = torch.tensor([]), torch.tensor([])
                                        for batch in train_loader:
                                            b = batch
                                            optimizer.zero_grad()
                                            b.x = b.x.to(torch.float32)  # Convert input features to float32
                                            b.y = b.y.to(torch.long)
                                            out = model(b, k)
                                            loss = criterion(out, b.y)
                                            pred = out.argmax(dim=1)
                                            predictions = torch.cat((predictions, pred), dim=0)
                                            reals = torch.cat((reals, batch.y), dim=0)
                                            loss.backward()
                                            optimizer.step()
                                            train_loss += loss.item() * b.num_graphs
                                            correct_predictions += int((pred == b.y).sum())
                                        end_time = time()

                                        train_loss = train_loss / len(train_loader.dataset)
                                        train_acc = accuracy_score(reals.cpu(), predictions.cpu())
                                        train_f1 = f1_score(reals.cpu(), predictions.cpu(), average='weighted')
                                        plot_confusion_matrix(reals.cpu(), predictions.cpu(), 'train', epoch, epoch_path)

                                        print(f'-> Training done in {end_time - start_time} s'
                                              f'\n\t-> Loss: {train_loss:.4f}'
                                              f'\n\t-> Accuracy: {(train_acc * 100):.4f}%'
                                              f'\n\t-> Weighted F1: {(train_f1 * 100):.4f}%')

                                        start_time = time()
                                        model.eval()
                                        test_loss, correct_predictions = 0, 0
                                        epoch_prefix_results = DataFrame(
                                            columns=['prefix_len', 'active_cases_len', 'y_pred', 'y_true', 'epoch'])
                                        with torch.no_grad():
                                            predictions, reals = torch.tensor([]), torch.tensor([])
                                            for batch in test_loader:
                                                batch.x = batch.x.to(torch.float32)
                                                batch.y = batch.y.to(torch.long)
                                                out = model(batch, k)
                                                loss = criterion(out, batch.y)
                                                pred = out.argmax(dim=1)
                                                predictions = torch.cat((predictions, pred), dim=0)
                                                reals = torch.cat((reals, batch.y), dim=0)
                                                test_loss += loss.item() * batch.num_graphs
                                                correct_predictions += int((pred == batch.y).sum())
                                                batch_results = DataFrame({'prefix_len': batch.prefix_len,
                                                                           'active_cases_len': batch.active_cases_len,
                                                                           'y_pred': pred,
                                                                           'y_true': batch.y, 'epoch': epoch})
                                                epoch_prefix_results = concat([epoch_prefix_results, batch_results],
                                                                              ignore_index=True)

                                        end_time = time()
                                        test_loss = test_loss / len(test_loader.dataset)
                                        test_acc = accuracy_score(reals.cpu(), predictions.cpu())
                                        test_f1 = f1_score(reals.cpu(), predictions.cpu(), average='weighted')
                                        plot_confusion_matrix(reals.cpu(), predictions.cpu(), 'test', epoch, epoch_path)

                                        print(f'-> Test done in {end_time - start_time} s'
                                              f'\n\t-> Loss: {test_loss:.4f}'
                                              f'\n\t-> Accuracy: {(test_acc * 100):.4f}%'
                                              f'\n\t-> Weighted F1: {(test_f1 * 100):.4f}%')

                                        if epoch == 0 or test_loss <= best_loss_test:
                                            best_loss_test = test_loss
                                            print(f'**** BEST TEST LOSS:{best_loss_test:.4f} ****')
                                            trigger_times = 0
                                        else:
                                            trigger_times += 1

                                        # saving infos..
                                        results_df.loc[len(results_df)] = [
                                            data_comb, str(test_loss <= best_loss_test), seed, total_epochs, epoch,
                                            batch_size, k, num_layers, num_neuron, lr, round(train_loss, 4),
                                            round(train_acc * 100, 3), round(train_f1 * 100, 3), round(test_loss, 4),
                                            round(test_acc * 100, 3), round(test_f1 * 100, 3)]
                                        results_df.to_csv(join(comb_path, 'results.csv'), header=True, index=False, sep=',')

                                        # saving prefix infos..
                                        epoch_prefix_metrics = calculate_metrics_by_prefixes(epoch_prefix_results, epoch_path)

                                        prefix_results_df = concat([prefix_results_df, epoch_prefix_results], ignore_index=True)
                                        current_epoch = prefix_results_df[prefix_results_df['epoch']==epoch]
                                        current_epoch.to_csv(join(epoch_path, 'prefix_results.csv'), header=True,
                                                                 index=False, sep=',')
                                        """zip_filename = 'prefix_results.zip'
                                        with zipfile.ZipFile(join(epoch_path, zip_filename), 'w') as zipf:
                                            zipf.write(join(epoch_path, 'prefix_results.csv'),
                                                       os.path.basename(join(epoch_path, 'prefix_results.csv')))
                                        os.remove(join(epoch_path, 'prefix_results.csv'))"""

                                        # saving epoch summary..
                                        with open(join(epoch_path, f'result_epoch_{epoch}.txt'), 'w') as file:
                                            file.write(f'\n\nMetrics of model ({epoch}/{total_epochs}):'
                                                       f'\n\t-> Train loss: {train_loss:.4f}'
                                                       f'\n\t-> Train accuracy: {(train_acc * 100):.4f}%'
                                                       f'\n\t-> Train weighted F1: {(train_f1 * 100):.4f}%'
                                                       f'\n\t-> Test loss: {test_loss:.4f}'
                                                       f'\n\t-> Test accuracy: {(test_acc * 100):.4f}%'
                                                       f'\n\t-> Test weighted F1: {(test_f1 * 100):.4f}%')
                                        if args.kfold != 1:
                                            accuracies_train.append(train_acc * 100)
                                            f1_scores_train.append(train_f1 * 100)
                                            accuracies_test.append(test_acc * 100)
                                            f1_scores_test.append(test_f1 * 100)

                                        if trigger_times >= patience:
                                            print(f'Early stopping!\nBest test loss: {best_loss_test:.4f}')
                                            break

                                    # plotting combination metrics..
                                    plot_comb_metrics(results_df, comb_path)
                                    calculate_metrics_by_prefixes(prefix_results_df, comb_path)
                                    # verificare
                                    if args.kfold != 1 and fold == args.kfold-1:
                                        with open(join(comb_path.rsplit('/',1)[0], f'fold_results.txt'), 'w') as file:
                                            file.write(f'\n\t-> Average Train accuracy: {(np.average(accuracies_train)):.4f}%'
                                                       f'\n\t-> Train weighted F1: {(np.average(f1_scores_train)):.4f}%'
                                                       f'\n\t-> Test accuracy: {(np.average(accuracies_test)):.4f}%'
                                                       f'\n\t-> Test weighted F1: {(np.average(f1_scores_test)):.4f}%')
                                        # shutil.make_archive(join(NET_RESULTS_PATH, current_timestamp, data_comb), 'zip',
                                        #                 join(NET_RESULTS_PATH, current_timestamp, data_comb))
                                        # shutil.rmtree(join(NET_RESULTS_PATH, current_timestamp, data_comb))

                                    if fold == args.kfold-1:
                                        actual_comb += 1
