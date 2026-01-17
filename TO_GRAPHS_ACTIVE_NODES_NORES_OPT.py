import torch
from torch_geometric.data import Data, InMemoryDataset
from pandas import Series, get_dummies, read_csv
import numpy as np
import networkx as nx
import pandas as pd
from os import listdir, remove
from os.path import join, exists, isfile
import torch.multiprocessing as mp
import os
import gc

from config import (load, DATASET_PROCESSED_PATH,
                    TARGET_PAR_FILE_PATH, TARGET_STD_FILE_PATH, ATTRIBUTES_FILE_PATH,
                    COMPLETE_IGS_NAME, DATASET_PATH, COMPLETE_IGS_FILE_PATH,
                    CATEGORICAL_ATTRIBUTES_FILE_PATH,
                    NUMERICAL_ATTRIBUTES_FILE_PATH, XES_NAME)

args = load()
per = args.per
att_numerici, att_categorici = [], []
processed_rows = 0

if os.path.exists('statistics.csv'):
    os.remove('statistics.csv')


class TraceDataset(InMemoryDataset):
    def __init__(self):
        super(TraceDataset, self).__init__(DATASET_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f'{COMPLETE_IGS_NAME}_par.pt']

    def process(self):
        data_list = []
        graphs, prefix_df = get_graphs()
        attr_event = dict_attr()
        target_std, _ = dict_target()
        stats = pd.DataFrame(columns=['reference_prefix_parallelisms', 'num_active_prefixes', 'num_active_nodes'])
        
        old_s_prefix = None
        old_f_prefix = None
        active_prefixes = []
        graph_stats = {}

        for index, reference_prefix_graph in enumerate(graphs):
            print(f'Processing graph {index + 1}/{len(graphs)}')

            # --- Calcolo timestamp e identificazione prefissi attivi ---
            graph_df = pd.DataFrame(
                [(n,
                pd.to_datetime(reference_prefix_graph.nodes[n]['start_timestamp']).tz_convert('UTC'),
                pd.to_datetime(reference_prefix_graph.nodes[n]['end_timestamp']).tz_convert('UTC'))
                for n in reference_prefix_graph.nodes],
                columns=['node_id', 'start_timestamp', 'end_timestamp']
            )
            s_prefix = pd.to_datetime(reference_prefix_graph.nodes[len(reference_prefix_graph.nodes) - 1]['start_timestamp']).tz_convert('UTC')
            f_prefix = pd.to_datetime(reference_prefix_graph.nodes[len(reference_prefix_graph.nodes) - 1]['end_timestamp']).tz_convert('UTC')
            timestamps = graph_df[(graph_df['start_timestamp'] < f_prefix) & (graph_df['end_timestamp'] > s_prefix)]
            if not timestamps.empty:
                s_prefix = min(timestamps['start_timestamp'])
                f_prefix = max(timestamps['end_timestamp'])

            if (old_s_prefix is None) or (s_prefix != old_s_prefix or f_prefix != old_f_prefix):
                old_s_prefix, old_f_prefix = s_prefix, f_prefix
                active_prefixes, graph_stats = get_active_prefixes(reference_prefix_graph, prefix_df, s_prefix, f_prefix, timestamps)
            
            if graph_stats:
                stats.loc[len(stats)] = graph_stats

            # --- Logica di costruzione del grafo ---
            if active_prefixes:
                #CASO A: CI SONO PREFISSI PARALLELI -> Costruiamo il Super-Grafo
                combined_graph = nx.Graph()
                node_mapping = {}
                
                #Aggiungi i NODI del prefisso di riferimento
                for node_id, attrs in reference_prefix_graph.nodes(data=True):
                    new_node_id = f"ref_{node_id}"
                    combined_graph.add_node(new_node_id, **attrs, type='reference')
                    node_mapping[new_node_id] = {'attrs': attrs, 'is_global': False, 'is_ref': True}
                
                #Aggiungi gli ARCHI del prefisso di riferimento
                for u, v in reference_prefix_graph.edges():
                    combined_graph.add_edge(f"ref_{u}", f"ref_{v}")

                #Aggiungi i prefissi attivi
                for i, active_graph in enumerate(active_prefixes):
                    # Aggiungi i nodi del prefisso attivo
                    for node_id, attrs in active_graph.nodes(data=True):
                        new_node_id = f"active_{i}_{node_id}"
                        combined_graph.add_node(new_node_id, **attrs, type='active')
                        node_mapping[new_node_id] = {'attrs': attrs, 'is_global': False, 'is_ref': False}
                    # Aggiungi gli archi del prefisso attivo
                    for u, v in active_graph.edges():
                        combined_graph.add_edge(f"active_{i}_{u}", f"active_{i}_{v}")

                # Aggiungi il nodo globale 'G' e i collegamenti
                global_node_id = 'G'
                combined_graph.add_node(global_node_id, type='global')
                node_mapping[global_node_id] = {'attrs': {}, 'is_global': True, 'is_ref': False}
                
                # Link a START del riferimento
                combined_graph.add_edge(global_node_id, 'ref_0')

                # Link all'ULTIMO NODO di ogni prefisso attivo
                for i, active_graph in enumerate(active_prefixes):
                    last_node_active_id = max(active_graph.nodes())
                    new_last_node_id = f"active_{i}_{last_node_active_id}"
                    if new_last_node_id in combined_graph:
                        combined_graph.add_edge(global_node_id, new_last_node_id)
                
                #Creazione dei Tensori dal Super-Grafo
                ordered_nodes = list(combined_graph.nodes())
                x_list = []
                num_base_features = len(list(attr_event.values())[0])
                for node_id in ordered_nodes:
                    info = node_mapping[node_id]
                    node_attrs = info['attrs']
                    node_features = []
                    if info['is_global']:
                        node_features = [0] * (num_base_features + 4)
                        node_features[-1] = 2
                    else:
                        node_features.extend(attr_event.get(node_attrs.get('concept:name'), [0]*num_base_features))
                        node_features.extend([float(node_attrs.get(attr, 0)) for attr in ['norm_time', 'trace_time', 'prev_event_time']])
                        node_features.append(1 if info['is_ref'] else 0)
                    x_list.append(node_features)

                x = torch.tensor(np.array(x_list), dtype=torch.float32)
                adj = nx.to_scipy_sparse_array(combined_graph, nodelist=ordered_nodes).tocoo()
                edge_index = torch.stack([torch.from_numpy(adj.row), torch.from_numpy(adj.col)], dim=0).long()

            else:
                #CASO B: NESSUN PREFISSO PARALLELO -> Grafo Standard
                x_list = []
                for node_id in reference_prefix_graph.nodes():
                    node = reference_prefix_graph.nodes[node_id]
                    node_features = []
                    node_features.extend(attr_event[node['concept:name']])
                    node_features.extend([float(node.get(attr, 0)) for attr in ['norm_time', 'trace_time', 'prev_event_time']])
                    node_features.append(1)
                    x_list.append(node_features)

                x = torch.tensor(np.array(x_list), dtype=torch.float32)
                adj = nx.to_scipy_sparse_array(reference_prefix_graph).tocoo()
                edge_index = torch.stack([torch.from_numpy(adj.row), torch.from_numpy(adj.col)], dim=0).long()

            y = torch.tensor([target_std[reference_prefix_graph.graph['target_std']]], dtype=torch.long)
            prefix_len = torch.tensor(len(reference_prefix_graph.nodes()), dtype=torch.long)
            active_cases_len = torch.tensor(sum(len(g.nodes) for g in active_prefixes), dtype=torch.long)
            
            # Prendiamo lo start_timestamp del primo evento (nodo 0) del prefisso di riferimento.
            start_timestamp_str = reference_prefix_graph.nodes[0]['start_timestamp']

            start_timestamp_val = pd.to_datetime(start_timestamp_str).timestamp()

            start_timestamp_tensor = torch.tensor([start_timestamp_val], dtype=torch.float64)
            
            data = Data(x=x, edge_index=edge_index, y=y, prefix_len=prefix_len, active_cases_len=active_cases_len, start_timestamp = start_timestamp_tensor)
            data_list.append(data)

            # Plot dei grafi per verifica
            #if active_prefixes:
                #plot_combined_graph(combined_graph, current_timestamp_dt=f_prefix)
            #else:
                #plot_combined_graph(reference_prefix_graph)

            
            del x, edge_index, y, prefix_len, active_cases_len, data
            gc.collect()


        stats.to_csv('statistics.csv')
        del stats
        gc.collect()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

"""
# --- Funzione di Plotting ---
def plot_combined_graph(graph):
    import matplotlib.pyplot as plt
    
    # Se il grafo è diretto (caso standard), lo convertiamo per il layout
    if nx.is_directed(graph):
        plot_graph = graph.to_undirected()
    else:
        plot_graph = graph

    pos = nx.spring_layout(plot_graph, seed=42, k=0.8)
    node_colors = []
    node_labels = {}
    
    is_combined_graph = any('type' in data for _, data in graph.nodes(data=True))

    for node, data in graph.nodes(data=True):
        node_labels[node] = data.get('concept:name', 'G' if node == 'G' else str(node))
        
        if is_combined_graph:
            type_val = data.get('type')
            if type_val == 'reference':
                node_colors.append('skyblue')
            elif type_val == 'active':
                node_colors.append('lightgreen')
            elif type_val == 'global':
                node_colors.append('salmon')
            else:
                node_colors.append('gray')
        else:
            node_colors.append('skyblue')

    plt.figure(figsize=(16, 12))
    nx.draw(graph, pos,
            with_labels=True,
            labels=node_labels,
            node_color=node_colors,
            node_size=600,
            font_size=8,
            width=0.9,
            edge_color='gray')
    
    plt.title("Visualizzazione del Grafo di Contesto")
    plt.show()
"""

def plot_combined_graph(graph, current_timestamp_dt=None):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import re 

    #Funzione per creare le sigle 
    def create_acronym(name):
        if name in ['START', 'END', 'G', 'N/A']:
            return name
        
        # Pulisce il nome da parti comuni e lo divide in "parole" basate sulle maiuscole
        clean_name = name.replace('RequestForPayment', '')
        words = re.sub(r"([A-Z])", r" \1", clean_name).split()
        
        if not words:
            return "RFP" #RequestForPayment
        
        # Crea una sigla dalle prime 3 lettere delle prime due parole
        if len(words) >= 2:
            return (words[0][:3] + words[1][:3]).upper()
        # Se c'è una sola parola, prendi fino a 6 lettere
        else:
            return words[0][:min(6, len(words[0]))].upper()

    # --- Grafi senza prefissi attivi in parallelo---
    if not any('type' in data for _, data in graph.nodes(data=True)):
        print("Grafico semplice rilevato, utilizzo il layout standard.")
        node_labels = {node: create_acronym(data.get('concept:name', str(node))) for node, data in graph.nodes(data=True)}
        pos = nx.spring_layout(graph.to_undirected(), seed=42)
        plt.figure(figsize=(12, 10))
        nx.draw(graph, pos, with_labels=True, labels=node_labels, node_color='skyblue', node_size=2000, font_size=8, font_weight='bold')
        plt.title("Visualizzazione Grafo Semplice")
        plt.show()
        return

    # --- PLOTTING con il nodo globale ---

    pos = {}
    lanes = defaultdict(list)
    node_timestamps = {}
    
    for node, data in graph.nodes(data=True):
        if node == 'G': continue
        
        if node.startswith('ref_'): lane_id = 'ref'
        else: lane_id = f"{node.split('_')[0]}_{node.split('_')[1]}"
            
        lanes[lane_id].append(node)
        node_timestamps[node] = pd.to_datetime(data['start_timestamp'])

    min_time = min(node_timestamps.values()) if node_timestamps else 0
    max_time = max(node_timestamps.values()) if node_timestamps else 1
    time_span = (max_time - min_time).total_seconds() if node_timestamps else 1
    if time_span == 0: time_span = 1

    sorted_lane_ids = sorted(lanes.keys(), key=lambda x: (x != 'ref', x))
    lane_y_coords = {lane_id: -i for i, lane_id in enumerate(sorted_lane_ids)}
    COLLISION_OFFSET = 1.2

    for lane_id in sorted_lane_ids:
        nodes, y_pos = lanes[lane_id], lane_y_coords[lane_id]
        nodes.sort(key=lambda n: int(n.split('_')[-1]) if n.startswith(('ref_', 'active_')) else 0)
        last_x_pos = -float('inf')
        for node in nodes:
            time_since_start = (node_timestamps[node] - min_time).total_seconds()
            x_pos = (time_since_start / time_span) * 10
            if x_pos - last_x_pos < COLLISION_OFFSET: x_pos = last_x_pos + COLLISION_OFFSET
            pos[node] = (x_pos, y_pos)
            last_x_pos = x_pos

    y_values = list(lane_y_coords.values())
    g_y_pos = (min(y_values) + max(y_values)) / 2 if y_values else 0
    pos['G'] = (-1.5, g_y_pos)

    node_colors = []
    node_labels = {}
    
    for node, data in graph.nodes(data=True):
        type_val = data.get('type')
        if type_val == 'global':
            node_labels[node] = 'G'
            node_colors.append('salmon')
        else:
            activity_name = data.get('concept:name', 'N/A')
            case_id_full = data.get('case:concept:name', 'N/A')
            
            acronym = create_acronym(activity_name)
            # Estrae solo la parte numerica del case ID
            case_id_num = ''.join(filter(str.isdigit, case_id_full))
            # Crea l'etichetta su due righe
            node_labels[node] = f"{acronym}\n({case_id_num})"
            
            if type_val == 'reference': node_colors.append('skyblue')
            elif type_val == 'active': node_colors.append('lightgreen')
            else: node_colors.append('gray')

    plt.figure(figsize=(24, max(10, len(lanes) * 1.5)))

    nx.draw(graph, pos,
            labels=node_labels,
            with_labels=True,
            node_color=node_colors,
            node_size=2800, 
            font_size=8,
            font_weight='bold',
            font_color='black',
            width=1.0,
            edge_color='gray')

    if current_timestamp_dt:
        time_since_start = (current_timestamp_dt - min_time).total_seconds()
        line_x_pos = (time_since_start / time_span) * 10
        plt.axvline(x=line_x_pos, color='r', linestyle='--', linewidth=2, label=f"Timestamp Attuale: {current_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        plt.legend()

    title_str = "Visualizzazione Temporale del Contesto dei Processi"
    if current_timestamp_dt:
        title_str += f"\n(Analisi al: {current_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')})"
    
    plt.title(title_str, fontsize=14)
    
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    
    plt.show()

def split_target(G, per):
    dict = {}
    tar_std, _ = dict_target()  # dizionario: chiave=attività (target), valore=codice progressivo
    for x in tar_std.keys():  # crea coppie chiave (codice progressivo) - lista vuota per ogni target
        dict[tar_std[x]] = []

    for x in G:
        dict[x.graph['target_std']].append(x)
        # pr rraggiunge alla lista (valore) nel dizionario alla chiave (codice progressivo) il Data che descrive il grafo
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

# Ottimizza get_active_prefixes
def get_active_prefixes(graph, pref_df, s_prefix, f_prefix, timestamps):
    def add_start(g):
        if nx.is_connected(g):
            return g  # Restituisce il grafo originale se non soddisfa i criteri
        else:
            connected_components = list(nx.connected_components(g))
            if {0} in connected_components:
                connected_components.remove({0})
            for component in connected_components:
                # Se la componente non è già connessa a 0, la colleghiamo
                if not any(0 in g.neighbors(node) for node in component):
                    # Aggiungiamo l'arco tra il nodo 0 e il primo nodo della componente
                    g.add_edge(0, next(iter(component)))  # Aggiungi un arco al nodo 0
            return g

    pref_df_filtered = pref_df[
        (pref_df['start_timestamp'] <= f_prefix) & (pref_df['end_timestamp'] >= s_prefix)
        ]

    active_prefixes = pref_df_filtered.loc[
        (pref_df_filtered['case:concept:name'] != graph.nodes[0]['case:concept:name'])
    ]

    active_nodes_range = active_prefixes.groupby('case:concept:name')['node1'].apply(list)
    active_prefixes = active_prefixes.loc[active_prefixes.groupby('case:concept:name')['node1'].idxmax()]

    filtered_graphs = [
        row['graph_structure'].subgraph([el - 1 for el in active_nodes_range.get(row['case:concept:name'], [])] + (
            [0] if not nx.is_connected(row['graph_structure'].subgraph(
                [el - 1 for el in active_nodes_range.get(row['case:concept:name'], [])]).copy()) else [])).copy()
        for _, row in active_prefixes.iterrows()
    ]

    """
    filtered_graphs = [
        row['graph_structure'].subgraph([el - 1 for el in active_nodes_range.get(row['case:concept:name'], [])]).copy()
        for _, row in active_prefixes.iterrows()
    ]
    """

    filtered_graphs = [g.copy() for g in filtered_graphs]

    if filtered_graphs:
        filtered_graphs = [add_start(g) for g in filtered_graphs]
    else:
        filtered_graphs = []

    graph_stats = {'reference_prefix_parallelisms': [len(nx.subgraph(graph,
                                                                     list(timestamps['node_id']) +
                                                                     [len(graph.nodes) - 1]).subgraph(c).copy())
                                                     for c in nx.connected_components(
            nx.subgraph(graph, list(timestamps['node_id']) + [len(graph.nodes) - 1]))],
                   'num_active_prefixes': len(filtered_graphs),
                   'num_active_nodes': sum([len(g.nodes) for g in filtered_graphs])}

    reference_prefix_parallelisms = [len(nx.subgraph(graph, list(timestamps['node_id']) +
                                                     [len(graph.nodes) - 1]).subgraph(c).copy())
                                     for c in nx.connected_components(
            nx.subgraph(graph, list(timestamps['node_id']) + [len(graph.nodes) - 1]))]

    if ((reference_prefix_parallelisms == [1, 1, 1, 1]) or (reference_prefix_parallelisms == [4]) or
            (reference_prefix_parallelisms == [11, 1]) or (reference_prefix_parallelisms == [1, 1, 5, 1]) or
            (reference_prefix_parallelisms == [1, 1, 4, 1])):
        print('da controllare')

    return filtered_graphs, graph_stats


# get features of active cases
def active_graphs_representation(graph, attr_event):
    node_features = np.zeros((len(graph.nodes), len(list(attr_event.values())[0]) + 4))
    for idx, node_id in enumerate(graph.nodes):
        node = graph.nodes[node_id]
        features = []
        features.extend(attr_event[node['concept:name']])
        for attr in ['norm_time', 'trace_time', 'prev_event_time']:
            if attr in node:
                features.append(float(node[attr]))
        features.append(0)
        node_features[idx] = features
    # node_features = torch.tensor(node_features, dtype=torch.float)
    adj = nx.to_scipy_sparse_array(graph).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int16)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int16)).to(torch.long)
    row = row.cpu()
    col = col.cpu()
    edge_index = torch.stack([row, col], dim=0)
    edge_index = edge_index.cpu()
    return node_features, edge_index


def read_dataset():
    graphs, attributes = [], []
    xp_indices = g_dataframe[g_dataframe['e_v'] == 'XP'].index
    for start, end in zip(xp_indices, xp_indices[1:]):
        sub_df = g_dataframe.iloc[start:end]
        for index, row in sub_df.iterrows():

            if row['e_v'] == 'XP':
                g = nx.DiGraph()

            if row['e_v'] == 'v':
                node_nr = int(float(row['node1']) - 1)
                node_attributes = {}
                to_ignore = ['e_v', 'node1', 'node2']
                for col_name in sub_df.columns[3:]:
                    if col_name in to_ignore:
                        continue
                    # Gestisco l'activity singolarmente, per assicurarmi che venga gestita sempre come una stringa
                    if col_name == 'concept:name':
                        activity = str(row[col_name])
                        node_attributes[col_name] = activity

                        # salvo tutte le Activity, mi serve per dopo quando devo creare
                        # il relativo dizionario (vedi 'dictattr')
                        if activity not in attributes:
                            attributes.append(activity)
                    # Gestisco tutti gli altri attributi in modo parametrico, precedentemente ho fatto dei controlli
                    # se gli attributi sono categorici o numerici
                    else:
                        col_value = row[col_name]
                        if col_name in att_categorici:
                            node_attributes[col_name] = str(col_value)
                        elif col_name in att_numerici:
                            node_attributes[col_name] = float(col_value)
                        else:
                            try:
                                node_attributes[col_name] = float(col_value)
                                if col_name not in att_numerici:
                                    att_numerici.append(col_name)
                            except (Exception,):
                                node_attributes[col_name] = col_value
                                if col_name not in att_categorici:
                                    att_categorici.append(col_name)

                # Aggiungi il nodo al grafo
                g.add_node(node_nr, **node_attributes)

            elif row['e_v'] == 'e':
                g.add_edge(int(float(row['node1']) - 1), int(float(row['node2']) - 1))

        if verify_graph(g):
            # salvo il grafo con gli archi invertiti
            graphs.append(g.reverse())

    with open(ATTRIBUTES_FILE_PATH, 'w') as f:
        for att in attributes:
            f.write(att + "\n")

    return graphs


def get_graphs():
    graphs = read_dataset()
    prefix_df = pd.DataFrame(columns=['node1', 'end_timestamp', 'start_timestamp',
                                      'prefix_start_timestamp', 'case:concept:name', 'graph_structure'])
    target_std, target_par, new_graphs = [], [], []
    for graph_num, graph in enumerate(graphs):
        print(f"Processing graphs: {graph_num + 1}/{len(graphs)}")
        sub_graph = nx.DiGraph(target_std='no', target_par='no')
        for node in list(graph.nodes()):
            graph_to_append = None
            sub_graph.graph['caseid'] = graph.nodes[node]['case:concept:name']
            sub_graph.graph['target_std'] = graph.nodes[node]['concept:name']
            sub_graph.graph['target_par'] = define_target(graph.copy(), sub_graph)

            if sub_graph.graph['target_std'] not in target_std:
                target_std.append(sub_graph.graph['target_std'])

            if sub_graph.graph['target_par'] not in target_par:
                target_par.append(sub_graph.graph['target_par'])

            if verify_graph(sub_graph):
                graph_to_append = sub_graph.copy().to_undirected()
                new_graphs.append(graph_to_append)

            attrs = graph.nodes[node]
            sub_graph.add_node(node, **attrs)
            # aggiungo gli archi per quel nodo al sottografo
            for n in graph.neighbors(node):
                sub_graph.add_edge(n, node)
            try:
                end_timestamp = graph_to_append.nodes[len(graph_to_append) - 1][
                    'end_timestamp']
                start_timestamp = graph_to_append.nodes[len(graph_to_append) - 1][
                    'start_timestamp']
                start_timestamp_prefix = graph_to_append.nodes(data=True)[0][
                    'start_timestamp']
                prefix_df.loc[len(prefix_df.index)] = [
                    int(len(graph_to_append)),
                    pd.Timestamp(f'{end_timestamp[:10]} {end_timestamp[10:]}'),
                    pd.Timestamp(f'{start_timestamp[:10]} {start_timestamp[10:]}'),
                    pd.Timestamp(f'{start_timestamp_prefix[:10]} {start_timestamp_prefix[10:]}'),
                    graph_to_append.nodes[0]['case:concept:name'],
                    graph_to_append
                ]
            except Exception:
                pass

    with open(TARGET_STD_FILE_PATH, "w") as f:
        for item in target_std:
            f.write(item + '\n')

    with open(TARGET_PAR_FILE_PATH, "w") as f:
        for item in target_par:
            f.write(item + '\n')

    return new_graphs, prefix_df


def verify_graph(g):
    # if nx.number_of_isolates(g) != 0:
    #    return False
    # check if some nodes missing "name_event" attribute.
    if len(nx.get_node_attributes(g, "concept:name")) != len(g.nodes()):
        # print('grafo senza concept name')
        return False
    # check if there's only a connected component
    elif nx.number_connected_components(g.to_undirected()) != 1:
        # print('grafo non connesso')
        return False
    # check number of nodes in graph
    elif len(g.nodes()) < 1:
        # print('grafo con nodi minori di 1')
        return False
    else:
        return True


# onehot encoding for the activities
def dict_attr():
    attr = []
    with open(ATTRIBUTES_FILE_PATH, "r") as f:
        for lines in f.readlines():
            lines = lines[:-1]
            attr.append(lines)  # ricrea la lista degli attributi
    s1 = Series(attr)  # crea una serie come valori le attività
    s2 = get_dummies(s1)  # crea dataframe con tante colonne quante le attività e valori solo 0 e 1
    onedictfeat = {}
    # crea dizionario: chiave=chiave dataframe, valore = dizionario (chiave=colonna dataframe, valore=0 o 1)
    s3 = s2.to_dict()
    for a, b in s3.items():
        onedictfeat[a] = list(b.values())  # nuovo dizionario (valore=lista valori con stessa chiave)
    # print("onedictfeat",onedictfeat)
    return onedictfeat


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
    i = 0
    with open(TARGET_PAR_FILE_PATH, "r") as f:
        for lines in f.readlines():
            lines = lines[:-1]
            target_par[lines] = i
            i += 1
    # print('target par', target_par)
    return target_std, target_par


def define_target(graph, subgraph):
    reverse = graph.reverse()  # inverte le direzioni degli archi del grafo direzionato (completo)
    possible_targets = []  # lista che conterrà i neighbors dei nodi del sotto-grafo
    subgraph_nodes = list(subgraph.nodes())  # lista dei nodi del sotto-grafo
    for node in subgraph_nodes:  # per ogni nodo del sotto-grafo, individua i neighbours e li inserisce in una lista
        possible_targets.extend(list(reverse.neighbors(node)))
    possible_targets = list(set(possible_targets) - set(subgraph_nodes))

    target = possible_targets.copy()
    for node in possible_targets:
        # per ogni possibile nodo target accerta che l'altro estremo degli archi entranti sia già un nodo del
        # sottografo, altrimenti lo elimina dai target
        for node_from, node_to in reverse.in_edges(node):
            if node_from not in subgraph_nodes:
                target.remove(node_to)
                break

    new_t = ''
    # sostituisce ogni nodo della lista target con la corrispettiva activity (attributo)
    for i in range(0, len(target)):
        targ_attr = graph.nodes[target[i]]['concept:name']
        new_t = new_t + str(targ_attr) + ' '
    target = new_t[:-1]

    return target


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # read dataframes and make conversions

    if exists(NUMERICAL_ATTRIBUTES_FILE_PATH):
        with open(NUMERICAL_ATTRIBUTES_FILE_PATH, 'r') as f:
            attributes = f.readlines()
            att_numerici += [attr.strip() for attr in attributes]

    if exists(CATEGORICAL_ATTRIBUTES_FILE_PATH):
        with open(CATEGORICAL_ATTRIBUTES_FILE_PATH, 'r') as f:
            attributes = f.readlines()
            att_categorici += [attr.strip() for attr in attributes]

    for item in listdir(DATASET_PROCESSED_PATH):
        if isfile(join(DATASET_PROCESSED_PATH, item)):
            remove(join(DATASET_PROCESSED_PATH, item))

    if len(listdir(DATASET_PROCESSED_PATH)) == 0:
        g_dataframe = read_csv(COMPLETE_IGS_FILE_PATH, sep=' ')
    G = TraceDataset()
