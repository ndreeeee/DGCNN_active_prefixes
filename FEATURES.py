import gc
import pm4py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval
from collections import Counter
from config import (G_FILE_PATH, XES_FILE_PATH, PARTIAL_IGS_FILE_PATH,
                    NUMERICAL_ATTRIBUTES_FILE_PATH, CATEGORICAL_ATTRIBUTES_FILE_PATH,
                    XES_NAME, COMPLETE_IGS_FILE_PATH)

max_value_resource = 0
trace_params = ['concept:name']
event_params = ['concept:name', 'time:timestamp']


def v_case(wfile, row, index, log):
    concept_name = row[2]
    # note: count = event_index + 1, to avoid same information by same "concept:name" value
    count = 1
    trace = log[index]
    for event_index, _ in enumerate(trace):
        current_event = trace[event_index]
        current_event_name = current_event['concept:name'].replace(' ', '')
        # when "concept:name" == tmp && the node count is right, we write in destination file the line + additional info
        # NOTE: we add 2 space for the execution of "temporal_calculated_features.py" file
        if concept_name == current_event_name and row[1] == str(count):

            # estraggo i valori degli attributi relativi alla traccia
            param_trace_values = ''
            for param in sorted(trace_params):
                parsed_value = trace.attributes[param].replace(' ', '')
                param_trace_values += f'{parsed_value} '
            param_trace_values = param_trace_values.strip()
            param_trace_values = param_trace_values.replace(' ', '_')

            # estraggo i valori degli attributi relativi all' evento
            param_event_values = ""
            for param in sorted(event_params):
                try:
                    parsed_value = str(current_event[param]).replace(" ", "")
                    param_event_values += f'{parsed_value} '
                    # se non presente viene impostato il valore -1
                except KeyError:
                    param_event_values += "-1 "
            param_event_values = param_event_values.strip()
            a = f'{row[0]} {int(row[1])}  {param_event_values} {param_trace_values}\n'
            wfile.writelines(a)
        count += 1


def add_info(log):
    # When we find 'XP' we increase the trace index by 1.
    # Each trace in .g start with 'XP' --> for the firs time we need to set trace_index= -1
    trace_index = -1
    to_write = open(PARTIAL_IGS_FILE_PATH, 'w')  # file in which we append trace + additionals information
    with open(G_FILE_PATH, "r") as to_read:  # file .g to which informations is to be added
        # scan line by line the .g file
        for line in to_read:
            # split line into a list where each word is a list item
            items_row = line.split()
            # control if line is empty; if it's true, the line isn't empty
            if line.strip():
                # if the list contain 'XP' we are in a new trace, --> we increase trace_index by 1
                if items_row[0] == 'XP':
                    to_write.writelines('XP\n')
                    trace_index = trace_index + 1
                # if the first element of list is 'v' --> the line refers to a node (event) -->
                # we have to add additional informations
                if items_row[0] == 'v':
                    v_case(to_write, items_row, trace_index, log)
                elif items_row[0] == 'e':
                    # if the first element of list is 'e' --> the line refers to an edge --> no additional informations.
                    # We only write the line in the destination file
                    # vertex1 = items_row[1].split('.')[0]
                    # vertex2 = items_row[2].split('.')[0]
                    row = f'{items_row[0]} {int(float(items_row[1]))} {int(float(items_row[2]))} {"".join(items_row[3:])}\n'
                    to_write.writelines(row)
            else:
                # empty line case
                to_write.writelines(' \n')
        to_write.close()


def get_json_string(resources):
    json_string = '{' + ','.join(f"'{resource}':'{occ}'" if isinstance(occ, str)
                                 else f"'{resource}':{occ}" for resource, occ in resources.items()) + '}'
    return json_string


def normalize_resources(resources_dict):
    parsed_dict = literal_eval(resources_dict)
    normalized_resources = {}
    for resource, value in parsed_dict.items():
        normalized_resources[resource] = value / max_value_resource if max_value_resource != 0 else 0
    return get_json_string(normalized_resources)


def process():
    log = pm4py.read_xes(XES_FILE_PATH)
    print('Adding infos to igs..')
    add_info(log)

    trace_params = ['case:concept:name']
    name_columns = ["e_v", "node1", "node2"] + sorted(event_params) + sorted(trace_params)

    df = pd.read_csv(PARTIAL_IGS_FILE_PATH, sep=" ", names=name_columns)
    # aggiungiamo il case:id agli archi
    df.loc[len(df), 'e_v'] = 'XP'
    xp_indexes = df.loc[df['e_v'] == 'XP'].index
    for start, end in zip(xp_indexes, xp_indexes[1:]):
        print(f'Adding case id to edges in graph: {start}-{end}')
        case_id = df.loc[start + 1, 'case:concept:name']
        df.loc[start + 1:end - 2, 'case:concept:name'] = case_id

    # cancelliamo gli archi duplicati
    df.loc[df['e_v'] == 'e'] = df.loc[df['e_v'] == 'e'].drop_duplicates()

    df['time:timestamp'] = df['time:timestamp'].apply(lambda x: str(x)[:18])
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%Y-%m-%d%H:%M:%S', utc=True)

    # prende i timestamp di inizio e di fine di ogni grafo
    case_timestamps = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max']).reset_index()
    case_timestamps.rename({'min': 'start_timestamp', 'max': 'end_timestamp'}, inplace=True)
    df.rename(columns={'time:timestamp': 'end_timestamp'}, inplace=True)

    print('Start with g_dataframe...\n')
    g_dataframe = pd.DataFrame(columns=list(df.columns))
    g_dataframe.loc[0] = np.array(np.nan * len(g_dataframe.columns))
    g_dataframe = pd.concat((g_dataframe, df), ignore_index=True)

    df_shift = pd.DataFrame(columns=list(df.columns))
    df_shift.loc[0] = np.array(np.nan * len(df_shift.columns))
    df_shift.loc[1] = np.array(np.nan * len(df_shift.columns))
    df_shift = pd.concat((df_shift, df), ignore_index=True)
    df_shift['e_v'] = df_shift['e_v'].fillna('')
    g_dataframe['start_timestamp'] = df_shift['end_timestamp'].copy()
    g_dataframe['start_timestamp'] = g_dataframe.apply(
        lambda x: x['end_timestamp'] if x['node1'] == 1 else x['start_timestamp'], axis=1)

    # iteriamo sugli archi delle attività parallele
    parallel_activities = (g_dataframe[g_dataframe['e_v'] == 'e'].groupby(['node1', 'case:concept:name'])
                           .filter(lambda x: len(x) > 1))
    for index, start_node, parallel_node, case_id in zip(
            parallel_activities.index, parallel_activities['node1'], parallel_activities['node2'],
            parallel_activities['case:concept:name']):
        print(f"Processing timestamp: {index}/{len(g_dataframe)}")

        current_case_id_graph = g_dataframe[g_dataframe['case:concept:name'] == case_id]

        parallel_node_index = current_case_id_graph[
            (current_case_id_graph['node1'] == parallel_node) &
            (current_case_id_graph['e_v'] == 'v')].index

        # si prende il timestamp finale del nodo prima dell'attività parallela
        finish_time_start_node = current_case_id_graph[
            (current_case_id_graph['e_v'] == 'v') &
            (current_case_id_graph['node1'] == start_node)]['end_timestamp'].iloc[0]

        # si aggiorna il tempo di inizio dell'attività parallela con quello finale della precedente
        g_dataframe.loc[parallel_node_index, 'start_timestamp'] = finish_time_start_node

        # si aggiorna il tempo di fine dell'attività parallela con il tempo di inizio minimo tra
        # i nodi successivi a quello parallelo
        name_nodes_next_to_parallel = current_case_id_graph[
            (current_case_id_graph['node1'] == parallel_node) &
            (current_case_id_graph['e_v'] == 'e')]['node2'].tolist()

        min_start_time_nodes_next_to_parallel = current_case_id_graph[
            (current_case_id_graph['node1'].isin(name_nodes_next_to_parallel)) &
            (current_case_id_graph['e_v'] == 'v')]['start_timestamp'].min()

        g_dataframe.loc[parallel_node_index, 'end_timestamp'] = min_start_time_nodes_next_to_parallel

    del [[df, df_shift]]
    gc.collect()

    g_dataframe['end_timestamp'].fillna(pd.NaT, inplace=True)
    g_dataframe['start_timestamp'].fillna(pd.NaT, inplace=True)
    g_dataframe['start_timestamp'].replace(0, pd.NaT, inplace=True)
    g_dataframe['end_timestamp'] = pd.to_datetime(g_dataframe['end_timestamp'], format='%Y-%m-%d %H:%M:%S',
                                                  errors='coerce')
    g_dataframe['start_timestamp'] = pd.to_datetime(g_dataframe['start_timestamp'], format='%Y-%m-%d %H:%M:%S',
                                                    errors='coerce')

    # features temporali

    print('Start with feature engineering...\n')
    print('norm_time...\n')
    g_dataframe['norm_time'] = ((g_dataframe['end_timestamp'].dt.dayofweek * 24 * 60) + (
            (g_dataframe['end_timestamp'].dt.hour * 60) + g_dataframe['end_timestamp'].dt.minute)) / 10080

    print('trace_time...\n')
    g_dataframe['trace_time'] = g_dataframe.groupby('case:concept:name', sort=False)['end_timestamp'].transform('min')
    g_dataframe['trace_time'] = g_dataframe['end_timestamp'] - g_dataframe['trace_time']
    g_dataframe['trace_time'] = (g_dataframe['trace_time'].dt.days * 24 * 60) + (
            g_dataframe['trace_time'].dt.seconds // 60)

    print('prev_event_time...\n')
    g_dataframe['prev_event_time'] = g_dataframe['end_timestamp'] - g_dataframe['start_timestamp']
    g_dataframe['prev_event_time'] = (g_dataframe['prev_event_time'].dt.days * 24 * 60) + (
            g_dataframe['prev_event_time'].dt.seconds // 60)

    print('Start with normalizing temporal features...\n')
    for i in ['norm_time', 'trace_time', 'prev_event_time']:
        g_dataframe[i] = g_dataframe[i].div(g_dataframe[i].max()).round(15)

    # features sulle risorse

    print('adding org:resource feature...\n')
    targetframe = pm4py.convert_to_dataframe(log)
    g_dataframe['org:resource'] = 'no_resource'
    selected_columns = ['org:resource']
    if 'Helpdesk' in XES_NAME:
        selected_columns = []

    att_categorici, att_numerici = [], []
    idxss = list(np.where(~g_dataframe['case:concept:name'].isnull()))[0]

    for column in selected_columns:
        for idx, element in zip(idxss, targetframe[column]):
            g_dataframe.loc[idx, str(column)] = element

        is_string = False
        column_values = g_dataframe[column].unique()
        for value in column_values:
            try:
                value.astype(float)
            except (Exception,):
                is_string = True
                continue

        if is_string:
            att_categorici.append(column)
        else:
            att_numerici.append(column)
            # Applico la normalizzazione sugli attributi categorici
            scaler = MinMaxScaler()
            # Trasforma i dati
            arr_normalized = scaler.fit_transform(arr.values.reshape(-1, 1))
            # Crea una Pandas Series dai dati normalizzati
            arr = pd.Series(arr_normalized.flatten())

    gc.collect()

    with open(NUMERICAL_ATTRIBUTES_FILE_PATH, 'w') as f:
        for attr in att_numerici:
            f.write(f'{attr}\n')

    with open(CATEGORICAL_ATTRIBUTES_FILE_PATH, 'w') as f:
        for attr in att_categorici:
            f.write(f'{attr}\n')

    g_dataframe['start_timestamp'] = pd.to_datetime(g_dataframe['start_timestamp'], format='%Y-%m-%d%H:%M:%S', utc=True)

    print('state features...\n')
    # rimpiazzo i nan e aggiungo le nuove colonne
    g_dataframe.loc[g_dataframe.loc[
        (g_dataframe['e_v'] == 'v') & (g_dataframe['org:resource'] == '')].index, 'org:resource'] = 'no_resource'
    resources = g_dataframe.loc[g_dataframe['e_v'] == 'v']['org:resource'].unique()
    g_dataframe[['nr_active_cases', 'nr_running_activities', 'workload_resources']] = np.nan, np.nan, np.nan
    global max_value_resource
    if 'Helpdesk' not in XES_NAME:
        resources_counter = Counter({resource: 0 for resource in resources})

    for index, type, event, time_start, case_id \
            in zip(g_dataframe.index, g_dataframe['e_v'],
                   g_dataframe['concept:name'], g_dataframe['start_timestamp'],
                   g_dataframe['case:concept:name']):
        if type != 'v':
            continue

        print(f"Processing event: {event} of case id: {case_id}"f"({index}/{len(g_dataframe)})")

        # conta i prefissi attivi

        print("Finding active cases..")
        active_cases = case_timestamps.loc[(
                (case_timestamps['min'] <= time_start) &
                (case_timestamps['max'] >= time_start)), 'case:concept:name'].tolist()
        nr_active_cases = len(active_cases)

        print(f"Finding running activities for active cases..")
        resources_in_case = Counter({r: 0 for r in resources})
        parallel_activities = g_dataframe.loc[(
                (g_dataframe['start_timestamp'] <= time_start) &
                (g_dataframe['end_timestamp'] >= time_start)), ['concept:name', 'org:resource']]

        nr_running_activities = len(parallel_activities['concept:name'])
        if 'Helpdesk' not in XES_NAME:
            print(f"Finding resources for running activities ..")
            resources_in_case.update(parallel_activities['org:resource'].tolist())
            resources_counter.update(parallel_activities['org:resource'].tolist())
            workload_resources = get_json_string(resources_in_case)
        if 'Helpdesk' not in XES_NAME:
            g_dataframe.loc[index, ['nr_active_cases', 'nr_running_activities',
                                    'workload_resources']] = nr_active_cases, nr_running_activities, workload_resources
        else:
            g_dataframe.loc[
                index, ['nr_active_cases', 'nr_running_activities']] = nr_active_cases, nr_running_activities

    print(f"Normalizing new features..")
    graph_nodes = g_dataframe.loc[g_dataframe['e_v'] == 'v']

    # normalizzo il numero di case concorrenti rispetto al massimo valore trovato nella colonna
    graph_nodes['nr_active_cases'] = graph_nodes['nr_active_cases'].div(graph_nodes['nr_active_cases'].max()).round(15)

    # normalizzo il numero di attività in esecuzione rispetto al massimo valore trovato nella colonna
    graph_nodes['nr_running_activities'] = graph_nodes['nr_running_activities'].div(
        graph_nodes['nr_running_activities'].max()).round(15)

    if 'Helpdesk' not in XES_NAME:
        max_value_resource = max(resources_counter.values())
        graph_nodes['workload_resources'] = graph_nodes['workload_resources'].apply(
            lambda item: normalize_resources(item))

    # trova tutte le possibili attività finali dei prefissi

    # creazione dei dataset per prefissi attivi spostato in TO_GRAPHS_ACTIVE_NODES_NORES.py
    """
    final_df = g_dataframe[g_dataframe['e_v'] == 'v'][['node1', 'concept:name', 'end_timestamp', 'case:concept:name',
                                                       'start_timestamp']]
    final_df.to_csv('active_cases/final_activities.csv', index=False)

    # trova tutti i possibili prefissi attivi assegnando alle attività il tempo di inizio del prefisso
    active_cases_df = final_df
    start_times = active_cases_df[active_cases_df['concept:name'] == 'START'][['case:concept:name', 'start_timestamp']]
    active_cases_df = active_cases_df.merge(start_times, on='case:concept:name', how='left')
    active_cases_df.drop(columns=['concept:name'], inplace=True)
    active_cases_df = active_cases_df.rename(columns={"start_timestamp_x": "start_timestamp",
                                                      "start_timestamp_y": "start_timestamp_prefix"})
    active_cases_df.to_csv('active_cases/active_cases.csv', index=False)
    """

    # il dataset prende in ingresso tutti i prefissi attivi, le statistiche non servono in questo caso
    try:
        g_dataframe.drop(columns=['nr_active_cases'], inplace=True)
    except Exception:
        print('no column named nr_active_cases')

    try:
        g_dataframe.drop(columns=['nr_active_activities'], inplace=True)
    except Exception:
        print('no column named nr_active_activities')

    try:
        g_dataframe.drop(columns=['nr_running_activities'], inplace=True)
    except Exception:
        print('no column named nr_running_activities')

    print(f"Saving final g..")
    g_dataframe.loc[g_dataframe['e_v'] == 'v'] = graph_nodes
    if 'Helpdesk' in XES_NAME:
        g_dataframe.drop(columns=['workload_resources', 'org:resource'], inplace=True)
    else:
        # il dataset prende in ingresso tutti i prefissi attivi, le statistiche non servono in questo caso
        g_dataframe.drop(columns=['workload_resources'], inplace=True)




    g_dataframe.to_csv(COMPLETE_IGS_FILE_PATH, index=False, header=True, sep=' ')


if __name__ == '__main__':
    process()
