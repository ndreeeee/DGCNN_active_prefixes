from pm4py import read_pnml
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.streaming.importer.xes import importer as xes_importer
from pm4py.objects.log.util import artificial, sorting
from pm4py import get_start_activities, get_end_activities, fitness_alignments, discover_petri_net_inductive
from time import time
from os.path import join, exists
from os import remove
from config import create_directories, G_FILE_PATH, XES_FILE_PATH, XES_PATH, NET_NAME
from pm4py import write_pnml, read_xes, write_xes


def extract_process_model():
    def has_initial_activity(log):
        start_activities = get_start_activities(log)
        return len(start_activities) == 1

    def add_initial_activity(log):
        log = artificial.insert_artificial_start_end(log, parameters={
            artificial.Parameters.PARAM_ARTIFICIAL_START_ACTIVITY: 'START',
            artificial.Parameters.PARAM_ARTIFICIAL_END_ACTIVITY: 'END'
        })
        # questa cosa alla bro viene fatta perchè pm4py non
        # permette di aggiungere solo l'attività iniziale o finale
        # piuttosto che modificare la funzione originale, ci accontentiamo
        for trace in log:
            trace._list.pop(len(trace._list) - 1)
        return log

    def has_final_activity(log):
        end_activities = get_end_activities(log)
        return len(end_activities) == 1

    def add_final_activity(log):
        log = artificial.insert_artificial_start_end(log, parameters={
            artificial.Parameters.PARAM_ARTIFICIAL_START_ACTIVITY: 'START',
            artificial.Parameters.PARAM_ARTIFICIAL_END_ACTIVITY: 'END'
        })

        for trace in log:
            trace._list.pop(0)
        return log

    with open(join(XES_PATH, 'log.txt'), 'w') as f:
        f.write(f'Processing: {XES_FILE_PATH}\n')

        log = read_xes(XES_FILE_PATH)
        f.write(f'Sorting log: {XES_FILE_PATH}\n')
        log = sorting.sort_timestamp_log(log)

        if not has_initial_activity(log):
            f.write(f'Adding initial activity for: {XES_FILE_PATH}\n')
            log = add_initial_activity(log)

        if not has_final_activity(log):
            f.write(f'Adding final activity for: {XES_FILE_PATH}\n')
            log = add_final_activity(log)

        f.write(f'Saving new log: {XES_FILE_PATH}\n')
        write_xes(log, XES_FILE_PATH)

        f.write(f'Searching noise that guarantees almost 90% of fitness for log: {XES_FILE_PATH}\n')
        f.flush()
        for noise in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            f.write(f'Extracting model with noise {noise * 100}% for: {XES_FILE_PATH}\n')
            net, initial_marking, final_marking = discover_petri_net_inductive(log, noise_threshold=noise)
            fitness = fitness_alignments(log, net, initial_marking, final_marking)['log_fitness']
            f.write(f'Model with noise: {noise * 100}% -> fitness: {fitness}\n')
            f.flush()
            if fitness >= 0.9:
                f.write('Done!\nSaving model..\n\n')
                write_pnml(net, initial_marking, final_marking, join(XES_PATH, NET_NAME))
                return net, initial_marking, final_marking


def find_causal_relationships(net):
    dict_succ = find_successors(net)
    result = []
    for key, item in dict_succ.items():
        for s in item:
            result.append((key.label, s.label))
    return result


def find_successors_of_transition(transition):
    sources = {transition}
    targets, visited = set(), set()
    while sources:
        source = sources.pop()
        if not (type(source) is PetriNet.Transition and source.label is not None):
            visited.add(source)
        for arc in source.out_arcs:
            if arc.target in visited:
                continue
            if type(arc.target) is PetriNet.Transition and arc.target.label is not None:
                targets.add(arc.target)
            else:
                sources.add(arc.target)
    return targets


def find_successors(net):
    return {transition: find_successors_of_transition(transition) for transition in net.transitions if
            transition.label is not None}


# restituisce in output due liste, una contenente le tracce allineate al modello,
# l'altra con le tracce allineata rispetto al log.
def pick_aligned_trace(trace, net, initial_marking, final_marking):
    aligned_traces = alignments.apply_trace(trace, net, initial_marking, final_marking)
    temp, al, temp1, fin = [], [], [], []
    id, id1 = 0, 0

    for edge in aligned_traces['alignment']:
        id += 1
        temp.append((id, edge[1]))
    al.append(temp)
    for edge in aligned_traces['alignment']:
        id1 += 1
        temp1.append((id1, edge[0]))
    fin.append(temp1)

    return al, fin


# La funzione mapping prende in input le due liste contenente le due tracce allineate.
# Crea quindi una lista di tuple di 3 elementi dove:
# - il primo elemento della tupla corrisponde al nome dell'attività
# - il secondo elemento della tupla corrisponde alla posizione dell'attività nella traccia presente nel log
# - il terzo elemento della tupla corrisponde alla posizione dell'attività nella traccia allineata rispetto al modello;
#   alle attività soggette a insertion (e quindi non presenti nel modello) viene assegnata la posizione a partire
#   dalla lunghezza della traccia (quindi se ho una traccia di 10 elementi, la prima insertion avrà posizione 10+1,
#   la seconda 10+2 ecc)
# Il mapping permette di capire quali sono le attività inserite nel grafo,
# quindi la funzione restituisce anche una lista ins contenente le insertion.
# crea il mapping, ritorna il map e la lista ins (lista degli inserimenti)
def mapping(L1, L2):
    map = [0] * len(L1)
    id1, id2 = 0, 0
    ins = []
    for i in range(len(L1)):
        e1 = L1[i]
        e2 = L2[i]
        if e1[1] == e2[1]:
            id1 += 1
            id2 += 1
            map[i] = (e1[1], id1, id2)
        elif e1[1] == '>>':  # insertion
            id1 += 1
            map[i] = (e2[1], id1, 0)
        elif e2[1] == '>>':  # deletion
            id2 += 1
            map[i] = (e1[1], 0, id2)

    for j in range(len(L1)):
        e1 = L1[j]
        e3 = map[j]
        if e1[1] == '>>':
            id2 += 1
            map[j] = (e3[0], e3[1], id2)
            ins.append((e3[0], e3[1], id2))

    return map, ins


# Every event of the trace is saved in a list V which represents the set of the nodes of the graph.
# An event is a pair of an ID (generated incrementally) and the activity label.
# The edges instead are saved as a pair of events in a list W.
# The algorithm is based on the definition 18 of the original paper.
def extract_instance_graph(trace, cr):
    V, W = [], []
    id = 1
    for event in trace:
        # V.append((id, event.get("concept:name")))
        V.append(event)
        id += 1
    for i in range(len(V)):
        for k in range(i + 1, len(V)):
            e1 = V[i]
            e2 = V[k]
            if (e1[1], e2[1]) in cr:
                flag_e1 = True
                for s in range(i + 1, k):
                    e3 = V[s]
                    if (e1[1], e3[1]) in cr:
                        flag_e1 = False
                        break
                flag_e2 = True
                for s in range(i + 1, k):
                    e3 = V[s]
                    if (e3[1], e2[1]) in cr:
                        flag_e2 = False
                        break

                if flag_e1 or flag_e2:
                    W.append((e1, e2))
    return V, W


# Prende in input la traccia ed elimina dalla stessa i move '>>'
def compliant_trace(trace):
    t = []
    id = 0
    for event in trace:
        if event[1] == '>>':
            continue
        else:
            id += 1
            t.append((id, event[1]))

    return t


# Prende in input la lista dei nodi, la lista degli archi, il mapping e la lista delle deletion.
# restituisce in output i nuovi nodi e archi ottenuti a seguito della deletion repair.
def del_repair(V, W, map, deletion):
    Pred, Succ, Erems, Eremp = [], [], [], []
    # W1, V1, d, W2 = [], [], [], []

    to_del = (deletion[2], deletion[0])
    # print('Da cancellare = ', to_del)
    for i in range(len(W)):
        e1 = W[i]
        e2 = e1[1]
        e3 = e1[0]
        if e2 == to_del:
            Eremp.append((e3, to_del))
        if e3 == to_del:
            Erems.append((to_del, e2))

    for a in Eremp:  # crea liste Pred e Succ
        Pred.append(a[0])
    for b in Erems:
        Succ.append(b[1])

    for ep in Eremp:
        W.remove(ep)
    for es in Erems:
        W.remove(es)

    V.remove(to_del)
    for p in Pred:
        for s in Succ:
            W.append((p, s))

    return V, W


def ins_repair(V, W, map, insertion, V_n, ins_list, Vpos):
    Eremp, Pred, Succ, pos_t, W_num, V1 = [], [], [], [], [], []
    # ins_num = []
    V.insert(insertion[1] - 1, (insertion[2], insertion[0]))
    Vpos.insert(insertion[1] - 1, (insertion[2], insertion[0]))
    pos_t.append(insertion[1])

    W_num = edge_number(W)
    # V_num = node_number(V)
    # ins_num = ins_list_num(ins_list)
    # print('ins num: ',ins_num)
    # print('Vpos agg: ', Vpos)

    for p in pos_t:  # numero dell'insertion
        # print('P=',p)
        # print('Len Vpos: ', len(Vpos))
        if p < len(Vpos):
            position = Vpos[p]  # posizione in cui va inserito il nodo
            # print('Position = ', position)
            pos = position[0]
        else:
            # Inserimento a ultimo posto. La posizione di inserimento è maggiore o uguale della lunghezza
            # di Vpos (che non considera nodi da cancellare)
            position = V[-1]
            pos = position[0]
            # print('ULtimo elemento ', Vpos[-2])
            # print(pos)
        # print('pos: ', pos)
        # in Vpos il nodo da inserire viene messo in posizione p-1 perchè il vettore parte da 0,
        # quindi il precedente lo trovo come p-2
        p_pred = Vpos[p - 2]
        pos_pred = p_pred[0]
        # print('P pred: ', p_pred)
        # linee 6-12 pseudocodice
        if is_path(pos_pred, pos, W_num, V):  # se c'è un cammino tra p-1 e p
            # print(is_path(pos_pred,pos,W_num,V))
            for i in range(len(W)):
                arc = W[i]
                a0 = arc[0]
                a1 = arc[1]
                if pos == a1[0] and (a0, a1) not in Eremp:
                    # se esiste un arco nel grafo che entra in posizione p e non è ancora stato inserito in Eremp si
                    # trovano gli archi entranti (e quindi i nodi Pred) nel nodo in posizione in cui va
                    # fatto l'inserimento
                    Eremp.append((a0, a1))
                    Pred.append(a0)
            for n in Pred:
                # linee 9-10 pseudocodice, si controllano eventuali parallelismi non considerati nel ciclo precedente
                for k in range(len(W)):
                    e = W[k]
                    e0 = e[0]
                    e1 = e[1]
                    if e0 == n and (e0, e1) not in Eremp:
                        Eremp.append((e0, e1))
        else:
            # linee 14-15 pseudocodice, l'insertion avviene all'interno di un parallelismo
            for m in range(len(W)):
                edge = W[m]
                edge0 = edge[0]
                edge1 = edge[1]
                if pos_pred == edge0[0] and (edge0, edge1) not in Eremp:
                    Eremp.append((edge0, edge1))
                    Pred.append(edge0)
                elif pos_pred == edge1[0] and pos_pred == V_n[-1]:
                    # insertion all'ultimo nodo del grafo
                    Pred.append(edge1)

    # linea 17 pseudocodice
    for erem in range(len(Eremp)):
        suc = Eremp[erem]
        suc1 = suc[1]
        if suc1 not in Succ:
            Succ.append(suc1)

    # print('Pred = ', Pred)
    # print('Succ = ', Succ)
    # print('Eremp = ', Eremp)

    # linea 18 pseudocodice
    for el in Eremp:
        if el in W:
            W.remove(el)

    for i in Pred:
        if (i, (insertion[2], insertion[0])) not in W:
            W.append((i, (insertion[2], insertion[0])))

    for s in Succ:
        if ((insertion[2], insertion[0]), s) not in W:
            W.append(((insertion[2], insertion[0]), s))

    #W_num = edge_number(W)
    #V_num = node_number(V)
    # print('V: ', V)
    # print('VPOS Finale: ', Vpos)
    # print('++++++++++++++')
    return V, W


# prende in input una lista di archi [(1,A),(2,B),...] e
# restituisce gli id [1,2,...]
def edge_number(W):
    W_number = []

    for i in range(len(W)):
        arc = W[i]
        a0 = arc[0]
        a1 = arc[1]
        W_number.append((a0[0], a1[0]))

    return W_number


# Restituisce una lista contenente solo il numero di ogni nodo.
def node_number(V):
    V_number = []
    for i in range(len(V)):
        nod = V[i]
        V_number.append(nod[0])

    return V_number


# verifica se tra due nodi dati in input esiste un cammino che li collega
def is_path(a, b, W, V):
    flag = False
    if (a, b) in W:
        flag = True
        return flag
    else:
        for c in range(len(V)):
            e = V[c]
            if (a, e[0]) in W:
                flag = is_path(e[0], b, W, V)
            else:
                continue

    return flag


# aggiorna le label dei nodi in base al mapping
def update_label(W, map, V):
    W1, V1 = [], []
    for i in range(len(W)):
        arc = W[i]
        a0 = arc[0]
        a1 = arc[1]
        for j in range(len(map)):
            e = map[j]
            if a0 == (e[2], e[0]):
                for k in range(len(map)):
                    f = map[k]
                    if a1 == (f[2], f[0]):
                        W1.append(((e[1], e[0]), (f[1], f[0])))

    for i1 in range(len(V)):
        node = V[i1]
        for j1 in range(len(map)):
            e = map[j1]
            if node == (e[2], e[0]):
                V1.append((e[1], e[0]))

    return W1, V1


def save_g_final(V, W, path, sort_labels):
    # quando salviamo l'ig, leviamo gli spazi sui nomi delle attività per via
    # della futura modalità di lettura del file, che sarà un csv separato da spazi
    with open(path, 'a') as f:
        f.write("XP\n")
        for n in V:
            f.write(f"v {n[0]} {n[1].replace(' ', '')}\n")
        if sort_labels:
            W.sort()
        for e in W:
            f.write(f"e {e[0][0]} {e[1][0]} {e[0][1].replace(' ', '')}__{e[1][1].replace(' ', '')}\n")
        f.write("\n")
        f.close()


def big(sort_labels=False, initial_marking=None, final_marking=None, net=None):
    if exists(G_FILE_PATH):
        remove(G_FILE_PATH)

    init_time = time()
    print("BIG algorithm started...")
    streaming_ev_object = xes_importer.apply(XES_FILE_PATH, variant=xes_importer.Variants.XES_TRACE_STREAM)

    print('Finding causal relationships..')
    cr = find_causal_relationships(net)
    n = 0
    # Aligned, L, Align, L1, A, A1, map, compliant, ins = ([], [], [], [], [], [],
    #                                                     [], [],[])
    print(f'Processing traces...')
    for index, trace in enumerate(streaming_ev_object):
        trace_start_time = time()
        n += 1
        # pos trace
        Aligned, A = pick_aligned_trace(trace, net, initial_marking, final_marking)
        Align = Aligned[0]
        A1 = A[0]
        # print('Aligned to model')
        # print(Align)
        # print('with invisible moves')
        # print(A1)
        map, ins = mapping(Align, A1)

        # compliant mi serve per generare l'ig in base al modello (rimuove dalla traccia allineata i move)
        compliant = compliant_trace(Align)
        # effettiva = compliant_trace(A1)
        # print(compliant)
        # print('Effettiva: ', effettiva)
        # print("map: ", map)
        # print("ins: ", ins)
        d = []
        # num = trace.attributes.get('concept:name')
        # id = trace.attributes.get('variant-index')

        # estrazione dell' IG su cui poi devo fare riparazione
        V, W = extract_instance_graph(compliant, cr)
        # print('V')
        # print(V)
        # print('W')
        # print(W)

        # ottengo liste di nodi e archi contenenti solo i valori numerici dei nodi
        V_n = node_number(V)
        # W_n = edge_number(W)

        for element in map:  # crea lista dei nodi da cancellare
            if element[1] == 0:
                d.append(element)

        # Vpos lista dei nodi utilizzata per repair. inizialmente si rimuovono da essa i nodi relativi a deletion.
        # In seguito viene passata in input alla funzione di insertion repair e ogni attività viene inserita
        # all'interno di Vpos
        # nella posizione di inserimento. così facendo vpos sarà sempre aggiornata a ogni inserimento.
        # Al termine delle insertion avrò la mia lista dei nodi aggiornata.
        Vpos = []
        for node in V:
            Vpos.append(node)

        for el in map:
            if el[1] == 0:
                Vpos.remove((el[2], el[0]))

        # print('Vpos = ', Vpos)
        # print('INSERTION REPAIR')
        for insertion in ins:
            V, W = ins_repair(V, W, map, insertion, V_n, ins, Vpos)

        # print('W repaired: ', W)
        # print('DELETION REPAIR')
        # print(d)
        for deletion in d:
            V, W = del_repair(V, W, map, deletion)

        # aggiorna le label dei nodi in base a quanto contenuto nel mapping
        W_new, V_new = update_label(W, map, V)

        # riordina le liste di nodi e archi in base agli id
        V_new.sort()
        W_new.sort()
        # print('V finale: ',V_new)
        # print('W finale: ',W_new)
        # dot = viewInstanceGraph(V,W)
        # dot.save()
        save_g_final(V_new, W_new, G_FILE_PATH, sort_labels)
        print(f'\tTrace {index} processed in  {time() - trace_start_time} seconds..')

    print(f'BIG algorithm completed in {time() - init_time} seconds..')


def process():
    # create folders at first use
    create_directories(False)
    ###
    if not exists(join(XES_PATH, NET_NAME)):
        net, initial_marking, final_marking = extract_process_model()
    else:
        net, initial_marking, final_marking = read_pnml(join(XES_PATH, NET_NAME))
    big(net=net, initial_marking=initial_marking, final_marking=final_marking)


if __name__ == '__main__':
    process()
