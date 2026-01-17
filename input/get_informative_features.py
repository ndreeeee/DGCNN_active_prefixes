import pandas as pd
import os

# features to keep
features_to_keep = ['e_v', 'node1', 'node2', 'concept:name', 'start_timestamp', 'end_timestamp',
                    'case:concept:name', 'norm_time', 'trace_time', 'prev_event_time']
path_to_g = 'g_all_features'

for file in os.listdir(path_to_g):
    all_features = pd.read_csv(path_to_g+'/'+file, sep=' ')
    if not file.__contains__('Helpdesk'):
        all_features = all_features[features_to_keep]
        all_features.to_csv('../output/dataset/g/'+file, sep=' ', index=False)
    else:
        all_features = all_features[features_to_keep[:10]]
        all_features.to_csv('../output/dataset/g/' + file, sep=' ', index=False)
