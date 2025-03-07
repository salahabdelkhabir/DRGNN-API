# %%
import json
import pandas as pd
import pickle

# #%%
# knowledge_graph = pd.read_csv('./knowledge_graph_v3.csv')
# with open('names2idx.pkl', 'rb') as f:
#     names2idx = pickle.load(f)

# #%%
# node_names_dict = {}
# for idx, row in knowledge_graph.iterrows():
#     node_name = row['x_name']
#     node_type = row['x_type']
#     node_id = row['x_id']


#     if node_type not in node_names_dict:
#         node_names_dict[node_type] = {}
#     # if node_name != "missing":
#     try:
#         node_idx = names2idx[node_type][str(node_id)]
#         node_names_dict[node_type][str(node_idx)] = node_name

#     except Exception:
#         print(node_type, node_id, 'not exist')
#         pass

#     node_name = row['y_name']
#     node_type = row['y_type']
#     node_id = row['y_id']


#     if node_type not in node_names_dict:
#         node_names_dict[node_type] = {}
#     # if node_name != "missing":
#     try:
#         node_idx = names2idx[node_type][str(node_id)]
#         node_names_dict[node_type][str(node_idx)] = node_name
#     except Exception:

#         print(node_type, node_id, 'not exist')
#         pass

# %%
knowledge_graph = pd.read_csv('./knowledge_graph_v3.csv')
knowledge_graph_proce = pd.read_csv('./knowledge_graph_v3_processed.csv')


# %%
id_idx_dict = {}
id_name_dict = {}
idx_name_dict = {}

for idx, row in knowledge_graph_proce.iterrows():
    node_type = row['x_type']
    node_id = row['x_id']
    node_idx = row['x_idx']

    if node_type not in id_idx_dict:
        id_idx_dict[node_type] = {}

    id_idx_dict[node_type][str(node_id)] = str(node_idx)

    node_type = row['y_type']
    node_id = row['y_id']
    node_idx = row['y_idx']

    if node_type not in id_idx_dict:
        id_idx_dict[node_type] = {}

    id_idx_dict[node_type][str(node_id)] = str(node_idx)

for idx, row in knowledge_graph.iterrows():
    node_name = row['x_name']
    node_type = row['x_type']
    node_id = row['x_id']

    try:
        node_idx = id_idx_dict[node_type][str(node_id)]
    except Exception:
        if '.0' in str(node_id):
            node_id = str(node_id).replace('.0', '')
        else:
            node_id = str(node_id) + '.0'
        try:
            node_idx = id_idx_dict[node_type][str(node_id)]
        except Exception:
            print(node_id, node_type)

    if node_type not in idx_name_dict:
        idx_name_dict[node_type] = {}

    idx_name_dict[node_type][str(node_idx)] = node_name

    node_name = row['y_name']
    node_type = row['y_type']
    node_id = row['y_id']

    try:
        node_idx = id_idx_dict[node_type][str(node_id)]
    except Exception:
        if '.0' in str(node_id):
            node_id = str(node_id).replace('.0', '')
        else:
            node_id = str(node_id) + '.0'
        try:
            node_idx = id_idx_dict[node_type][str(node_id)]
        except Exception:
            print(node_id, node_type)

    if node_type not in idx_name_dict:
        idx_name_dict[node_type] = {}

    idx_name_dict[node_type][str(node_idx)] = node_name


# %%
f = open('node_name_dict.json', 'w')
json.dump(idx_name_dict, f)
f.close()

# %%
#########################
# # cover node id to node names
##########################
graph = pd.read_csv('./gate_score_all_random_sigmoid.csv')

id_name_dict = {}


def convert2str(x):
    try:
        y = float(x)
        if '_' in x:
            # if a merge node xx_xx_xx
            y = float(x.split('_')[0])
        return str(y)
    except:
        return str(x)


for idx, row in graph.iterrows():
    node_name = row['x_name'].split('\\')[0]
    node_type = row['x_type']
    node_id = row['x_id']
    # node_id = convert2str(node_id)

    if node_type not in id_name_dict:
        id_name_dict[node_type] = {}

    id_name_dict[node_type][node_id] = node_name

    node_name = row['y_name'].split('\\')[0]
    node_type = row['y_type']
    node_id = row['y_id']
    # node_id = convert2str(node_id)

    if node_type not in id_name_dict:
        id_name_dict[node_type] = {}

    id_name_dict[node_type][node_id] = node_name

f = open('../data/node_name_dict.json', 'w')
json.dump(id_name_dict, f)
f.close()

# %%

#############################
# get subset attention fro the sample results
##############################
predictions = pd.read_pickle('result.pkl')
diseases = [key for key in predictions["rev_indication"]]

# %%
# reduce attention csv size
attentions = pd.read_csv('attention_all.csv', dtype={
                         'x_id': 'string', 'y_id': 'string'})

attentions = attentions[~attentions['relation'].str.contains(
    'contraindication')]

attentions = attentions[~attentions['relation'].str.contains(
    'anatomy_protein_absent')]

attentions = attentions[abs(attentions['layer1_att']) +
                        abs(attentions['layer2_att']) > 0.1]

attentions.to_csv('attention_prune.csv')

# %% conver drug id to index
table = pd.read_csv('./attention_all.csv')
drug_rows = table[table['x_type'] == 'drug']
max_drug_index = max(drug_rows['x_idx'].unique())

# %%
drug_idx2id = [idx for idx in range(max_drug_index + 1)]

for i, row in drug_rows.iterrows():
    index = row['x_idx']
    id = row['x_id']
    drug_idx2id[index] = id

with open('drug_idx2id.json', 'w') as f:
    json.dump(drug_idx2id, f)
# %%
