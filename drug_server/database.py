# %%
import logging

from flask.globals import session
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

import pandas as pd
import os
import json

from config import SERVER_ROOT
# %%
from flask import current_app, g


def get_db():
    if 'db' not in g:
        db = Neo4jApp(server=current_app.config['GNN'], database='neo4j')
        db.create_session()
        g.db = db
    return g.db


# %%
class Neo4jApp:
    k1 = 5  # upper limit of children for root node
    k2 = 5  # upper limit of children for hop-1 nodes
    # path_thr = 45  # upper limit of path numbers
    path_thr = 5  # upper limit for each metapath
    top_n = 200  # write the predicted top n drugs to the graph database
    # # Removed from graph base to reduce computation time
    not_cool_node_pre = ['CYP']
    not_cool_rel = ['rev_contraindication', 'contraindication', 'drug_drug', 'rev_off-label use', 'off-label use', 'anatomy_protein_absent', 'rev_anatomy_protein_absent', 'disease_phenotype_negative', 'rev_disease_phenotype_negative']


    def __init__(self, server, uri=None, password='reader_password', user='reader', datapath='./colab_delivery/', database='drug'):
        self.node_types = [
        "anatomy",
        "biological_process",
        "cellular_component",
        "disease",
        "drug",
        "effect/phenotype",
        "exposure",
        "gene/protein",
        "molecular_function",
        "pathway"
        ]
        self.batch_size = 5000

        # Default connection settings for different servers
        server_configs = {
            'txgnn_v2': {
                'uri': 'bolt://localhost:7687',  # Replace with actual Neo4j bolt URI
            }
            # Add more server configurations as needed
        }

        # If no URI is provided, try to get it from server configurations
        if uri is None:
            if server not in server_configs:
                raise ValueError(f"No default configuration for server: {server}")
            uri = server_configs[server]['uri']

        self.data_path = datapath
        self.database = database

        try:
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password), encrypted=False, max_connection_lifetime=400)
        except Exception as e:
            print(f"Failed to create the driver for {server}: {e}")
            raise

        self.current_disease = None
        self.drugs = None
        self.session = None
        # self.data_path = 'https://drug-gnn-models.s3.us-east-2.amazonaws.com/collaboration_delivery/'

        
        

    def create_session(self):
        self.session = self.driver.session(database=self.database)

    def close_session(self):
        self.session.close()

    def close_driver(self):
        if self.driver is not None:
            self.driver.close()

    def clean_database(self):
        if not self.session:
            self.create_session()

        def delete_all(tx, node):
            tx.run('MATCH (n: `{}`) DETACH DELETE n'.format(node))

        for node_type in self.node_types:
            self.session.write_transaction(delete_all, node_type)
        print('delete all nodes')

    def remove_not_cool_nodes_edges(self):
        if not self.session:
            self.create_session()

        for pre in Neo4jApp.not_cool_node_pre:
            self.session.write_transaction(
                    lambda tx: tx.run('MATCH (n: `gene/protein`) WHERE n.name STARTS WITH "{}" DETACH DELETE n'.format(pre)))
        for rel in Neo4jApp.not_cool_rel:
            self.session.write_transaction(
                lambda tx: tx.run('MATCH ()-[r:{}]->() DELETE r'.format(rel)))

    def create_index(self):

        if not self.session:
            self.create_session()

        def create_singel_index(tx, node_label):
            tx.run(
                'CREATE INDEX IF NOT EXISTS FOR (n: `{}` ) ON (n.id)'.format(node_label))

        for node_type in self.node_types:
            self.session.write_transaction(
                create_singel_index, node_type)

    def init_database(self):
        self.clean_database()
        self.create_index()
        print('build attention graph...')
        self.build_attention('graphmask_output_indication.csv')
        print('add predictions...')
        self.add_prediction()
        print('database initialization finished')

    def build_attention(self, filename):

        if not self.session:
            self.create_session()

        attention_path = os.path.join(self.data_path, filename)
        print('read attention file')
        if filename.endswith('.csv'):
            attentions = pd.read_csv(attention_path, dtype={
                'x_id': 'string', 'y_id': 'string'})
        elif filename.endswith('pkl'):
            attentions = pd.read_pickle(attention_path)


        lines = []
        x_type = ''
        y_type = ''
        relation = ''

        def commit_batch_attention(tx, x_type, y_type, relation, lines):
            query = (
                'UNWIND $lines as line '
                'MERGE (node1: `{x_type}` {{ id: line.x_id }}) '
                'ON CREATE SET node1.name = line.x_name '
                'MERGE (node2: `{y_type}` {{ id: line.y_id }}) '
                'ON CREATE SET node2.name = line.y_name  '
                'MERGE (node1)-[e: `{relation}` ]->(node2) '
                'ON CREATE SET e.layer1_att = line.layer1_att, e.layer2_att= line.layer2_att '
            ).format(x_type=x_type,  y_type=y_type, relation=relation)
            tx.run(query, lines=lines)

        def delete_empty_edge(tx):
            query = (
                'MATCH (node1)-[e { layer1_att: 0.0, layer2_att: 0.0 } ]->(node2) '
                'DELETE e'
            )
            tx.run(query)

        print('build attention graph')
        session = self.session
        for idx, row in attentions.iterrows():
            # print(idx, lines)
            if idx % self.batch_size == 0:
                # fulfil batchsize, commit lines
                if len(lines) > 0:
                    # session.write_transaction(
                    #     commit_batch_attention, x_type, y_type, relation, lines=lines)
                    try:
                        commit_batch_attention(
                            session, x_type, y_type, relation, lines=lines)
                        delete_empty_edge(session)
                    except Exception:
                        print(idx, 'can not commit')
                x_type = row['x_type']
                y_type = row['y_type']
                relation = row['relation']
                lines = []
            elif row['x_type'] == x_type and row['y_type'] == y_type and relation == relation:
                # add new line
                lines += [{
                    'x_id': row['x_id'], 'y_id': row['y_id'],
                    'x_name': row['x_name'], 'y_name': row['y_name'],
                    'layer1_att': row['layer1_att'],
                    'layer2_att': row['layer2_att']
                }]
            else:
                # commit previous lines, change x y type
                # session.write_transaction(
                #     commit_batch_attention, x_type, y_type, relation, lines=lines)
                commit_batch_attention(
                    session, x_type, y_type, relation, lines=lines)
                delete_empty_edge(session)
                x_type = row['x_type']
                y_type = row['y_type']
                relation = row['relation']
                lines = []
        # session.write_transaction(
        #     commit_batch_attention, x_type, y_type, relation, lines=lines)
        commit_batch_attention(session, x_type, y_type, relation, lines=lines)
        # session.write_transaction(delete_empty_edge)
        delete_empty_edge(session)

    def remove_prediction(self):
        if not self.session:
            self.create_session()
        # query = ('match (:disease)-[e:Prediction]->(:drug) delete e')
        query = ('match (n:disease) remove n.predictions')
        self.session.run(query)

    def add_prediction(self, filename='full_graph_split1_eval.pkl'):

        if not self.session:
            self.create_session()

        drugs_with_indication = pd.read_pickle(os.path.join(
            self.data_path, 'drug_indication_subset.pkl'))

        prediction = pd.read_pickle(os.path.join(
            self.data_path, filename))['prediction']

        def commit_batch_prediction(tx, lines):
            query = (
                'UNWIND $lines as line '
                'MATCH (node: disease { id: line.disease_id }) '
                'SET node.predictions = line.predictions '
                'RETURN node'
            )
            tx.run(query, lines=lines)

        lines = []

        for disease in prediction:
            drugs = prediction[disease]
            drugs = [k for k in drugs.items() if k[0] in drugs_with_indication]
            top_drugs = sorted(
                drugs, key=lambda item: item[1], reverse=True
            )[:Neo4jApp.top_n]

            if len(lines) >= self.batch_size:
                self.session.write_transaction(
                    commit_batch_prediction, lines=lines)
                lines = []
            else:
                lines += [
                    {'disease_id': disease, 'predictions': json.dumps(
                        [[drug[0], float(drug[1])] for drug in top_drugs])}  # drug[:2] -> drug_id, score
                ]
        self.session.write_transaction(commit_batch_prediction, lines=lines)

    def query_diseases(self):

        if not self.session:
            self.create_session()

        def commit_diseases_query(tx):
            query = (
                'MATCH (node:disease) '
                'RETURN node.id'
            )
            results = tx.run(query)
            return list([k[0] for k in results])

        def commit_treatable_diseases_query(tx):
            query = (
                'MATCH (node:disease)-[e:rev_indication]->(:drug) '
                'RETURN node.id'
            )
            results = tx.run(query)
            # set to remove duplicate
            return list(set([k[0] for k in results]))

        all_disease = self.session.read_transaction(commit_diseases_query)
        treatable_diseases = self.session.read_transaction(
            commit_treatable_diseases_query)

        results = [[d, True if d in treatable_diseases else False]
                   for d in all_disease]
        return results

    def query_predicted_drugs(self, disease_id, query_n):

        def commit_pred_drugs_query(tx, disease_id):
            # query = (
            #     'MATCH (:disease { id: $id })-[edge:Prediction]->(node:drug)'
            #     'RETURN node, edge ORDER BY edge.score DESC LIMIT $top_n'
            # )
            query = (
                'MATCH (node:disease { id: $id })'
                'RETURN node.predictions'
            )
            results = tx.run(query, id=disease_id)

            predicted_drugs = json.loads(results.data()[0]['node.predictions'])[:query_n]

            return predicted_drugs

        def commit_known_drug_query(tx, disease_id):
            query = (
                'MATCH (:disease { id: $id })-[e:rev_indication]->(node:drug) '
                'RETURN node.id'
            )
            results = tx.run(query, id=disease_id)

            # set to remove duplicate
            return list(set([k[0] for k in results]))

        predicted_drugs = self.session.read_transaction(
            commit_pred_drugs_query, disease_id)

        known_drugs = self.session.read_transaction(
            commit_known_drug_query, disease_id)

        drugs = [
            {'score': drug[1], 'id': drug[0],
                "known": True if drug[0] in known_drugs else False}
            for drug in predicted_drugs
        ]

        self.current_disease = disease_id
        self.drugs = drugs

        return drugs

    @staticmethod
    def get_tree(results, node_type, node_id):
        """
        Params:
            reuslts: return from query, Array<[{node, rel}, [node, rel]]>
            node_type: type of root node
            node_id: id of root node
        Return:
            tree
        """
        def insertChild(children, items, depth, skip_nodes):
            if depth >= len(items):
                return children
            children_ids = list(map(lambda n: n['nodeId'], children))
            node = items[depth]['node']
            if node['id'] in skip_nodes and depth > 0:
                return children
            rel = items[depth]['rel']
            try:
                index = children_ids.index(node['id'])

            except Exception:  # item does not exist, insert a child
                if depth == 0:
                    score = 1
                    edgeInfo = ''
                elif depth == 1:
                    score = (rel['layer1_att'] + rel['layer2_att'])
                    edgeInfo = rel['edge_info'] if rel['edge_info'] else rel.type
                else:
                    score = (rel['layer1_att'])
                    edgeInfo = rel['edge_info'] if rel['edge_info'] else rel.type
                children.append({
                    'nodeId': node['id'],
                    'nodeType': Neo4jApp.get_node_labels(node)[0],
                    'score':  score,
                    'edgeInfo': edgeInfo,
                    'children': []
                })
                index = 0
            children[index]['children'] = insertChild(
                children[index]['children'], items, depth + 1, skip_nodes)

            return children

        children = []

        for i in range(len(results)):
            children = insertChild(children, results[i], 0, [node_id])

        if len(children) == 0:
            return {}
        return children[0]

    @staticmethod
    def commit_batch_attention_query(tx, node_type, root_nodes, k1, k2, rel1, rel2):

        query = (
            'UNWIND $nodes as root_node ',
            'MATCH  (node: {node_type} {{ id: root_node.id }})<-[rel: {rel1}]-(neighbor) '.format(
                node_type=node_type, rel1=rel1),
            # 'WHERE NOT (node)-[:Prediction]-(neighbor) '
            'WITH node, neighbor, rel '
            'ORDER BY (rel.layer1_att + rel.layer2_att + coalesce(rel.case_att, 0) ) DESC '
            'WITH node, ',
            'collect([ neighbor, rel])[0..{k1}] AS neighbors_and_rels '.format(
                k1=k1) if k1 else 'collect([ neighbor, rel]) AS neighbors_and_rels ',
            'UNWIND neighbors_and_rels AS neighbor_and_rel '
            'WITH node, '
            'neighbor_and_rel[0] AS neighbor, '
            'neighbor_and_rel[1] AS rel '
            'MATCH(neighbor)<-[rel2: {rel2}]-(neighbor2) '.format(rel2=rel2),
            # 'WHERE NOT (neighbor)-[:Prediction]-(neighbor2) '
            'WITH node, neighbor, rel, neighbor2, rel2 '
            'ORDER BY (rel2.layer1_att + coalesce(rel2.case_att, 0) ) DESC '
            'WITH node, neighbor, rel, ',
            'collect([neighbor2, rel2])[0..{k2}] AS neighbors_and_rels2 '.format(
                k2=k2) if k2 else 'collect([neighbor2, rel2]) AS neighbors_and_rels2 ',
            'UNWIND neighbors_and_rels2 AS neighbor_and_rel2 '
            'RETURN node, neighbor, rel, neighbor_and_rel2[0] AS neighbor2, neighbor_and_rel2[1] AS rel2 '
        )

        query = ' '.join(query)

        results = tx.run(query, nodes=root_nodes)
        results = [
            [
                {'node': record['node'], 'rel': 'none'},  # root node
                {'node': record['neighbor'], 'rel': record['rel']},  # hop1
                {'node': record['neighbor2'], 'rel': record['rel2']}  # hop2
            ]
            for record in results
        ]
        return results
    
    @staticmethod
    def commit_edge_type_query(tx, node_type, node_id):
        query = (
            'MATCH (node: {node_type} {{ id: "{node_id}" }})<-[e1]-()<-[e2]-() '.format(node_type=node_type, node_id=node_id),
            'RETURN DISTINCT type(e1) AS EdgeType1, type(e2) AS EdgeType2'
        )
        query = ' '.join(query)
        results = tx.run(query)

        return [[record['EdgeType1'],  record['EdgeType2']] for record in results]

    def query_attention(self, node_id, node_type):
        if not self.session:
            self.create_session()

        edge_types = self.session.read_transaction(
            Neo4jApp.commit_edge_type_query, node_type, node_id)
        
        results = []
        
        for edge_type in edge_types:
            rel1, rel2 = edge_type

            res = self.session.read_transaction(
                Neo4jApp.commit_batch_attention_query, node_type, [{'id': node_id}], Neo4jApp.k1, Neo4jApp.k2, rel1, rel2)
 
            results += res

        tree = self.get_tree(results, node_type, node_id)

        return results, tree

    # called by API
    def query_attention_pair(self, disease_id, drug_id):
        if not self.session:
            self.create_session()

        # drug_paths = self.session.read_transaction(
        #     Neo4jApp.commit_batch_attention_query, "drug", [{'id': drug_id}], 30, 20)

        # disease_paths = self.session.read_transaction(
        #     Neo4jApp.commit_batch_attention_query, "disease", [{'id': disease_id}], Neo4jApp.k1, Neo4jApp.k2)

        # def topk_paths(paths, k):
        #     '''
        #     Params:
        #     path: Array<[{node, rel}, [node, rel]]>
        #     k: top k
        #     Return: top k paths
        #     '''
        #     if k is None:
        #         return paths
        #     paths.sort(key=lambda x: x[1]['rel']['layer2_att']+x[1]['rel']
        #                ['layer1_att'] + x[2]['rel']['layer1_att'], reverse=True)
        #     return paths[:k]
        # drug_paths = topk_paths(drug_paths, Neo4jApp.k1 * Neo4jApp.k2)
        # disease_paths = topk_paths(disease_paths, Neo4jApp.k1 * Neo4jApp.k2)


        def convert(e, i):
            return {'edgeInfo': e['edge_info'] if e['edge_info'] else e.type, 'score': e['layer1_att'] + e['layer2_att'] if i == 0 else e['layer1_att']}

        disease_paths, disease_tree = self.query_attention(
            disease_id, 'disease')
        drug_paths, drug_tree = self.query_attention(drug_id, 'drug')
        
        paths = []
        existing_path = []

        for disease_path in disease_paths:
            for drug_path in drug_paths:
                for idx_a, item_a in enumerate(disease_path):
                    for idx_b, item_b in enumerate(drug_path):
                        node_a = item_a['node']
                        node_b = item_b['node']
                        type_a = Neo4jApp.get_node_labels(node_a)[0]
                        type_b = Neo4jApp.get_node_labels(node_b)[0]
                        if type_a == type_b and node_a['id'] == node_b['id']:
                            # find a path, update path
                            nodes = [
                                {
                                    'nodeId': item['node']['id'],
                                    'nodeType': Neo4jApp.get_node_labels(item['node'])[0]
                                } for item in disease_path[:idx_a+1] +
                                drug_path[:idx_b][::-1]]

                            node_ids = [node['nodeId'] for node in nodes]

                            # if duplicated items in path, ignore
                            if len(node_ids) > len(set(node_ids)):
                                continue
                            path_string = '-'.join(node_ids)
                            if path_string in existing_path:
                                pass
                            else:
                                existing_path.append(path_string)
                                # the edge calculation is tricky here
                                edges = [convert(item['rel'], i) for i, item in enumerate(disease_path[1:idx_a+1])] + (
                                    [convert(item['rel'], i) for i, item in enumerate(
                                        drug_path[1:idx_b+1])
                                     ][::-1]
                                )

                                path = {
                                    'nodes': nodes,
                                    'edges': edges,
                                    'avg_score': sum([e['score'] for e in edges])/len(edges)
                                }
                                paths.append(path)
        
        # for all paths, keep top n for each metapath based on path score
        metapaths = {}
        for path in paths:
            metapath = '-'.join([node['nodeType'] for node in path['nodes']])
            if metapath in metapaths:
                metapaths[metapath].append(path)
                metapaths[metapath].sort(key=lambda x: x['avg_score'], reverse=True)
                if len(metapaths[metapath]) > Neo4jApp.path_thr:
                    metapaths[metapath] = metapaths[metapath][:Neo4jApp.path_thr]
            else:
                metapaths[metapath] = [path]

        # flatten metapaths
        sorted_paths = []
        for metapath in metapaths:
            sorted_paths += metapaths[metapath]

        attention = {}
        attention['{}:{}'.format('disease', disease_id)] = disease_tree
        attention['{}:{}'.format('drug', drug_id)
                  ] = drug_tree

        # sort paths by score
        paths.sort(key=lambda x: sum(
            [e['score'] for e in x['edges']])/len(x['edges']), reverse=True)

        return {'attention': attention, "paths": sorted_paths}

    @staticmethod
    def get_node_labels(node):
        return list(node.labels)

    def query_metapath_summary(self, top_n):

        if not self.session:
            self.create_session()

        assert self.current_disease is not None, 'should assign a disease id first'

        if not self.drugs:
            self.query_predicted_drugs(self.current_disease, top_n)

        drug_paths = self.session.read_transaction(
            Neo4jApp.commit_batch_attention_query, 'drug', self.drugs)
        disease_paths = self.session.read_transaction(
            Neo4jApp.commit_batch_attention_query,
            'disease',
            [{'id': self.current_disease}]
        )

        metapaths = []
        existing_paths = []
        metapath_keys = {}

        for disease_path in disease_paths:

            for drug_path in drug_paths:
                current_drug = drug_path[0]['node']['id']
                drug_idx = [drug['id']
                            for drug in self.drugs].index(current_drug)

                for idx_a, item_a in enumerate(disease_path):
                    for idx_b, item_b in enumerate(drug_path):
                        node_a = item_a['node']
                        node_b = item_b['node']
                        type_a = Neo4jApp.get_node_labels(node_a)[0]
                        type_b = Neo4jApp.get_node_labels(node_b)[0]
                        if type_a == type_b and node_a['id'] == node_b['id']:
                            # find a path, update metapath
                            items = disease_path[:idx_a+1] + \
                                drug_path[:idx_b][::-1]
                            node_ids = [item['node']['id']
                                        for item in items]

                            # if duplicated items in path, ignore
                            if len(node_ids) > len(set(node_ids)):
                                continue
                            path = '-'.join(node_ids)

                            metapath = list(
                                map(lambda item: Neo4jApp.get_node_labels(item['node'])[0], items))
                            metapath_key = '-'.join(metapath)

                            if path in existing_paths:
                                continue
                            else:
                                existing_paths.append(path)

                                if metapath_key in metapath_keys:
                                    metapaths[metapath_keys[metapath_key]
                                              ]['count'][drug_idx] += 1
                                    metapaths[metapath_keys[metapath_key]
                                              ]['sum'] += 1
                                else:
                                    metapath_keys[metapath_key] = len(
                                        metapaths)
                                    count = [0 for i in self.drugs]
                                    count[drug_idx] += 1
                                    metapaths.append(
                                        {
                                            'nodeTypes': metapath,
                                            'count': count,
                                            'sum': 1
                                        }
                                    )

        metapaths.sort(key=lambda x: x['sum'], reverse=True)
        return metapaths

# %%


if __name__ == '__main__':
    db = Neo4jApp(server='txgnn_v2', user='neo4j', 
                  database='neo4j', datapath='TxGNNExplorer_v2')
    db.init_database()
# %%
