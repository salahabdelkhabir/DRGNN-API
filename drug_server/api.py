import json
import numpy as np

import flask
from flask import request, jsonify, Blueprint, current_app, g
from utils import better_json_encoder

from database import get_db

api = Blueprint('api', __name__)

api.json_encoder = better_json_encoder(flask.json.JSONEncoder)

######################
# API Starts here
######################


@api.route('/test', methods=['GET'])
def test():
    return 'api test'


@api.route('/diseases', methods=['GET'])
def get_diseases():
    '''
    :return: diseaseID[]
    '''
    db = get_db()
    return jsonify(db.query_diseases())


@api.route('/attention', methods=['GET'])
def get_attention():
    '''
    :return: {'key': attentionTree}
    E.g.: [base_url]/api/attention?disease=0&drug=0
    '''
    disease_id = request.args.get('disease', None, type=str)
    drug_id = request.args.get('drug', None, type=str)

    db = get_db()
    attention = {}
    attention['disease'] = db.query_attention(disease_id, 'disease')
    attention['drug'] = db.query_attention(drug_id, 'drug')

    return jsonify(attention)


@api.route('/attention_pair', methods=['GET'])
def get_attention_pair():
    '''
    :return: {'attention': {key: attentionTree}, 'paths': path[]}
    E.g.: [base_url]/api/attention_pair?disease=0&drug=0
    '''
    disease_id = request.args.get('disease', None, type=str)
    drug_id = request.args.get('drug', None, type=str)

    db = get_db()
    res = db.query_attention_pair(disease_id, drug_id)

    return jsonify(res)


@api.route('/drug_predictions', methods=['GET'])
def get_drug_predictions():
    '''
    get drug predictions
    E.g.: [base_url]/api/drug_predictions?disease_id=1687.0&top_n=30

    :return: {
        predictions:{score:number, id: string }[],
        metapath_summary: {node_types: string[], count: number}[]
        }
    '''
    disease_id = request.args.get('disease_id', None, type=str)
    db = get_db()
    QUERY_N = 200
    predictions = db.query_predicted_drugs(
        disease_id=disease_id, query_n = QUERY_N)

     summary = db.query_metapath_summary(top_n=top_n)
     return jsonify({'predictions': predictions, 'metapath_summary': summary})

@api.route('/link_pred', methods=['GET'])
def get_link_pred():
    '''
    E.g.: [base_url]/api/drug_score?disease_id=5263.0&drug_id=DB06700
    '''
    
    disease_id = request.args.get('disease_id', None, type=str)
    drug_id = request.args.get('drug_id', None, type=str)
    db = get_db()
    predictions = db.query_drug_disease_pair(disease_id=disease_id, drug_id=drug_id)
    return jsonify(predictions)

    return jsonify(predictions)
