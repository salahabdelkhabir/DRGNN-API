import os
# from model_loader_static import ModelLoader
SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    FRONT_ROOT = os.path.join(SERVER_ROOT, 'build')
    DATA_FOLDER = os.path.join(SERVER_ROOT, 'txgnn_data_v2')
    STATIC_FOLDER = os.path.join(SERVER_ROOT, 'build/static')
    GNN = 'txgnn_v2'
    # MODEL_LOADER = ModelLoader(os.path.join(SERVER_ROOT, 'colab_delivery/'))
    # MODEL_LOADER = ModelLoader('s3://drug-gnn-models/collaboration_delivery/')


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    pass


class TestingConfig(Config):
    pass
