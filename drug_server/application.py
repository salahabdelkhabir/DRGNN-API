from vis import vis
from api import api
from config import Config, ProductionConfig, DevelopmentConfig, SERVER_ROOT

from flask_cors import CORS
from flask import Flask, jsonify, g

import argparse

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    import simplejson as json
except ImportError:
    import json


def create_app(config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Update configs
    if app.config['ENV'] == 'production':
        app.config.from_object(ProductionConfig)
    elif app.config['ENV'] == 'development':
        app.config.from_object(DevelopmentConfig)
    else:
        app.config.from_object(Config)

    # print(config)
    app.config.update(config)

    @app.teardown_appcontext
    def close_db(error):
        if hasattr(g, 'neo4j_db'):
            g.neo4j_db.close_session()

    @app.route('/config')
    def config():
        return jsonify([str(k) for k in list(app.config.items())])

    @app.route('/hello')
    def hello():
        return 'hello world'

    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(vis, url_prefix='/')

    return app


parser = argparse.ArgumentParser()
parser.add_argument('--host', default='0.0.0.0',
                    help='Port in which to run the API')
parser.add_argument('--port', default=8002,
                    help='Port in which to run the API')
parser.add_argument('--debug', action="store_const", default=True, const=True,
                    help='If true, run Flask in debug mode')

_args, unknown = parser.parse_known_args()

if _args.debug:
    os.environ['FLASK_ENV'] = 'development'

application = create_app(vars(_args))


if __name__ == '__main__':
    application.run(
        debug=_args.debug,
        host=_args.host,
        port=int(_args.port)
    )
