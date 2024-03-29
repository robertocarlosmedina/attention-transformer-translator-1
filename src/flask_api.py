from flask_restful import Api, Resource, reqparse
from flask import Flask, jsonify
from flask_cors import CORS
from src.seq_to_seq_transformer import Sequence_to_Sequence_Transformer


app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)

request_put_args = reqparse.RequestParser()
request_put_args.add_argument("sentence", type=str, help="Sentece to be translated.")
seq_to_seq_trans = Sequence_to_Sequence_Transformer()


class Translation(Resource):

    def post(self, source, target):
        # print(source, " => ",target)
        args = request_put_args.parse_args()
        valid = False
        source_sentence = args["sentence"]
        target_sentence = ""
        if source_sentence:
            valid = True
        target_sentence = seq_to_seq_trans.translate_sentence(source_sentence)
        data = {"data": [{"translation": target_sentence,"valid": valid}]}
        return jsonify(data)


class Resfull_API:
    @staticmethod
    def start():
        api.add_resource(Translation, "/translate/<string:source>/<string:target>")
        app.run(debug=False)
