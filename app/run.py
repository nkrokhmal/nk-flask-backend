import werkzeug
from flask import Flask, request
from flask_restplus import reqparse
from flask_restx import Api, Resource, swagger
from utils.ftpclient.client import SftpClient
from utils.postgresclient.client import PostgresClient
from datetime import datetime
import json

host = '35.228.186.127'
username = 'nkrokhmal'
password = 'kloppolk_2018'
sftp_client = SftpClient(host=host, username=username, password=password)

app = Flask(__name__)

api = Api(app=app, version='0.1', title='AlisaProject API', description='API for building model')
namespace = api.namespace('api', description='Main APIs')

psql_client = PostgresClient(dbname='aproject', user='nkrokhmal', password='kloppolk_2018', host='35.228.186.127')


@namespace.route('/savemodel/', methods=["POST"])
class ModelTest(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument(
        'ModelFile', type=werkzeug.FileStorage, location='files', required=True)
    parser.add_argument(
        'DistributionFile', type=werkzeug.FileStorage, location='files', required=True)
    parser.add_argument(
        'ModelName', type=str, location='form', required=True)
    parser.add_argument(
        'Parameters', type=str, location='form', required=True)

    @api.expect(parser=parser)
    @api.doc(parser=parser)
    def post(self):
        args = self.parser.parse_args()
        model_file = args['ModelFile']
        model_name = args['ModelName']
        params = args['Parameters']
        distribution_file = args['DistributionFile']

        model_path = 'models/{}.mat'.format(model_name)
        distribution_path = 'distributions/{}.jpg'.format(model_name)

        sftp_client.upload_file_fo(model_file, model_path)
        sftp_client.upload_file_fo(distribution_file, distribution_path)

        cur = psql_client.client.cursor()
        req = 'INSERT INTO "{}" (model_name, model_path, pressure_distribution_path, creation_time, params, status_id) VALUES (%s, %s, %s, %s, %s, %s)' \
            .format('Models')
        values = (model_name, model_path, distribution_path, datetime.utcnow(), params, 1)
        cur.execute(req, values)
        psql_client.client.commit()
        cur.close()
        return model_name


@namespace.route('/models/')
class SaveModel(Resource):
    def myconverter(self, dt):
        if isinstance(dt, datetime):
            return dt.__str__()

    def get_data(self):
        cursor = psql_client.client.cursor()
        cursor.execute('SELECT * FROM "{}" LIMIT 0'.format('Models'))
        column_names = [desc[0] for desc in cursor.description]
        cursor.execute('SELECT * FROM "{}" WHERE "status_id" = 1'.format('Models'))
        records = cursor.fetchall()
        result = [{k: v for k, v in zip(column_names, record)} for record in records]
        cursor.close()
        return result

    def get(self):
        result = self.get_data()
        return json.dumps(result, default=self.myconverter)



@namespace.route('/')
class Test(Resource):
    def get(self):
        return {
            "status": "Got new data"
        }

    def post(self):
        return {
            "status": "Posted new data"
        }


if __name__ == '__main__':
    app.run(debug=True, port=6666, threaded=True, host='0.0.0.0')
