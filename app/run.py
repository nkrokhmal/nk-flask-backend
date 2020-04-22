import werkzeug
from flask import Flask, request
from flask_restplus import reqparse
from flask_restx import Api, Resource, swagger
from utils.ftpclient.client import SftpClient
from utils.postgresclient.client import PostgresClient
from utils.rfcclient.client import Object, Wave, Spectrum, Coordinates, Points
from datetime import datetime
import json
import numpy as np
import io
import os
import base64
from flask_cors import CORS
import time


host = '35.228.186.127'
username = 'nkrokhmal'
password = 'kloppolk_2018'
sftp_client = SftpClient(host=host, username=username, password=password)

app = Flask(__name__)
CORS(app)

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
        values = (model_name, 'http://' + host + '/' + model_path, 'http://' + host + '/' + distribution_path, datetime.utcnow(), params, 1)
        cur.execute(req, values)
        psql_client.client.commit()
        cur.close()
        return model_name


@namespace.route('/scatterer/', methods=['POST'])
class Scatterer(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('Radius', type=float, location='form', required=False)
    parser.add_argument('LongitudinalSpeed', type=float, location='form', required=False)
    parser.add_argument('TransverseSpeed', type=float, location='form', required=False)
    parser.add_argument('DensityOfScatterer', type=float, location='form', required=False)
    parser.add_argument('Frequency', type=float, location='form', required=False)
    parser.add_argument('SpeedOfSound', type=float, location='form', required=False)
    parser.add_argument('DensityOfMedium', type=float, location='form', required=False)
    parser.add_argument('Dx', type=float, location='form', required=False)
    parser.add_argument('Type', type=str, location='form', required=False)
    parser.add_argument('From', type=float, location='form', required=False)
    parser.add_argument('To', type=float, location='form', required=False)
    parser.add_argument('Step', type=float, location='form', required=False)
    parser.add_argument('ModelPath', type=str, location='form', required=False)
    parser.add_argument('ModelName', type=str, location='form', required=False)

    @api.expect(parser=parser)
    @api.doc(parser=parser)
    def post(self):
        args = self.parser.parse_args()
        radius = args['Radius']
        longitudinal = args['LongitudinalSpeed']
        transverse = args['TransverseSpeed']
        density_of_scatterer = args['DensityOfScatterer']
        frequency = args['Frequency']
        speed_of_sound = args['SpeedOfSound']
        density_of_medium = args['DensityOfMedium']
        dx = args['Dx']
        params_type = args['Type']
        from_coordinate = args['From']
        to_coordinate = args['To']
        step = args['Step']
        model_path = args['ModelPath']
        model_id = args['ModelName']

        cur_time = time.strftime('%Y%m%d%H%M%S')
        force_dict_name = '{}_{}.npy'.format(model_id, cur_time)
        force_image_name = '{}_{}.png'.format(model_id, cur_time)
        force_dict_path = '/opt/download/{}'.format(force_dict_name)
        force_image_path = '/opt/download/{}'.format(force_image_name)

        obj = Object(a=radius, rho=density_of_scatterer, c_l=longitudinal, c_t=transverse)
        wave = Wave(f=frequency, c=speed_of_sound, rho=density_of_medium)
        spectrum = Spectrum(dx=dx)

        if params_type == 'X':
            coordinates = Coordinates(x=np.arange(from_coordinate, to_coordinate, step), y=np.array([0.0]), z=np.array([0.0]))
        elif params_type == 'Y':
            coordinates = Coordinates(x=np.array([0.0]), y=np.arange(from_coordinate, to_coordinate, step), z=np.array([0.0]))
        else:
            coordinates = Coordinates(x=np.array([0.0]), y=np.array([0.0]), z=np.arange(from_coordinate, to_coordinate, step))

        local_path = '/opt/download/{}.mat'.format(model_id)
        sftp_client.download_file_local(local_path, model_path)

        points = Points(coordinates, obj, wave, spectrum, local_path)
        force, force_x, force_y, force_z, scat_p = points.calculate_force()
        fig = points.build_rad_force(force)
        fig.savefig(force_image_path)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        result_image = base64.b64encode(buffer.getvalue())
        os.remove(local_path)

        '''add data to ftp server'''
        force_dict = {
            'force_x': force_x,
            'force_y': force_y,
            'force_z': force_z,
            'force': force
        }
        np.save('/opt/download/{}_{}'.format(model_id, cur_time), force)

        print(force_dict_path)
        sftp_client.upload_file(force_dict_path, 'force/{}'.format(force_dict_name))
        print(force_image_path)
        sftp_client.upload_file(force_image_path, 'force_image/{}'.format(force_image_name))

        '''add data to postgresql'''
        cur = psql_client.client.cursor()
        req = 'INSERT INTO "{}" (x_force, y_force, z_force, force_data_path, force_image_path, model_id) VALUES (%s, %s, %s, %s, %s, %s)' \
            .format('ModelResults')
        force_data_path = 'http://' + host + '/' + 'force/{}'.format(force_dict_name)
        force_image_path = 'http://' + host + '/' + 'force_image/{}'.format(force_image_name)
        values = (force_x, force_y, force_z, force_data_path, force_image_path, model_id)
        cur.execute(req, values)
        psql_client.client.commit()
        cur.close()


        return result_image.decode('ascii')


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

@namespace.route('/model/<model_id>')
class DeleteModel(Resource):
    def delete(self, model_id):
        print(model_id)
        cursor = psql_client.client.cursor()
        req = 'UPDATE "{}" SET "status_id" = 2 WHERE "id" = {}'.format('Models', model_id)
        print(req)
        cursor.execute(req)
        updated_rows = cursor.rowcount
        print(updated_rows)
        psql_client.client.commit()
        cursor.close()



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
    app.run(debug=True, port=8818, threaded=True, host='0.0.0.0')
