import pysftp
from io import BytesIO


class SftpClient():
    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        self.sftp_client = pysftp.Connection(host=host, username=username, password=password, cnopts=cnopts)
        self.upload_path = '/home/nkrokhmal/storage/'

    def upload_file(self, local_path, remote_path):
        self.sftp_client.put(localpath=local_path, remotepath=self.upload_path + remote_path)

    def upload_file_remote(self, local_path, remote_path):
        self.sftp_client.put(localpath=local_path, remotepath=remote_path)

    def upload_file_fo(self, file, remote_path):
        self.sftp_client.putfo(file, remotepath=self.upload_path + remote_path)

    def download_file(self, remote_path):
        file = BytesIO()
        self.sftp_client.getfo(self.upload_path + remote_path, file)
        file.seek(0)
        return file

    def download_file_local(self, local_path, remote_path):
        self.sftp_client.get(self.upload_path + remote_path, localpath=local_path)

