import psycopg2

class PostgresClient:
    def __init__(self, dbname, user, password, host):
        self.client = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
