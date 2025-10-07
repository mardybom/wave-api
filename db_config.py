import psycopg2
import psycopg2.extras
import os

def _conninfo():
    return {
        "host": os.environ["DB_HOST"],
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "port": os.environ["DB_PORT"],
        "sslmode": os.environ["DB_SSLMODE"],
    }

def get_db_connection():
    return psycopg2.connect(**_conninfo(), cursor_factory=psycopg2.extras.RealDictCursor)
