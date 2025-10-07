"""
Database Configuration Module

Provides helper functions to construct connection parameters
and return a psycopg2 database connection.
"""

import os
import psycopg2
import psycopg2.extras


def _conninfo() -> dict:
    """Return connection information from environment variables."""
    return {
        "host": os.environ["DB_HOST"],
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "port": os.environ["DB_PORT"],
        "sslmode": os.environ["DB_SSLMODE"],
    }


def get_db_connection():
    """Create and return a psycopg2 connection using RealDictCursor."""
    return psycopg2.connect(
        **_conninfo(),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
