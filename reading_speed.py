"""
Reading Speed Database Logic

Provides helper function to fetch the next unique reading passage
for a given difficulty level, using PostgreSQL with cursor tracking.
"""

import psycopg2
import psycopg2.extras
from db_config import get_db_connection


def fetch_next_reading_row(level: str):
    """
    Return the next unique row for the given level (Easy, Medium, Hard).
    Wraps back to the first when the end is reached.

    Requires existing tables:
        reading_speed(
            id SERIAL PRIMARY KEY,
            level TEXT,
            text TEXT,
            word_count INTEGER
        )

        reading_speed_cursors(
            level TEXT PRIMARY KEY,
            last_reading_id INTEGER NOT NULL
        )
    """

    sql_select = """
        WITH cur AS (
            SELECT last_reading_id
            FROM reading_speed_cursors
            WHERE level = %s
        ),
        nxt AS (
            SELECT id, text, level, word_count
            FROM reading_speed
            WHERE level = %s
              AND id > COALESCE((SELECT last_reading_id FROM cur), 0)
            ORDER BY id ASC
            LIMIT 1
        )
        SELECT * FROM nxt
        UNION ALL
        SELECT id, text, level, word_count
        FROM (
            SELECT id, text, level, word_count
            FROM reading_speed
            WHERE level = %s
            ORDER BY id ASC
            LIMIT 1
        ) wrap
        WHERE NOT EXISTS (SELECT 1 FROM nxt);
    """

    sql_update_cursor = """
        INSERT INTO reading_speed_cursors (level, last_reading_id)
        VALUES (%s, %s)
        ON CONFLICT (level)
        DO UPDATE SET last_reading_id = EXCLUDED.last_reading_id;
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Lock per difficulty level to prevent concurrent double-serving
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (level,))

            # Fetch the next (or wrapped) reading row
            cur.execute(sql_select, (level, level, level))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None

            # Update cursor to this id
            next_id = row["id"]
            cur.execute(sql_update_cursor, (level, next_id))
            conn.commit()

            return row

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()