"""
Dyslexia Myths Database Logic

Provides helper function to fetch the next batch of myths and truths
from the dyslexia_myths table. Wraps to the beginning when the end
is reached and maintains progress in myth_cursors.
"""

import psycopg2
import psycopg2.extras

from db_config import get_db_connection


def fetch_next_myth_row(batch_size: int = 10):
    """
    Return the next batch of myths (wraps after reaching the end).

    Tracks progress using:
        myth_cursors(
            singleton INTEGER PRIMARY KEY,
            last_myth_id INTEGER NOT NULL
        )
    """
    sql_select = """
        WITH cur AS (
            SELECT last_myth_id
            FROM myth_cursors
            WHERE singleton = 1
        ),
        nxt AS (
            SELECT id, myth, truth
            FROM dyslexia_myths
            WHERE id > COALESCE((SELECT last_myth_id FROM cur), 0)
            ORDER BY id ASC
            LIMIT %s
        ),
        wrap AS (
            SELECT id, myth, truth
            FROM dyslexia_myths
            ORDER BY id ASC
            LIMIT %s
        )
        SELECT * FROM nxt
        UNION ALL
        SELECT * FROM wrap
        WHERE (SELECT COUNT(*) FROM nxt) < %s;
    """

    sql_update_cursor = """
        INSERT INTO myth_cursors (singleton, last_myth_id)
        VALUES (1, %s)
        ON CONFLICT (singleton)
        DO UPDATE SET last_myth_id = EXCLUDED.last_myth_id;
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Prevent concurrent reads
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('dyslexia_myths_stream'));")

            # Fetch next batch or wrap
            cur.execute(sql_select, (batch_size, batch_size - 1, batch_size))
            rows = cur.fetchall()

            if not rows:
                conn.rollback()
                return []

            # Update cursor to last myth ID fetched
            last_id = rows[-1]["id"]
            cur.execute(sql_update_cursor, (last_id,))
            conn.commit()

            return rows

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()