"""
Sentence Rearranging Database Logic

Provides helper function to fetch the next unique sentence row
for a given difficulty level, using PostgreSQL with cursor tracking.
"""

import psycopg2
import psycopg2.extras

from db_config import get_db_connection


def fetch_next_sentence_row(level: str):
    """
    Return the next unique row for the given difficulty_level.
    Wraps to the first row when the end of the list is reached.

    Requires existing tables:
        sentence_cursors(
            difficulty_level TEXT PRIMARY KEY,
            last_sentence_id INTEGER NOT NULL
        )
        sentence_jumbling(
            sentence_id SERIAL PRIMARY KEY,
            original_sentence TEXT,
            jumbled_sentence TEXT,
            difficulty_level TEXT
        )
    """
    sql_select = """
        WITH cur AS (
            SELECT last_sentence_id
            FROM sentence_cursors
            WHERE difficulty_level = %s
        ),
        nxt AS (
            SELECT sentence_id, original_sentence, jumbled_sentence, difficulty_level
            FROM sentence_jumbling
            WHERE difficulty_level = %s
              AND sentence_id > COALESCE((SELECT last_sentence_id FROM cur), 0)
            ORDER BY sentence_id ASC
            LIMIT 1
        )
        SELECT * FROM nxt
        UNION ALL
        SELECT sentence_id, original_sentence, jumbled_sentence, difficulty_level
        FROM (
            SELECT sentence_id, original_sentence, jumbled_sentence, difficulty_level
            FROM sentence_jumbling
            WHERE difficulty_level = %s
            ORDER BY sentence_id ASC
            LIMIT 1
        ) wrap
        WHERE NOT EXISTS (SELECT 1 FROM nxt);
    """

    sql_update_cursor = """
        INSERT INTO sentence_cursors (difficulty_level, last_sentence_id)
        VALUES (%s, %s)
        ON CONFLICT (difficulty_level)
        DO UPDATE SET last_sentence_id = EXCLUDED.last_sentence_id;
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Lock per difficulty level to prevent concurrent double-serving
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (level,))

            # Fetch the next or wrapped row
            cur.execute(sql_select, (level, level, level))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None

            # Update cursor to this sentence_id
            next_id = row["sentence_id"]
            cur.execute(sql_update_cursor, (level, next_id))
            conn.commit()

            return row

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()