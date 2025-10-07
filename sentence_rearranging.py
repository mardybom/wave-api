import os
import psycopg2
import psycopg2.extras
from db_config import get_db_connection

def fetch_next_sentence_row(level: str):
    """
    Returns the next unique row for the given difficulty_level (wraps to first when needed).
    Requires an existing cursor table:
      sentence_cursors(
        difficulty_level difficulty_level PRIMARY KEY,
        last_sentence_id INTEGER NOT NULL
      )
    No DDL is executed here; this function only reads.
    """
    # 1) Find next row for this level (or wrap to first)
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

    # 2) Advance the cursor to the id we just served
    sql_update_cursor = """
        INSERT INTO sentence_cursors(difficulty_level, last_sentence_id)
        VALUES (%s, %s)
        ON CONFLICT (difficulty_level)
        DO UPDATE SET last_sentence_id = EXCLUDED.last_sentence_id;
    """

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Per-level advisory lock so concurrent requests don't double-serve the same row
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (level,))

            # Fetch the next (or wrap) row for this level
            cur.execute(sql_select, (level, level, level))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None

            next_id = row["sentence_id"]

            # Update cursor to this sentence_id
            cur.execute(sql_update_cursor, (level, next_id))
            conn.commit()
            return row
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()