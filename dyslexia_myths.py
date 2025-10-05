import psycopg2
import psycopg2.extras
from db import get_db_connection

def fetch_next_myth_row():
    """
    Returns the next myth row (wraps after the last one).
    Tracks progress in myth_cursors (singleton=1, last_myth_id).
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
            LIMIT 1
        )
        SELECT * FROM nxt
        UNION ALL
        SELECT id, myth, truth
        FROM (
            SELECT id, myth, truth
            FROM dyslexia_myths
            ORDER BY id ASC
            LIMIT 1
        ) wrap
        WHERE NOT EXISTS (SELECT 1 FROM nxt);
    """

    sql_update_cursor = """
        INSERT INTO myth_cursors (singleton, last_myth_id)
        VALUES (1, %s)
        ON CONFLICT (singleton)
        DO UPDATE SET last_myth_id = EXCLUDED.last_myth_id;
    """
    # connect to db
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Single-stream advisory lock (no key arg needed)
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('dyslexia_myths_stream'));")

            cur.execute(sql_select)
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None

            next_id = row["id"]

            cur.execute(sql_update_cursor, (next_id,))
            conn.commit()
            return row
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
