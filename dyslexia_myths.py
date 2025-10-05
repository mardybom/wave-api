import psycopg2
import psycopg2.extras
from db import get_db_connection

def fetch_next_myth_row(batch_size=10):
    """
    Returns the next batch of myths (wraps after the end).
    Tracks progress in myth_cursors (singleton=1, last_myth_id).
    """

    sql_select = f"""
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
            LIMIT {batch_size}
        ),
        wrap AS (
            SELECT id, myth, truth
            FROM dyslexia_myths
            ORDER BY id ASC
            LIMIT {batch_size - 1}
        )
        SELECT * FROM nxt
        UNION ALL
        SELECT * FROM wrap
        WHERE (SELECT COUNT(*) FROM nxt) < {batch_size};
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

            cur.execute(sql_select)
            rows = cur.fetchall()

            if not rows:
                conn.rollback()
                return []

            # Update cursor to last myth ID fetched
            last_id = rows[-1]["id"]

            cur.execute(sql_update_cursor, (last_id,))
            conn.commit()

            return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
