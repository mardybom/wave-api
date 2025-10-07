import base64
import random
import re
import pronouncing
from db_config import get_db_connection

# ---- ITR3: ARPAbet helpers ----
def _arpabet_from_cmudict(word: str) -> str | None:

    phones = pronouncing.phones_for_word(word.lower())
    return phones[0] if phones else None

def label_to_arpabet(label: str) -> str:
    """
    Convert a (possibly multi-word) label to ARPAbet.
    Joins per-word ARPAbet with ' | ' (word separator).
    """
    words = re.findall(r"[A-Za-z']+", label)
    arpabet_chunks = []
    for w in words:
        p = _arpabet_from_cmudict(w)
        if p:
            arpabet_chunks.append(p)

    return " | ".join(arpabet_chunks)


def format_label(label: str) -> str:

    # Replace underscores and hyphens with spaces
    label = label.replace("_", " ").replace("-", " ")

    # Remove unwanted symbols except spaces
    label = re.sub(r"[^A-Za-z0-9 ]+", "", label)
    return label.strip()

def generate_rearranged_labels(correct_label: str, count: int = 4):
    """
    Generate up to `count` unique jumbled versions of the correct label.
    Avoids factorial permutations by using random shuffling.
    """
 
    chars = list(correct_label)
    fake_labels = set()

    attempts = 0
    max_attempts = 50  # safety cap
    while len(fake_labels) < count and attempts < max_attempts:
        attempts += 1
        random.shuffle(chars)
        shuffled = "".join(chars)
        if shuffled.lower() != correct_label.lower():
            fake_labels.add(shuffled)

    return list(fake_labels)

def fetch_random_image_row():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT image_id, image_byte, image_label
                FROM image_labeling
                ORDER BY random()
                LIMIT 1;
            """)
            row = cur.fetchone()
            if not row:
                return None

            formatted_label = format_label(row["image_label"])
            fake_labels = generate_rearranged_labels(formatted_label)

            # Encode image bytes to base64
            base64_img = base64.b64encode(row["image_byte"]).decode("utf-8")

            # Build options (correct + fakes)
            options = [formatted_label] + fake_labels
            random.shuffle(options)

            # compute ARPAbet for the formatted label
            arpabet = label_to_arpabet(formatted_label)

            return {
                "image_id": row["image_id"],
                "image_label": formatted_label,
                "image_base64": base64_img,
                "options": options,
                "arpabet": arpabet,          # <--- added field
            }
    finally:
        conn.close()