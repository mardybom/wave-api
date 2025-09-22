import base64
import random
import re
import itertools
from db import get_db_connection

def format_label(label: str) -> str:
    # Replace underscores and hyphens with spaces
    label = label.replace("_", " ").replace("-", " ")
    # Remove unwanted symbols except spaces
    label = re.sub(r"[^A-Za-z0-9 ]+", "", label)
    return label.strip()

def generate_rearranged_labels(correct_label: str, count: int = 4):
    chars = list(correct_label)

    # Generate permutations (as strings)
    perms = {"".join(p) for p in itertools.permutations(chars)}

    # Remove the original correct label
    perms.discard(correct_label)

    # Convert to list for random.sample
    perms_list = list(perms)

    # Pick up to `count` random permutations
    fake_labels = random.sample(perms_list, min(count, len(perms_list)))

    return fake_labels

def fetch_random_image_row():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT image_id, image_byte, image_label FROM image_labeling ORDER BY random() LIMIT 1;")
            row = cur.fetchone()
            if not row:
                return None

            formatted_label = format_label(row["image_label"])
            fake_labels = generate_rearranged_labels(formatted_label)

            # Encode image bytes to base64
            base64_img = base64.b64encode(row["image_byte"]).decode("utf-8")

            # Shuffle correct + fake options
            options = [formatted_label] + fake_labels
            random.shuffle(options)

            return {
                "image_id": row["image_id"],
                "image_label": formatted_label,
                "image_base64": base64_img,
                "options": options
            }
    finally:
        conn.close()
