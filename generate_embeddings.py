import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# paths
RESUME_DIR = "Figure1_100Samples/Resumes"
EMB_DIR = "Figure1_100Samples/Embeddings"

os.makedirs(EMB_DIR, exist_ok=True)

# load embedding model (same for all)
model = SentenceTransformer("all-MiniLM-L6-v2")

# map filename -> text column
TEXT_COLUMN_MAP = {
    "original.csv": "CV",
    "claude-3-5-sonnet.csv": "Claude-3.5-Sonnet",
    "deepseek_v3.csv": "Clean_Deepseek_V3",
    "deepseek67b.csv": "Clean_Deepseek_67B",
    "gpt-35-turbo.csv": "GPT-3.5-Turbo",
    "gpt4o.csv": "GPT4o",
    "gpt4omini.csv": "GPT4omini",
    "llama3-70B.csv": "Llama3-70B",
    "mixtral-8x7b.csv": "Mixtral-8x7B",
}

for filename, text_col in TEXT_COLUMN_MAP.items():
    csv_path = os.path.join(RESUME_DIR, filename)
    out_path = os.path.join(
        EMB_DIR, filename.replace(".csv", ".npy")
    )

    print(f"Processing {filename}")

    df = pd.read_csv(csv_path)

    # enforce correct alignment
    df = df.sort_values("id").reset_index(drop=True)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in {filename}. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_col].astype(str).tolist()

    if len(texts) != 100:
        print(f"WARNING: {filename} has {len(texts)} rows")

    embeddings = model.encode(texts, show_progress_bar=True)

    np.save(out_path, embeddings)

    print(f"Saved {out_path} with shape {embeddings.shape}\n")

print("All embeddings regenerated.")
