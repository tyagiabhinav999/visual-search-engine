import pandas as pd
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from PIL import Image
from src.config import PROCESSED_IMAGES_DIR, STYLES_CSV, INDEX_PATH, EMBEDDING_DIM
from src.encoder import ImageEncoder


def build_index():
    print("Loading metadata...")

    # read csv
    bad_lines = []
    def bad_line_logger(line):
        bad_lines.append(line)
        return None

    df = pd.read_csv(
        STYLES_CSV,
        engine="python",
        on_bad_lines=bad_line_logger
    )
    print(f"Loaded {len(df)} rows")
    print(f"Skipped {len(bad_lines)} malformed rows")

    # filter (match csv rows to actual images)
    print(f"Original CSV count: {len(df)}")

    valid_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying Files"):
        image_filename = str(row['id']) + ".jpg"
        image_path = PROCESSED_IMAGES_DIR / image_filename

        if image_path.exists():
            # Convert row to dict and add image path
            item =  row.to_dict()
            item['image_path'] = image_path
            valid_rows.append(item)

    df_clean = pd.DataFrame(valid_rows)
    print(f"Cleaned Data count: {len(df_clean)} (Dropped {len(df) - len(df_clean)} missing files)")

    # FAISS setup
    print("Initializing FAISS HNSW Index...")
    M = 32              # Number of connections per node 
    ef_construction = 40 # Depth of search during build
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, M)
    index.hnsw.efConstruction = ef_construction

    # PROCESSING LOOP
    print("Encoding Images ...")
    encoder = ImageEncoder()
    batch_size = 64
    vectors_buffer = []

    # Iterate through clean data
    for i, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Indexing"):
        try:
            # Load Image
            img_path = row['image_path']
            img = Image.open(img_path).convert('RGB')
            
            # Encode (returns 512-dim numpy array)
            vec = encoder.encode(img)
            vectors_buffer.append(vec)
            
            # Flush batch to FAISS
            if len(vectors_buffer) >= batch_size:
                # Convert list of arrays -> 2D Matrix (Batch_Size, 512)
                batch_matrix = np.array(vectors_buffer).astype('float32')
                index.add(batch_matrix)
                vectors_buffer = [] # Clear memory
                
        except Exception as e:
            print(f"\nError processing {row['id']}: {e}")

    # Add any remaining vectors
    if vectors_buffer:
        batch_matrix = np.array(vectors_buffer).astype('float32')
        index.add(batch_matrix)

    print(f"Index Built! Total vectors: {index.ntotal}")
    
    # Save ARTIFACTS and FAISS Graph
    faiss.write_index(index, str(INDEX_PATH))
    
    # Save the Metadata Mapping (ID -> Product Info)
    # We save the clean dataframe so index 0 maps to row 0
    metadata_path = str(INDEX_PATH).replace(".bin", "_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(df_clean, f)
    
    print("------------------------------------------------")
    print(f"✅ SAVED: {INDEX_PATH}")
    print(f"✅ SAVED: {metadata_path}")
    print("You are ready to search!")


if __name__ == '__main__':
    build_index()