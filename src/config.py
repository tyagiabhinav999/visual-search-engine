from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_IMAGES_DIR = DATA_DIR / "processed" / "images"
STYLES_CSV = DATA_DIR / "styles.csv"
INDEX_PATH = DATA_DIR / "faiss_index.bin"

# Model Settings
# We use a small, efficient CLIP model supported by ONNX
MODEL_ID = "openai/clip-vit-base-patch32"
ONNX_PATH = BASE_DIR / "models" / "onnx" # Where we save the compiled model
EMBEDDING_DIM = 512