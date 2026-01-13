from PIL import Image
import numpy as np
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import CLIPProcessor
from src.config import MODEL_ID, ONNX_PATH

class ImageEncoder:
    def __init__(self):
        print(f'Loading CLIP via ONNX Runtime: {MODEL_ID}')

        # load processor
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

        # load onnx model
        # we use FeatureExtraction because it respects CLIPModel architecture which includes the projection head.
        if ONNX_PATH.exists() and (ONNX_PATH / 'model.onnx').exists():
            print("Loading existing ONNX model from disk...")
            self.model = ORTModelForFeatureExtraction.from_pretrained(ONNX_PATH)
        else:
            print("Exporting model to ONNX...")
            # create 'model.onnx' in the folder
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                MODEL_ID,
                export=True
            )
            self.model.save_pretrained(ONNX_PATH)
            self.processor.save_pretrained(ONNX_PATH)

        print("ONNX Model loaded successfully!")




