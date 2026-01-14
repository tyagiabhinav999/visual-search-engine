import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from src.config import MODEL_ID, ONNX_PATH

class CLIPVisionTower(nn.Module):
    """
    A minimal wrapper for Production Export.
    It forces the model to return ONLY the projected embedding (512-dim).
    """
    def __init__(self, model_id):
        super().__init__()
        # Load the model with the projection head
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        # Freeze weights to ensure deterministic export
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        # strictly access .image_embeds, forces the ONNX tracer to include the linear projection layer (768 -> 512)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.image_embeds

class ImageEncoder:
    def __init__(self):
        self.onnx_folder = ONNX_PATH
        self.onnx_file = self.onnx_folder / "model.onnx"
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

        # Pipeline Check
        if not self.onnx_file.exists():
            print("Exporting Explicit Vision Tower...")
            self._export_model()
        else:
            print("Loading Optimized Vision Tower from disk...")

        # Load Inference Session (Raw ONNX Runtime)
        # We do not use 'optimum' wrappers here to avoid any hidden post-processing.
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.onnx_file), providers=providers)
        
        # Validation (Strict Production Check)
        self._validate_graph()

    def _export_model(self):
        # Create directory
        self.onnx_folder.mkdir(parents=True, exist_ok=True)

        # A. Initialize the Wrapper
        print("Loading PyTorch weights...")
        vision_tower = CLIPVisionTower(MODEL_ID)
        
        # Create Dummy Input (1, 3, 224, 224)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export
        print("Tracing computation graph...")
        torch.onnx.export(
            vision_tower,
            dummy_input,
            str(self.onnx_file),
            input_names=["pixel_values"],
            output_names=["image_embeds"],  # Naming the output explicitly
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"}
            },
            opset_version=18, 
            do_constant_folding=True
        )
        
        # Save Processor (for resizing logic)
        self.processor.save_pretrained(self.onnx_folder)
        print("Export Complete.")

    def _validate_graph(self):
        """Checks input/output names and shapes matches expectation."""
        inputs = [i.name for i in self.session.get_inputs()]
        outputs = [o.name for o in self.session.get_outputs()]
        
        print("Graph Signature:")
        print(f"   - Inputs: {inputs}")   # Expected: ['pixel_values']
        print(f"   - Outputs: {outputs}") # Expected: ['image_embeds']
        
        if "image_embeds" not in outputs[0]:
            # fail-safe, if graph doesn't return the embedding, crash early.
            raise RuntimeError(f"Critical Export Error: Output is named {outputs[0]}, expected 'image_embeds'")

    def encode(self, image: Image.Image):
        # Preprocess, return_tensors="np" gives us efficient numpy arrays
        inputs = self.processor(images=image, return_tensors="np")
        
        # Inference, map strictly: Graph Input 'pixel_values' 
        onnx_inputs = {"pixel_values": inputs["pixel_values"]}
        
        # Run
        outputs = self.session.run(None, onnx_inputs)
        
        # Post-Process
        image_features = outputs[0] # (1, 512)
        
        # L2 Normalize
        norm = np.linalg.norm(image_features, axis=-1, keepdims=True)
        return (image_features / norm).flatten()

if __name__ == "__main__":
    # Unit Test
    img = Image.new("RGB", (224, 224), color="red")
    encoder = ImageEncoder()
    vec = encoder.encode(img)
    
    print(f"Vector Shape: {vec.shape}")
    
    if vec.shape == (512,):
        print("SUCCESS: Production Graph Verified (512-dim).")
    else:
        print(f"❌ FAILURE: Got {vec.shape}")