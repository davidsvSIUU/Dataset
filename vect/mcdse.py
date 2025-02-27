# mcdse.py
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from huggingface_hub import snapshot_download
from typing import List, Union, Optional
from PIL import Image
import os
import tempfile
from tqdm import tqdm
import logging
import math

# Remplacer l'import relatif par la définition directe de BaseEmbeddingModel
class BaseEmbeddingModel:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device

    def cleanup(self):
        pass
        
logger = logging.getLogger(__name__)

class MCDSEModel(BaseEmbeddingModel):
    """
    MCDSE (Multilingual Contrastive Dense Screen Embedding) model implementation using mcdse-2b-v1.
    """
    
    def __init__(self, 
                model_path: str = "marco/mcdse-2b-v1",
                device: Optional[str] = None,
                batch_size: int = 1,
                use_flash_attention: bool = False,
                dimension: int = 768,
                use_fake: bool = True,  # Nouveau paramètre
                **kwargs):
        """
        Initialize the MCDSE model.
        """
        super().__init__(model_path, device)
        self.use_fake = use_fake  # ← Désactive le chargement du modèle

        if not self.use_fake:
            # Logique de chargement du modèle (à garder pour un usage réel)
            if device is None:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                self.dimension = dimension
        
        # Définir explicitement le device après la logique
        self.device = device        
        # Initialize temp_dir first
        self.temp_dir = tempfile.mkdtemp()
        self.model = None
        self.processor = None
        
        try:
            # Set up model configuration
            model_kwargs = {
                "attn_implementation": "eager"  # Force le mode CPU
            }

            if self.device == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                })
            if use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Download the model
            snapshot_download(model_path)

            # Set dimension (supports MRL: 256 to 1536)
            self.dimension = min(max(256, dimension), 1536)
            
            # Initialize model and processor
            min_pixels = 1 * 28 * 28
            max_pixels = 960 * 28 * 28
            model_kwargs["attn_implementation"] = "eager"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            # Set padding configuration
            self.model.padding_side = "left"
            self.processor.tokenizer.padding_side = "left"
            
            # Set prompts
            self.document_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
            self.query_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"
            
            # Set model parameters
            self.batch_size = batch_size
            
            logger.info(f"Initialized MCDSE model on {device}")
            logger.info(f"Model loaded with batch_size={batch_size}, dimension={self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCDSE model: {str(e)}")
            self.cleanup()
            raise

    def _smart_resize(self, height: int, width: int) -> tuple[int, int]:
        """Calculate optimal dimensions for image resizing."""
        min_pixels = 1 * 28 * 28
        max_pixels = 960 * 28 * 28
        
        h_bar = max(28, round(height / 28) * 28)
        w_bar = max(28, round(width / 28) * 28)
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / 28) * 28
            w_bar = math.floor(width / beta / 28) * 28
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / 28) * 28
            w_bar = math.ceil(width * beta / 28) * 28
        return h_bar, w_bar

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to optimal dimensions."""
        new_size = self._smart_resize(image.height, image.width)
        return image.resize(new_size)

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode text queries into embeddings."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not properly initialized")
            
        try:
            all_embeddings = []
            dummy_image = Image.new('RGB', (56, 56))
            
            for i in tqdm(range(0, len(queries), self.batch_size),
                         desc="Encoding queries"):
                batch_queries = queries[i:i + self.batch_size]
                
                inputs = self.processor(
                    text=[self.query_prompt % x for x in batch_queries],
                    images=[dummy_image for _ in batch_queries],
                    videos=None,
                    padding='longest',
                    return_tensors='pt'
                ).to(self.device)
                
                cache_position = torch.arange(0, len(batch_queries))
                inputs = self.model.prepare_inputs_for_generation(
                    **inputs, cache_position=cache_position, use_cache=False)
                
                with torch.no_grad():
                    output = self.model(
                        **inputs,
                        return_dict=True,
                        output_hidden_states=True
                    )
                
                embeddings = output.hidden_states[-1][:, -1]
                normalized_embeddings = torch.nn.functional.normalize(
                    embeddings[:, :self.dimension], p=2, dim=-1)
                all_embeddings.append(normalized_embeddings)
            
            return torch.cat(all_embeddings, dim=0)
            
        except Exception as e:
            logger.error(f"Error encoding queries: {str(e)}")
            raise

    def encode_documents(self, documents: List[Image.Image]) -> torch.Tensor:
        """Encode document images into embeddings."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not properly initialized")
            
        try:
            all_embeddings = []
            
            for i in tqdm(range(0, len(documents), self.batch_size),
                         desc="Encoding documents"):
                batch_docs = documents[i:i + self.batch_size]
                
                # Resize images
                resized_docs = [self._resize_image(img) for img in batch_docs]
                
                inputs = self.processor(
                    text=[self.document_prompt] * len(batch_docs),
                    images=resized_docs,
                    videos=None,
                    padding='longest',
                    return_tensors='pt'
                ).to(self.device)
                
                cache_position = torch.arange(0, len(batch_docs))
                inputs = self.model.prepare_inputs_for_generation(
                    **inputs, cache_position=cache_position, use_cache=False)
                
                with torch.no_grad():
                    output = self.model(
                        **inputs,
                        return_dict=True,
                        output_hidden_states=True
                    )
                
                embeddings = output.hidden_states[-1][:, -1]
                normalized_embeddings = torch.nn.functional.normalize(
                    embeddings[:, :self.dimension], p=2, dim=-1)
                all_embeddings.append(normalized_embeddings)
            
            return torch.cat(all_embeddings, dim=0)
            
        except Exception as e:
            logger.error(f"Error encoding documents: {str(e)}")
            raise

    def fake_encode_documents(self, documents: List[Image.Image]) -> torch.Tensor:
        """Create random embeddings for testing."""
        return torch.rand(len(documents), self.dimension)
    
    def fake_encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Create random embeddings for testing."""
        return torch.rand(len(queries), self.dimension)

    def compute_similarity(self, 
                        query_embedding: torch.Tensor,
                        document_embedding: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and document embeddings."""
        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding.unsqueeze(1)
        if len(document_embedding.shape) == 2:
            document_embedding = document_embedding.unsqueeze(0)
        
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding,
            document_embedding,
            dim=-1
        )
        
        return similarities

    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()