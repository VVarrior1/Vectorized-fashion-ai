import os
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import openai
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, text_model_name="all-MiniLM-L6-v2", use_openai=False):
        """
        Initialize embedding models for text and images
        
        Args:
            text_model_name: Name of the sentence-transformers model for text embeddings
            use_openai: Whether to use OpenAI for text embeddings (requires API key)
        """
        self.use_openai = use_openai
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize text embedding model
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required when use_openai=True")
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            self.text_model = SentenceTransformer(text_model_name).to(self.device)
        
        # Initialize image embedding model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def get_text_embedding(self, text):
        """Generate embedding for a text string"""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding)
        return np.array(self.text_model.encode(text))
    
    def get_image_embedding(self, image_path):
        """Generate embedding for an image from its file path"""
        try:
            if not os.path.exists(image_path):
                return np.zeros(512)
                
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs).cpu().numpy()[0]
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(512)
    
    def get_combined_embedding(self, text, image_path, text_weight=0.5):
        """Generate a combined embedding from text and image"""
        text_emb = self.get_text_embedding(text)
        image_emb = self.get_image_embedding(image_path)
        
        if text_emb.shape != image_emb.shape:
            text_emb = text_emb / np.linalg.norm(text_emb)
            image_emb = image_emb / np.linalg.norm(image_emb)
            combined = np.concatenate([text_emb, image_emb])
        else:
            combined = text_weight * text_emb + (1 - text_weight) * image_emb
            
        return combined / np.linalg.norm(combined)
    
    def batch_get_text_embeddings(self, texts):
        """Generate embeddings for a list of texts"""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return np.array([item.embedding for item in response.data])
        return np.array(self.text_model.encode(texts))
    
    def batch_get_image_embeddings(self, image_paths):
        """Generate embeddings for a list of image paths"""
        return np.stack([self.get_image_embedding(path) for path in image_paths]) 