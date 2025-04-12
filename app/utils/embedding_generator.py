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
        
        if use_openai:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when use_openai=True")
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            # Load text embedding model
            self.text_model = SentenceTransformer(text_model_name)
        
        # Load CLIP model for image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        if not use_openai:
            self.text_model.to(self.device)
        
        # Default CLIP embedding size
        self.default_clip_dimension = 512
    
    def get_text_embedding(self, text):
        """Generate embedding for a text string"""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding)
        else:
            # Ensure we return a numpy array
            embedding = self.text_model.encode(text)
            return np.array(embedding)
    
    def get_image_embedding(self, image_path):
        """Generate embedding for an image from its file path"""
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                return np.zeros(self.default_clip_dimension)
                
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize and convert to numpy
            image_embedding = image_features.cpu().numpy()[0]
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            
            return np.array(image_embedding)
        except Exception as e:
            print(f"Error generating image embedding for {image_path}: {str(e)}")
            return np.zeros(self.default_clip_dimension)
    
    def get_combined_embedding(self, text, image_path, text_weight=0.5):
        """Generate a combined embedding from text and image"""
        text_embedding = self.get_text_embedding(text)
        image_embedding = self.get_image_embedding(image_path)
        
        # Ensure both are numpy arrays
        text_embedding = np.array(text_embedding)
        image_embedding = np.array(image_embedding)
        
        # Ensure embeddings have the same dimensions
        if text_embedding.shape != image_embedding.shape:
            # Use simple concatenation and normalize
            text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)
            image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
            combined = np.concatenate([text_embedding_norm, image_embedding_norm])
            return np.array(combined / np.linalg.norm(combined))
        
        # Weighted combination
        combined = text_weight * text_embedding + (1 - text_weight) * image_embedding
        return np.array(combined / np.linalg.norm(combined))
    
    def batch_get_text_embeddings(self, texts):
        """Generate embeddings for a list of texts"""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        else:
            embeddings = self.text_model.encode(texts)
            return np.array(embeddings)
    
    def batch_get_image_embeddings(self, image_paths):
        """Generate embeddings for a list of image paths"""
        # First, get the first valid embedding to determine the shape
        embeddings = []
        embedding_size = None
        
        # Try to get a valid embedding to determine the shape
        for path in image_paths:
            embedding = self.get_image_embedding(path)
            if embedding is not None and not np.all(embedding == 0):
                embedding_size = embedding.shape[0]
                break
        
        # If no valid embedding was found, use default size
        if embedding_size is None:
            embedding_size = self.default_clip_dimension
            
        # Now process all images
        for image_path in image_paths:
            embedding = self.get_image_embedding(image_path)
            
            # Ensure embedding is not None and has the correct shape
            if embedding is None or embedding.shape[0] != embedding_size:
                # Create zero embedding with correct shape
                embedding = np.zeros(embedding_size)
                
            embeddings.append(embedding)
            
        # Convert list to numpy array
        # Make sure all embeddings have the same shape before creating array
        return np.stack(embeddings) 