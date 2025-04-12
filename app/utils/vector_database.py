import os
import numpy as np
import pandas as pd
import pickle
import json
import warnings

# Try to import FAISS, but handle the case where it's not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Using fallback numpy-based similarity search.")

class VectorDatabase:
    def __init__(self, dimension=None):
        """
        Initialize a vector database using FAISS if available, or fallback to numpy
        
        Args:
            dimension: Dimension of the embeddings to store (required for initialization)
        """
        self.index = None
        self.dimension = dimension
        self.metadata = []
        self.product_data = None
        self.embeddings = []  # For fallback implementation
    
    def create_index(self, embeddings=None):
        """Create a new index with the given dimension"""
        if embeddings is not None:
            # Convert to numpy array if it's a list
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            self.dimension = embeddings.shape[1]
        
        if self.dimension is None:
            raise ValueError("Dimension must be provided either during initialization or when creating index")
        
        if FAISS_AVAILABLE:
            # Create a flat L2 index (exact search)
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # If embeddings are provided, add them to the index
            if embeddings is not None and len(embeddings) > 0:
                self.add_embeddings(embeddings)
        else:
            # Fallback implementation
            self.index = "numpy_fallback"
            if embeddings is not None and len(embeddings) > 0:
                # Make sure embeddings is a numpy array
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                self.embeddings = embeddings.astype(np.float32)
            
        return self
    
    def add_embeddings(self, embeddings, metadata_list=None):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings to add
            metadata_list: List of metadata dictionaries for each embedding
        """
        if self.index is None:
            self.create_index(embeddings)
            return self
        
        # Ensure embeddings are in the correct format (float32)
        # Convert to numpy array if it's a list
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        embeddings = np.array(embeddings).astype(np.float32)
        
        if FAISS_AVAILABLE and self.index != "numpy_fallback":
            # Add embeddings to the FAISS index
            self.index.add(embeddings)
        else:
            # Fallback: store embeddings in memory
            if len(self.embeddings) == 0:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Store metadata if provided
        if metadata_list:
            self.metadata.extend(metadata_list)
        else:
            # Create empty metadata if not provided
            self.metadata.extend([{} for _ in range(len(embeddings))])
            
        return self
    
    def search(self, query_embedding, k=5):
        """
        Search for similar embeddings in the index
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of results with metadata
        """
        if self.index is None:
            raise ValueError("Index has not been created yet")
        
        # Ensure query_embedding is in the correct format
        # Convert to numpy array if it's a list
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        query_embedding = np.array(query_embedding).astype(np.float32)
        
        # Reshape if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        if FAISS_AVAILABLE and self.index != "numpy_fallback":
            distances, indices = self.index.search(query_embedding, k)
            distances, indices = distances[0], indices[0]
        else:
            # Fallback: calculate distances manually
            if len(self.embeddings) == 0:
                return []
                
            # Calculate L2 distances
            sq_query = np.sum(query_embedding**2, axis=1)
            sq_embeddings = np.sum(self.embeddings**2, axis=1)
            dot_product = np.dot(query_embedding, self.embeddings.T)
            distances = sq_query.reshape(-1, 1) + sq_embeddings - 2 * dot_product
            distances = distances[0]
            
            # Get indices of k smallest distances
            if len(distances) <= k:
                indices = np.arange(len(distances))
            else:
                indices = np.argsort(distances)[:k]
                distances = distances[indices]
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices):
            if idx >= 0 and idx < len(self.metadata):  # Ensure valid index
                distance = float(distances[i]) if i < len(distances) else float('inf')
                result = {
                    "distance": distance,
                    "index": int(idx),
                    "metadata": self.metadata[idx]
                }
                
                # Add full product data if available
                if self.product_data is not None and 'id' in self.metadata[idx]:
                    product_id = self.metadata[idx]['id']
                    product = self.product_data[self.product_data['id'] == product_id]
                    if not product.empty:
                        result["product"] = product.iloc[0].to_dict()
                        
                results.append(result)
                
        return results
    
    def load_product_data(self, products_df):
        """Load product data for enriching search results"""
        self.product_data = products_df
        return self
    
    def save(self, directory, prefix="vector_db"):
        """Save the vector database to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        if FAISS_AVAILABLE and self.index != "numpy_fallback":
            # Save FAISS index
            index_path = os.path.join(directory, f"{prefix}_index.faiss")
            faiss.write_index(self.index, index_path)
        else:
            # Save embeddings as numpy array
            embeddings_path = os.path.join(directory, f"{prefix}_embeddings.npy")
            np.save(embeddings_path, self.embeddings)
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{prefix}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config_path = os.path.join(directory, f"{prefix}_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "dimension": self.dimension,
                "faiss_available": FAISS_AVAILABLE
            }, f)
        
        return self
    
    @classmethod
    def load(cls, directory, prefix="vector_db"):
        """Load a vector database from disk"""
        # Load config
        config_path = os.path.join(directory, f"{prefix}_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(dimension=config.get("dimension"))
        
        # Load index
        if FAISS_AVAILABLE and config.get("faiss_available", False):
            # Load FAISS index
            index_path = os.path.join(directory, f"{prefix}_index.faiss")
            if os.path.exists(index_path):
                instance.index = faiss.read_index(index_path)
            else:
                # If FAISS index doesn't exist, try to load embeddings
                embeddings_path = os.path.join(directory, f"{prefix}_embeddings.npy")
                if os.path.exists(embeddings_path):
                    instance.embeddings = np.load(embeddings_path)
                    instance.index = "numpy_fallback"
        else:
            # Load embeddings as numpy array
            embeddings_path = os.path.join(directory, f"{prefix}_embeddings.npy")
            if os.path.exists(embeddings_path):
                instance.embeddings = np.load(embeddings_path)
                instance.index = "numpy_fallback"
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{prefix}_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            instance.metadata = pickle.load(f)
        
        return instance 